# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-op chisel callbacks driven by the ttmlir runtime debug hooks.

Validators raise ChiselFailure on the first failed shape/dtype/PCC check.
chisel_safe wraps the per-op handler and turns the failure into a structured
record - any subsequent checks on the same op are skipped. Other exceptions
are recorded as chisel_bug.
"""
import logging
from contextlib import contextmanager
from enum import Enum
from typing import Iterator, Optional

from _ttmlir_runtime import runtime as tt_runtime
from _ttmlir_runtime.binary import Binary
from _ttmlir_runtime.runtime import CallbackContext, OpContext, TensorRef
from ttmlir.ir import Value

from golden import GoldenMapTensor

from .context import ChiselContext, get_instance
from .exceptions import IrRuntimeMismatch
from .executor import (
    execute_golden_from_pool,
    execute_golden_with_ssa_inputs,
)
from .op_configs import ChiselOpConfig
from .ops import (
    SSAName,
    get_inplace_input_refs,
    get_op_inputs,
    get_op_outputs,
)
from .report import (
    ChiselRecord,
    GoldenEvictedPayload,
    GoldenPromotedPayload,
    NoGoldenPayload,
    NumericsMode,
    SkippedNumericsPayload,
)
from .safety import chisel_safe
from .utils import cached_retrieve_tensor, get_op_asm, invalidate_device_cache
from .validators import check_numerics, check_shape_dtype

logger = logging.getLogger("chisel")


class CallbackPhase(Enum):
    PRE = "pre"
    POST = "post"


@contextmanager
def _op_callback(
    ctx: ChiselContext,
    rt_binary: Binary,
    rt_program_context: CallbackContext,
    rt_op_context: OpContext,
    *,
    phase: CallbackPhase,
) -> Iterator[None]:
    """Open callback scope (and, on PRE, op scope); close them symmetrically."""
    program = ctx.begin_callback(rt_binary, rt_program_context, rt_op_context)
    try:
        if phase is CallbackPhase.PRE:
            program.begin_op()
        yield
    finally:
        if phase is CallbackPhase.POST:
            program.end_op()
        ctx.end_callback()


def _assert_op_matches_runtime(ctx: ChiselContext) -> None:
    """Raise IrRuntimeMismatch if chisel and the runtime point at different ops."""
    rt_debug = tt_runtime.get_op_debug_str(ctx.rt_op_context)
    op = ctx.op
    if rt_debug.strip() == get_op_asm(op).strip():
        return
    raise IrRuntimeMismatch(op, "ir_vs_runtime_op", rt_debug)


def _validate_and_retrieve_tensor(
    ctx: ChiselContext, mlir_value: Value, rt_tensor_ref: TensorRef
) -> GoldenMapTensor:
    """Validate IR vs ref shape/dtype, fetch tensor (via device cache), revalidate."""
    op = ctx.op
    check_shape_dtype(op, "mlir_vs_tensor_ref", mlir_value, rt_tensor_ref)
    ssa = mlir_value.get_name(ctx.asm_state)
    tensor = cached_retrieve_tensor(ctx, ssa, rt_tensor_ref, ctx.mesh_shape)
    check_shape_dtype(op, "mlir_vs_runtime_tensor", mlir_value, tensor)
    return tensor


@chisel_safe
def _default_pre_op(ctx: ChiselContext, config: ChiselOpConfig) -> None:
    """Stash host copies of device inputs and seed function args into the pool."""
    op = ctx.op
    if config.no_golden:
        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="golden_not_implemented",
                payload=NoGoldenPayload(),
            )
        )
        return

    _assert_op_matches_runtime(ctx)
    asm_state = ctx.asm_state
    pool = ctx.golden_tensor_pool

    mlir_op_inputs = get_op_inputs(op)
    for mlir_input, rt_tensor_ref in zip(mlir_op_inputs, ctx.input_refs, strict=True):
        tensor = _validate_and_retrieve_tensor(ctx, mlir_input, rt_tensor_ref)
        ssa = mlir_input.get_name(asm_state)
        ctx.stashed_inputs[ssa] = tensor
        # Seed only SSAs not yet produced by a prior op's golden (i.e. function args).
        if ssa in pool:
            continue

        pool[ssa] = tensor
        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="golden_promoted",
                ssa=ssa,
                payload=GoldenPromotedPayload(),
            )
        )


def _emit_pcc(
    ctx: ChiselContext,
    op,
    ssa: SSAName,
    mlir_value: Value,
    golden_out: GoldenMapTensor,
    device_tensor: GoldenMapTensor,
    *,
    mode: NumericsMode,
    skip_pcc: bool,
    role: Optional[str] = None,
) -> None:
    """Shape/dtype + PCC for one (golden, device) pair under `mode`.

    `role` is set for in-place mutated-operand comparisons; it's threaded
    into the emitted record so consumers can distinguish SSA outputs from
    mutated operands.
    """
    check_shape_dtype(op, "mlir_vs_golden", mlir_value, golden_out)
    if skip_pcc:
        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="numerics",
                ssa=ssa,
                payload=SkippedNumericsPayload(mode=mode, role=role),
            )
        )
        return
    check_numerics(ctx, op, ssa, golden_out, device_tensor, mode=mode, role=role)


def _evict_inplace_no_golden(ctx: ChiselContext) -> None:
    """For each mutated tensor operand on `ctx.op`, drop both pools and record.

    Called from `_default_post_op` when the op has no golden but the IR
    declares MemWrite effects. The on-device contents have diverged from
    any cached host copy and from any prior golden, so chisel can't
    reason about this SSA anymore.
    """
    op = ctx.op
    asm_state = ctx.asm_state
    golden_pool = ctx.golden_tensor_pool
    device_pool = ctx.device_tensor_pool
    for role, ssa, _ref in get_inplace_input_refs(op, ctx.input_refs, asm_state):
        device_pool.pop(ssa, None)
        golden_pool.pop(ssa, None)
        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="golden_evicted",
                ssa=ssa,
                payload=GoldenEvictedPayload(),
            )
        )


@chisel_safe
def _default_post_op(ctx: ChiselContext, config: ChiselOpConfig) -> None:
    """Run isolation + accumulation goldens; shape/dtype + PCC each output.

    For ops with no golden but IR-declared MemWrite effects, evict each
    mutated SSA from both pools (see `_evict_inplace_no_golden`).
    """
    if config.no_golden:
        # IR-driven dispatch: only no-golden ops that mutate operands need
        # the eviction sweep; pure no-golden ops fall through to return.
        _evict_inplace_no_golden(ctx)
        return

    op = ctx.op
    asm_state = ctx.asm_state

    mlir_op_outputs = get_op_outputs(op)
    inplace_refs = get_inplace_input_refs(op, ctx.input_refs, asm_state)
    if not mlir_op_outputs and not inplace_refs:
        return

    # Goldens return ssa_count + n_inplace tensors. Run iso/accum once each;
    # below we split each result list into (ssa_outs, inplace_outs).
    ssa_count = len(mlir_op_outputs)

    if ctx.checks_config.isolation:
        iso_all = execute_golden_with_ssa_inputs(op, ctx.stashed_inputs, asm_state)
        iso_ssa_outs = iso_all[:ssa_count]
        iso_inplace_outs = iso_all[ssa_count:]
    else:
        iso_ssa_outs = [None] * ssa_count
        iso_inplace_outs = [None] * len(inplace_refs)

    if ctx.checks_config.accumulation:
        accum_all = execute_golden_from_pool(op, ctx.golden_tensor_pool, asm_state)
        accum_ssa_outs = accum_all[:ssa_count]
        accum_inplace_outs = accum_all[ssa_count:]
    else:
        accum_ssa_outs = [None] * ssa_count
        accum_inplace_outs = [None] * len(inplace_refs)

    # Validate SSA outputs and in-place mutated operands with one loop. Each
    # entry is (role, mlir_value, tensor_ref, iso_out, accum_out). `role`
    # comes from the OpView's RESULT_NAMES for SSA outputs and from
    # OPERAND_NAMES (via get_inplace_input_refs) for in-place operands.
    result_names = getattr(type(op), "RESULT_NAMES", None) or []
    entries = [
        (
            result_names[i] if i < len(result_names) else None,
            mlir_out,
            ref,
            iso,
            accum,
        )
        for i, (mlir_out, ref, iso, accum) in enumerate(
            zip(
                mlir_op_outputs,
                ctx.output_refs,
                iso_ssa_outs,
                accum_ssa_outs,
                strict=True,
            )
        )
    ]
    if inplace_refs:
        ssa_to_value = {
            inp.get_name(asm_state): inp for inp in get_op_inputs(op)
        }
        entries.extend(
            (role, ssa_to_value[ssa], ref, iso, accum)
            for (role, ssa, ref), iso, accum in zip(
                inplace_refs, iso_inplace_outs, accum_inplace_outs, strict=True
            )
        )

    for role, mlir_value, tensor_ref, iso_out, accum_out in entries:
        ssa = mlir_value.get_name(asm_state)
        # For in-place operands the cached PRE copy is stale; for SSA
        # outputs the SSA is freshly produced and unlikely to be cached.
        # Invalidate unconditionally - it's a no-op when absent.
        invalidate_device_cache(ctx, ssa)
        device_tensor = _validate_and_retrieve_tensor(ctx, mlir_value, tensor_ref)

        if iso_out is not None:
            _emit_pcc(
                ctx,
                op,
                ssa,
                mlir_value,
                iso_out,
                device_tensor,
                mode=NumericsMode.ISOLATED,
                skip_pcc=config.skip_pcc,
                role=role,
            )

        if accum_out is not None:
            _emit_pcc(
                ctx,
                op,
                ssa,
                mlir_value,
                accum_out,
                device_tensor,
                mode=NumericsMode.ACCUMULATED,
                skip_pcc=config.skip_pcc,
                role=role,
            )
            # Keep the program chain coherent: write the (possibly mutated)
            # tensor back into the golden pool. For SSA outputs
            # execute_golden_from_pool already wrote this value; for
            # in-place operands the executor returned the tensor but did
            # not store it, so we do it here. The overwrite is idempotent
            # in the SSA-output case.
            ctx.golden_tensor_pool[ssa] = accum_out


def run_op_callback(
    rt_binary: Binary,
    rt_program_context: CallbackContext,
    rt_op_context: OpContext,
    *,
    phase: CallbackPhase,
) -> None:
    """Dispatch the per-op pre/post handler (config override falls back to default)."""
    ctx = get_instance()
    with _op_callback(ctx, rt_binary, rt_program_context, rt_op_context, phase=phase):
        config = ctx.get_op_config(ctx.op)
        if phase is CallbackPhase.PRE:
            pre_fn = config.pre_op or _default_pre_op
            pre_succeeded = pre_fn(ctx, config)
            ctx.pre_failed = pre_succeeded is False
        else:
            # PRE recorded a failure - skip POST to avoid cascading into a
            # chisel_bug from incomplete state.
            if ctx.pre_failed:
                return
            post_fn = config.post_op or _default_post_op
            post_fn(ctx, config)
