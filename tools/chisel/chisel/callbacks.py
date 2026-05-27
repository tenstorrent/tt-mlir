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
from .utils import get_op_asm, retrieve_tensor
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
    """Validate IR vs ref shape/dtype, pull tensor from device, revalidate."""
    op = ctx.op
    check_shape_dtype(op, "mlir_vs_tensor_ref", mlir_value, rt_tensor_ref)
    tensor = retrieve_tensor(ctx.rt_program_context, rt_tensor_ref, ctx.mesh_shape)
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
        # TODO(ndrakulic): Right now we are pulling input device tensors to host potentially multiple times
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
    """Shape/dtype + PCC for one (golden, device) pair under `mode`."""
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
    """For each mutated tensor operand on `ctx.op`, drop the golden and record."""
    op = ctx.op
    asm_state = ctx.asm_state
    golden_pool = ctx.golden_tensor_pool
    for role, ssa, _ref in get_inplace_input_refs(op, ctx.input_refs, asm_state):
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
    """Run isolation + accumulation goldens; shape/dtype + PCC each output."""
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

    modes: list[NumericsMode] = []
    if ctx.checks_config.isolation:
        modes.append(NumericsMode.ISOLATED)
    if ctx.checks_config.accumulation:
        modes.append(NumericsMode.ACCUMULATED)

    # Validate SSA outputs and in-place mutated operands with one loop.
    result_names = getattr(type(op), "RESULT_NAMES", None) or []
    entries: list[tuple[Optional[str], Value, TensorRef]] = [
        (result_names[i] if i < len(result_names) else None, mlir_out, ref)
        for i, (mlir_out, ref) in enumerate(
            zip(mlir_op_outputs, ctx.output_refs, strict=True)
        )
    ]
    if inplace_refs:
        ssa_to_value = {inp.get_name(asm_state): inp for inp in get_op_inputs(op)}
        entries.extend(
            (role, ssa_to_value[ssa], ref) for role, ssa, ref in inplace_refs
        )

    # Pull every entry's device tensor once; the device-side bytes don't
    # change between iso and accum golden runs (goldens run host-side),
    # so we share the retrieved tensor across all enabled modes.
    device_tensors = [
        _validate_and_retrieve_tensor(ctx, mlir_value, tensor_ref)
        for _, mlir_value, tensor_ref in entries
    ]

    # Each enabled mode runs its golden once; the result is aligned with
    # [SSA outputs ..., in-place operands ...]. Accumulation re-publishes
    # to the golden pool below: execute_golden_from_pool writes SSA outputs
    # back itself, but in-place operands are returned without being stored,
    # so we need to push them in to keep the chain coherent.
    for mode in modes:
        if mode is NumericsMode.ISOLATED:
            all_outs = execute_golden_with_ssa_inputs(
                op, ctx.stashed_inputs, asm_state
            )
        else:
            all_outs = execute_golden_from_pool(
                op, ctx.golden_tensor_pool, asm_state
            )

        for idx, (role, mlir_value, _tensor_ref) in enumerate(entries):
            ssa = mlir_value.get_name(asm_state)
            golden_out = all_outs[idx]
            _emit_pcc(
                ctx,
                op,
                ssa,
                mlir_value,
                golden_out,
                device_tensors[idx],
                mode=mode,
                skip_pcc=config.skip_pcc,
                role=role,
            )
            if mode is NumericsMode.ACCUMULATED:
                ctx.golden_tensor_pool[ssa] = golden_out


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
