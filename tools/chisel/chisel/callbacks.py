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
from typing import Iterator

from _ttmlir_runtime import runtime as tt_runtime
from _ttmlir_runtime.binary import Binary
from _ttmlir_runtime.runtime import CallbackContext, OpContext, TensorRef
from ttmlir.ir import Value

from golden import GoldenMapTensor

from .context import ChiselContext, get_instance
from .exceptions import IrRuntimeMismatch
from .executor import execute_golden_with_ssa_inputs
from .op_configs import ChiselOpConfig
from .ops import (
    SSAName,
    get_inplace_vals,
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
) -> None:
    """Shape/dtype + PCC for one (golden, device) pair under `mode`."""
    check_shape_dtype(op, "mlir_vs_golden", mlir_value, golden_out)
    if skip_pcc:
        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="numerics",
                ssa=ssa,
                payload=SkippedNumericsPayload(mode=mode),
            )
        )
        return
    check_numerics(ctx, op, ssa, golden_out, device_tensor, mode=mode)


def _get_inplace_input_refs(
    op, input_refs: list[TensorRef], asm_state
) -> list[tuple[Value, TensorRef]]:
    """Pair each in-place mutated tensor operand with its runtime TensorRef."""
    inplace_vals = get_inplace_vals(op)
    if not inplace_vals:
        return []
    inplace_ssas = {val.get_name(asm_state) for val in inplace_vals}
    out: list[tuple[Value, TensorRef]] = []
    for mlir_in, ref in zip(get_op_inputs(op), input_refs):
        if mlir_in.get_name(asm_state) in inplace_ssas:
            out.append((mlir_in, ref))
    return out


def _evict_inplace_no_golden(ctx: ChiselContext) -> None:
    """For each mutated tensor operand on `ctx.op`, drop both pools and record."""
    op = ctx.op
    asm_state = ctx.asm_state
    golden_pool = ctx.golden_tensor_pool
    for val in get_inplace_vals(op):
        ssa = val.get_name(asm_state)
        golden_pool.pop(ssa, None)
        invalidate_device_cache(ctx, ssa)
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
        # Drop pooled goldens for any in-place mutated operand: the device
        # tensor was just modified but we have no golden to track it, so a
        # stale pool entry would produce false PCC failures on later ops.
        _evict_inplace_no_golden(ctx)
        return

    if not (ctx.checks_config.isolation or ctx.checks_config.accumulation):
        return

    op = ctx.op
    asm_state = ctx.asm_state

    mlir_op_outputs = get_op_outputs(op)
    inplace_refs = _get_inplace_input_refs(op, ctx.input_refs, asm_state)
    if not mlir_op_outputs and not inplace_refs:
        return

    modes: list[NumericsMode] = []
    if ctx.checks_config.isolation:
        modes.append(NumericsMode.ISOLATED)
    if ctx.checks_config.accumulation:
        modes.append(NumericsMode.ACCUMULATED)

    entries: list[tuple[Value, TensorRef]] = list(
        zip(mlir_op_outputs, ctx.output_refs, strict=True)
    )
    entries.extend(inplace_refs)

    # Invalidate the device cache for every entry, then pull each device tensor
    # once: outputs are freshly produced and in-place operands were just mutated,
    # so any cached host copy is stale. The mode loop below reuses these.
    entry_ssas = [mlir_value.get_name(asm_state) for mlir_value, _ in entries]
    for ssa in entry_ssas:
        invalidate_device_cache(ctx, ssa)
    device_tensors = [
        _validate_and_retrieve_tensor(ctx, mlir_value, tensor_ref)
        for mlir_value, tensor_ref in entries
    ]

    for mode in modes:
        ssa_inputs = (
            ctx.stashed_inputs
            if mode is NumericsMode.ISOLATED
            else ctx.golden_tensor_pool
        )
        all_outs = execute_golden_with_ssa_inputs(op, ssa_inputs, asm_state)

        for idx, (mlir_value, _tensor_ref) in enumerate(entries):
            ssa = entry_ssas[idx]
            golden_out = all_outs[idx]
            device_tensor = device_tensors[idx]
            _emit_pcc(
                ctx,
                op,
                ssa,
                mlir_value,
                golden_out,
                device_tensor,
                mode=mode,
                skip_pcc=config.skip_pcc,
            )
            if mode is NumericsMode.ISOLATED:
                continue

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
