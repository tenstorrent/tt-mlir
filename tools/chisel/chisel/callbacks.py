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
from typing import Dict, Iterator, List

from _ttmlir_runtime import runtime as tt_runtime
from _ttmlir_runtime.binary import Binary
from _ttmlir_runtime.runtime import CallbackContext, OpContext, TensorRef
from ttmlir.ir import OpView, Value

from golden import GoldenMapTensor

from .context import ChiselContext, get_instance
from .exceptions import IrRuntimeMismatch
from .executor import build_role_keyed_inputs, execute_golden
from .op_configs import ChiselOpConfig
from .ops import SSAName, get_op_inputs, get_op_outputs
from .report import ChiselRecord, NoGoldenPayload, SkippedNumericsPayload
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
    op = ctx.op
    check_shape_dtype(op, "mlir_vs_tensor_ref", mlir_value, rt_tensor_ref)
    tensor = retrieve_tensor(ctx.rt_program_context, rt_tensor_ref)
    check_shape_dtype(op, "mlir_vs_runtime_tensor", mlir_value, tensor)
    return tensor


def _run_isolation_golden(
    op: OpView,
    mlir_outputs: List[Value],
    asm_state,
    ssa_inputs: Dict[SSAName, GoldenMapTensor],
) -> List[GoldenMapTensor]:
    role_inputs = build_role_keyed_inputs(op, ssa_inputs, asm_state)
    iso_outs = execute_golden(op, role_inputs)
    for mlir_output, iso_out in zip(mlir_outputs, iso_outs, strict=True):
        check_shape_dtype(op, "mlir_vs_golden", mlir_output, iso_out)
    return iso_outs


@chisel_safe
def _default_pre_op(ctx: ChiselContext, config: ChiselOpConfig) -> None:
    """Stash a host copy of every device input so POST can drive the golden."""
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

    mlir_op_inputs = get_op_inputs(op)
    for mlir_input, rt_tensor_ref in zip(mlir_op_inputs, ctx.input_refs, strict=True):
        # TODO(ndrakulic): Right now we are pulling input device tensors to host potentially multiple times
        tensor = _validate_and_retrieve_tensor(ctx, mlir_input, rt_tensor_ref)
        ctx.stashed_inputs[mlir_input.get_name(asm_state)] = tensor


@chisel_safe
def _default_post_op(ctx: ChiselContext, config: ChiselOpConfig) -> None:
    """Run the golden, validate shapes/dtypes, and PCC-check each output."""
    if config.no_golden:
        return

    skip_isolated_pcc = config.skip_isolated_pcc
    op = ctx.op
    asm_state = ctx.asm_state

    mlir_op_outputs = get_op_outputs(op)
    if not mlir_op_outputs:
        return

    iso_outs = _run_isolation_golden(op, mlir_op_outputs, asm_state, ctx.stashed_inputs)

    for mlir_output, output_ref, iso_out in zip(
        mlir_op_outputs, ctx.output_refs, iso_outs, strict=True
    ):
        device_tensor = _validate_and_retrieve_tensor(ctx, mlir_output, output_ref)
        ssa = mlir_output.get_name(asm_state)
        if skip_isolated_pcc:
            ctx.write_record(
                ChiselRecord(
                    op=op.name,
                    check="numerics",
                    ssa=ssa,
                    payload=SkippedNumericsPayload(),
                )
            )
            continue

        check_numerics(ctx, op, ssa, iso_out, device_tensor)


def run_op_callback(
    rt_binary: Binary,
    rt_program_context: CallbackContext,
    rt_op_context: OpContext,
    *,
    phase: CallbackPhase,
) -> None:
    """Dispatch the PRE or POST handler for the current op."""
    ctx = get_instance()
    with _op_callback(ctx, rt_binary, rt_program_context, rt_op_context, phase=phase):
        config = ctx.get_op_config(ctx.op)
        if phase is CallbackPhase.PRE:
            pre_succeeded = _default_pre_op(ctx, config)
            ctx.pre_failed = not pre_succeeded
        else:
            # PRE recorded a failure - skip POST to avoid cascading into a
            # chisel_bug from incomplete state.
            if ctx.pre_failed:
                return
            _default_post_op(ctx, config)
