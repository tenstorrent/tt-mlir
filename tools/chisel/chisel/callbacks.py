# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Four DebugHooks callbacks for program-level accumulation testing.

Program-level signature: (binary, program_context)
Op-level signature:      (binary, program_context, op_context)

Each op output produces two PCC records:
  - golden_vs_runtime_tensor      — isolation golden (device-stashed inputs)
  - accum_golden_vs_runtime_tensor — accumulation golden (pool inputs)

Either check can be disabled via ctx.isolation_check / ctx.accum_check.
"""
import logging

from .checker import ChiselChecker
from .context import ChiselContext
from .exceptions import record_check
from .executor import execute_golden, execute_golden_from_pool
from .op_configs import get_op_config
from .ops import get_op_inputs, get_op_outputs
from .utils import (
    chisel_safe,
    debug_wrap,
    retrieve_torch_tensor,
    write_torch_tensor_to_pool,
)

logger = logging.getLogger("chisel")


# ---------------------------------------------------------------------------
# Program-level callbacks
# ---------------------------------------------------------------------------


@chisel_safe
def chisel_pre_program_callback(binary, program_context):
    ChiselContext.get_instance().preprogram(binary, program_context)


@chisel_safe
def chisel_post_program_callback(binary, program_context):
    ChiselContext.get_instance().postprogram(binary, program_context)


# ---------------------------------------------------------------------------
# Op-level helpers
# ---------------------------------------------------------------------------


def _default_pre_op(binary, program_context, op_context) -> None:
    """Default preOp body (runs after op iterator has already advanced).

    1. For each input: validate MLIR IR type against TensorRef metadata
    2. Retrieve device input tensors to host
    3. For each input: validate MLIR IR type against retrieved tensor
    4. Stash inputs in program.stashed_inputs for isolation golden in postOp
    5. Seed golden_tensor_pool for inputs not yet present (program inputs)
    """
    from ttrt import runtime as tt_runtime

    ctx = ChiselContext.get_instance()
    program = ctx.current_program
    binary_state = ctx.current_binary
    op = program.current_op
    checker = ChiselChecker(ctx, op.name)

    program.stashed_inputs = {}
    op_inputs = get_op_inputs(op)
    input_refs = tt_runtime.get_op_input_refs(op_context, program_context)
    asm_state = binary_state.ir_module.get_asm_state(program.program_name)

    for mlir_input, tensor_ref in zip(op_inputs, input_refs):
        name = mlir_input.get_name(asm_state)
        checker.check_shape_dtype(name, "mlir_vs_tensor_ref", mlir_input, tensor_ref)
        tensor = None
        with record_check([name], "retrieve_input", checker, log_prefix=op.name):
            tensor = retrieve_torch_tensor(program_context, tensor_ref)
        if tensor is None:
            continue
        checker.check_shape_dtype(name, "mlir_vs_runtime_tensor", mlir_input, tensor)
        program.stashed_inputs[name] = tensor
        # Seed pool for program inputs (never produced by a prior golden op)
        if name not in program.golden_tensor_pool:
            program.golden_tensor_pool[name] = tensor


def _default_post_op(
    binary, program_context, op_context, *, skip_pcc: bool = False, skip_accum_pcc: bool = False
) -> None:
    """Default postOp body: dual-check validation against isolation and accumulation golden.

    For each op output:
    1. check_shape_dtype(mlir_vs_tensor_ref)     — MLIR IR shape/dtype vs flatbuffer TensorRef
    2. Retrieve device tensor
    3. check_shape_dtype(mlir_vs_runtime_tensor) — MLIR IR shape/dtype vs actual tensor

    Then two golden checks (each gated by ctx.isolation_check / ctx.accum_check):
    Isolation golden (device-stashed inputs):
    4. execute_golden with program.stashed_inputs
    5. check_shape_dtype(mlir_vs_golden)         — MLIR IR shape/dtype vs isolation golden
    6. check_golden_vs_runtime_tensor            — PCC/atol/rtol (skip if skip_pcc)

    Accumulation golden (pool inputs):
    7. execute_golden_from_pool with program.golden_tensor_pool
    8. check_golden_vs_runtime_tensor(accum=True) — PCC/atol/rtol (skip if skip_accum_pcc)
    """
    from ttrt import runtime as tt_runtime

    ctx = ChiselContext.get_instance()
    program = ctx.current_program
    binary_state = ctx.current_binary
    op = program.current_op
    op_name = op.name
    checker = ChiselChecker(ctx, op_name)
    ir_module = binary_state.ir_module
    program_name = program.program_name

    op_outputs = get_op_outputs(op)
    if not op_outputs:
        program.stashed_inputs = None
        return

    output_refs = tt_runtime.get_op_output_refs(op_context, program_context)
    asm_state = ir_module.get_asm_state(program_name)

    # --- Execute golden functions ONCE per op, before the per-output loop ---

    # Compute isolation golden if either the isolation check is on OR the op is
    # marked for skipping — skip mode writes the isolation golden back into the
    # device pool, so it needs the tensor even when isolation_check is off.
    # Error/skip records are still gated on isolation_check so disabling the
    # check means a clean report.
    skip_this_op = ctx.should_skip(op)
    slots = [out.get_name(asm_state) for out in op_outputs]

    iso_result = None
    if ctx.isolation_check or skip_this_op:
        with record_check(
            slots, "golden", checker,
            log_prefix=op_name, record=ctx.isolation_check,
        ):
            iso_result = execute_golden(
                op.opview, ir_module, program_name, program.stashed_inputs
            )

    acc_result = None
    if ctx.accum_check:
        with record_check(slots, "accum_golden", checker, log_prefix=op_name):
            acc_result = execute_golden_from_pool(
                op.opview, ir_module, program_name, program.golden_tensor_pool
            )

    # --- Per-output validation loop ---

    iso_outs = iso_result if iso_result is not None else [None] * len(op_outputs)
    acc_outs = acc_result if acc_result is not None else [None] * len(op_outputs)

    for mlir_output, output_ref, iso_out, acc_out in zip(
        op_outputs, output_refs, iso_outs, acc_outs, strict=True
    ):
        name = mlir_output.get_name(asm_state)

        checker.check_shape_dtype(name, "mlir_vs_tensor_ref", mlir_output, output_ref)

        device_tensor = None
        with record_check([name], "retrieve_output", checker, log_prefix=op_name):
            device_tensor = retrieve_torch_tensor(program_context, output_ref)
        if device_tensor is None:
            continue

        checker.check_shape_dtype(name, "mlir_vs_runtime_tensor", mlir_output, device_tensor)

        if ctx.isolation_check and iso_out is not None:
            checker.check_shape_dtype(name, "mlir_vs_golden", mlir_output, iso_out)
            if skip_pcc:
                checker.record(name, "golden_vs_runtime_tensor", "skipped_pcc")
            else:
                checker.check_golden_vs_runtime_tensor(name, iso_out, device_tensor)

        if acc_out is not None and not skip_accum_pcc:
            checker.check_golden_vs_runtime_tensor(name, acc_out, device_tensor, accum=True)

    if skip_this_op:
        _apply_skip(
            program_context, op, op_outputs, output_refs, asm_state, iso_result, checker
        )

    program.stashed_inputs = None


def _apply_skip(
    program_context, op, op_outputs, output_refs, asm_state, iso_result, checker
) -> None:
    """Overwrite device output(s) with the isolation golden.

    Isolation (not accumulation) is the correct source: it represents this
    single op's kernel being replaced while preserving any upstream PCC drift
    that was already present in the device inputs. Accumulation would retro-
    actively erase upstream error, which is not what skip mode means.
    """
    if iso_result is None:
        for mlir_output in op_outputs:
            checker.record(
                mlir_output.get_name(asm_state), "skip_on_device", "skipped_no_golden"
            )
        return

    for i, (mlir_output, output_ref) in enumerate(
        zip(op_outputs, output_refs, strict=True)
    ):
        name = mlir_output.get_name(asm_state)
        tensor = iso_result[i]
        with record_check(
            [name], "skip_on_device", checker,
            log_prefix=op.name, success="applied",
        ):
            write_torch_tensor_to_pool(program_context, output_ref, tensor)


# ---------------------------------------------------------------------------
# Op-level callbacks
# ---------------------------------------------------------------------------


@chisel_safe
@debug_wrap
def chisel_pre_op_callback(binary, program_context, op_context):
    """
    Pre-operation callback: advance op iterator, then dispatch to per-op or default preOp.

    The op iterator is always advanced here before dispatch so that custom pre_op
    implementations can rely on ctx.current_program.current_op being set.
    """
    ctx = ChiselContext.get_instance()
    ctx.current_program.current_op = next(ctx.current_program.op_iter)

    config = get_op_config(ctx.current_program.current_op.opview)
    if config.pre_op is not None:
        config.pre_op(binary, program_context, op_context)
    else:
        _default_pre_op(binary, program_context, op_context)


@chisel_safe
@debug_wrap
def chisel_post_op_callback(binary, program_context, op_context):
    """
    Post-operation callback: dispatch to per-op or default postOp.

    When a custom post_op is registered for the current op type it replaces the
    default body entirely. Otherwise the default postOp runs with skip_pcc and
    skip_accum_pcc taken from the op's ChiselOpConfig.
    """
    ctx = ChiselContext.get_instance()
    config = get_op_config(ctx.current_program.current_op.opview)
    if config.post_op is not None:
        config.post_op(binary, program_context, op_context)
    else:
        _default_post_op(
            binary, program_context, op_context,
            skip_pcc=config.skip_pcc,
            skip_accum_pcc=config.skip_accum_pcc,
        )
