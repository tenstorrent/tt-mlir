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
import functools
import logging
import traceback

from .checker import ChiselChecker
from .context import ChiselContext
from .executor import execute_golden, execute_golden_from_pool
from .op_configs import get_op_config
from .ops import get_op_inputs, get_op_outputs
from .utils import debug_wrap, retrieve_torch_tensor, write_torch_tensor_to_pool

logger = logging.getLogger("chisel")

DEBUG = False


def with_pytest_subtests(subtests):
    """Decorator that wraps a post_op callback so each op runs as a pytest subtest.

    Pairs with ctx.strict=True: the subtest fixture catches the AssertionError
    raised on mismatch, records the failure, and continues to the next op.

    Usage:
        ctx.strict = True
        tt_runtime.DebugHooks.get(
            pre_op=chisel_pre_op_callback,
            post_op=with_pytest_subtests(subtests)(chisel_post_op_callback),
            pre_program=chisel_pre_program_callback,
            post_program=chisel_post_program_callback,
        )
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(binary, program_context, op_context):
            ctx = ChiselContext.get_instance()
            program = ctx.current_program
            op_name = program.current_op.name if program and program.current_op else "unknown"
            with subtests.test(op=op_name):
                fn(binary, program_context, op_context)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Program-level callbacks
# ---------------------------------------------------------------------------


def chisel_pre_program_callback(binary, program_context):
    ChiselContext.get_instance().preprogram(binary, program_context)


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
    4. Stash inputs in ctx._stashed_inputs for isolation golden in postOp
    5. Seed golden_tensor_pool for inputs not yet present (program inputs)
    """
    from ttrt import runtime as tt_runtime

    ctx = ChiselContext.get_instance()
    program = ctx.current_program
    binary_state = ctx.current_binary
    op = program.current_op
    checker = ChiselChecker(ctx, op.name)

    ctx._stashed_inputs = {}
    op_inputs = get_op_inputs(op)
    input_refs = tt_runtime.get_op_input_refs(op_context, program_context)
    asm_state = binary_state.ir_module.get_asm_state(program.program_name)

    for mlir_input, tensor_ref in zip(op_inputs, input_refs):
        name = mlir_input.get_name(asm_state)
        checker.check_mlir_vs_tensor_ref(name, mlir_input, tensor_ref)
        try:
            tensor = retrieve_torch_tensor(program_context, tensor_ref)
            checker.check_mlir_vs_runtime_tensor(name, mlir_input, tensor)
            ctx._stashed_inputs[name] = tensor
            # Seed pool for program inputs (never produced by a prior golden op)
            if name not in program.golden_tensor_pool:
                program.golden_tensor_pool[name] = tensor
        except Exception:
            tb = traceback.format_exc()
            logger.error(
                f"{op.name} {name}: failed to retrieve input tensor\n{tb}"
            )
            checker._record(name, "retrieve_input", "error", traceback=tb)


def _default_post_op(
    binary, program_context, op_context, *, skip_pcc: bool = False, skip_accum_pcc: bool = False
) -> None:
    """Default postOp body: dual-check validation against isolation and accumulation golden.

    For each op output:
    1. check_mlir_vs_tensor_ref       — MLIR IR shape/dtype vs flatbuffer TensorRef
    2. Retrieve device tensor
    3. check_mlir_vs_runtime_tensor   — MLIR IR shape/dtype vs actual tensor

    Then two golden checks (each gated by ctx.isolation_check / ctx.accum_check):
    Isolation golden (device-stashed inputs):
    4. execute_golden with ctx._stashed_inputs
    5. check_mlir_vs_golden           — MLIR IR shape/dtype vs isolation golden
    6. check_golden_vs_runtime_tensor — PCC/atol/rtol (skip if skip_pcc)

    Accumulation golden (pool inputs):
    7. execute_golden_from_pool with program.golden_tensor_pool
    8. check_accum_golden_vs_runtime_tensor — PCC/atol/rtol (skip if skip_accum_pcc)
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
        ctx._stashed_inputs = None
        return

    output_refs = tt_runtime.get_op_output_refs(op_context, program_context)
    asm_state = ir_module.get_asm_state(program_name)

    # --- Execute golden functions ONCE per op, before the per-output loop ---

    iso_result = None
    # Compute isolation golden if either the isolation check is on OR the op is
    # marked for skipping — skip mode writes the isolation golden back into the
    # device pool, so it needs the tensor even when isolation_check is off.
    # Error/skip records are still gated on isolation_check so disabling the
    # check means a clean report.
    skip_this_op = ctx.should_skip(op)
    if ctx.isolation_check or skip_this_op:
        try:
            iso_result = execute_golden(
                op.opview, ir_module, program_name, ctx._stashed_inputs
            )
        except RuntimeError:
            logger.debug(f"{op_name}: no golden implementation, skipping isolation check")
            if ctx.isolation_check:
                for out in op_outputs:
                    checker._record(out.get_name(asm_state), "golden", "skipped")
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{op_name}: isolation golden execution failed\n{tb}")
            if ctx.isolation_check:
                for out in op_outputs:
                    checker._record(out.get_name(asm_state), "golden", "error", traceback=tb)

    acc_result = None
    if ctx.accum_check:
        try:
            acc_result = execute_golden_from_pool(
                op.opview, ir_module, program_name, program.golden_tensor_pool
            )
        except (RuntimeError, KeyError) as e:
            logger.debug(f"{op_name}: accumulation golden skipped ({type(e).__name__}: {e})")
            for out in op_outputs:
                checker._record(out.get_name(asm_state), "accum_golden", "skipped")
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{op_name}: accumulation golden execution failed\n{tb}")
            for out in op_outputs:
                checker._record(out.get_name(asm_state), "accum_golden", "error", traceback=tb)

    # --- Per-output validation loop ---

    for i, (mlir_output, output_ref) in enumerate(zip(op_outputs, output_refs, strict=True)):
        name = mlir_output.get_name(asm_state)

        checker.check_mlir_vs_tensor_ref(name, mlir_output, output_ref)

        try:
            device_tensor = retrieve_torch_tensor(program_context, output_ref)
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{op_name} {name}: failed to retrieve device output tensor\n{tb}")
            checker._record(name, "retrieve_output", "error", traceback=tb)
            continue

        checker.check_mlir_vs_runtime_tensor(name, mlir_output, device_tensor)

        if ctx.isolation_check and iso_result is not None:
            iso_out = iso_result[i] if isinstance(iso_result, (list, tuple)) else iso_result
            checker.check_mlir_vs_golden(name, mlir_output, iso_out)
            if skip_pcc:
                checker._record(name, "golden_vs_runtime_tensor", "skipped_pcc")
            else:
                checker.check_golden_vs_runtime_tensor(name, iso_out, device_tensor)

        if acc_result is not None:
            acc_out = acc_result[i] if isinstance(acc_result, (list, tuple)) else acc_result
            if skip_accum_pcc:
                checker._record(name, "accum_golden_vs_runtime_tensor", "skipped_pcc")
            else:
                checker.check_accum_golden_vs_runtime_tensor(name, acc_out, device_tensor)

    if skip_this_op:
        _apply_skip(
            program_context, op, op_outputs, output_refs, asm_state, iso_result, checker
        )

    ctx._stashed_inputs = None


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
            checker._record(
                mlir_output.get_name(asm_state), "skip_on_device", "skipped_no_golden"
            )
        return

    for i, (mlir_output, output_ref) in enumerate(
        zip(op_outputs, output_refs, strict=True)
    ):
        name = mlir_output.get_name(asm_state)
        tensor = iso_result[i] if isinstance(iso_result, (list, tuple)) else iso_result
        try:
            write_torch_tensor_to_pool(program_context, output_ref, tensor)
            checker._record(name, "skip_on_device", "applied")
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{op.name} {name}: skip_on_device write failed\n{tb}")
            checker._record(name, "skip_on_device", "error", traceback=tb)


# ---------------------------------------------------------------------------
# Op-level callbacks
# ---------------------------------------------------------------------------


@debug_wrap(debug=DEBUG)
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


@debug_wrap(debug=DEBUG)
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
