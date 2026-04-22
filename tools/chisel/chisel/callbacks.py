# tools/chisel/chisel/callbacks.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DebugHooks callbacks for per-op isolation testing.

Two plain functions compatible with DebugHooks op-level callbacks.
No program-level callbacks in this PR.
"""
import functools
import logging
import traceback

from .checker import ChiselChecker
from .context import ChiselContext
from .executor import execute_golden
from .ops import get_op_inputs, get_op_outputs
from .utils import debug_wrap, retrieve_torch_tensor

logger = logging.getLogger("chisel")
logger.setLevel("DEBUG")

DEBUG = False


def with_pytest_subtests(subtests):
    """Decorator that wraps a post_op callback so each op runs as a pytest subtest.

    Pairs with ctx.strict=True: the subtest fixture catches the AssertionError
    raised on mismatch, records the failure, and continues to the next op.

    Usage:
        ctx.strict = True
        tt_runtime.DebugHooks.get(
            chisel_pre_op_callback,
            with_pytest_subtests(subtests)(chisel_post_op_callback),
        )
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(binary, program_context, op_context):
            ctx = ChiselContext.get_instance()
            op_name = ctx._current_op.name if ctx._current_op else "unknown"
            with subtests.test(op=op_name):
                fn(binary, program_context, op_context)

        return wrapper

    return decorator


@debug_wrap(debug=DEBUG)
def chisel_pre_op_callback(binary, program_context, op_context):
    """
    Pre-operation callback: advance op iterator and stash device input tensors.

    1. Advance op_iter to get current MLIR op
    2. For each input: validate MLIR IR type against TensorRef metadata
    3. Retrieve device input tensors to host
    4. For each input: validate MLIR IR type against retrieved tensor
    5. Stash inputs in ctx._stashed_inputs for postOp
    """
    from ttrt import runtime as tt_runtime

    ctx = ChiselContext.get_instance()
    ctx.ensure_ir_module(binary, program_context)
    ctx._current_op = next(ctx.op_iter)

    checker = ChiselChecker(ctx, ctx._current_op.name)

    ctx._stashed_inputs = {}
    op_inputs = get_op_inputs(ctx._current_op)
    input_refs = tt_runtime.get_op_input_refs(op_context, program_context)
    asm_state = ctx.ir_module.get_asm_state(ctx._current_program_name)

    for mlir_input, tensor_ref in zip(op_inputs, input_refs):
        name = mlir_input.get_name(asm_state)
        checker.check_mlir_vs_tensor_ref(name, mlir_input, tensor_ref)
        try:
            tensor = retrieve_torch_tensor(program_context, tensor_ref)
            checker.check_mlir_vs_runtime_tensor(name, mlir_input, tensor)
            ctx._stashed_inputs[name] = tensor
        except Exception:
            tb = traceback.format_exc()
            logger.error(
                f"{ctx._current_op.name} {name}: failed to retrieve input tensor\n{tb}"
            )
            checker._record(name, "retrieve_input", "error", traceback=tb)


@debug_wrap(debug=DEBUG)
def chisel_post_op_callback(binary, program_context, op_context):
    """
    Post-operation callback: capture device output and validate against golden.

    For each output:
    1. check_mlir_vs_tensor_ref       — MLIR IR shape/dtype vs flatbuffer TensorRef
    2. Retrieve device tensor
    3. check_mlir_vs_runtime_tensor   — MLIR IR shape/dtype vs actual tensor
    4. execute_golden                 — run CPU reference (skip if no mapping)
    5. check_mlir_vs_golden           — MLIR IR shape/dtype vs golden tensor
    6. check_golden_vs_runtime_tensor — full comparison: shape, dtype, PCC, atol, rtol
    """
    from ttrt import runtime as tt_runtime

    ctx = ChiselContext.get_instance()
    op_name = ctx._current_op.name
    checker = ChiselChecker(ctx, op_name)

    op_outputs = get_op_outputs(ctx._current_op)
    if not op_outputs:
        ctx._stashed_inputs = None
        return

    output_refs = tt_runtime.get_op_output_refs(op_context, program_context)
    asm_state = ctx.ir_module.get_asm_state(ctx._current_program_name)

    for mlir_output, output_ref in zip(op_outputs, output_refs, strict=True):
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

        try:
            golden = execute_golden(
                ctx._current_op.opview,
                ctx.ir_module,
                ctx._current_program_name,
                ctx._stashed_inputs,
            )
        except RuntimeError:
            checker._record(name, "golden", "skipped")
            logger.debug(f"{op_name} {name}: no golden implementation, skipping")
            continue
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{op_name} {name}: golden execution failed\n{tb}")
            checker._record(name, "golden", "error", traceback=tb)
            continue

        checker.check_mlir_vs_golden(name, mlir_output, golden)
        checker.check_golden_vs_runtime_tensor(name, golden, device_tensor)

    ctx._stashed_inputs = None
