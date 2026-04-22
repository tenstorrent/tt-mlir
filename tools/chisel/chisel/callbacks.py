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
import json
import logging

import torch

from golden.metrics import compute_atol, compute_pcc, compute_rtol

from .context import ChiselContext
from .executor import execute_golden
from .ops import get_op_inputs, get_op_outputs
from .utils import retrieve_torch_tensor

logger = logging.getLogger("chisel")
logger.setLevel("DEBUG")

_PCC_THRESHOLD = 0.99
DEBUG=False

def debug_wrap(*, debug: bool = False):
    """
    Decorator factory: use as @debug_wrap(debug=DEBUG_FLAG).
    Helpful to use with runtime callbacks to get more info about errors.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:  # noqa: BLE001
                if debug:  # flag decided at *definition* time
                    import pdb
                    import traceback

                    traceback.print_exc()
                    pdb.set_trace()
                raise

        return wrapper

    return decorator

def _append_result(ctx: ChiselContext, record: dict) -> None:
    if ctx.results_path is None:
        return
    with open(ctx.results_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def check_op_output(
    golden: torch.Tensor,
    device: torch.Tensor,
    op_name: str,
    output_name: str,
    ctx: ChiselContext,
) -> None:
    """Validate device output against golden: shape, dtype, then PCC/ATOL/RTOL.

    Raises AssertionError on first failure when ctx.strict is True, otherwise logs warnings.
    """
    if golden.shape != device.shape:
        msg = (
            f"{op_name} {output_name}: shape MISMATCH "
            f"expected={tuple(golden.shape)} actual={tuple(device.shape)}"
        )
        _append_result(ctx, {
            "op": op_name, "output": output_name, "status": "shape_mismatch",
            "expected_shape": list(golden.shape), "actual_shape": list(device.shape),
        })
        if ctx.strict:
            raise AssertionError(msg)
        logger.warning(msg)
        return

    if golden.dtype != device.dtype:
        msg = (
            f"{op_name} {output_name}: dtype MISMATCH "
            f"expected={golden.dtype} actual={device.dtype}"
        )
        _append_result(ctx, {
            "op": op_name, "output": output_name, "status": "dtype_mismatch",
            "expected_dtype": str(golden.dtype), "actual_dtype": str(device.dtype),
        })
        if ctx.strict:
            raise AssertionError(msg)
        logger.warning(msg)
        return

    pcc = compute_pcc(golden, device)
    atol = compute_atol(golden, device)
    rtol = compute_rtol(golden, device)

    if pcc >= _PCC_THRESHOLD:
        _append_result(ctx, {
            "op": op_name, "output": output_name, "status": "ok",
            "pcc": pcc, "atol": atol, "rtol": rtol,
        })
        logger.info(
            f"{op_name} {output_name}: OK  "
            f"pcc={pcc:.6f} atol={atol:.6e} rtol={rtol:.6e}"
        )
    else:
        msg = (
            f"{op_name} {output_name}: PCC FAIL "
            f"pcc={pcc:.6f} (threshold={_PCC_THRESHOLD}) "
            f"atol={atol:.6e} rtol={rtol:.6e}"
        )
        _append_result(ctx, {
            "op": op_name, "output": output_name, "status": "pcc_fail",
            "pcc": pcc, "atol": atol, "rtol": rtol,
        })
        if ctx.strict:
            raise AssertionError(msg)
        logger.warning(msg)

@debug_wrap(debug=DEBUG)
def chisel_pre_op_callback(binary, program_context, op_context):
    """
    Pre-operation callback: advance op iterator and stash device input tensors.

    1. Advance op_iter to get current MLIR op
    2. Copy device input tensors to host
    3. Stash inputs in ctx._stashed_inputs for postOp
    """
    from ttrt import runtime as tt_runtime

    ctx = ChiselContext.get_instance()
    ctx.ensure_ir_module(binary, program_context)
    ctx._current_op = next(ctx.op_iter)

    # Copy device inputs to host, keyed by SSA name
    ctx._stashed_inputs = {}
    op_inputs = get_op_inputs(ctx._current_op)
    input_refs = tt_runtime.get_op_input_refs(op_context, program_context)
    asm_state = ctx.ir_module.get_asm_state(ctx._current_program_name)

    for mlir_input, tensor_ref in zip(op_inputs, input_refs):
        name = mlir_input.get_name(asm_state)
        ctx._stashed_inputs[name] = retrieve_torch_tensor(program_context, tensor_ref)

@debug_wrap(debug=DEBUG)
def chisel_post_op_callback(binary, program_context, op_context):
    """
    Post-operation callback: capture device output and validate against golden.

    1. Skip ops with no outputs
    2. Capture device output tensor
    3. Run golden execution and call check_op_output (shape, dtype, PCC, ATOL, RTOL)
    4. Skip validation if no golden implementation exists for the op
    """
    from ttrt import runtime as tt_runtime

    ctx = ChiselContext.get_instance()
    op_name = ctx._current_op.name
    _append_result(ctx, {"op": op_name, "status": "testing"})
    # Skip ops with no outputs
    op_outputs = get_op_outputs(ctx._current_op)
    if len(op_outputs) == 0:
        ctx._stashed_inputs = None
        return

    # Capture device output tensors
    output_refs = tt_runtime.get_op_output_refs(op_context, program_context)
    device_tensors = [
        retrieve_torch_tensor(program_context, ref) for ref in output_refs
    ]

    asm_state = ctx.ir_module.get_asm_state(ctx._current_program_name)

    for mlir_output, device_torch in zip(op_outputs, device_tensors, strict=True):
        name = mlir_output.get_name(asm_state)
        try:
            golden_torch = execute_golden(
                ctx._current_op.opview,
                ctx.ir_module,
                ctx._current_program_name,
                ctx._stashed_inputs,
            )
        except RuntimeError as e:
            _append_result(ctx, {"op": op_name, "output": name, "status": "skipped"})
            logger.debug(f"{op_name} {name}: no golden implementation, skipping")
            continue
        check_op_output(golden_torch, device_torch, op_name, name, ctx)

    ctx._stashed_inputs = None


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

