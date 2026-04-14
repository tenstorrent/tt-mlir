# tools/chisel/chisel/callbacks.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DebugHooks callbacks for per-op isolation testing.

Two plain functions compatible with DebugHooks op-level callbacks.
No program-level callbacks in this PR.
"""
import logging

from .context import ChiselContext
from .ops import get_op_inputs, get_op_outputs
from .executor import execute_golden
from .utils import retrieve_torch_tensor
from golden.metrics import compute_pcc, compute_atol, compute_rtol

logger = logging.getLogger("chisel")


def chisel_pre_op_callback(binary, program_context, op_context):
    """
    Pre-operation callback: advance op iterator and stash device input tensors.

    1. Advance op_iter to get current MLIR op
    2. Copy device input tensors to host
    3. Stash inputs in ctx._stashed_inputs for postOp
    """
    import _ttmlir_runtime as tt_runtime
    print("chisel_pre_op_callback")
    ctx = ChiselContext.get_instance()
    ctx.ensure_ir_module(binary, program_context)
    ctx._current_op = next(ctx.op_iter)

    # Copy device inputs to host, keyed by SSA name
    ctx._stashed_inputs = {}
    op_inputs = get_op_inputs(ctx._current_op)
    input_refs = tt_runtime.runtime.get_op_input_refs(op_context, program_context)
    asm_state = ctx.ir_module.get_asm_state()

    for mlir_input, tensor_ref in zip(op_inputs, input_refs):
        name = mlir_input.get_name(asm_state)
        ctx._stashed_inputs[name] = retrieve_torch_tensor(program_context, tensor_ref)


def chisel_post_op_callback(binary, program_context, op_context):
    """
    Post-operation callback: run golden, capture device output, compare, log.

    1. Run golden function with stashed inputs
    2. Capture device output tensor
    3. Compare golden vs device (PCC, atol, rtol)
    4. Log metrics
    5. Discard golden output (no pool storage)
    """
    print("chisel_post_op_callback")
    import _ttmlir_runtime as tt_runtime
    ctx = ChiselContext.get_instance()

    # Skip ops with no outputs
    if len(get_op_outputs(ctx._current_op)) == 0:
        ctx._stashed_inputs = None
        print("Skipping for the OP", ctx._current_op)
        return

    # Execute golden
    golden_result = execute_golden(
        ctx._current_op, ctx.ir_module, ctx._stashed_inputs
    )

    # Capture device output
    output_ref = tt_runtime.runtime.get_op_output_ref(op_context, program_context)
    device_torch = retrieve_torch_tensor(program_context, output_ref)

    # Compare
    pcc = compute_pcc(golden_result, device_torch)
    atol = compute_atol(golden_result, device_torch)
    rtol = compute_rtol(golden_result, device_torch)
    print(pcc)
    # Log
    op_name = ctx._current_op.name
    logger.info(f"{op_name}: PCC={pcc:.6f}, atol={atol:.6e}, rtol={rtol:.6e}")

    # Discard — no pool storage in isolation mode
    ctx._stashed_inputs = None
