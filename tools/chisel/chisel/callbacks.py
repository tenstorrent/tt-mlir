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
from .utils import retrieve_torch_tensor

logger = logging.getLogger("chisel")


def chisel_pre_op_callback(binary, program_context, op_context):
    """
    Pre-operation callback: advance op iterator and stash device input tensors.

    1. Advance op_iter to get current MLIR op
    2. Copy device input tensors to host
    3. Stash inputs in ctx._stashed_inputs for postOp
    """
    import _ttmlir_runtime as tt_runtime

    ctx = ChiselContext.get_instance()
    ctx.ensure_ir_module(binary, program_context)
    ctx._current_op = next(ctx.op_iter)

    # Copy device inputs to host, keyed by SSA name
    ctx._stashed_inputs = {}
    op_inputs = get_op_inputs(ctx._current_op)
    input_refs = tt_runtime.runtime.get_op_input_refs(op_context, program_context)
    asm_state = ctx.ir_module.get_asm_state(ctx._current_program_name)

    for mlir_input, tensor_ref in zip(op_inputs, input_refs):
        name = mlir_input.get_name(asm_state)
        ctx._stashed_inputs[name] = retrieve_torch_tensor(program_context, tensor_ref)


def chisel_post_op_callback(binary, program_context, op_context):
    """
    Post-operation callback: capture device output, check shape against MLIR.

    1. Skip ops with no outputs
    2. Capture device output tensor
    3. Compare shape against MLIR-declared shape
    4. Log result
    """
    import _ttmlir_runtime as tt_runtime

    ctx = ChiselContext.get_instance()

    # Skip ops with no outputs
    op_outputs = get_op_outputs(ctx._current_op)
    if len(op_outputs) == 0:
        ctx._stashed_inputs = None
        return

    # Capture device output tensors
    # TODO(ndrakulic): Replace with get_op_output_refs (plural) once multi-output API lands
    output_refs = [
        tt_runtime.runtime.get_op_output_ref(op_context, program_context)
    ]
    device_tensors = [
        retrieve_torch_tensor(program_context, ref)
        for ref in output_refs
        if ref is not None
    ]

    op_name = ctx._current_op.name
    asm_state = ctx.ir_module.get_asm_state(ctx._current_program_name)

    for mlir_output, device_torch in zip(op_outputs, device_tensors, strict=True):
        name = mlir_output.get_name(asm_state)
        expected_shape = tuple(mlir_output.type.shape)
        actual_shape = tuple(device_torch.shape)

        if expected_shape == actual_shape:
            logger.info(f"{op_name} {name}: shape OK {actual_shape}")
        else:
            logger.warning(
                f"{op_name} {name}: shape MISMATCH expected={expected_shape} actual={actual_shape}"
            )

    ctx._stashed_inputs = None
