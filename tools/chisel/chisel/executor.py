# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Golden execution for a single TTNN op via CHISEL_GOLDEN_MAPPINGS.

Each golden function has the interface:
    fn(op: Operation, inputs: Dict[str, GoldenMapTensor], asm_state: AsmState) -> GoldenMapTensor

The executor wraps host tensors as GoldenMapTensor, calls the golden function,
and normalizes the result to a tuple of torch.Tensor with one entry per op
output. Single-tensor goldens on multi-output ops (e.g. sort/topk return only
values) are broadcast to every output slot so downstream consumers of any
output find a value.
"""
from typing import Dict, Tuple

import torch
from ttmlir.ir import Operation

from golden import get_chisel_golden_function, GoldenMapTensor

from .exceptions import (
    GoldenExecutionError,
    GoldenInputMissing,
    NoGoldenImplementation,
)
from .ops import IRModule, get_op_inputs, get_op_outputs
from .tensors import TensorPool


def execute_golden(
    op: Operation, ir_module: IRModule, inputs: dict
) -> Tuple[torch.Tensor, ...]:
    """
    Execute a TTNN op on CPU via CHISEL_GOLDEN_MAPPINGS.

    Args:
        op: The MLIR operation to execute.
        ir_module: The IRModule containing the operation (for SSA name resolution).
        inputs: Dict mapping SSA names to torch.Tensor (device inputs copied to host).

    Returns:
        Tuple[torch.Tensor, ...] — one tensor per op output. Single-tensor
        goldens on multi-output ops are broadcast to every slot.

    Raises:
        NoGoldenImplementation: If no golden mapping exists for the op type.
        GoldenExecutionError: If the golden function itself raises.
        TypeError: If the golden returns a type we cannot interpret as tensors.
    """
    golden_fn = get_chisel_golden_function(type(op))
    if golden_fn is None:
        raise NoGoldenImplementation(
            f"No golden implementation for {type(op).__name__}"
        )

    # Wrap input tensors as GoldenMapTensor keyed by SSA name
    op_inputs = get_op_inputs(op)
    asm_state = ir_module.get_asm_state()
    golden_inputs: Dict[str, GoldenMapTensor] = {}
    for inp in op_inputs:
        name = inp.get_name(asm_state)
        golden_inputs[name] = GoldenMapTensor({0: inputs[name]}, (1, 1))

    try:
        result = golden_fn(op, golden_inputs, asm_state)
    except Exception as e:
        raise GoldenExecutionError(
            f"golden for {type(op).__name__} raised: {e}"
        ) from e

    if isinstance(result, GoldenMapTensor):
        tensors: Tuple[torch.Tensor, ...] = (result.golden_map_tensor_as_torch_tensors()[0],)
    elif isinstance(result, torch.Tensor):
        tensors = (result,)
    elif isinstance(result, (list, tuple)):
        tensors = tuple(result)
    else:
        raise TypeError(
            f"Golden for {type(op).__name__} returned unsupported type "
            f"{type(result).__name__}"
        )

    num_outputs = len(get_op_outputs(op))
    if len(tensors) == 1 and num_outputs > 1:
        tensors = tensors * num_outputs
    return tensors


def execute_golden_from_pool(
    op: Operation,
    ir_module: IRModule,
    tensor_pool: TensorPool,
) -> Tuple[torch.Tensor, ...]:
    """
    Pool-aware golden execution.

    Pulls input tensors from tensor_pool by SSA name, calls execute_golden,
    stores each output tensor back in the pool under its SSA name, and
    returns the tuple.

    ir_module is used to obtain asm_state for SSA name resolution of both
    inputs (pool lookup) and outputs (pool store).

    Raises:
        GoldenInputMissing: If a required input is not present in tensor_pool.
        NoGoldenImplementation / GoldenExecutionError: Propagated from execute_golden.
    """
    asm_state = ir_module.get_asm_state()
    op_inputs = get_op_inputs(op)
    try:
        inputs = {
            inp.get_name(asm_state): tensor_pool[inp.get_name(asm_state)]
            for inp in op_inputs
        }
    except KeyError as e:
        raise GoldenInputMissing(f"missing input in pool: {e}") from e

    result = execute_golden(op, ir_module, inputs)

    for out_val, res_tensor in zip(get_op_outputs(op), result, strict=True):
        tensor_pool[out_val.get_name(asm_state)] = res_tensor

    return result
