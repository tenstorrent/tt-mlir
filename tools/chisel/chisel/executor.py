# tools/chisel/chisel/executor.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Golden execution for a single TTNN op via CHISEL_GOLDEN_MAPPINGS.

Each golden function has the interface:
    fn(op: Operation, inputs: Dict[str, GoldenMapTensor], asm_state: AsmState) -> GoldenMapTensor

The executor wraps host tensors as GoldenMapTensor, calls the golden function,
and extracts the result back to a torch.Tensor.
"""
from typing import Dict

import torch
from ttmlir.ir import Operation

from golden import get_chisel_golden_function, GoldenMapTensor

from .ops import IRModule, get_op_inputs


def execute_golden(
    op: Operation, ir_module: IRModule, function_name: str, inputs: dict
) -> torch.Tensor:
    """
    Execute a TTNN op on CPU via CHISEL_GOLDEN_MAPPINGS.

    Args:
        op: The MLIR operation to execute.
        ir_module: The IRModule containing the operation (for SSA name resolution).
        function_name: Name of the function containing the op (for AsmState lookup).
        inputs: Dict mapping SSA names to torch.Tensor (device inputs copied to host).

    Returns:
        torch.Tensor — the golden output.

    Raises:
        RuntimeError: If no golden implementation exists for the op type.
    """
    golden_fn = get_chisel_golden_function(type(op))
    if golden_fn is None:
        raise RuntimeError(f"No golden implementation for {type(op).__name__}")

    # Wrap input tensors as GoldenMapTensor keyed by SSA name
    op_inputs = get_op_inputs(op)
    asm_state = ir_module.get_asm_state(function_name)
    golden_inputs: Dict[str, GoldenMapTensor] = {
        inp.get_name(asm_state): GoldenMapTensor({0: inputs[inp.get_name(asm_state)]}, (1, 1))
        for inp in op_inputs
    }

    result = golden_fn(op, golden_inputs, asm_state)

    # Extract torch.Tensor from GoldenMapTensor
    if isinstance(result, GoldenMapTensor):
        return result.golden_map_tensor_as_torch_tensors()[0]
    if isinstance(result, torch.Tensor):
        return result
    return result
