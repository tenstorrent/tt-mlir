# tools/chisel/chisel/executor.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Golden execution for a single TTNN op via GOLDEN_MAPPINGS.

Uses inspect.signature() to generically match MLIR op attributes to golden
function parameters. This avoids per-op special cases.
"""
import inspect
from typing import Any, List, get_origin, get_args

import torch
from ttmlir.ir import Operation

from golden import get_golden_function, GoldenMapTensor

from chisel.ops import IRModule, get_op_inputs, get_op_outputs


# Parameter names that indicate the output type (always last positional param)
_OUTPUT_TYPE_PARAMS = {"output_type_mlir", "output_ranked_tensor_type"}


def _is_list_annotation(annotation) -> bool:
    """Check if a type annotation is List[...] or list[...]."""
    origin = get_origin(annotation)
    if origin is list:
        return True
    try:
        import typing
        if origin is getattr(typing, "List", None):
            return True
    except Exception:
        pass
    return False


def _build_golden_args(
    golden_fn,
    golden_inputs: List[GoldenMapTensor],
    op: Operation,
    output_type,
) -> list:
    """
    Build the positional argument list for a TTNN golden function.

    Strategy:
    1. Inspect golden_fn signature to get parameter names.
    2. If the first param expects a list, pass all inputs as one list arg.
       Otherwise, pass inputs as individual positional args.
    3. For params between inputs and the last param:
       - If name ends with '_attr', look up op.attributes[name.removesuffix('_attr')]
       - Otherwise, look up op.attributes[name]
    4. Last param is always the output type.
    """
    sig = inspect.signature(golden_fn)
    params = list(sig.parameters.values())

    if not params:
        return []

    args = []
    param_idx = 0

    # Determine if first param expects a list (variadic tensor inputs)
    first_param = params[0]
    if _is_list_annotation(first_param.annotation) and len(golden_inputs) != 1:
        args.append(golden_inputs)
        param_idx = 1
    else:
        for tensor in golden_inputs:
            args.append(tensor)
            param_idx += 1

    # Remaining params: attributes and output type
    for param in params[param_idx:]:
        name = param.name
        if name in _OUTPUT_TYPE_PARAMS:
            args.append(output_type)
        elif name.endswith("_attr"):
            attr_key = name.removesuffix("_attr")
            args.append(op.attributes[attr_key])
        elif name in op.attributes:
            args.append(op.attributes[name])
        else:
            # Assume it's the output type (last param convention)
            args.append(output_type)

    return args


def execute_golden(op: Operation, ir_module: IRModule, inputs: dict) -> torch.Tensor:
    """
    Execute a TTNN op on CPU via GOLDEN_MAPPINGS.

    Args:
        op: The MLIR operation to execute.
        ir_module: The IRModule containing the operation (for SSA name resolution).
        inputs: Dict mapping SSA names to torch.Tensor (device inputs copied to host).

    Returns:
        torch.Tensor — the golden output.

    Raises:
        RuntimeError: If no golden implementation exists for the op type.
    """
    golden_fn = get_golden_function(type(op))
    if golden_fn is None:
        raise RuntimeError(f"No golden implementation for {type(op).__name__}")

    # Wrap input tensors as GoldenMapTensor (single-device)
    op_inputs = get_op_inputs(op)
    asm_state = ir_module.get_asm_state()
    golden_inputs = [
        GoldenMapTensor({0: inputs[inp.get_name(asm_state)]}, (1, 1))
        for inp in op_inputs
    ]

    # Get output type from first result
    op_outputs = get_op_outputs(op)
    output_type = op_outputs[0].type if op_outputs else None

    # Build args and call
    args = _build_golden_args(golden_fn, golden_inputs, op, output_type)
    result = golden_fn(*args)

    # Extract torch.Tensor from GoldenMapTensor
    if isinstance(result, GoldenMapTensor):
        return result.golden_map_tensor_as_torch_tensors()[0]
    if isinstance(result, torch.Tensor):
        return result
    return result
