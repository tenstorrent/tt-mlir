# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Golden Executor for TTIR operations with GOLDEN_MAPPINGS integration.

This module provides the GoldenExecutor class which executes operations in the golden
(reference) execution path using PyTorch as the reference implementation.

Key Features:
- Integrates with GOLDEN_MAPPINGS from builder_golden.py for operation execution
- Provides automatic conversion from operation names (e.g., "ttir.abs") to operation
  classes (e.g., ttir.AbsOp) to leverage the comprehensive GOLDEN_MAPPINGS dictionary
- Falls back to the legacy ttir_to_torch_mapping for operations not yet in GOLDEN_MAPPINGS
- Manages tensor values through a tensor pool with support for multiple execution contexts

Operation Function Resolution:
1. Attempts to resolve operation functions from GOLDEN_MAPPINGS (preferred)
   - Converts operation name to operation class using _get_op_class_from_name()
   - Uses get_golden_function() for special cases like ToLayoutOp
2. Falls back to legacy ttir_to_torch_mapping if GOLDEN_MAPPINGS lookup fails
3. Raises ValueError if operation is not supported

Example operation name conversions:
- "ttir.abs" → ttir.AbsOp
- "ttir.add" → ttir.AddOp
- "ttir.get_dimension_size" → ttir.GetDimensionSizeOp
- "stablehlo.add" → stablehlo.AddOp
- "d2m.to_layout" → d2m.ToLayoutOp
"""
from typing import Tuple
from ttmlir.ir import Operation
import torch
import importlib

from .ops import get_op_outputs, get_op_inputs
from .tensors import TensorPool, TensorValue
from .enums import ExecutionType
from .registry import Registry

from ..utils.mapping import ttir_to_torch_mapping

# Try to import GOLDEN_MAPPINGS from builder_golden
try:
    from builder.base.builder_golden import GOLDEN_MAPPINGS, get_golden_function

    HAS_GOLDEN_MAPPINGS = True
except ImportError:
    HAS_GOLDEN_MAPPINGS = False


def _get_op_class_from_name(op_name: str):
    """
    Convert operation name (e.g., "ttir.abs") to operation class (e.g., ttir.AbsOp).

    Args:
        op_name (str): Operation name like "ttir.abs" or "stablehlo.add"

    Returns:
        type: The operation class, or None if not found
    """
    if not HAS_GOLDEN_MAPPINGS:
        return None

    # Parse dialect and operation name
    parts = op_name.split(".")
    if len(parts) < 2:
        return None

    dialect_name = parts[0]  # e.g., "ttir", "stablehlo", "d2m"
    op_short_name = ".".join(parts[1:])  # e.g., "abs", "to_layout"

    # Convert from snake_case to CamelCase and add "Op" suffix
    # e.g., "abs" -> "AbsOp", "get_dimension_size" -> "GetDimensionSizeOp"
    words = op_short_name.split("_")
    class_name = "".join(word.capitalize() for word in words) + "Op"

    try:
        # Import the dialect module
        dialect_module = importlib.import_module(f"ttmlir.dialects.{dialect_name}")
        # Get the operation class
        op_class = getattr(dialect_module, class_name, None)
        return op_class
    except (ImportError, AttributeError):
        return None


def _get_operation_function(op: Operation, **kwargs):
    """
    Get the golden function for an operation using GOLDEN_MAPPINGS or fallback.

    Args:
        op (Operation): The MLIR operation
        **kwargs: Additional arguments for get_golden_function

    Returns:
        callable: The operation function or None if not found
    """
    if HAS_GOLDEN_MAPPINGS:
        # Try to get operation class and look up in GOLDEN_MAPPINGS
        op_class = _get_op_class_from_name(op.name)
        if op_class is not None:
            # Use get_golden_function for special cases like ToLayoutOp
            golden_fn = get_golden_function(op_class, **kwargs)
            if golden_fn is not None:
                return golden_fn

    # Fallback to old ttir_to_torch_mapping
    if op.name in ttir_to_torch_mapping:
        return ttir_to_torch_mapping[op.name]

    return None


class GoldenExecutor:
    """
    Executes operations in the golden (reference) execution path.

    This class is responsible for executing operations using PyTorch as the reference
    implementation. It maintains state about the execution and manages tensor values
    through a tensor pool.

    Args:
        registry (Registry): The registry containing operation and module information
        golden_tensor_pool (TensorPool): Pool for managing tensor values in the golden path

    Attributes:
        registry (Registry): Reference to the registry containing operation information
        last_golden_executed: Tracks the last executed operation
        op_locations (list): Sorted list of operation locations
        loc_iter: Iterator over operation locations
        golden_op_iter: Iterator over golden operations
        golden_tensor_pool (TensorPool): Pool for managing tensor values
    """

    def __init__(self, registry: Registry, golden_tensor_pool: TensorPool):
        self.registry: Registry = registry
        self.last_golden_executed = None  # Tracks the last executed operation

        # Initialize operation tracking
        self.op_locations = sorted(
            self.registry.modules[ExecutionType.GOLDEN].last_loc_line.keys()
        )
        self.loc_iter = iter(self.op_locations)
        self.golden_op_iter = iter(
            self.registry.modules[ExecutionType.GOLDEN].get_function_ops()
        )
        self.golden_tensor_pool = (
            golden_tensor_pool  # Manages tensor values in the golden path
        )

    def execute(self, op: Operation, skip_op: bool = False) -> any:
        """
        Execute a single operation in the golden path.

        This method handles the execution of a single operation, including:
        - Operation validation and mapping to PyTorch equivalents
        - Input tensor retrieval
        - Operation execution
        - Output tensor storage and management

        Args:
            op (Operation): The operation to execute
            skip_op (bool, optional): If True, it will not try to load cached result

        Returns:
            any: The result of the operation execution, or None if the operation has no return value

        Raises:
            ValueError: If the operation is not found in the TTIR to PyTorch mapping
        """
        print(f"Starting execution of operation: {op.name}")
        print(f"Operation ASM: {op.get_asm(enable_debug_info=True)}")

        # Handle special case for empty operations
        op_name = op.name
        if op_name == "ttir.empty":
            return None

        # Get the operation function using GOLDEN_MAPPINGS (preferred) or fallback
        mapping_obj = _get_operation_function(op)
        if mapping_obj is None:
            raise ValueError(f"Unknown op: {op.name}")

        # If it's a wrapped OpMapping object, call it with (op, inputs)
        # Otherwise, it's a callable that expects tensor inputs
        if hasattr(mapping_obj, "__call__"):
            if hasattr(mapping_obj, "torch_op"):  # OpMapping from ttir_to_torch_mapping
                # Get operation outputs and check if we can use cached results
                outputs = get_op_outputs(op)

                # Prepare input tensors, filtering out empty tensors
                inputs_mlir = [
                    input
                    for input in get_op_inputs(op)
                    if not (
                        hasattr(input, "owner")
                        and hasattr(input.owner, "name")
                        and input.owner.name == "ttir.empty"
                    )
                ]
                input_names = [input.get_name() for input in inputs_mlir]
                print(f"Input names: {input_names}")

                # Retrieve input tensors from the pool
                inputs = [
                    self.golden_tensor_pool[name].execution_data for name in input_names
                ]

                # Execute the operation using the mapped function
                op_result = mapping_obj(op, inputs)
            else:  # Golden function from GOLDEN_MAPPINGS
                # Get operation outputs and check if we can use cached results
                outputs = get_op_outputs(op)

                # Prepare input tensors, filtering out empty tensors
                inputs_mlir = [
                    input
                    for input in get_op_inputs(op)
                    if not (
                        hasattr(input, "owner")
                        and hasattr(input.owner, "name")
                        and input.owner.name == "ttir.empty"
                    )
                ]
                input_names = [input.get_name() for input in inputs_mlir]
                print(f"Input names: {input_names}")

                # Retrieve input tensors from the pool
                inputs = [
                    self.golden_tensor_pool[name].execution_data for name in input_names
                ]

                # Execute the operation using the golden function
                op_result = mapping_obj(*inputs)
        else:
            raise ValueError(f"Invalid operation mapping for {op.name}")

        # Handle function returns specially
        if op.name == "func.return":
            return op_result

        # Process operation outputs
        for output in outputs:
            tensor_name = output.get_name()
            if op_result is not None:
                print(f"Output shape: {tensor_name} = {op_result.shape}")

            # Store or update the output tensor in the pool
            if tensor_name not in self.golden_tensor_pool:
                golden_value = TensorValue(tensor_name, op_result, ExecutionType.GOLDEN)
                golden_value.set_execution_data()
                self.golden_tensor_pool[tensor_name] = golden_value
            else:
                self.golden_tensor_pool[tensor_name].set_execution_data(op_result)

        print(f"Finished execution of operation: {op.name}")
        return op_result

    def execute_golden(self, device_op_location: Tuple[int, int], op_asm: str) -> None:
        """
        Execute operations in the golden path up to a specified location.

        This method executes operations in sequence until it reaches the operation
        corresponding to the specified device operation location. This is used to
        synchronize execution between the golden and device paths.

        Args:
            device_op_location (Tuple[int, int]): The target operation location to execute up to
            op_asm (str): Assembly representation of the operation (for debugging)

        Raises:
            AssertionError: If the execution goes past the expected location
        """
        # Get the target operation index for the given location
        target_index = self.registry.modules[ExecutionType.GOLDEN].last_loc_line[
            device_op_location
        ]

        # Execute operations in sequence until we reach the target
        for i, op in enumerate(self.golden_op_iter):
            self.execute(op)
            print(f"Executing op: {op.name}")

            # Check if we've reached the target operation
            if i == target_index:
                break

            # Safety check to prevent infinite loops
            if i > target_index:
                raise AssertionError(
                    f"Execution went past the expected location. "
                    f"Current index: {i}, Target index: {target_index}"
                )
