# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Golden Executor for TTIR operations using builder_golden.

This module provides the GoldenExecutor class which executes operations in the golden
(reference) execution path using builder_golden implementations exclusively.

Key Features:
- Integrates with GOLDEN_MAPPINGS from builder_golden.py for operation execution
- Automatic conversion from operation names (e.g., "ttir.abs") to operation classes
  (e.g., ttir.AbsOp) to lookup builder_golden implementations
- Fails immediately if an operation does not have a builder_golden implementation
- Manages tensor values through a tensor pool with support for multiple execution contexts

Operation Function Resolution:
1. Converts operation name to operation class using _get_op_class_from_name()
2. Looks up the operation in builder_golden.GOLDEN_MAPPINGS
3. Raises ValueError if operation is not found - no fallback allowed

Example operation name conversions:
- "ttir.abs" → ttir.AbsOp
- "ttir.add" → ttir.AddOp
- "ttir.get_dimension_size" → ttir.GetDimensionSizeOp
- "stablehlo.add" → stablehlo.AddOp
"""
from typing import Tuple
from ttmlir.ir import Operation
import torch
import importlib

from .ops import get_op_outputs, get_op_inputs
from .tensors import TensorPool, TensorValue
from .enums import ExecutionType
from .registry import Registry

from builder.base.builder_golden import GOLDEN_MAPPINGS, get_golden_function


# Mapping of abbreviated operation names to their full class names
OPERATION_NAME_ALIASES = {
    "eq": "equal",
    "gt": "greater_than",
    "lt": "less_than",
    "ge": "greater_equal",
    "le": "less_equal",
    "ne": "not_equal",
}


def _get_op_class_from_name(op_name: str):
    """
    Convert operation name (e.g., "ttir.abs") to operation class (e.g., ttir.AbsOp).

    Args:
        op_name (str): Operation name like "ttir.abs" or "stablehlo.add"

    Returns:
        type: The operation class, or None if not found
    """
    # Parse dialect and operation name
    parts = op_name.split(".")
    if len(parts) < 2:
        return None

    dialect_name = parts[0]  # e.g., "ttir", "stablehlo", "d2m"
    op_short_name = ".".join(parts[1:])  # e.g., "abs", "to_layout"

    # Check if this is an abbreviated operation name and expand it
    if op_short_name in OPERATION_NAME_ALIASES:
        op_short_name = OPERATION_NAME_ALIASES[op_short_name]

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


class GoldenExecutor:
    """
    Executes operations in the golden (reference) execution path.

    This class is responsible for executing operations using builder_golden as the reference
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
        - Operation validation and mapping to builder_golden implementations
        - Input tensor retrieval
        - Operation execution
        - Output tensor storage and management

        Args:
            op (Operation): The operation to execute
            skip_op (bool, optional): If True, it will not try to load cached result

        Returns:
            any: The result of the operation execution, or None if the operation has no return value

        Raises:
            ValueError: If the operation is not found in builder_golden GOLDEN_MAPPINGS
        """
        print(f"Starting execution of operation: {op.name}")
        print(f"Operation ASM: {op.get_asm(enable_debug_info=True)}")

        # Handle special case for empty operations
        op_name = op.name
        if op_name == "ttir.empty":
            return None

        # Get the operation class from operation name
        op_class = _get_op_class_from_name(op_name)
        if op_class is None:
            raise ValueError(
                f"Could not resolve operation class for {op_name}. "
                f"Operation must be in the form 'dialect.operation_name'"
            )

        # Get the golden function from builder_golden - no fallback allowed
        golden_fn = get_golden_function(op_class)
        if golden_fn is None:
            raise ValueError(
                f"No builder_golden implementation found for operation: {op_name}. "
                f"All operations must have a builder_golden implementation. "
                f"Operation class: {op_class}"
            )

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

        # Retrieve input tensors from the pool
        inputs = [self.golden_tensor_pool[name].execution_data for name in input_names]

        # Execute the operation using the golden function
        try:
            op_result = golden_fn(*inputs) if inputs else golden_fn()
        except Exception as e:
            print(f"Error executing golden function for {op_name}: {e}")
            raise

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
