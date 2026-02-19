# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Golden Executor for TTNN operations using golden library.

This module provides the GoldenExecutor class which executes TTNN operations in the golden
(reference/CPU) execution path using golden library implementations exclusively.

Key Features:
- Integrates with GOLDEN_MAPPINGS from golden library for operation execution
- Automatic conversion from operation names (e.g., "ttnn.abs") to operation classes
  (e.g., ttnn.AbsOp) to lookup golden library implementations
- Fails immediately if an operation does not have a golden library implementation
- Manages tensor values through a tensor pool with support for multiple execution contexts

Operation Function Resolution:
1. Looks up the operation type in golden.GOLDEN_MAPPINGS
2. Raises ValueError if operation is not found - no fallback allowed

Note: This executor works with TTNN operations only. TTIR-specific operations are not supported.
"""
from typing import Tuple
from ttmlir.ir import Operation
import torch
import importlib

from .ops import get_op_outputs, get_op_inputs
from .tensors import TensorPool, TensorValue
from .enums import ExecutionType
from .registry import Registry

from golden import (
    GOLDEN_MAPPINGS,
    get_golden_function,
    GoldenMapTensor,
    unpack_mlir_attr,
)


# Mapping of abbreviated operation names to their full class names
OPERATION_NAME_ALIASES = {
    "eq": "equal",
    "gt": "greater_than",
    "lt": "less_than",
    "ge": "greater_equal",
    "le": "less_equal",
    "ne": "not_equal",
}


class GoldenExecutor:
    """
    Executes TTNN operations in the golden (CPU reference) execution path.

    This class is responsible for executing TTNN operations using golden library implementations
    as the CPU reference. It maintains state about the execution and manages tensor values
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

        op_name = op.name

        # Handle func.return specially - just return the input tensor value
        if op_name == "func.return":
            inputs_mlir = get_op_inputs(op)
            if inputs_mlir:
                input_names = [input.get_name() for input in inputs_mlir]
                return self.golden_tensor_pool[input_names[0]].execution_data
            return None

        # Validate operation is supported in GOLDEN_MAPPINGS
        if type(op) not in GOLDEN_MAPPINGS:
            raise ValueError(
                f"Unknown op: {op.name}. "
                f"Operation type {type(op)} not found in GOLDEN_MAPPINGS. "
                f"All TTNN operations must have a golden implementation."
            )

        # Get the golden function - no fallback allowed
        golden_fn = get_golden_function(type(op))
        if golden_fn is None:
            raise ValueError(
                f"No golden implementation found for operation: {op_name}. "
                f"Operation class: {type(op)}"
            )

        # Get operation outputs
        outputs = get_op_outputs(op)

        # Get input tensors
        inputs_mlir = get_op_inputs(op)
        input_names = [input.get_name() for input in inputs_mlir]

        # Retrieve input tensors from the pool
        inputs = [self.golden_tensor_pool[name].execution_data for name in input_names]

        # Execute the operation using the golden function
        # TTNN operations have standardized golden implementations
        try:
            op_result = golden_fn(*inputs) if inputs else golden_fn()
        except Exception as e:
            print(f"Error executing golden function for {op_name}: {e}")
            raise

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
