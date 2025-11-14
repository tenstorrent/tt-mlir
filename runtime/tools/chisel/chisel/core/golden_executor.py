# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Golden Executor for TTIR operations using golden library.

This module provides the GoldenExecutor class which executes operations in the golden
(reference) execution path using golden library implementations exclusively.

Key Features:
- Integrates with GOLDEN_MAPPINGS from golden library for operation execution
- Automatic conversion from operation names (e.g., "ttir.abs") to operation classes
  (e.g., ttir.AbsOp) to lookup golden library implementations
- Fails immediately if an operation does not have a golden library implementation
- Manages tensor values through a tensor pool with support for multiple execution contexts

Operation Function Resolution:
1. Looks up the operation type in golden.GOLDEN_MAPPINGS
2. Raises ValueError if operation is not found - no fallback allowed
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

        # Handle special cases that don't need golden functions
        op_name = op.name
        if op_name == "ttir.empty":
            return None

        if op_name == "func.return":
            # For func.return, just return the input tensor value from the pool
            inputs_mlir = get_op_inputs(op)
            if inputs_mlir:
                input_names = [input.get_name() for input in inputs_mlir]
                return self.golden_tensor_pool[input_names[0]].execution_data
            return None

        # Validate operation is supported
        if type(op) not in GOLDEN_MAPPINGS:
            raise ValueError(f"Unknown op: {op.name}")

        # Get the golden function from builder_golden - no fallback allowed
        golden_fn = get_golden_function(type(op))
        if golden_fn is None:
            raise ValueError(
                f"No builder_golden implementation found for operation: {op_name}. "
                f"All operations must have a builder_golden implementation. "
                f"Operation class: {type(op)}"
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

        # Collect operation attributes to pass as kwargs
        # Unpack MLIR attributes to Python types for golden functions
        kwargs = {}
        for named_attr in op.attributes:
            attr_name = named_attr.name
            attr_value = named_attr.attr
            try:
                kwargs[attr_name] = unpack_mlir_attr(attr_value)
            except ValueError:
                # If unpacking fails, pass the raw attribute
                # (some golden functions may handle MLIR attributes directly)
                kwargs[attr_name] = attr_value

        # Execute the operation using the golden function
        try:
            if op_name == "ttir.dot_general":
                batch_dims_lhs = []
                batch_dims_rhs = []
                contract_dims_lhs = []
                contract_dims_rhs = []

                if "batch_dims_lhs" in op.attributes:
                    batch_dims_lhs = [int(x) for x in op.attributes["batch_dims_lhs"]]
                if "batch_dims_rhs" in op.attributes:
                    batch_dims_rhs = [int(x) for x in op.attributes["batch_dims_rhs"]]
                if "contract_dims_lhs" in op.attributes:
                    contract_dims_lhs = [
                        int(x) for x in op.attributes["contract_dims_lhs"]
                    ]
                if "contract_dims_rhs" in op.attributes:
                    contract_dims_rhs = [
                        int(x) for x in op.attributes["contract_dims_rhs"]
                    ]

                # Simple implementation for common case: no batching, use torch.tensordot
                if len(batch_dims_lhs) == 0 and len(batch_dims_rhs) == 0:
                    # For simple case with no batching, use torch.tensordot directly
                    try:
                        op_result = torch.tensordot(
                            inputs[0],
                            inputs[1],
                            dims=(contract_dims_lhs, contract_dims_rhs),
                        )
                    except Exception as e:
                        raise
                else:
                    # For batched case, wrap inputs as GoldenMapTensor and use golden function
                    dot_kwargs = {
                        "batch_dims_lhs": batch_dims_lhs,
                        "batch_dims_rhs": batch_dims_rhs,
                        "contract_dims_lhs": contract_dims_lhs,
                        "contract_dims_rhs": contract_dims_rhs,
                    }
                    wrapped_inputs = []
                    for inp in inputs:
                        if isinstance(inp, GoldenMapTensor):
                            wrapped_inputs.append(inp)
                        else:
                            wrapped_inputs.append(GoldenMapTensor({0: inp}, (1, 1)))
                    op_result = golden_fn(*wrapped_inputs, **dot_kwargs)
            elif op_name == "ttir.broadcast":
                # Special handling for broadcast: torch.broadcast_to needs the target shape
                # The second input (from ttir.empty) was filtered out, so we get the shape
                # from the operation's result type
                if outputs and len(outputs) > 0:
                    output_type = outputs[0].type
                    if hasattr(output_type, "shape"):
                        target_shape = list(output_type.shape)
                        op_result = golden_fn(inputs[0], target_shape)
                    else:
                        # Fallback if we can't get shape
                        op_result = golden_fn(*inputs, **kwargs)
                else:
                    op_result = golden_fn(*inputs, **kwargs)
            elif op_name == "ttir.pad":
                # Extract required attributes for pad operation
                padding_attr = op.attributes["padding"]
                value_attr = op.attributes["value"]
                value = float(value_attr.value)

                # Convert padding ArrayAttr to list of integers
                padding = [int(x) for x in padding_attr]
                kwargs = {}
                if "padding" in op.attributes:
                    kwargs["padding"] = padding
                if "value" in op.attributes:
                    kwargs["value"] = value

                op_result = golden_fn(*inputs, **kwargs)
            else:
                # Pass attributes as kwargs to all golden functions
                # if inputs:
                #     op_result = golden_fn(*inputs, **kwargs)
                # else:
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
