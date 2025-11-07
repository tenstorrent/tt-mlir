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

from golden import GOLDEN_MAPPINGS, get_golden_function
from golden.mapping import GoldenMapTensor


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

        # Handle special case for empty operations and function returns
        op_name = op.name
        if op_name == "ttir.empty":
            return None

        # Handle function returns specially - they don't need golden execution
        if op_name == "func.return":
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

        # Debug logging for problematic operations
        if op_name in [
            "ttir.convolution",
            "ttir.pooling",
            "ttir.batch_norm_inference",
            "ttir.add",
            "ttir.permute",
            "ttir.dot_general",
        ]:
            print(f"\n=== DEBUG {op_name.upper()} INPUTS ===")
            for i, (name, tensor) in enumerate(zip(input_names, inputs)):
                print(f"Input {i} ({name}):")
                print(f"  Type: {type(tensor)}")
                if hasattr(tensor, "shape"):
                    print(f"  Shape: {tensor.shape}")
                if hasattr(tensor, "shard_map"):
                    print(f"  Is GoldenMapTensor: True")
                    print(f"  Number of shards: {len(tensor.shard_map)}")
                    for shard_id, shard in tensor.shard_map.items():
                        print(f"    Shard {shard_id}: shape = {shard.shape}")
                else:
                    print(f"  Is GoldenMapTensor: False")
            if op_name == "ttir.batch_norm_inference":
                print(
                    f"  Expected output shape from MLIR: {outputs[0].type if outputs else 'N/A'}"
                )
            print(f"=== END DEBUG ===\n")

        # Execute the operation using the golden function
        try:
            # Special case for ttir.pad - extract padding and value attributes
            if op_name == "ttir.pad":
                padding_attr = op.attributes["padding"]
                value_attr = op.attributes["value"]

                # Convert padding ArrayAttr to list of integers
                padding = [int(x) for x in padding_attr]

                # Convert FloatAttr to Python float
                # FloatAttr has a .value property that gives the float value
                value = float(value_attr.value)

                op_result = golden_fn(*inputs, padding, value)
            # Special case for ttir.pooling - extract pooling attributes
            elif op_name == "ttir.pooling":
                pooling_method = op.attributes["pooling_method"]
                window_dimensions = op.attributes["window_dimensions"]
                window_strides = op.attributes["window_strides"]
                padding = op.attributes["padding"]
                window_dilations = op.attributes["window_dilations"]

                op_result = golden_fn(
                    *inputs,
                    pooling_method=pooling_method,
                    window_dimensions=window_dimensions,
                    window_strides=window_strides,
                    padding=padding,
                    window_dilations=window_dilations,
                )
            # Special case for ttir.convolution - extract convolution attributes
            elif op_name == "ttir.convolution":
                # Extract required attributes
                kwargs = {}
                if "padding" in op.attributes:
                    kwargs["padding"] = op.attributes["padding"]
                if "window_strides" in op.attributes:
                    kwargs["window_strides"] = op.attributes["window_strides"]
                if "input_dilation" in op.attributes:
                    kwargs["input_dilation"] = op.attributes["input_dilation"]
                if "weight_dilation" in op.attributes:
                    kwargs["weight_dilation"] = op.attributes["weight_dilation"]
                # Note: convolution_layout is skipped because ConvolutionLayoutAttr doesn't have
                # Python bindings yet. For now, we assume standard NCHW layout which is PyTorch's default.
                # TODO (dloke): Add Python bindings for ConvolutionLayoutAttr to support custom layouts
                if "feature_group_count" in op.attributes:
                    # Convert IntegerAttr to Python int
                    kwargs["feature_group_count"] = int(
                        op.attributes["feature_group_count"]
                    )
                if "batch_group_count" in op.attributes:
                    # Convert IntegerAttr to Python int
                    kwargs["batch_group_count"] = int(
                        op.attributes["batch_group_count"]
                    )
                if "window_reversal" in op.attributes:
                    kwargs["window_reversal"] = op.attributes["window_reversal"]

                op_result = golden_fn(*inputs, **kwargs)
            # Special case for ttir.conv2d - extract conv2d attributes
            elif op_name == "ttir.conv2d":
                # Extract required attributes
                kwargs = {}
                if "stride" in op.attributes:
                    kwargs["stride"] = op.attributes["stride"]
                if "padding" in op.attributes:
                    kwargs["padding"] = op.attributes["padding"]
                if "dilation" in op.attributes:
                    kwargs["dilation"] = op.attributes["dilation"]
                if "groups" in op.attributes:
                    # Convert IntegerAttr to Python int
                    kwargs["groups"] = int(op.attributes["groups"])

                op_result = golden_fn(*inputs, **kwargs)
            # Special case for reduction operations - extract reduction attributes
            elif op_name in [
                "ttir.sum",
                "ttir.mean",
                "ttir.max",
                "ttir.min",
                "ttir.prod",
            ]:
                kwargs = {}
                if "dim_arg" in op.attributes:
                    # Convert to list of ints (or single int for some ops)
                    dim_arg = [int(x) for x in op.attributes["dim_arg"]]
                    # Some ops expect a single int, others a list
                    if op_name in ["ttir.max", "ttir.min"] and len(dim_arg) == 1:
                        kwargs["dim_arg"] = dim_arg[0]
                    else:
                        kwargs["dim_arg"] = dim_arg
                if "keep_dim" in op.attributes:
                    # Convert BoolAttr to Python bool
                    kwargs["keep_dim"] = bool(op.attributes["keep_dim"])

                op_result = golden_fn(*inputs, **kwargs)
            # Special case for ttir.dot_general - extract dimension attributes
            elif op_name == "ttir.dot_general":
                print(f"\n[DEBUG dot_general] ===== STARTING dot_general =====")
                print(f"[DEBUG dot_general] Number of inputs: {len(inputs)}")
                for i, inp in enumerate(inputs):
                    print(
                        f"[DEBUG dot_general] Input {i} raw shape: {inp.shape}, type: {type(inp).__name__}"
                    )

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

                print(f"[DEBUG dot_general] Extracted attributes:")
                print(f"[DEBUG dot_general]   batch_dims_lhs: {batch_dims_lhs}")
                print(f"[DEBUG dot_general]   batch_dims_rhs: {batch_dims_rhs}")
                print(f"[DEBUG dot_general]   contract_dims_lhs: {contract_dims_lhs}")
                print(f"[DEBUG dot_general]   contract_dims_rhs: {contract_dims_rhs}")

                # Simple implementation for common case: no batching, use torch.tensordot
                if len(batch_dims_lhs) == 0 and len(batch_dims_rhs) == 0:
                    # For simple case with no batching, use torch.tensordot directly
                    print(
                        f"[DEBUG dot_general] Input 0 shape: {inputs[0].shape}, type: {type(inputs[0])}"
                    )
                    print(
                        f"[DEBUG dot_general] Input 1 shape: {inputs[1].shape}, type: {type(inputs[1])}"
                    )
                    print(f"[DEBUG dot_general] contract_dims_lhs: {contract_dims_lhs}")
                    print(f"[DEBUG dot_general] contract_dims_rhs: {contract_dims_rhs}")
                    print(f"[DEBUG dot_general] batch_dims_lhs: {batch_dims_lhs}")
                    print(f"[DEBUG dot_general] batch_dims_rhs: {batch_dims_rhs}")
                    if hasattr(inputs[0], "shards"):
                        print(
                            f"[DEBUG dot_general] Input 0 is sharded - shards keys: {inputs[0].shards.keys()}"
                        )
                        for k, shard in inputs[0].shards.items():
                            print(
                                f"[DEBUG dot_general]   Shard {k}: shape {shard.shape}"
                            )
                    if hasattr(inputs[1], "shards"):
                        print(
                            f"[DEBUG dot_general] Input 1 is sharded - shards keys: {inputs[1].shards.keys()}"
                        )
                        for k, shard in inputs[1].shards.items():
                            print(
                                f"[DEBUG dot_general]   Shard {k}: shape {shard.shape}"
                            )

                    print(f"[DEBUG dot_general] About to call torch.tensordot...")
                    try:
                        op_result = torch.tensordot(
                            inputs[0],
                            inputs[1],
                            dims=(contract_dims_lhs, contract_dims_rhs),
                        )
                        print(
                            f"[DEBUG dot_general] SUCCESS! Result shape: {op_result.shape}"
                        )
                    except Exception as e:
                        print(f"[DEBUG dot_general] FAILED with error: {e}")
                        print(f"[DEBUG dot_general] Error type: {type(e).__name__}")
                        raise
                else:
                    # For batched case, wrap inputs as GoldenMapTensor and use golden function
                    kwargs = {
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
                    op_result = golden_fn(*wrapped_inputs, **kwargs)
            elif op_name == "ttir.permute":
                # Special debug for permute operation
                print(f"\n[DEBUG PERMUTE] ===== PERMUTE OPERATION =====")
                print(f"[DEBUG PERMUTE] Number of inputs: {len(inputs)}")
                for i, inp in enumerate(inputs):
                    print(
                        f"[DEBUG PERMUTE] Input {i} shape: {inp.shape}, type: {type(inp).__name__}"
                    )
                print(f"[DEBUG PERMUTE] Operation attributes: {op.attributes}")
                if "permutation" in op.attributes:
                    permutation = op.attributes["permutation"]
                    print(f"[DEBUG PERMUTE] Permutation attribute: {permutation}")
                    print(f"[DEBUG PERMUTE] Permutation type: {type(permutation)}")
                    # Try to extract the actual permutation values
                    try:
                        perm_values = [int(x) for x in permutation]
                        print(f"[DEBUG PERMUTE] Permutation values: {perm_values}")
                    except Exception as e:
                        print(
                            f"[DEBUG PERMUTE] Could not extract permutation values: {e}"
                        )

                # Call golden function with permutation as kwarg
                kwargs = {"permutation": op.attributes["permutation"]}
                print(f"[DEBUG PERMUTE] Calling golden_fn with kwargs: {kwargs}")
                op_result = golden_fn(*inputs, **kwargs)
                print(
                    f"[DEBUG PERMUTE] Result shape: {op_result.shape}, type: {type(op_result).__name__}"
                )
                print(f"[DEBUG PERMUTE] ===== END PERMUTE =====\n")
            else:
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

                # Debug logging for problematic operation outputs
                if op_name in [
                    "ttir.pooling",
                    "ttir.batch_norm_inference",
                    "ttir.convolution",
                    "ttir.permute",
                ]:
                    print(f"\n=== DEBUG {op_name.upper()} OUTPUT ===")
                    print(f"Operation: {op.name}")
                    print(f"Tensor name: {tensor_name}")
                    print(f"Result Type: {type(op_result)}")
                    print(f"Result Shape: {op_result.shape}")
                    if hasattr(op_result, "shard_map"):
                        print(f"Is GoldenMapTensor: True")
                        print(f"Number of shards: {len(op_result.shard_map)}")
                        for shard_id, shard in op_result.shard_map.items():
                            print(f"  Shard {shard_id}: shape = {shard.shape}")
                    else:
                        print(f"Is GoldenMapTensor: False")
                    print(f"Expected output shape from MLIR: {output.type}")
                    if op_name == "ttir.convolution":
                        # Extract convolution parameters for debugging
                        print(f"Convolution attributes:")
                        for attr_name in [
                            "padding",
                            "window_strides",
                            "input_dilation",
                            "weight_dilation",
                        ]:
                            if attr_name in op.attributes:
                                print(f"  {attr_name}: {op.attributes[attr_name]}")
                    print(f"=== END DEBUG ===\n")

                # Debug logging for specific tensor
                if tensor_name == "%40":
                    print(f"\n=== DEBUG TENSOR %40 OUTPUT ===")
                    print(f"Operation: {op.name}")
                    print(f"Result Type: {type(op_result)}")
                    print(f"Result Shape: {op_result.shape}")
                    if hasattr(op_result, "shard_map"):
                        print(f"Is GoldenMapTensor: True")
                        print(f"Number of shards: {len(op_result.shard_map)}")
                        for shard_id, shard in op_result.shard_map.items():
                            print(f"  Shard {shard_id}: shape = {shard.shape}")
                    else:
                        print(f"Is GoldenMapTensor: False")
                    print(f"=== END DEBUG ===\n")

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
        # Check if the device operation location exists in the golden module
        # This can happen when device operations have locations that don't correspond
        # to any golden operation (e.g., after operation fusion or merging)
        if (
            device_op_location
            not in self.registry.modules[ExecutionType.GOLDEN].last_loc_line
        ):
            # Find the golden operations that correspond to this device operation group
            if device_op_location not in self.registry.op_groups:
                return

            golden_ops = self.registry.op_groups[device_op_location].ops[
                ExecutionType.GOLDEN
            ]
            if len(golden_ops) == 0:
                return

            # Execute all golden operations in this group
            for golden_op in golden_ops:
                self.execute(golden_op)
            return

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
