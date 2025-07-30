# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple
from ttmlir.ir import Operation
from ttmlir.dialects import ttir
import torch
import sys
import os

from .ops import get_op_outputs, get_op_inputs
from .tensors import TensorPool, TensorValue
from .enums import ExecutionType
from .registry import Registry

# Import golden mappings from ttir_golden.py
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "tools",
            "ttir-builder",
        )
    )
)
from ttir_golden import get_golden_function


def _create_op_name_to_class_mapping():
    """Create a mapping from operation names to TTIR operation classes."""
    return {
        "ttir.abs": ttir.AbsOp,
        "ttir.add": ttir.AddOp,
        "ttir.arange": ttir.ArangeOp,
        "ttir.argmax": ttir.ArgMaxOp,
        "ttir.atan": ttir.AtanOp,
        "ttir.broadcast": ttir.BroadcastOp,
        "ttir.cbrt": ttir.CbrtOp,
        "ttir.ceil": ttir.CeilOp,
        "ttir.clamp": ttir.ClampScalarOp,
        "ttir.concat": ttir.ConcatOp,
        "ttir.conv2d": ttir.Conv2dOp,
        "ttir.cos": ttir.CosOp,
        "ttir.cumsum": ttir.CumSumOp,
        "ttir.div": ttir.DivOp,
        "ttir.embedding": ttir.EmbeddingOp,
        "ttir.eq": ttir.EqualOp,
        "ttir.exp": ttir.ExpOp,
        "ttir.expm1": ttir.Expm1Op,
        "ttir.fill_cache": ttir.FillCacheOp,
        "ttir.floor": ttir.FloorOp,
        "ttir.gather": ttir.GatherOp,
        "ttir.ge": ttir.GreaterEqualOp,
        "ttir.gelu": ttir.GeluOp,
        "ttir.gt": ttir.GreaterThanOp,
        "ttir.is_finite": ttir.IsFiniteOp,
        "ttir.le": ttir.LessEqualOp,
        "ttir.linear": ttir.LinearOp,
        "ttir.log": ttir.LogOp,
        "ttir.log1p": ttir.Log1pOp,
        "ttir.logical_and": ttir.LogicalAndOp,
        "ttir.logical_not": ttir.LogicalNotOp,
        "ttir.logical_or": ttir.LogicalOrOp,
        "ttir.logical_xor": ttir.LogicalXorOp,
        "ttir.lt": ttir.LessThanOp,
        "ttir.matmul": ttir.MatmulOp,
        "ttir.max": ttir.MaxOp,
        "ttir.maximum": ttir.MaximumOp,
        "ttir.max_pool2d": ttir.MaxPool2dOp,
        "ttir.mean": ttir.MeanOp,
        "ttir.min": ttir.MinOp,
        "ttir.minimum": ttir.MinimumOp,
        "ttir.multiply": ttir.MultiplyOp,
        "ttir.ne": ttir.NotEqualOp,
        "ttir.neg": ttir.NegOp,
        "ttir.ones": ttir.OnesOp,
        "ttir.pad": ttir.PadOp,
        "ttir.permute": ttir.PermuteOp,
        "ttir.pow": ttir.PowOp,
        "ttir.prod": ttir.ProdOp,
        "ttir.reciprocal": ttir.ReciprocalOp,
        "ttir.reduce_and": ttir.ReduceAndOp,
        "ttir.reduce_or": ttir.ReduceOrOp,
        "ttir.relu": ttir.ReluOp,
        "ttir.remainder": ttir.RemainderOp,
        "ttir.repeat": ttir.RepeatOp,
        "ttir.repeat_interleave": ttir.RepeatInterleaveOp,
        "ttir.reshape": ttir.ReshapeOp,
        "ttir.reverse": ttir.ReverseOp,
        "ttir.rsqrt": ttir.RsqrtOp,
        "ttir.sigmoid": ttir.SigmoidOp,
        "ttir.sign": ttir.SignOp,
        "ttir.sin": ttir.SinOp,
        "ttir.slice": ttir.SliceOp,
        "ttir.softmax": ttir.SoftmaxOp,
        "ttir.sqrt": ttir.SqrtOp,
        "ttir.squeeze": ttir.SqueezeOp,
        "ttir.subtract": ttir.SubtractOp,
        "ttir.sum": ttir.SumOp,
        "ttir.tan": ttir.TanOp,
        "ttir.tanh": ttir.TanhOp,
        "ttir.to_layout": ttir.ToLayoutOp,
        "ttir.transpose": ttir.TransposeOp,
        "ttir.typecast": ttir.TypecastOp,
        "ttir.unsqueeze": ttir.UnsqueezeOp,
        "ttir.update_cache": ttir.UpdateCacheOp,
        "ttir.upsample2d": ttir.Upsample2dOp,
        "ttir.view_layout": ttir.ViewLayoutOp,
        "ttir.where": ttir.WhereOp,
        "ttir.zeros": ttir.ZerosOp,
    }


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
        op_name_to_class (dict): Mapping from operation names to TTIR operation classes
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

        # Create mapping from operation names to classes for golden function lookup
        self.op_name_to_class = _create_op_name_to_class_mapping()

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
            ValueError: If the operation is not found in the golden mappings
        """
        print(f"Starting execution of operation: {op.name}")
        print(f"Operation ASM: {op.get_asm(enable_debug_info=True)}")

        # Handle special case for empty operations
        op_name = op.name
        if op_name == "ttir.empty":
            return None

        # Handle function return operations
        if op_name == "func.return":
            # For func.return, we just return the input tensor(s)
            inputs_mlir = get_op_inputs(op)
            if inputs_mlir:
                input_names = [input.get_name() for input in inputs_mlir]
                inputs = [
                    self.golden_tensor_pool[name].execution_data for name in input_names
                ]
                return inputs[0] if len(inputs) == 1 else inputs
            return None

        # Get the TTIR operation class for this operation
        if op_name not in self.op_name_to_class:
            raise ValueError(f"Unknown op: {op.name}")

        op_class = self.op_name_to_class[op_name]

        golden_function = get_golden_function(op_class)

        # Get operation outputs
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
        inputs = [self.golden_tensor_pool[name].execution_data for name in input_names]

        # Extract operation attributes for golden functions that need them
        op_attrs = {}
        for attr in op.attributes:
            op_attrs[attr.name] = attr.attr

        # Execute the operation using the golden function
        # Golden functions expect inputs as *args and attributes as **kwargs
        op_result = golden_function(*inputs, **op_attrs)

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
