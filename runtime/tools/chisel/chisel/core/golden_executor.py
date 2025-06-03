# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple
from ttmlir.ir import Operation
import torch

from .tensors import TensorPool, TensorValue
from .enums import ExecutionType
from .registry import Registry
from .ops import Op
from .tensors import get_op_inputs, get_op_outputs
from ..utils.mapping import ttir_to_torch_mapping
import pdb


class GoldenExecutor:
    def __init__(self, registry: Registry, golden_tensor_pool: TensorPool):
        self.registry = registry
        self.last_golden_executed = None
        # sorted location list
        self.op_locations = sorted(self.registry.op_groups.keys())
        self.loc_iter = iter(self.op_locations)
        self.golden_tensor_pool = golden_tensor_pool

    def execute(self, op: Op):
        print(f"Executing operation: {op.name}")
        print(f"Operation ASM: {op.asm}")
        print(f"Operation location: {op.location}")
        print(f"Executing op: {op.name}")

        if op.name not in ttir_to_torch_mapping:
            # TODO: enable to enter debug mode so you can add on the fly mapping if missing
            raise ValueError(f"Unknown op: {op.name}")

        mapping = ttir_to_torch_mapping[op.name]

        outputs = op.outputs
        inputs = [
            self.golden_tensor_pool[input.name].execution_data
            for i, input in enumerate(op.inputs)
            if not (
                (i == len(op.inputs) - 1)
                and (input.name not in self.golden_tensor_pool)
            )
        ]
        print(
            f"Input shapes: {[(inp.name, x.shape if x is not None else None) for inp, x in zip(op.inputs, inputs)]}"
        )
        print(f"Inputs: {inputs}")
        op_result = mapping(op, inputs)
        if op.name == "func.return":
            return op_result

        for output in outputs:
            tensor_name = output.name
            if op_result is not None:
                print(f"Output shape: {tensor_name} = {op_result.shape}")

            self.golden_tensor_pool[tensor_name] = TensorValue(
                tensor_name, op_result, ExecutionType.GOLDEN
            )
            # check if all values are nan
            # if torch.isnan(op_result).any() or torch.isinf(op_result).any():
            #     print(f"Tensor {tensor_name} has nan values")
            #     print(f"Tensor values: {op_result}")
            #     raise ValueError(f"Tensor {tensor_name} has nan values")
            print(f"Added tensor {tensor_name} to golden tensor pool")
        return op_result

    def execute_golden(self, device_op_location: Tuple[int, int], op_asm: str) -> bool:
        last_device_op = self.registry.get_last(
            device_op_location, ExecutionType.DEVICE
        )
        if last_device_op is None:
            print(f"No last device op found for location {device_op_location}")
            return False
        if last_device_op.asm != op_asm:
            print(f"ASM mismatch at {device_op_location}")
            print(f"Expected: {op_asm}")
            print(f"Got: {last_device_op.asm}")
            return False

        to_execute = []
        for loc in self.loc_iter:
            if loc <= device_op_location:
                to_execute.append(loc)
            if loc >= device_op_location:
                break

        print(f"Executing golden ops from groups: {to_execute}")
        for loc in to_execute:
            for op in self.registry.op_groups[loc].ops[ExecutionType.GOLDEN]:
                self.execute(op)
        return True
