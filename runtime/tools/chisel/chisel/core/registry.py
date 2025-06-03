# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import Dict, Tuple

from ttmlir.ir import Operation

from .tensors import TensorDescriptor, get_function_inputs, get_op_outputs
from .ops import IRModule, Op
from .enums import ExecutionType, Status
from ..utils.location import hash_location


class OpGroup:
    def __init__(self, id):
        self.id = id
        self.ops = {ExecutionType.GOLDEN: [], ExecutionType.DEVICE: []}
        self.status = Status.PENDING

    def add_op(self, op: Op):
        self.ops[op.execution_type].append(op)

    def __len__(self):
        return len(self.ops)

    def items(self):
        return self.ops.items()

    def get_last(self, kind: ExecutionType, with_output: bool = True):
        if with_output:
            for op in self.ops[kind][::-1]:
                if len(op.outputs) > 0:
                    return op
        else:
            return self.ops[kind][-1]


class Registry:
    def __init__(self, modules: Dict[ExecutionType, IRModule]):
        self.tensors = defaultdict(dict)
        self.tensor_to_location: Dict[ExecutionType, Dict[str, Tuple[int, int]]] = {
            ExecutionType.GOLDEN: {},
            ExecutionType.DEVICE: {},
        }
        self.op_groups = {}

        for execution_type, module in modules.items():
            print(f"Adding inputs for {execution_type}")
            for arg in get_function_inputs(module.get_main_op(), execution_type):
                print(f"Adding input {arg.name} for {execution_type}")
                self.add_tensor(arg, execution_type)

            print(f"Adding ops for {execution_type}")
            for op in module.main_body_ops():
                self.add_op(op, execution_type)
                for output in op.outputs:
                    self.add_tensor(output, execution_type)

    def add_tensor(self, tensor: TensorDescriptor, kind: ExecutionType):
        self.tensor_to_location[kind][tensor.name] = tensor.location_hash
        self.tensors[tensor.location_hash][kind] = tensor

    def get_tensor(self, tensor: TensorDescriptor, kind: ExecutionType):
        return self.tensors[tensor.location_hash][kind]

    def add_op(self, op: Op, execution_type: ExecutionType):
        if op.location not in self.op_groups:
            self.op_groups[op.location] = OpGroup(op.location)
        self.op_groups[op.location].add_op(op)

    def find_op(
        self, location_hash: Tuple[int, int], asm: str, execution_type: ExecutionType
    ):
        for op in self.op_groups[location_hash].ops[execution_type]:
            if op.asm == asm:
                return op
        return None

    def get_last(
        self,
        group_id: Tuple[int, int],
        execution_type: ExecutionType,
        with_output: bool = True,
    ):
        return self.op_groups[group_id].get_last(execution_type, with_output)

    def print(self):
        print("\n" * 2)
        print("--------------------------------")
        print("Op Groups")
        print("--------------------------------")
        for group_id in sorted(self.op_groups):
            op_groups = self.op_groups[group_id]
            print(f"Group {group_id}:")
            for execution_type, ops in op_groups.items():
                print(f"\t{execution_type}:")
                for op in ops:
                    print(f"\t\t{op.name} {op.location}")

        print("\n" * 2)
        print("--------------------------------")
        print("Tensors")
        print("--------------------------------")
        for group_id, tensors in self.tensors.items():
            print(f"Group {group_id}:")
            for execution_type, tensor in tensors.items():
                print(
                    f"\t{execution_type}: {tensor.name} {tensor.dtype} {tensor.shape}"
                )
