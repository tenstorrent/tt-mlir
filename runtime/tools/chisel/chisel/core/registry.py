# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from operator import __getitem__
from collections import defaultdict
from typing import Dict, Tuple, List
from functools import cache

from ttmlir.ir import Operation

from .ops import IRModule, get_op_outputs, get_op_inputs
from .enums import ExecutionType, Status
from ..utils.location import hash_location, UNKNOWN_LOCATION


class OpGroup:
    def __init__(self, id):
        self.id = id
        self.ops: Dict[ExecutionType, List[Operation]] = {
            ExecutionType.GOLDEN: [],
            ExecutionType.DEVICE: [],
        }

    def add_op(self, op: Operation, execution_type: ExecutionType) -> None:
        self.ops[execution_type].append(op)

    def __getitem__(self, kind: ExecutionType):
        if kind not in self.ops:
            return []
        return self.ops[kind]

    def get_last_op(
        self, kind: ExecutionType, with_output: bool = True
    ) -> Operation | None:
        if len(self.ops[kind]) == 0:
            return None
        if not with_output:
            return self.ops[kind][-1]

        for op in self.ops[kind][::-1]:
            if len(get_op_outputs(op)) > 0:
                return op


class Registry:
    def __init__(self, modules: Dict[ExecutionType, IRModule]) -> None:
        # TODO: check what is actual type of tensors, it is not Operation
        self.tensors: Dict[
            Tuple[int, int], Dict[ExecutionType, Operation]
        ] = defaultdict(dict)
        self.tensor_to_location: Dict[ExecutionType, Dict[str, Tuple[int, int]]] = {
            ExecutionType.GOLDEN: {},
            ExecutionType.DEVICE: {},
        }
        self.op_groups: Dict[Tuple[int, int], OpGroup] = {}

        self.modules = modules
        self.module_iters = {
            execution_type: iter(enumerate(module.get_function_ops()))
            for execution_type, module in modules.items()
        }
        self.last_loaded_loc: Tuple[int, int] = UNKNOWN_LOCATION

        for execution_type, module in modules.items():
            module.populate_last_loc_line()
            for arg in module.get_function_inputs():
                self.add_tensor(arg, execution_type)

    def init_ops_until(self, location: Tuple[int, int]):
        if self.last_loaded_loc >= location:
            return
        for execution_type, module in self.modules.items():
            for i, op in self.module_iters[execution_type]:
                self._add_op(op, execution_type)
                for output in get_op_outputs(op):
                    self.add_tensor(output, execution_type)

                if i >= module.last_loc_line.get(location, -1):
                    self.last_loaded_loc = location
                    break

        self._merge_empty_golden_groups()

    def should_compare(
        self,
        op: Operation,
        location_hash: Tuple[int, int],
        execution_type: ExecutionType,
    ):
        last_op = self.get_last_op(location_hash, execution_type)
        if last_op is None:
            return False
        return last_op == op

    def add_tensor(self, tensor, kind: ExecutionType):
        locatin_hash = hash_location(tensor.location)
        self.tensor_to_location[kind][tensor.get_name()] = locatin_hash
        self.tensors[locatin_hash][kind] = tensor

    def get_tensor(self, location, kind: ExecutionType):
        return self.tensors[location][kind]

    def find_op(self, location, asm: str, execution_type: ExecutionType):
        for op in self.op_groups[location].ops[execution_type]:
            if op.get_asm(enable_debug_info=True) == asm:
                return op
        return None

    def get_group(self, group_id: Tuple[int, int], execution_type: ExecutionType):
        return self.op_groups[group_id][execution_type]

    def get_group_output(
        self, group_id: Tuple[int, int], execution_type: ExecutionType
    ):
        if group_id not in self.tensors:
            print(f"Group {group_id} does not exist")
            return None
        if execution_type not in self.tensors[group_id]:
            print(f"Execution type {execution_type} does not exist in group {group_id}")
            return None
        return self.tensors[group_id][execution_type]

    def get_group_inputs(
        self, group_id: Tuple[int, int], execution_type: ExecutionType
    ):
        tensors = set()
        for op in self.op_groups[group_id][execution_type]:
            tensors.update([t for t in get_op_inputs(op)])
        for op in self.op_groups[group_id][execution_type]:
            outputs = get_op_outputs(op)
            if len(outputs) == 0:
                continue
            for t in outputs:
                if t not in tensors:
                    continue
                tensors.remove(t)
        return list(tensors)

    @cache
    def get_last_op(
        self,
        group_id: Tuple[int, int],
        execution_type: ExecutionType,
        with_output: bool = True,
    ) -> Operation | None:
        return self.op_groups[group_id].get_last_op(execution_type, with_output)

    def _add_op(self, op: Operation, execution_type: ExecutionType):
        location_hash = hash_location(op.location)
        if location_hash not in self.op_groups:
            self.op_groups[location_hash] = OpGroup(location_hash)
        self.op_groups[location_hash].add_op(op, execution_type)

    def _merge_empty_golden_groups(self):
        """
        Ida behing this is that if the golden ops were fused together,
        then the new op would get the last golden op's location.
        """
        # TODO: improve if its too slow
        # Groups are keyed by (line, col); sorting gives textual order.
        sorted_ids = sorted(self.op_groups.keys())
        idx = 0
        while idx < len(sorted_ids) - 1:  # last group has no “next”
            gid = sorted_ids[idx]
            group = self.op_groups[gid]

            # Only GOLDEN ops?
            if len(group.ops[ExecutionType.DEVICE]) != 0:
                idx += 1
                continue

            next_gid = sorted_ids[idx + 1]
            next_group = self.op_groups[next_gid]

            next_group.ops[ExecutionType.GOLDEN] = (
                group.ops[ExecutionType.GOLDEN] + next_group.ops[ExecutionType.GOLDEN]
            )

            del self.op_groups[gid]
            sorted_ids.pop(idx)
