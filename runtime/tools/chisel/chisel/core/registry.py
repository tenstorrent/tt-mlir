# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torchgen.api.native import name
from collections import defaultdict
from typing import Dict, Tuple

from ttmlir.ir import Operation

from .ops import IRModule
from .enums import ExecutionType, Status
from ..utils.location import hash_location
from .tensors import get_op_outputs

import pandas as pd
from pathlib import Path
import os


class OpGroup:
    def __init__(self, id):
        self.id = id
        self.ops = {ExecutionType.GOLDEN: [], ExecutionType.DEVICE: []}
        self.status = Status.PENDING

    def add_op(self, op: Operation, execution_type: ExecutionType):
        self.ops[execution_type].append(op)

    def __len__(self):
        return len(self.ops)

    def items(self):
        return self.ops.items()

    def get_last(self, kind: ExecutionType, with_output: bool = True):
        if with_output:
            for op in self.ops[kind][::-1]:
                if len(get_op_outputs(op)) > 0:
                    return op
        else:
            return self.ops[kind][-1]


class Registry:
    def __init__(self, modules: Dict[ExecutionType, IRModule], name: str = ""):
        self.tensors = defaultdict(dict)
        self.tensor_to_location: Dict[ExecutionType, Dict[str, Tuple[int, int]]] = {
            ExecutionType.GOLDEN: {},
            ExecutionType.DEVICE: {},
        }
        self.op_groups = {}
        self.name = name

        for execution_type, module in modules.items():
            print(f"Adding inputs for {execution_type}")
            for arg in module.get_function_inputs():
                # print(f"Adding input {arg.get_name()} for {execution_type}")
                self.add_tensor(arg, execution_type)

            print(f"Adding ops for {execution_type}")
            for op in module.main_body_ops():
                self.add_op(op, execution_type)
                for output in get_op_outputs(op):
                    self.add_tensor(output, execution_type)

        self._merge_empty_golden_groups()

    def add_tensor(self, tensor, kind: ExecutionType):
        self.tensor_to_location[kind][tensor.get_name()] = tensor.location
        self.tensors[tensor.location][kind] = tensor

    def get_tensor(self, tensor, kind: ExecutionType):
        return self.tensors[tensor.location][kind]

    def add_op(self, op: Operation, execution_type: ExecutionType):
        location_hash = hash_location(op.location)
        if location_hash not in self.op_groups:
            self.op_groups[location_hash] = OpGroup(location_hash)
        self.op_groups[location_hash].add_op(op, execution_type)

    def find_op(self, location, asm: str, execution_type: ExecutionType):
        for op in self.op_groups[location].ops[execution_type]:
            if op.get_asm(enable_debug_info=True) == asm:
                return op
        return None

    def get_last(
        self,
        group_id: Tuple[int, int],
        execution_type: ExecutionType,
        with_output: bool = True,
    ):
        return self.op_groups[group_id].get_last(execution_type, with_output)

    def _merge_empty_golden_groups(self):
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

            # Move GOLDEN ops over.
            next_group.ops[ExecutionType.GOLDEN] = (
                group.ops[ExecutionType.GOLDEN] + next_group.ops[ExecutionType.GOLDEN]
            )

            # Remove the empty group and update our traversal list.
            del self.op_groups[gid]
            sorted_ids.pop(idx)  # keep idx at same position

    def dump_registry(self, out_path="registry_dump.xlsx"):
        """
        Export the registry as an Excel spreadsheet.

        Columns:
        • Location   –  (line, col) of the op group
        • Golden ops –  newline-separated list of golden op asm
        • Device ops –  newline-separated list of device op asm
        """
        rows = []
        for loc, group in sorted(self.op_groups.items()):
            golden_ops = [
                op.get_asm(enable_debug_info=False).strip()
                for op in group.ops.get(ExecutionType.GOLDEN, [])
            ]
            device_ops = [
                op.get_asm(enable_debug_info=False).strip()
                for op in group.ops.get(ExecutionType.DEVICE, [])
            ]

            rows.append(
                {
                    "Location": f"{loc[0]}:{loc[1]}"
                    if isinstance(loc, tuple)
                    else str(loc),
                    "Golden ops": "\n".join(golden_ops),
                    "Device ops": "\n".join(device_ops),
                }
            )

        df = pd.DataFrame(rows, columns=["Location", "Golden ops", "Device ops"])

        out_path = Path(out_path).with_suffix(".xlsx")
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="registry", index=False)

            # simple formatting tweaks
            wb = writer.book
            fmt = wb.add_format({"text_wrap": True, "valign": "top"})
            ws = writer.sheets["registry"]
            ws.set_column(0, 0, 15)  # Location column width
            ws.set_column(1, 2, 60, fmt)  # wrap the long lists

        print(f"Registry written to {out_path.resolve()}")
