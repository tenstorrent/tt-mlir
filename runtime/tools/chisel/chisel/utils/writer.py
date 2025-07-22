# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import csv
from typing import List

from ttmlir.ir import Operation

from ..core.enums import ExecutionType


class ReportWriter:
    def __init__(self, file_path, asm_state) -> None:
        self.file_path = file_path
        self.column_names = [
            "location",
            "golden_ops",
            "device_ops",
            "golden_output",
            "device_output",
            "golden_inputs",
            "device_inputs",
            "pcc",
            "abs_error",
            "rel_error",
            "golden_output_tensor",
            "device_output_tensor",
        ]
        self.asm_state = asm_state
        self._init_file()

    def _format_ops(self, ops: List[Operation], kind: ExecutionType):
        # import pdb; pdb.set_trace()
        return "\n".join([op.get_asm(enable_debug_info=True) for op in ops])

    def _format_tensor(self, tensor, kind: ExecutionType):
        if tensor is None:
            return ""
        return f"{tensor.get_name(self.asm_state[kind])}:{tensor.type.shape}, {tensor.type.element_type}"

    def _format_tensors(self, tensors: List[Operation], kind: ExecutionType):
        return "\n".join(
            sorted([self._format_tensor(tensor, kind) for tensor in tensors])
        )

    def write_row(self, **kwargs):
        assert "location" in kwargs
        assert "golden_ops" in kwargs
        assert "device_ops" in kwargs
        kwargs["golden_ops"] = self._format_ops(
            kwargs["golden_ops"], ExecutionType.GOLDEN
        )
        kwargs["device_ops"] = self._format_ops(
            kwargs["device_ops"], ExecutionType.DEVICE
        )
        kwargs["golden_output"] = self._format_tensor(
            kwargs["golden_output"], ExecutionType.GOLDEN
        )
        kwargs["device_output"] = self._format_tensor(
            kwargs["device_output"], ExecutionType.DEVICE
        )
        if "golden_inputs" in kwargs:
            kwargs["golden_inputs"] = self._format_tensors(
                kwargs["golden_inputs"], ExecutionType.GOLDEN
            )
        if "device_inputs" in kwargs:
            kwargs["device_inputs"] = self._format_tensors(
                kwargs["device_inputs"], ExecutionType.DEVICE
            )
        line = [kwargs.get(col, " ") for col in self.column_names]
        self._write([line])

    def _init_file(self):
        with open(self.file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(self.column_names)

    def _write(self, lines):
        with open(self.file_path, "a") as f:
            writer = csv.writer(f)
            for line in lines:
                writer.writerow(line)
