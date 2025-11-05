# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import csv
from typing import List

import torch
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
        return "\n".join([op.get_asm(enable_debug_info=True) for op in ops])

    def _format_tensor(self, tensor, kind: ExecutionType):
        if tensor is None:
            return ""
        return f"{tensor.get_name(self.asm_state[kind])}:{tensor.type.shape}, {tensor.type.element_type}"

    def _format_tensors(self, tensors: List[Operation], kind: ExecutionType):
        return "\n".join(
            sorted([self._format_tensor(tensor, kind) for tensor in tensors])
        )

    def _format_tensor_data(self, tensor_data):
        """Format tensor data for CSV output, handling both torch.Tensor and GoldenMapTensor."""
        if tensor_data is None:
            return ""

        # Check if it's a GoldenMapTensor by checking for _shard_map attribute
        if hasattr(tensor_data, "_shard_map"):
            # For GoldenMapTensor, get the first shard's tensor data
            # or combine all shards if there are multiple
            shard_map = tensor_data._shard_map
            if len(shard_map) == 1:
                # Single shard case - just return the tensor representation
                tensor_data = next(iter(shard_map.values()))
            else:
                # Multiple shards - could represent as a dict or combine
                # For now, just return the first shard for simplicity
                tensor_data = next(iter(shard_map.values()))

        # Now tensor_data should be a torch.Tensor or compatible tensor
        if isinstance(tensor_data, torch.Tensor):
            return str(tensor_data)

        # Fallback for other types
        return str(tensor_data)

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
        # Format tensor data columns to convert GoldenMapTensor objects to actual tensor strings
        if "golden_output_tensor" in kwargs:
            kwargs["golden_output_tensor"] = self._format_tensor_data(
                kwargs["golden_output_tensor"]
            )
        if "device_output_tensor" in kwargs:
            kwargs["device_output_tensor"] = self._format_tensor_data(
                kwargs["device_output_tensor"]
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
