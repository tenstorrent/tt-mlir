# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from enum import Enum

import numpy as np
import torch

from ttrt.runtime import create_tensor, memcpy, DataType


class TensorStatus(Enum):
    NOT_INITIALIZED = 0
    TT_SYNCED = 1
    TT_OVERWRITTEN = 2
    CPU_SYNCED = 3
    CPU_OVERWRITTEN = 4


np_dtype_mapping = {
    DataType.Float32: np.float32,
    DataType.Float16: np.float16,
    DataType.BFloat16: np.float16,
    DataType.UInt32: np.uint32,
    DataType.UInt16: np.uint16,
    DataType.UInt8: np.uint8,
    DataType.Int32: np.int32,
}

torch_dtype_mapping = {
    DataType.Float32: torch.float32,
    DataType.Float16: torch.float16,
    DataType.BFloat16: torch.bfloat16,
    DataType.UInt32: torch.uint32,
    DataType.UInt16: torch.uint16,
    DataType.UInt8: torch.uint8,
    DataType.Int32: torch.int32,
}


class TensorValue:
    def __init__(self, name, tensor_ref=None):
        self.name = name

        self.tensor_ref = tensor_ref
        self.current_data: Optional[np.ndarray] = None
        self._tt_data: Optional[np.ndarray] = None
        self._cpu_data: Optional[torch.Tensor] = None
        self.status = TensorStatus.NOT_INITIALIZED

    def set_device_data(self, data: np.ndarray, program_context):
        # TODO: handle bfloat16
        self.current_data = data
        if isinstance(data, np.ndarray):
            data_ptr = data.ctypes.data
        if isinstance(data, torch.Tensor):
            data_ptr = data.data_ptr()

        np_dtype = np_dtype_mapping[self.tensor_ref.tensor.get_dtype()]
        if isinstance(data, np.ndarray):
            stride = list(np.array(data.strides, dtype=np.int32) // data.itemsize)
        elif isinstance(data, torch.Tensor):
            stride = list(
                np.array(data.stride(), dtype=np.int32) // data.element_size()
            )
        shape = list(data.shape)
        size = np.prod(data.shape)
        dtype = self.tensor_ref.tensor.get_dtype()
        src_rtensor = create_tensor(data_ptr, shape, stride, size, dtype)
        memcpy(self.tensor_ref.tensor, src_rtensor)

        # Update pool
        self.tensor_ref.update_tensor(program_context)
        self._tt_data = data
        self.status = TensorStatus.TT_OVERWRITTEN

    @property
    def tt_data(self):
        if self.tensor_ref is None:
            return
        if self._tt_data is not None:
            return self._tt_data

        buffer = self.tensor_ref.tensor.get_data_buffer()
        dtype = self.tensor_ref.tensor.get_dtype()
        shape = self.tensor_ref.tensor.get_shape()

        if dtype == DataType.BFloat16:
            raw_data = np.frombuffer(buffer, dtype=np.uint16)
            uint32_array = np.array(raw_data, dtype=np.uint32) << 16
            np_array = uint32_array.view(np.float32).reshape(shape)
            self._tt_data = torch.from_numpy(np_array).to(torch.bfloat16)
        else:
            np_array = np.frombuffer(buffer, dtype=np_dtype_mapping[dtype]).reshape(
                shape
            )
            torch_dtype = torch_dtype_mapping.get(dtype)
            self._tt_data = torch.from_numpy(np_array)
            if torch_dtype is not None:
                self._tt_data = self._tt_data.to(torch_dtype)

        if self.status == TensorStatus.NOT_INITIALIZED:
            self.status = TensorStatus.TT_SYNCED
        return self._tt_data

    def set_cpu_data(self, data: torch.Tensor):
        self.current_data = data
        self._cpu_data = data
        self.status = TensorStatus.CPU_OVERWRITTEN

    @property
    def cpu_data(self):
        return self._cpu_data

    def __repr__(self) -> str:
        return f"TensorValue({self.name=})"
