# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any
import torch

from ..utils.runtime_utils import update_device_tensor, get_torch_tensor
from .enums import ExecutionType

from ttrt.runtime import Tensor, TensorRef, retrieve_tensor_from_pool


class TensorValue:
    """
    This class has a dual purpose if the ExecutionType is DEVICE:
        1. It is used to store the data of the tensor, with which it will be compared
        2. If the execution_data is not None, it is used to replace the data of the tensor

    If the ExecutionType is GOLDEN:
        1. It is used to store the data of the tensor, with which it will be compared
        2. execution_data is used for GoldenExecutor as input
    """

    def __init__(
        self, name: str, data: Any, execution_type: ExecutionType, tensor_ref=None
    ):
        self.name = name
        self.data = data
        self.execution_type = execution_type
        self.tensor_ref: TensorRef | None = tensor_ref
        self.tensor: Tensor | None = None
        self.execution_data = None

    def set_execution_data(self, data: Any = None):
        if data is None:
            self.execution_data = self.data
            return
        self.execution_data = data

    def update_tensor_in_pool(self, program_context):
        assert self.execution_type == ExecutionType.DEVICE
        assert self.execution_data is not None
        assert self.tensor_ref is not None
        if self.tensor is None:
            self.tensor = retrieve_tensor_from_pool(program_context, self.tensor_ref)
        update_device_tensor(
            program_context, self.tensor_ref, self.tensor, self.execution_data
        )
        self.data = self.execution_data
        self.execution_data = None

    def retrieve_tensor_from_pool(self, program_context):
        if self.tensor is None:
            self.tensor = retrieve_tensor_from_pool(program_context, self.tensor_ref)
        if self.tensor is None:
            return
        self.data = get_torch_tensor(self.tensor)
        return self.data

    def __str__(self):
        return f"TensorValue(name={self.name}, data={self.data}, execution_type={self.execution_type}, execution_data={self.execution_data})"

    def __repr__(self):
        return self.__str__()


class TensorPool(dict):
    def __init__(self, caching=False, output_dir=None):
        super().__init__()
        self.caching = caching
        self.output_dir = output_dir
        if caching:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if not self.caching:
            return
        if isinstance(value, TensorValue):
            # HACK: Torch dont support saving uint16, uint32, uint64 https://github.com/pytorch/pytorch/issues/58734
            if value.data.dtype in [torch.uint16, torch.uint32, torch.uint64]:
                return
            # TODO: potentially enable to not override existing files
            torch.save(value.data, self.output_dir / f"{key}.pt")
