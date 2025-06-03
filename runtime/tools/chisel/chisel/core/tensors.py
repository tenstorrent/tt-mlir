# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from functools import cache
from typing import Any, List, Tuple

import torch
from ttmlir.ir import Operation, BlockArgument, Value, Type

from ttrt.runtime import (
    Tensor,
    create_owned_tensor,
    update_tensor,
    get_op_output_ref,
    get_tensor,
)

from .enums import ExecutionType
from ..utils.location import hash_location, parse_op_location
from ..utils.mapping import ttrt_dtype_maps


@dataclass
class TensorDescriptor:
    name: str
    dtype: str
    shape: List[int]
    location_hash: Tuple[int, int]
    execution_type: ExecutionType


def get_tensor_descriptor(op: Operation, execution_type: ExecutionType):
    return TensorDescriptor(
        name=op.get_name(),
        dtype=str(op.type.element_type),
        shape=op.type.shape,
        location_hash=hash_location(op.location),
        execution_type=execution_type,
    )


def get_op_outputs(op: Operation, execution_type: ExecutionType):
    outputs = []
    for result in op.results:
        if hasattr(result.type, "shape") and hasattr(result.type, "element_type"):
            outputs.append(get_tensor_descriptor(result, execution_type))
    return outputs


def get_op_inputs(op: Operation, execution_type: ExecutionType):
    inputs = []
    for operand in op.operands:
        if hasattr(operand.type, "shape") and hasattr(operand.type, "element_type"):
            inputs.append(get_tensor_descriptor(operand, execution_type))
    return inputs


def get_function_inputs(op: Operation, execution_type: ExecutionType):
    inputs = []
    for arg in op.arguments:
        inputs.append(get_tensor_descriptor(arg, execution_type))
    return inputs


def get_torch_tensor(tensor):
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    dtype = ttrt_dtype_maps[str(rt_dtype)]
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=dtype)
    # TODO: This is a hack to keep the tensor alive until the program context is destroyed
    torch_tensor = torch_tensor.reshape(shape).clone()
    return torch_tensor


def update_device_tensor(
    program_context, tensor_ref, dst_tensor, src_tensor: torch.Tensor
):
    data_ptr = src_tensor.data_ptr()
    shape = dst_tensor.get_shape()
    stride = dst_tensor.get_stride()
    dtype = dst_tensor.get_dtype()
    size = torch.numel(src_tensor)
    tensor = create_owned_tensor(data_ptr, shape, stride, size, dtype)
    update_tensor(program_context, tensor_ref, tensor)


class TensorValue:
    def __init__(self, name: str, data: Any, execution_type: ExecutionType):
        self.name = name
        self.data = data
        self.execution_type = execution_type
        self.tensor_ref: TensorRef | None = None
        self.tensor: Tensor | None = None
        self.execution_data = data

    def set_execution_data(self, data: Any = None):
        if data is None:
            data = self.data
        self.execution_data = data

    def update_tensor(self, program_context):
        if self.tensor_ref is None:
            return
        update_device_tensor(program_context, self.tensor_ref, self.data)

    def __str__(self):
        return f"TensorValue(name={self.name}, data={self.data}, execution_type={self.execution_type}, execution_data={self.execution_data})"

    def __repr__(self):
        return self.__str__()


class TensorPool:
    def __init__(self):
        self.tensors = {}

    def __getitem__(self, key: str):
        return self.tensors[key]

    def __setitem__(self, key: str, value):
        self.tensors[key] = value

    def __contains__(self, key: str):
        return key in self.tensors
