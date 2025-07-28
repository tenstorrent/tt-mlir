# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

from chisel.utils.mapping import ttrt_dtype_maps

from ttrt.runtime import (
    create_owned_host_tensor,
    update_tensor_in_pool,
    Tensor as RtTensor,
    TensorRef,
    ProgramContext,
)


def get_torch_tensor(tensor: RtTensor) -> torch.Tensor:
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    dtype = ttrt_dtype_maps[str(rt_dtype)]
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=dtype)
    # HACK: This is a hack to keep the tensor alive until the program context is destroyed
    torch_tensor = torch_tensor.reshape(shape).clone()
    return torch_tensor


def update_device_tensor(
    program_context: ProgramContext,
    tensor_ref: TensorRef,
    dst_tensor: RtTensor,
    src_tensor: torch.Tensor,
) -> None:
    data_ptr = src_tensor.data_ptr()
    shape = dst_tensor.get_shape()
    stride = dst_tensor.get_stride()
    dtype = dst_tensor.get_dtype()
    size = torch.numel(src_tensor)
    tensor = create_owned_host_tensor(data_ptr, shape, stride, size, dtype)
    update_tensor_in_pool(program_context, tensor_ref, tensor)
