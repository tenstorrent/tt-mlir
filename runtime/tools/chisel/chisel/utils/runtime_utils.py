# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

from ttrt.runtime import (
    create_owned_host_tensor,
    update_tensor_in_pool,
    Tensor as RtTensor,
    TensorRef,
)


ttir_dtype_maps = {
    "i32": torch.int32,
    "i64": torch.int64,
    "f32": torch.float32,
    "f64": torch.float64,
    "si32": torch.int32,
    "i1": torch.bool,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "ui32": torch.uint32,
}


ttrt_dtype_maps = {
    "DataType.Float32": torch.float32,
    "DataType.BFloat16": torch.bfloat16,
    "DataType.UInt32": torch.uint32,
    "DataType.UInt16": torch.uint16,
    "DataType.UInt8": torch.uint8,
    "DataType.Int32": torch.int32,
}


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
    program_context,
    tensor_ref: TensorRef,
    dst_tensor: RtTensor,
    src_tensor: torch.Tensor,
    tensor_value=None,  # Optional TensorValue to keep tensor alive
) -> torch.Tensor:
    """
    Update a device tensor in the pool with data from a source torch tensor.

    Returns the tensor that should be kept alive (either the original or a clone).
    """
    print(f"DEBUG update_device_tensor called:")
    print(f"  src_tensor shape: {src_tensor.shape}")
    print(f"  src_tensor dtype: {src_tensor.dtype}")
    print(f"  src_tensor sample values BEFORE: {src_tensor.flatten()[:5]}")
    print(f"  src_tensor is contiguous: {src_tensor.is_contiguous()}")

    # Ensure tensor is contiguous - make a copy if needed
    if not src_tensor.is_contiguous():
        src_tensor = src_tensor.contiguous()
        print(f"  Made tensor contiguous")

    # IMPORTANT: Get data pointer BEFORE any other operations
    data_ptr = src_tensor.data_ptr()
    shape = dst_tensor.get_shape()
    stride = dst_tensor.get_stride()
    dtype = dst_tensor.get_dtype()
    size = torch.numel(src_tensor)

    print(f"  dst_tensor shape: {shape}")
    print(f"  dst_tensor dtype: {dtype}")
    print(f"  Creating owned host tensor with size: {size}")
    print(f"  data_ptr: {data_ptr}")

    tensor = create_owned_host_tensor(data_ptr, shape, stride, size, dtype)
    update_tensor_in_pool(program_context, tensor_ref, tensor)

    print(f"  Tensor updated in pool")
    print(f"  src_tensor sample values AFTER: {src_tensor.flatten()[:5]}")

    # Return the tensor to keep it alive
    return src_tensor
