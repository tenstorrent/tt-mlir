# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import ttrt
import ttrt.runtime
from ttrt.common.util import *
from ..utils import DeviceContext, assert_pcc


@pytest.mark.parametrize(
    "mesh_shape",
    [[1, 1], [1, 2]],
)
@pytest.mark.parametrize(
    "mesh_offset",
    [[0, 0]],
)
@pytest.mark.parametrize(
    "num_hw_cqs",
    [1, 2],
)
@pytest.mark.parametrize(
    "enable_program_cache",
    [False, True],
)
@pytest.mark.parametrize(
    "l1_small_size",
    [0, 1024, 2048],
)
@pytest.mark.parametrize(
    "trace_region_size",
    [0, 1024, 2048],
)
def test_open_mesh_device(
    helper,
    mesh_shape,
    mesh_offset,
    num_hw_cqs,
    enable_program_cache,
    l1_small_size,
    trace_region_size,
):
    num_devices = ttrt.runtime.get_num_available_devices()
    assert num_devices == 2, f"Expected 2 devices, got {num_devices}"
    options = ttrt.runtime.MeshDeviceOptions()
    options.mesh_shape = mesh_shape
    options.mesh_offset = mesh_offset
    options.num_hw_cqs = num_hw_cqs
    options.enable_program_cache = enable_program_cache
    options.l1_small_size = l1_small_size
    options.trace_region_size = trace_region_size
    device = ttrt.runtime.open_mesh_device(options)
    device_ids = device.get_device_ids()
    assert device_ids == list(range(math.prod(mesh_shape)))
    assert device.get_mesh_shape() == mesh_shape
    assert device.get_num_hw_cqs() == num_hw_cqs
    assert device.is_program_cache_enabled() == enable_program_cache
    assert device.get_l1_small_size() == l1_small_size
    assert device.get_trace_region_size() == trace_region_size
    assert device.get_num_dram_channels() != 0
    assert device.get_dram_size_per_channel() != 0
    assert device.get_l1_size_per_core() != 0
    ttrt.runtime.close_mesh_device(device)


# Converts runtime device tensor to torch tensor.
# It first moves device tensor to host, and creates torch tensor from host tensor.
def device_tensor_to_torch_tensor(runtime_tensor):
    host_tensor = ttrt.runtime.to_host(runtime_tensor, untilize=True)
    assert len(host_tensor) == 1

    shape = host_tensor[0].get_shape()
    host_tensor[0].get_dtype()
    torch_tensor = torch.zeros(
        shape, dtype=ttrt_datatype_to_torch_dtype(host_tensor[0].get_dtype())
    )
    ttrt.runtime.memcpy(torch_tensor.data_ptr(), host_tensor[0])
    return torch_tensor


# Testing get_device_tensors API.
# Creates random torch tensor and moves it to device. Then retrieves tensor from device and compare it with the original tensor.
@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_get_device_tensors(shape, dtype):
    runtime_dtype = Binary.Program.to_data_type(dtype)

    torch_tensor = torch.randn(shape, dtype=dtype)
    runtime_tensor = ttrt.runtime.create_owned_host_tensor(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        list(torch_tensor.stride()),
        torch_tensor.element_size(),
        runtime_dtype,
    )

    device_layout = ttrt.runtime.test.get_dram_interleaved_tile_layout(runtime_dtype)

    with DeviceContext(mesh_shape=[1, 2]) as device:
        device_tensor = ttrt.runtime.to_layout(runtime_tensor, device, device_layout)
        device_tensors = ttrt.runtime.get_device_tensors(device_tensor)

        assert (
            len(device_tensors) == 2
        ), f"Expected 2 device tensors, got {len(device_tensors)}"

        for t in device_tensors:
            torch_output_tensor = device_tensor_to_torch_tensor(t)
            assert_pcc(torch_tensor, torch_output_tensor, threshold=0.99)

        ttrt.runtime.deallocate_tensor(device_tensor, force=True)


# Testing get_device_tensors with create_multi_device_host_tensor API.
# Creates 2 random torch tensors, creates multi device host tensor from torch tensors and moves it to device.
# Then retrieves tensor from device and compares it with the original tensors.
@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_get_device_tensors_multi_device(shape, dtype):
    runtime_dtype = Binary.Program.to_data_type(dtype)

    # Create two different tensors for two devices
    torch_tensor_0 = torch.randn(shape, dtype=dtype)
    torch_tensor_1 = torch.randn(shape, dtype=dtype)

    # Create multi-device tensor with different data per device
    multi_device_tensor = ttrt.runtime.create_multi_device_host_tensor(
        [torch_tensor_0.data_ptr(), torch_tensor_1.data_ptr()],
        list(shape),
        list(torch_tensor_0.stride()),
        torch_tensor_0.element_size(),
        runtime_dtype,
        {},
        [1, 2],
    )

    device_layout = ttrt.runtime.test.get_dram_interleaved_tile_layout(runtime_dtype)

    with DeviceContext(mesh_shape=[1, 2]) as device:
        device_tensor = ttrt.runtime.to_layout(
            multi_device_tensor, device, device_layout
        )
        device_tensors = ttrt.runtime.get_device_tensors(device_tensor)

        assert (
            len(device_tensors) == 2
        ), f"Expected 2 device tensors, got {len(device_tensors)}"

        # Get shards back from device
        host_shards = []
        for t in device_tensors:
            host_shards.append(device_tensor_to_torch_tensor(t))

        # Verify each shard matches the original input tensor for that device
        assert_pcc(torch_tensor_0, host_shards[0], threshold=0.99)
        assert_pcc(torch_tensor_1, host_shards[1], threshold=0.99)

        ttrt.runtime.deallocate_tensor(device_tensor, force=True)
