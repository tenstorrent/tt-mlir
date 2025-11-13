# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import (
    _get_ttnn_op,
    all_close_check,
    memory_configs_equal,
    create_dram_tensor,
    create_sharded_tile_tensor,
    run_op_test,
)


@ttnn_jit.jit(max_grid=(0, 0), debug=True, enable_cache=True)
def muladd(input_tensor_a, input_tensor_b, input_tensor_c):

    return ttnn.add(ttnn.multiply(input_tensor_b, input_tensor_c), input_tensor_a)


@pytest.mark.parametrize(
    "h, w",
    [(32, 32)],
)
def test_muladd_l1_trace(h, w):
    device = ttnn.open_device(device_id=0, trace_region_size=10240)

    # setup inputs
    dtype = torch.bfloat16
    max_grid = (0, 0)

    torch_input_a = torch.randn((h, w), dtype=dtype)
    torch_input_b = torch.randn((h, w), dtype=dtype)
    torch_input_c = torch.randn((h, w), dtype=dtype)

    core_range = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max_grid[0], max_grid[1])
    )
    core_range_set = ttnn.CoreRangeSet([core_range])

    # TTNN grids are (Width, Height), while tensor shapes are (Height, Width).
    shard_shape_x = h if max_grid[1] == 0 else h // (max_grid[1] + 1)
    shard_shape_y = w if max_grid[0] == 0 else w // (max_grid[0] + 1)

    shard_spec = ttnn.ShardSpec(
        grid=core_range_set,
        shard_shape=[shard_shape_x, shard_shape_y],
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    tensor_spec = ttnn.TensorSpec(
        shape=(h, w),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        memory_layout=memory_config.memory_layout,  # Extract memory_layout
        shard_spec=memory_config.shard_spec,  # Extract shard_spec
        buffer_type=memory_config.buffer_type,  # Extract buffer_type
    )

    # warmup program cache
    print("Warming up JIT program cache...")
    # host tensors
    input_a = ttnn.from_torch(torch_input_a, spec=tensor_spec)
    input_b = ttnn.from_torch(torch_input_b, spec=tensor_spec)
    input_c = ttnn.from_torch(torch_input_c, spec=tensor_spec)

    # allocate device tensors
    input_a_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)
    input_b_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)
    input_c_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)

    # copy data to device
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(input_b, input_b_tensor)
    ttnn.copy_host_to_device_tensor(input_c, input_c_tensor)

    # should cache the program
    output_tensor = muladd(input_a_tensor, input_b_tensor, input_c_tensor)
    ttnn.synchronize_device(device)

    # Capture trace
    print("Capturing trace...")

    # copy to device
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(input_b, input_b_tensor)
    ttnn.copy_host_to_device_tensor(input_c, input_c_tensor)

    tid = ttnn.begin_trace_capture(device)
    # should work?
    output_tensor = muladd(input_a_tensor, input_b_tensor, input_c_tensor)
    ttnn.end_trace_capture(device, tid)
    ttnn.synchronize_device(device)

    # Execute trace
    print("Executing trace...")

    # copy again
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(input_b, input_b_tensor)
    ttnn.copy_host_to_device_tensor(input_c, input_c_tensor)

    # execute trace multiple times
    for _ in range(5):
        ttnn.execute_trace(device, tid, blocking=True)

    ttnn.synchronize_device(device)

    for _ in range(25):
        ttnn.execute_trace(device, tid, blocking=True)

    ttnn.release_trace(device, tid)
    ttnn.close_device(device)


@pytest.mark.parametrize(
    "h, w",
    [(32, 32)],
)
def test_muladd_dram_trace(h, w):
    device = ttnn.open_device(device_id=0, trace_region_size=10240)

    # setup inputs
    dtype = torch.bfloat16

    torch_input_a = torch.randn((h, w), dtype=dtype)
    torch_input_b = torch.randn((h, w), dtype=dtype)
    torch_input_c = torch.randn((h, w), dtype=dtype)

    tensor_spec = ttnn.TensorSpec(
        shape=(h, w),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    # warmup program cache
    print("Warming up JIT program cache...")
    # host tensors
    input_a = ttnn.from_torch(torch_input_a, spec=tensor_spec)
    input_b = ttnn.from_torch(torch_input_b, spec=tensor_spec)
    input_c = ttnn.from_torch(torch_input_c, spec=tensor_spec)

    # allocate device tensors
    input_a_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)
    input_b_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)
    input_c_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)

    # copy data to device
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(input_b, input_b_tensor)
    ttnn.copy_host_to_device_tensor(input_c, input_c_tensor)

    # should cache the program
    output_tensor = muladd(input_a_tensor, input_b_tensor, input_c_tensor)
    ttnn.synchronize_device(device)

    # Capture trace
    print("Capturing trace...")

    # copy to device
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(input_b, input_b_tensor)
    ttnn.copy_host_to_device_tensor(input_c, input_c_tensor)

    tid = ttnn.begin_trace_capture(device)
    # should work?
    output_tensor = muladd(input_a_tensor, input_b_tensor, input_c_tensor)
    ttnn.end_trace_capture(device, tid)
    ttnn.synchronize_device(device)

    # Execute trace
    print("Executing trace...")

    # copy again
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(input_b, input_b_tensor)
    ttnn.copy_host_to_device_tensor(input_c, input_c_tensor)

    # execute trace multiple times
    for _ in range(5):
        ttnn.execute_trace(device, tid, blocking=True)

    ttnn.synchronize_device(device)

    for _ in range(25):
        ttnn.execute_trace(device, tid, blocking=True)

    ttnn.release_trace(device, tid)
    ttnn.close_device(device)
