# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import (
    get_block_sharding_grid,
)

SHAPES = [
    # M, K, N
    (512, 512, 512),
    (512, 1024, 1024),
    (512, 1024, 2048),
    (1024, 1024, 1024),
    (1024, 1024, 2048),
    (1024, 2048, 2048),
    (2048, 2048, 2048),
]


@ttnn_jit.jit(debug=False, enable_cache=True, graph_capture=False)
def matmul(input_tensor_a, input_tensor_b):
    return ttnn.matmul(input_tensor_a, input_tensor_b)


@pytest.mark.parametrize(
    "m, k, n",
    SHAPES,
)
def test_matmul_block_sharded_trace(m, k, n):
    device = ttnn.open_device(device_id=0, trace_region_size=81920)

    # setup inputs
    dtype = torch.bfloat16
    shape_a = (m, k)
    shape_b = (k, n)

    torch_input_a = torch.randn(shape_a, dtype=dtype)
    torch_input_b = torch.randn(shape_b, dtype=dtype)

    # Use helper to get a valid grid size
    grid_a = get_block_sharding_grid(shape_a)
    grid_b = get_block_sharding_grid(shape_b)

    # Tensor A Spec
    core_range_a = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_a[0], grid_a[1])
    )
    shard_spec_a = ttnn.ShardSpec(
        grid=ttnn.CoreRangeSet([core_range_a]),
        shard_shape=(m // (grid_a[1] + 1), k // (grid_a[0] + 1)),
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config_a = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec_a,
    )
    tensor_spec_a = ttnn.TensorSpec(
        shape=shape_a,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        memory_layout=memory_config_a.memory_layout,
        shard_spec=memory_config_a.shard_spec,
        buffer_type=memory_config_a.buffer_type,
    )

    # Tensor B Spec
    core_range_b = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_b[0], grid_b[1])
    )
    shard_spec_b = ttnn.ShardSpec(
        grid=ttnn.CoreRangeSet([core_range_b]),
        shard_shape=(k // (grid_b[1] + 1), n // (grid_b[0] + 1)),
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config_b = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec_b,
    )
    tensor_spec_b = ttnn.TensorSpec(
        shape=shape_b,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        memory_layout=memory_config_b.memory_layout,
        shard_spec=memory_config_b.shard_spec,
        buffer_type=memory_config_b.buffer_type,
    )

    # warmup program cache
    print("Warming up JIT program cache...")
    # host tensors
    input_a = ttnn.from_torch(torch_input_a, spec=tensor_spec_a)
    input_b = ttnn.from_torch(torch_input_b, spec=tensor_spec_b)

    # allocate device tensors
    input_a_tensor = ttnn.allocate_tensor_on_device(tensor_spec_a, device)
    input_b_tensor = ttnn.allocate_tensor_on_device(tensor_spec_b, device)

    # copy data to device
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(input_b, input_b_tensor)

    # should cache the program
    output_tensor = matmul(input_a_tensor, input_b_tensor)
    ttnn.synchronize_device(device)

    # Capture trace
    print("Capturing trace...")
    # copy to device
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(input_b, input_b_tensor)

    tid = ttnn.begin_trace_capture(device)
    output_tensor = matmul(input_a_tensor, input_b_tensor)
    ttnn.end_trace_capture(device, tid)
    ttnn.synchronize_device(device)

    # Execute trace
    print("Executing trace...")
    # copy again
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(input_b, input_b_tensor)

    # execute trace multiple times
    for _ in range(5):
        ttnn.execute_trace(device, tid, blocking=True)
    ttnn.synchronize_device(device)

    for _ in range(25):
        ttnn.execute_trace(device, tid, blocking=True)

    ttnn.release_trace(device, tid)
    ttnn.close_device(device)
