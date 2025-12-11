# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import (
    get_block_sharding_grid,
    create_torch_tensor,
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


def matmul_no_jit(input_tensor_a, input_tensor_b):
    return ttnn.matmul(input_tensor_a, input_tensor_b)


@pytest.mark.parametrize(
    "m, k, n",
    SHAPES,
)
def test_matmul_block_sharded_trace(m, k, n):
    device = ttnn.open_device(device_id=0, trace_region_size=20480)

    # setup inputs
    dtype = torch.float32
    ttnn_dtype = ttnn.DataType.FLOAT32
    shape_a = (m, k)
    shape_b = (k, n)

    torch_input_a = create_torch_tensor(shape_a, dtype=dtype)
    torch_input_b = create_torch_tensor(shape_b, dtype=dtype)

    # Use helper to get a valid grid size
    grid_list_a = get_block_sharding_grid(shape_a)
    grid_list_b = get_block_sharding_grid(shape_b)

    grid_a = ttnn.CoreGrid(x=grid_list_a[0] + 1, y=grid_list_a[1] + 1)
    grid_b = ttnn.CoreGrid(x=grid_list_b[0] + 1, y=grid_list_b[1] + 1)

    print(f"Using core grid A: {grid_a}, core grid B: {grid_b}")

    # Create MemoryConfig using the modern, strategy-based helper
    # This is the method used by the smoketest's helpers.
    memory_config_a = ttnn.create_sharded_memory_config(
        shape=shape_a,
        core_grid=grid_a,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    memory_config_b = ttnn.create_sharded_memory_config(
        shape=shape_b,
        core_grid=grid_b,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )

    # Tensor A Spec
    tensor_spec_a = ttnn.TensorSpec(
        shape=shape_a,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_layout=memory_config_a.memory_layout,
        shard_spec=memory_config_a.shard_spec,
        buffer_type=memory_config_a.buffer_type,
    )

    # Tensor B Spec
    tensor_spec_b = ttnn.TensorSpec(
        shape=shape_b,
        dtype=ttnn_dtype,
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


# COPIED FROM TT-METAL
def comp_pcc(golden, calculated, pcc=0.99):
    import numpy as np

    # 1. Convert to torch tensors
    golden = torch.Tensor(golden)
    calculated = torch.Tensor(calculated)

    # 2. Handle dtype mismatches
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    # 3. Handle special cases
    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        return True, 1.0  # Both NaN

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        return False, 0.0  # One NaN, one not

    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        return False, 0.0  # One all zero, one not

    # 4. Mask infs and nans
    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    # 5. Convert bfloat16 to float32 for better precision
    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)

    # 6. Compute PCC using numpy correlation
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    return cal_pcc >= pcc, cal_pcc


@pytest.mark.parametrize(
    "m, k, n",
    SHAPES,
)
def test_matmul_block_sharded_compare(device, m, k, n):

    # setup inputs
    dtype = torch.bfloat16
    shape_a = (m, k)
    shape_b = (k, n)
    print(f"Testing matmul compare with shapes A: {shape_a}, B: {shape_b}")

    torch_input_a = torch.randn(shape_a, dtype=dtype)
    torch_input_b = torch.randn(shape_b, dtype=dtype)

    # Use helper to get a valid grid size
    grid_list_a = get_block_sharding_grid(shape_a)
    grid_list_b = get_block_sharding_grid(shape_b)

    print("Grid A:", grid_list_a)
    print("Grid B:", grid_list_b)

    grid_a = ttnn.CoreGrid(x=grid_list_a[0] + 1, y=grid_list_a[1] + 1)
    grid_b = ttnn.CoreGrid(x=grid_list_b[0] + 1, y=grid_list_b[1] + 1)

    # Create MemoryConfig using the modern, strategy-based helper
    # This is the method used by the smoketest's helpers.
    memory_config_a = ttnn.create_sharded_memory_config(
        shape=shape_a,
        core_grid=grid_a,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    memory_config_b = ttnn.create_sharded_memory_config(
        shape=shape_b,
        core_grid=grid_b,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )

    ttnn_tensor_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config_a,
    )

    ttnn_tensor_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config_b,
    )

    output_tensor = matmul(ttnn_tensor_a, ttnn_tensor_b)
    ttnn_tensor_a = ttnn.to_memory_config(ttnn_tensor_a, ttnn.DRAM_MEMORY_CONFIG)
    ttnn_tensor_b = ttnn.to_memory_config(ttnn_tensor_b, ttnn.DRAM_MEMORY_CONFIG)
    print("ttnn tensor a memory config:", ttnn_tensor_a.memory_config())
    print("ttnn tensor b memory config:", ttnn_tensor_b.memory_config())

    golden_output = matmul_no_jit(ttnn_tensor_a, ttnn_tensor_b)
    print("output tensor:", output_tensor)

    pcc = ttnn.pearson_correlation_coefficient(
        golden_output.cpu().to_torch(), output_tensor.cpu().to_torch()
    )
    print("pcc: ", pcc)
    assert pcc > 0.99, f"PCC: {pcc} is less than 0.99"
