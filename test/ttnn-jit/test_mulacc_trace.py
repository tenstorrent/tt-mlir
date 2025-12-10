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

DRAM_SHAPES = [
    (256, 256),
    (256, 512),
    (512, 512),
    (512, 1024),
    (512, 2048),
    (1024, 1024),
    # other shapes
    (1705, 320),
    (1705, 640),
    (4095, 640),
    (4095, 1280),
    (8190, 640),
    (8190, 1280),
]


@ttnn_jit.jit(debug=False, enable_cache=True, graph_capture=False)
def muladd(input_tensor_a, input_tensor_b, input_tensor_c):

    return ttnn.add(ttnn.multiply(input_tensor_b, input_tensor_c), input_tensor_a)


def muladd_no_jit(input_tensor_a, input_tensor_b, input_tensor_c):

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
    DRAM_SHAPES,
)
def test_muladd_dram_trace(h, w):
    device = ttnn.open_device(device_id=0, trace_region_size=20480)
    shape = (h, w)
    # setup inputs
    dtype = torch.bfloat16

    torch_input_a = torch.randn(shape, dtype=dtype)
    torch_input_b = torch.randn(shape, dtype=dtype)
    torch_input_c = torch.randn(shape, dtype=dtype)

    tensor_spec = ttnn.TensorSpec(
        shape=shape,
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
    "h, w",
    DRAM_SHAPES,
)
def test_muladd_dram_compare(h, w):
    device = ttnn.open_device(device_id=0)

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

    input_a_tensor = ttnn.from_torch(torch_input_a, spec=tensor_spec, device=device)
    input_b_tensor = ttnn.from_torch(torch_input_b, spec=tensor_spec, device=device)
    input_c_tensor = ttnn.from_torch(torch_input_c, spec=tensor_spec, device=device)

    op_jit = muladd
    output_tensor = op_jit(input_a_tensor, input_b_tensor, input_c_tensor)
    golden = muladd_no_jit(input_a_tensor, input_b_tensor, input_c_tensor)

    matching = torch.allclose(
        output_tensor.cpu().to_torch(), golden.cpu().to_torch(), atol=1, rtol=1
    )
    print("Tensors are matching:", matching)

    matching_pcc, pcc = comp_pcc(
        golden.cpu().to_torch(), output_tensor.cpu().to_torch()
    )
    print(f"PCC: {pcc}")
    assert matching_pcc
