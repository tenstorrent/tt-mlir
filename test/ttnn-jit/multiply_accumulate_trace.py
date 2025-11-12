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


@ttnn_jit.jit(max_grid=(7, 7), debug=False)
def muladd(input_tensor_a, input_tensor_b, input_tensor_c):
    mul_out = ttnn.multiply(input_tensor_b, input_tensor_c)
    return ttnn.add(mul_out, input_tensor_a)


@pytest.mark.parametrize(
    "h, w",
    [(32, 32), (64, 64), (128, 128)],
)
def test_muladd_trace(h, w):
    device = ttnn.open_device(device_id=0, trace_region_size=10240)

    # setup inputs
    dtype = ttnn.bfloat16
    max_grid = (7, 7)

    torch_input_a = torch.randn((h, w), dtype=dtype)
    torch_input_b = torch.randn((h, w), dtype=dtype)
    torch_input_c = torch.randn((h, w), dtype=dtype)

    core_range = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max_grid[0], max_grid[1])
    )
    core_range_set = ttnn.CoreRangeSet([core_range])
    tensor_spec = ttnn.TensorSpec(
        shape=(h, w),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.L1,
    ).block_sharded(core_range_set)

    # warmup program cache
    input_a = ttnn.from_torch(torch_input_a, spec=tensor_spec)
    input_b = ttnn.from_torch(torch_input_b, spec=tensor_spec)
    input_c = ttnn.from_torch(torch_input_c, spec=tensor_spec)

    op_jit = muladd

    input_a_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)
    input_b_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)
    input_c_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)

    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(input_b, input_b_tensor)
    ttnn.copy_host_to_device_tensor(input_c, input_c_tensor)

    output_tensor = op_jit(input_a_tensor, input_b_tensor, input_c_tensor)

    ttnn.synchronize_device(device)

    # Capture trace

    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(input_b, input_b_tensor)
    ttnn.copy_host_to_device_tensor(input_c, input_c_tensor)

    tid = ttnn.begin_trace_capture(device)
    output_tensor = op_jit(input_a_tensor, input_b_tensor, input_c_tensor)
    ttnn.end_trace_capture(device, tid)
    ttnn.synchronize_device(device)

    # Execute trace

    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(input_b, input_b_tensor)
    ttnn.copy_host_to_device_tensor(input_c, input_c_tensor)

    for _ in range(5):
        ttnn.execute_trace(device, tid, blocking=True)

    ttnn.synchronize_device(device)

    for _ in range(25):
        ttnn.execute_trace(device, tid, blocking=True)

    ttnn.release_trace(device, tid)
    ttnn.close_device(device)
