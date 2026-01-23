# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import ttnn_jit
import torch

import pytest


@ttnn_jit.jit(debug=True, enable_cache=True)
def digamma(input_tensor):
    t_log_out = ttnn.log(input_tensor)  # negative log is not useful here

    # 1/2(z)
    tmp_reciprocal = ttnn.reciprocal(input_tensor)
    output = ttnn.multiply(tmp_reciprocal, 0.5)
    tmp = ttnn.multiply(tmp_reciprocal, tmp_reciprocal)
    val_square = tmp

    # (1/12) * x^2
    output = ttnn.subtract(output, ttnn.multiply(tmp, 0.083333333))

    # (1/120) * x^4
    tmp = ttnn.multiply(tmp, val_square)
    output = ttnn.add(output, ttnn.multiply(tmp, 0.008333333333333333))

    # (1/252) * x^6
    tmp = ttnn.multiply(tmp, val_square)
    output = ttnn.subtract(output, ttnn.multiply(tmp, 0.003968253968253968))

    # (1/240) * x^8
    tmp = ttnn.multiply(tmp, val_square)
    output = ttnn.add(output, ttnn.multiply(tmp, 0.004166666666666667))

    # (1/132) * x^10
    tmp = ttnn.multiply(tmp, val_square)
    output = ttnn.subtract(output, ttnn.multiply(tmp, 0.007575757575757576))

    # (691/32760) * x^12
    tmp = ttnn.multiply(tmp, val_square)
    output = ttnn.add(output, ttnn.multiply(tmp, 0.021092796092796094))

    # (1/12) * x^14
    tmp = ttnn.multiply(tmp, val_square)
    output = ttnn.subtract(output, ttnn.multiply(tmp, 0.08333333333333333))

    return ttnn.subtract(t_log_out, output)


@pytest.mark.parametrize(
    "h, w",
    [
        (256, 256),
        (256, 512),
        (512, 512),
        (512, 1024),
        (512, 2048),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        digamma,
        ttnn_jit.jit(debug=True, enable_cache=True)(digamma.func),
    ],
)
def test_digamma_metal_trace(h, w, op):
    device = ttnn.open_device(device_id=0, trace_region_size=240000)

    dtype = torch.bfloat16
    torch_tensor_a = torch.rand((h, w), dtype=dtype) * 100

    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))
    core_range_set = ttnn.CoreRangeSet([core_range])
    tensor_spec = ttnn.TensorSpec(
        shape=(h, w),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.L1,
    ).block_sharded(core_range_set)

    input_a = ttnn.from_torch(torch_tensor_a, spec=tensor_spec)

    # Warmup program caches
    input_a_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)

    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)

    output_tensor = op(input_a_tensor)
    ttnn.synchronize_device(device)

    # Capture trace
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)

    tid = ttnn.begin_trace_capture(device)
    output_tensor = op(input_a_tensor)
    ttnn.end_trace_capture(device, tid)
    ttnn.synchronize_device(device)

    # Execute trace
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)

    ttnn.execute_trace(device, tid, blocking=True)

    ttnn.release_trace(device, tid)
    golden = ttnn.digamma(input_a_tensor)

    print(f"output_tensor\n: {output_tensor}")
    print(f"golden\n: {golden}")

    pcc = ttnn.pearson_correlation_coefficient(
        golden.cpu().to_torch(), output_tensor.cpu().to_torch()
    )
    print(f"PCC: {pcc}")
    assert pcc > 0.99

    ttnn.close_device(device)


@pytest.mark.parametrize(
    "h, w",
    [
        (256, 256),
        (256, 512),
        (512, 512),
        (512, 1024),
        (512, 2048),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        digamma,
        ttnn_jit.jit(debug=True, enable_cache=True)(digamma.func),
    ],
)
def test_digamma_compare(h, w, op):
    device = ttnn.open_device(device_id=0)

    # h, w = 1024, 1024
    dtype = torch.bfloat16
    max_grid = (7, 7)
    torch_tensor_a = torch.rand((h, w), dtype=dtype) * 100

    core_range = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max_grid[0], max_grid[1])
    )
    core_range_set = ttnn.CoreRangeSet([core_range])
    tensor_spec = ttnn.TensorSpec(
        shape=(h, w),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.L1,
    ).block_sharded(core_range_set)

    input_a = ttnn.from_torch(torch_tensor_a, spec=tensor_spec, device=device)
    output_tensor = op(input_a)
    golden = ttnn.digamma(input_a)

    print(f"output_tensor\n: {output_tensor}")
    print(f"golden\n: {golden}")

    pcc = ttnn.pearson_correlation_coefficient(
        golden.cpu().to_torch(), output_tensor.cpu().to_torch()
    )
    ttnn.close_device(device)
    print(f"PCC: {pcc}")
    assert pcc > 0.99
