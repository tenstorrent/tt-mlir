# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import ttnn_jit
import torch

import pytest


@ttnn_jit.jit(max_grid=(7, 7), debug=False)
def digamma_trace(input_a, one_tensor):
    t_log_out = ttnn.log(input_a)  # negative log is not useful here

    # 1/2(z)
    tmp_reciprocal = ttnn.divide(
        one_tensor, input_a
    )  # reciprocal op not supported in TTNNToTTIR pass.
    output = ttnn.multiply(tmp_reciprocal, 0.5)
    tmp = ttnn.multiply(
        tmp_reciprocal, tmp_reciprocal
    )  # square op not supported in TTIR dialect yet.
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


@ttnn_jit.jit(max_grid=(7, 7), debug=False)
def digamma(input_a):
    t_log_out = ttnn.log(input_a)  # negative log is not useful here

    # 1/2(z)
    tmp_reciprocal = ttnn.reciprocal(input_a)
    output = ttnn.multiply(tmp_reciprocal, 0.5)
    tmp = ttnn.pow(tmp_reciprocal, 2)
    val_square = tmp

    # (1/12) * x^2
    output = ttnn.subtract(output, ttnn.multiply(tmp, 0.083333333))

    # (1/120) * x^4
    tmp = ttnn.pow(tmp_reciprocal, 4)
    output = ttnn.add(output, ttnn.multiply(tmp, 0.008333333333333333))

    # (1/252) * x^6
    tmp = ttnn.pow(tmp_reciprocal, 6)
    output = ttnn.subtract(output, ttnn.multiply(tmp, 0.003968253968253968))

    # (1/240) * x^8
    tmp = ttnn.pow(tmp_reciprocal, 8)
    output = ttnn.add(output, ttnn.multiply(tmp, 0.004166666666666667))

    # (1/132) * x^10
    tmp = ttnn.pow(tmp_reciprocal, 10)
    output = ttnn.subtract(output, ttnn.multiply(tmp, 0.007575757575757576))

    # (691/32760) * x^12
    tmp = ttnn.pow(tmp_reciprocal, 12)
    output = ttnn.add(output, ttnn.multiply(tmp, 0.021092796092796094))

    # (1/12) * x^14
    tmp = ttnn.pow(tmp_reciprocal, 14)
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
def test_digamma_trace(h, w):
    device = ttnn.open_device(device_id=0, trace_region_size=10240)

    # h, w = 512, 2048
    dtype = torch.bfloat16
    max_grid = (7, 7)
    torch_tensor_a = torch.rand((h, w), dtype=dtype) * 100
    torch_one_tensor = torch.full((h, w), 1.0, dtype=dtype)

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

    input_a = ttnn.from_torch(torch_tensor_a, spec=tensor_spec)
    one_ = ttnn.from_torch(torch_one_tensor, spec=tensor_spec)

    # op_jit = ttnn_jit.jit(backend="ttnn", max_grid=max_grid, debug=True)(digamma)
    op_jit = digamma_trace

    # Warmup program caches
    input_a_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)
    one_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)

    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(one_, one_tensor)

    output_tensor = op_jit(input_a_tensor, one_tensor)
    ttnn.synchronize_device(device)

    # Capture trace
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(one_, one_tensor)

    tid = ttnn.begin_trace_capture(device)
    output_tensor = op_jit(input_a_tensor, one_tensor)
    ttnn.end_trace_capture(device, tid)
    ttnn.synchronize_device(device)

    # Execute trace
    ttnn.copy_host_to_device_tensor(input_a, input_a_tensor)
    ttnn.copy_host_to_device_tensor(one_, one_tensor)

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
    [
        (256, 256),
        (256, 512),
        (512, 512),
        (512, 1024),
        (512, 2048),
        (1024, 1024),
    ],
)
def test_digamma_compare(h, w):
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

    # op_jit = ttnn_jit.jit(backend="ttnn", max_grid=max_grid, debug=True)(digamma)
    op_jit = digamma
    output_tensor = op_jit(
        input_a,
    )
    golden = ttnn.digamma(input_a)

    print(f"output_tensor\n: {output_tensor}")
    print(f"golden\n: {golden}")

    matching = torch.allclose(
        output_tensor.cpu().to_torch(), golden.cpu().to_torch(), atol=1, rtol=1
    )
    print(f"Tensors are matching: {matching}")

    matching_pcc, pcc = comp_pcc(
        golden.cpu().to_torch(), output_tensor.cpu().to_torch()
    )
    print(f"PCC: {pcc}")
    assert matching_pcc
