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

COMMON_SHAPE_GRID_PARAMS = [
    (32, 32, (0, 0)),
    (32, 64, (0, 0)),
    (64, 64, (0, 0)),
    (64, 128, (0, 0)),
    (128, 128, (0, 0)),
    (256, 256, (7, 7)),
    (256, 512, (7, 7)),
    (512, 512, (7, 7)),
    (512, 1024, (7, 7)),
    (1024, 1024, (7, 7)),
    (1024, 2048, (7, 7)),
]

COMMON_SHAPE_PARAMS = [(32, 32), (32, 64), (64, 64), (64, 128), (128, 128)]


# ------------------------------------------------------------
# Composite ops
# ------------------------------------------------------------
def cosh(input_tensor):
    e_pos_x = ttnn.exp(input_tensor)
    e_neg_x = ttnn.exp(ttnn.neg(input_tensor))
    nr_term = ttnn.add(e_pos_x, e_neg_x)
    output = ttnn.multiply(nr_term, 0.5)
    return output


def sinh(input_tensor):
    e_pos_x = ttnn.exp(input_tensor)
    e_neg_x = ttnn.exp(ttnn.neg(input_tensor))
    nr_term = ttnn.subtract(e_pos_x, e_neg_x)
    output = ttnn.multiply(nr_term, 0.5)
    return output


def mul_add(input_tensor_a, input_tensor_b, input_tensor_c):
    matmul_result = ttnn.multiply(input_tensor_b, input_tensor_c)
    output = ttnn.add(matmul_result, input_tensor_a)
    return output


@pytest.mark.parametrize("h , w, max_grid", COMMON_SHAPE_GRID_PARAMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op", [cosh, sinh, mul_add])
def test_composite_ops_l1(device, h, w, max_grid, dtype, op):
    num_inputs = 1
    if op is mul_add:
        num_inputs = 3
    if op is mul_add and (h, w) == (256, 512) and dtype is torch.bfloat16:
        pytest.xfail("OOM error.")
    run_op_test(
        device, h, w, max_grid, dtype, op, num_inputs, buffer_type=ttnn.BufferType.L1
    )


@pytest.mark.parametrize("h , w", COMMON_SHAPE_PARAMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op", [cosh, sinh, mul_add])
def test_composite_ops_dram(device, h, w, dtype, op):
    num_inputs = 1
    if op is mul_add:
        num_inputs = 3
    run_op_test(
        device,
        h,
        w,
        max_grid=(0, 0),
        dtype=dtype,
        op=op,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.DRAM,
    )


PASSING_LARGE_SHAPES_DTYPES_L1 = [
    (512, 2048, torch.bfloat16),
    (512, 4096, torch.bfloat16),
    (512, 8192, torch.bfloat16),
    (1024, 2048, torch.bfloat16),
    (1024, 4096, torch.bfloat16),
]


@pytest.mark.parametrize("hidden_dim, seq_len, dtype", PASSING_LARGE_SHAPES_DTYPES_L1)
def test_large_shapes_muladd_l1(device, hidden_dim, seq_len, dtype):

    max_grid = (7, 7)

    A = create_sharded_tile_tensor(device, seq_len, hidden_dim, max_grid, dtype)
    B = create_sharded_tile_tensor(device, seq_len, hidden_dim, max_grid, dtype)
    C = create_sharded_tile_tensor(device, seq_len, hidden_dim, max_grid, dtype)

    # JIT path
    op_jit = ttnn_jit.jit(debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result, atol=1e-1, rtol=1e-1)


PASSING_LARGE_SHAPES_DTYPES_DRAM = [
    (512, 2048, torch.float32),
    (512, 4096, torch.float32),
    (512, 8192, torch.float32),
    (1024, 2048, torch.float32),
    (1024, 4096, torch.float32),
    (2048, 2048, torch.float32),
    (4096, 1024, torch.float32),
    (512, 2048, torch.bfloat16),
    (512, 4096, torch.bfloat16),
    (512, 8192, torch.bfloat16),
    (512, 16384, torch.bfloat16),
    (1024, 2048, torch.bfloat16),
    (1024, 4096, torch.bfloat16),
    (1024, 8192, torch.bfloat16),
    (2048, 4096, torch.bfloat16),
]


@pytest.mark.parametrize("hidden_dim, seq_len, dtype", PASSING_LARGE_SHAPES_DTYPES_DRAM)
def test_large_shapes_muladd_dram(device, hidden_dim, seq_len, dtype):

    A = create_dram_tensor(device, seq_len, hidden_dim, dtype)
    B = create_dram_tensor(device, seq_len, hidden_dim, dtype)
    C = create_dram_tensor(device, seq_len, hidden_dim, dtype)

    # JIT path
    op_jit = ttnn_jit.jit(debug=True, max_grid=(0, 0))(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("h, w, max_grid", COMMON_SHAPE_GRID_PARAMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.xfail(
    reason="Broadcasting requires either h or w to be 1, but sharded tensor must be at least 32 x 32. Assert error."
)
def test_muladd_broadcast_jit_l1(device, h, w, max_grid, dtype):

    if max_grid == (7, 7):
        pytest.skip(
            "Fatal error in /tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/api/tt-metalium/math.hpp:27, 'Divide by zero error in div_up'"
        )

    A = create_sharded_tile_tensor(device, h, w, max_grid, dtype)
    B = create_sharded_tile_tensor(device, h, w, max_grid, dtype)
    # broadcast C
    C = create_sharded_tile_tensor(device, 1, w, max_grid, dtype)

    # JIT path
    op_jit = ttnn_jit.jit(debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)


@pytest.mark.parametrize(
    "h, w", [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.xfail(reason="All tests failing allclose.")
def test_muladd_broadcast_jit_dram(device, h, w, dtype):

    max_grid = (0, 0)
    A = create_dram_tensor(device, h, w, dtype)
    B = create_dram_tensor(device, h, w, dtype)
    # broadcast C
    C = create_dram_tensor(device, 1, w, dtype)

    # JIT path
    op_jit = ttnn_jit.jit(debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)
