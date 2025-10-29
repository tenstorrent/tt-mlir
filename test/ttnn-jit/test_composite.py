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
)

COMMON_SHAPE_GRID_PARAMS = [
    (32, 32, (0, 0)),
    (32, 64, (0, 0)),
    (64, 64, (0, 0)),
    (64, 128, (0, 0))
]

COMMON_SHAPE_PARAMS = [
    (32, 32),
    (32, 64),
    (64, 64),
    (64, 128)
]

def run_op_test(
    device, h, w, max_grid, dtype, op, num_inputs, buffer_type=ttnn.BufferType.L1
):
    if buffer_type == ttnn.BufferType.L1:
        inputs = [
            create_sharded_tile_tensor(device, h, w, max_grid, dtype)
            for _ in range(num_inputs)
        ]
    else:
        inputs = [create_dram_tensor(device, h, w, dtype) for _ in range(num_inputs)]
    print("inputs", inputs)
    golden_op = _get_ttnn_op(op)

    op_jit = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(op)
    output_tensor = op_jit(*inputs)
    golden_tensor = (golden_op or op)(*inputs)

    assert memory_configs_equal(
        output_tensor.memory_config(), golden_tensor.memory_config()
    )
    assert all_close_check(output_tensor, golden_tensor)


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
    run_op_test(
        device, h, w, max_grid, dtype, op, num_inputs, buffer_type=ttnn.BufferType.L1
    )

@pytest.mark.parametrize("h , w", COMMON_SHAPE_PARAMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op", [cosh, sinh, mul_add])
def test_composite_ops_dram(device, h, w, dtype, op):
    max_grid = (0, 0)
    num_inputs = 1
    if op is mul_add:
        num_inputs = 3
    run_op_test(
        device, h, w, max_grid, dtype, op, num_inputs, buffer_type=ttnn.BufferType.DRAM
    )