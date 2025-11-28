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

BLOCK_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    ((64, 64), (0, 0)),
    ((64, 128), (0, 0)),
    ((256, 256), (7, 7)),
    ((256, 512), (7, 7)),
    ((512, 512), (7, 7)),
    ((512, 1024), (7, 7)),
    ((1024, 1024), (7, 7)),
    ((2, 512, 2048), (7, 7)),
]

HEIGHT_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    ((256, 64), (7, 0)),
    ((256, 64), (0, 7)),
    ((2048, 128), (7, 7)),
    ((2, 192, 32), (1, 5)),
]

WIDTH_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    ((64, 256), (7, 0)),
    ((64, 256), (0, 7)),
    ((128, 2048), (7, 7)),
    ((2, 32, 384), (1, 5)),
]

SHARDED_SHAPE_GRID_LAYOUTS = (
    [
        (shape, grid, ttnn.TensorMemoryLayout.BLOCK_SHARDED)
        for shape, grid in BLOCK_SHARDED_SHAPE_GRIDS
    ]
    + [
        (shape, grid, ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
        for shape, grid in HEIGHT_SHARDED_SHAPE_GRIDS
    ]
    + [
        (shape, grid, ttnn.TensorMemoryLayout.WIDTH_SHARDED)
        for shape, grid in WIDTH_SHARDED_SHAPE_GRIDS
    ]
)

DRAM_SHAPES = [(32, 32), (32, 64), (64, 64), (64, 128), (128, 128), (1024, 1024)]


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


@pytest.mark.parametrize(
    "shape, max_grid, memory_layout",
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{layout}"
        for shape, grid, layout in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op", [cosh, sinh, mul_add])
def test_composite_ops_l1(device, shape, max_grid, dtype, op, memory_layout):
    num_inputs = 1
    if op is mul_add:
        # num_inputs = 3
        pytest.xfail(
            "mul_add fails allclose, see https://github.com/tenstorrent/tt-mlir/issues/5873"
        )
    if op is mul_add and shape == (256, 512) and dtype is torch.bfloat16:
        pytest.xfail("OOM error.")
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs,
        buffer_type=ttnn.BufferType.L1,
        memory_layout=memory_layout,
    )


@pytest.mark.parametrize("shape", DRAM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op", [cosh, sinh, mul_add])
def test_composite_ops_dram(device, shape, dtype, op):
    num_inputs = 1
    if op is mul_add:
        # num_inputs = 3
        pytest.xfail(
            "mul_add fails allclose, see https://github.com/tenstorrent/tt-mlir/issues/5873"
        )
    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=op,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.DRAM,
    )


PASSING_LARGE_SHAPES_DTYPES_L1 = [
    ((2048, 512), torch.bfloat16),
    ((4096, 512), torch.bfloat16),
    ((8192, 512), torch.bfloat16),
    ((2048, 1024), torch.bfloat16),
]


@pytest.mark.parametrize("shape, dtype", PASSING_LARGE_SHAPES_DTYPES_L1)
@pytest.mark.xfail(
    reason="mul_add fails allclose, see https://github.com/tenstorrent/tt-mlir/issues/5873"
)
def test_large_shapes_muladd_l1(device, shape, dtype):

    num_inputs = 3

    run_op_test(
        device,
        shape,
        max_grid=(7, 7),
        dtype=dtype,
        op=mul_add,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.L1,
    )


PASSING_LARGE_SHAPES_DTYPES_DRAM = [
    ((2048, 512), torch.float32),
    ((4096, 512), torch.float32),
    ((8192, 512), torch.float32),
    ((2048, 1024), torch.float32),
    ((4096, 1024), torch.float32),
    ((2048, 2048), torch.float32),
    ((1024, 4096), torch.float32),
    ((2048, 512), torch.bfloat16),
    ((4096, 512), torch.bfloat16),
    ((8192, 512), torch.bfloat16),
    ((16384, 512), torch.bfloat16),
    ((2048, 1024), torch.bfloat16),
    ((4096, 1024), torch.bfloat16),
    ((8192, 1024), torch.bfloat16),
    ((4096, 2048), torch.bfloat16),
]


@pytest.mark.parametrize("shape, dtype", PASSING_LARGE_SHAPES_DTYPES_DRAM)
@pytest.mark.xfail(
    reason="mul_add fails allclose, see https://github.com/tenstorrent/tt-mlir/issues/5873"
)
def test_large_shapes_muladd_dram(device, shape, dtype):

    num_inputs = 3

    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=mul_add,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.DRAM,
    )


@pytest.mark.parametrize("shape, max_grid", BLOCK_SHARDED_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.xfail(
    reason="Broadcasting requires either h or w to be 1, but sharded tensor must be at least 32 x 32. Assert error."
)
def test_muladd_broadcast_jit_l1(device, shape, max_grid, dtype):

    if max_grid == (7, 7):
        pytest.skip(
            "Fatal error in /tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/api/tt-metalium/math.hpp:27, 'Divide by zero error in div_up'"
        )

    A = create_sharded_tile_tensor(device, shape, max_grid, dtype)
    B = create_sharded_tile_tensor(device, shape, max_grid, dtype)
    # broadcast C
    C = create_sharded_tile_tensor(device, 1, shape[1], max_grid, dtype)

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
    "shape", [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.xfail(reason="All tests failing allclose.")
def test_muladd_broadcast_jit_dram(device, shape, dtype):

    max_grid = (0, 0)
    A = create_dram_tensor(device, shape, dtype)
    B = create_dram_tensor(device, shape, dtype)
    # broadcast C
    C = create_dram_tensor(device, 1, shape[1], dtype)

    # JIT path
    op_jit = ttnn_jit.jit(debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)
