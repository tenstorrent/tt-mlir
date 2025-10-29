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

# Generates all shapes with 1 to 4 tiles per core in each dimension, with every grid from single core to 8x8.
# TTNN grids are (Width, Height), while tensor shapes are (Height, Width).
BLOCK_SHARDED_SHAPE_GRIDS = [
    (h * 32 * (grid_h + 1), w * 32 * (grid_w + 1), (grid_w, grid_h))
    for h in range(1, 5)
    for w in range(1, 5)
    for grid_h in range(8)
    for grid_w in range(8)
]

# TODO (5415): These grids fail for all shapes.
GRIDS_FAILING_ALL_SHAPES = [(1, 1), (1, 2), (1, 3), (3, 2), (3, 3), (5, 3)]

DRAM_INTERLEAVED_SHAPE_GRIDS = (
    [(32 * (2**x), 2048) for x in range(6)]
    + [(2048, 32 * (2**x)) for x in range(6)]
    + [(2**x, 2**x) for x in range(5, 12)]
    + [
        (32, 64),
        (64, 32),
        (32, 256),
        (256, 32),
    ]
)


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


def abs(input_tensor):
    return ttnn.abs(input_tensor)


@pytest.mark.skip(reason="Failing non-deterministicly in CI. Issue: #5550")
@pytest.mark.parametrize(
    "h , w, max_grid",
    BLOCK_SHARDED_SHAPE_GRIDS,
    ids=[f"{h}-{w}-{grid}" for h, w, grid in BLOCK_SHARDED_SHAPE_GRIDS],
)
@pytest.mark.parametrize("op", [abs])
def test_l1_block_sharded_shapes(device, h, w, max_grid, op):
    if max_grid in GRIDS_FAILING_ALL_SHAPES:
        pytest.xfail("Grid fails for all shapes. Issue: #5415")
    if (h, w) == (256, 672) and max_grid == (6, 1):
        pytest.xfail("Golden verification fails. Issue: #5550")

    run_op_test(
        device,
        h,
        w,
        max_grid,
        torch.float16,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
    )


@pytest.mark.skip(reason="Failing non-deterministicly in CI. Issue: #5550")
@pytest.mark.parametrize(
    "h , w",
    DRAM_INTERLEAVED_SHAPE_GRIDS,
    ids=[f"{h}-{w}" for h, w in DRAM_INTERLEAVED_SHAPE_GRIDS],
)
@pytest.mark.parametrize("op", [abs])
def test_dram_interleaved_shapes(device, h, w, op):
    max_grid = (0, 0)
    run_op_test(
        device,
        h,
        w,
        max_grid,
        torch.float16,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
    )
