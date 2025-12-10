# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest
import itertools

from utils import (
    _get_ttnn_op,
    all_close_check,
    memory_configs_equal,
    create_dram_tensor,
    create_sharded_tile_tensor,
    run_op_test,
)


BLOCK_SHARDED_SHAPE_GRIDS = []

# Generates all rank 2 shapes with 3 tiles per core in each dimension, with every grid from single core to 8x8.
# TTNN grids are (Width, Height), while tensor shapes are (Height, Width).
BLOCK_SHARDED_SHAPE_GRIDS.extend(
    [
        ((h * 32 * (grid_h + 1), w * 32 * (grid_w + 1)), (grid_w, grid_h))
        for h, w, grid_h, grid_w in itertools.product([3], [3], range(8), range(8))
    ]
)

# Generates all rank 3 shapes with 1 to 2 tiles per core in the 2nd and 3rd rank, with every grid from single core to 8x8.
BLOCK_SHARDED_SHAPE_GRIDS.extend(
    [
        ((batch, h * 32 * (grid_h + 1), w * 32 * (grid_w + 1)), (grid_w, grid_h))
        for batch, h, w, grid_h, grid_w in itertools.product(
            [1, 8], [3], [3], range(8), range(8)
        )
    ]
)

# Generates all rank 4 shapes with 1 to 2 tiles per core in the 3rd and 4th rank, with every grid from single core to 8x8.
BLOCK_SHARDED_SHAPE_GRIDS.extend(
    [
        (
            (batch1, batch2, h * 32 * (grid_h + 1), w * 32 * (grid_w + 1)),
            (grid_w, grid_h),
        )
        for batch1, batch2, h, w, grid_h, grid_w in itertools.product(
            [1, 2], [1, 4], [3], [3], range(8), range(8)
        )
    ]
)

DRAM_INTERLEAVED_SHAPE_GRIDS = []

# 2D shapes
DRAM_INTERLEAVED_SHAPE_GRIDS.extend(
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

# 3D shapes
DRAM_INTERLEAVED_SHAPE_GRIDS.extend(
    [
        (batch, h, w)
        for batch, h, w in itertools.product(
            [1, 8], [32, 64, 128, 256, 512], [32, 64, 128, 256, 512]
        )
    ]
)

# 4D shapes
DRAM_INTERLEAVED_SHAPE_GRIDS.extend(
    [
        (batch1, batch2, h, w)
        for batch1, batch2, h, w in itertools.product(
            [1, 2], [1, 4], [32, 64, 128, 256], [32, 64, 128, 256]
        )
    ]
)


def abs(input_tensor):
    return ttnn.abs(input_tensor)


@pytest.mark.parametrize(
    "shape, max_grid",
    BLOCK_SHARDED_SHAPE_GRIDS,
    ids=[f"{shape}-{grid}" for shape, grid in BLOCK_SHARDED_SHAPE_GRIDS],
)
@pytest.mark.parametrize("op", [abs])
def test_l1_block_sharded_shapes(device, shape, max_grid, op):
    run_op_test(
        device,
        shape,
        max_grid,
        torch.float16,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        enable_cache=True,
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )


HEIGHT_SHARDED_SHAPE_GRIDS = []
HEIGHT_SHARDED_SHAPE_GRIDS.extend(
    [
        ((h * 32 * (grid_w + 1) * (grid_h + 1), w * 32), (grid_w, grid_h))
        for h, w, grid_h, grid_w in itertools.product([3], [3], range(8), range(8))
    ]
)

HEIGHT_SHARDED_SHAPE_GRIDS.extend(
    [
        ((batch, h * 32 * (grid_w + 1) * (grid_h + 1), w * 32), (grid_w, grid_h))
        for batch, h, w, grid_h, grid_w in itertools.product(
            [1, 8], [3], [3], range(8), range(8)
        )
    ]
)
HEIGHT_SHARDED_SHAPE_GRIDS.extend(
    [
        (
            (batch1, batch2, h * 32 * (grid_w + 1) * (grid_h + 1), w * 32),
            (grid_w, grid_h),
        )
        for batch1, batch2, h, w, grid_h, grid_w in itertools.product(
            [1, 2], [1, 4], [3], [3], range(8), range(8)
        )
    ]
)


@pytest.mark.parametrize(
    "shape, max_grid",
    HEIGHT_SHARDED_SHAPE_GRIDS,
    ids=[f"{shape}-{grid}" for shape, grid in HEIGHT_SHARDED_SHAPE_GRIDS],
)
@pytest.mark.parametrize("op", [abs])
def test_l1_height_sharded_shapes(device, shape, max_grid, op):
    run_op_test(
        device,
        shape,
        max_grid,
        torch.float16,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        enable_cache=True,
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )


WIDTH_SHARDED_SHAPE_GRIDS = []

WIDTH_SHARDED_SHAPE_GRIDS.extend(
    [
        ((h * 32, w * 32 * (grid_h + 1) * (grid_w + 1)), (grid_w, grid_h))
        for h, w, grid_h, grid_w in itertools.product([3], [3], range(8), range(8))
    ]
)

WIDTH_SHARDED_SHAPE_GRIDS.extend(
    [
        ((batch, h * 32, w * 32 * (grid_h + 1) * (grid_w + 1)), (grid_w, grid_h))
        for batch, h, w, grid_h, grid_w in itertools.product(
            [1, 8], [3], [3], range(8), range(8)
        )
    ]
)
WIDTH_SHARDED_SHAPE_GRIDS.extend(
    [
        (
            (batch1, batch2, h * 32, w * 32 * (grid_h + 1) * (grid_w + 1)),
            (grid_w, grid_h),
        )
        for batch1, batch2, h, w, grid_h, grid_w in itertools.product(
            [1, 2], [1, 4], [3], [3], range(8), range(8)
        )
    ]
)


@pytest.mark.parametrize(
    "shape, max_grid",
    WIDTH_SHARDED_SHAPE_GRIDS,
    ids=[f"{shape}-{grid}" for shape, grid in WIDTH_SHARDED_SHAPE_GRIDS],
)
@pytest.mark.parametrize("op", [abs])
def test_l1_width_sharded_shapes(device, shape, max_grid, op):
    run_op_test(
        device,
        shape,
        max_grid,
        torch.float16,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        enable_cache=True,
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    )


@pytest.mark.parametrize(
    "shape",
    DRAM_INTERLEAVED_SHAPE_GRIDS,
    ids=[f"{shape}" for shape in DRAM_INTERLEAVED_SHAPE_GRIDS],
)
@pytest.mark.parametrize("op", [abs])
def test_dram_interleaved_shapes(device, shape, op):
    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        torch.float16,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        enable_cache=True,
    )
