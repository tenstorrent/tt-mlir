# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest
import itertools

from utils import (
    run_op_test,
    all_close_check,
)

from op_definitions import (
    abs,
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


@pytest.mark.parametrize(
    "shape, max_grid",
    BLOCK_SHARDED_SHAPE_GRIDS,
    ids=[f"{shape}-{grid}" for shape, grid in BLOCK_SHARDED_SHAPE_GRIDS],
)
@pytest.mark.parametrize("op", [abs])
def test_l1_block_sharded_shapes(device, shape, max_grid, op):
    output_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=max_grid[0] + 1, y=max_grid[1] + 1),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )
    run_op_test(
        device,
        shape,
        max_grid,
        torch.float16,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        enable_cache=True,
        shard_strategy=ttnn.ShardStrategy.BLOCK,
        memory_config=output_memory_config,
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


SKIPPED_HEIGHT_SHARDED_CASES = [
    ((1728, 96), (2, 5)),
    ((2880, 96), (4, 5)),
    ((2016, 96), (2, 6)),
    ((3360, 96), (4, 6)),
    ((4032, 96), (5, 6)),
    ((3840, 96), (4, 7)),
    ((4608, 96), (5, 7)),
    ((5376, 96), (6, 7)),
]


@pytest.mark.parametrize(
    "shape, max_grid",
    HEIGHT_SHARDED_SHAPE_GRIDS,
    ids=[f"{shape}-{grid}" for shape, grid in HEIGHT_SHARDED_SHAPE_GRIDS],
)
@pytest.mark.parametrize("op", [abs])
def test_l1_height_sharded_shapes(device, shape, max_grid, op):
    if (shape, max_grid) in SKIPPED_HEIGHT_SHARDED_CASES:
        pytest.skip("Known failing shape/grid case, Issue #7157.")
    output_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=max_grid[0] + 1, y=max_grid[1] + 1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=False,
    )

    run_op_test(
        device,
        shape,
        max_grid,
        torch.float16,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        enable_cache=True,
        shard_strategy=ttnn.ShardStrategy.HEIGHT,
        memory_config=output_memory_config,
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

SKIPPED_WIDTH_SHARDED_CASES = [
    ((96, 1728), (2, 5)),
    ((96, 2880), (4, 5)),
    ((96, 2016), (2, 6)),
    ((96, 3360), (4, 6)),
    ((96, 4032), (5, 6)),
    ((96, 3840), (4, 7)),
    ((96, 4608), (5, 7)),
    ((96, 5376), (6, 7)),
]


@pytest.mark.parametrize(
    "shape, max_grid",
    WIDTH_SHARDED_SHAPE_GRIDS,
    ids=[f"{shape}-{grid}" for shape, grid in WIDTH_SHARDED_SHAPE_GRIDS],
)
@pytest.mark.parametrize("op", [abs])
def test_l1_width_sharded_shapes(device, shape, max_grid, op):
    if (shape, max_grid) in SKIPPED_WIDTH_SHARDED_CASES:
        pytest.skip("Known failing shape/grid case, Issue #7157.")
    output_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=max_grid[0] + 1, y=max_grid[1] + 1),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=False,
    )
    run_op_test(
        device,
        shape,
        max_grid,
        torch.float16,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        enable_cache=True,
        shard_strategy=ttnn.ShardStrategy.WIDTH,
        memory_config=output_memory_config,
    )


def get_sharded_layout(shape):
    strategy = ttnn.ShardStrategy.BLOCK
    if shape[-2] // shape[-1] >= 8:
        strategy = ttnn.ShardStrategy.HEIGHT
    elif shape[-1] // shape[-2] >= 8:
        strategy = ttnn.ShardStrategy.WIDTH

    grid = []
    if strategy == ttnn.ShardStrategy.BLOCK:
        for shard_dim in [shape[-1], shape[-2]]:
            for grid_dim in range(8, 0, -1):
                if (shard_dim // 32) % grid_dim == 0:
                    grid.append(grid_dim)
                    break
    elif strategy == ttnn.ShardStrategy.WIDTH:
        shard_shape = shape[-1] // 32
        for [grid_x, grid_y] in itertools.product(range(8, 0, -1), range(8, 0, -1)):
            if shard_shape % (grid_x * grid_y) == 0:
                grid = [grid_x, grid_y]
                break
    elif strategy == ttnn.ShardStrategy.HEIGHT:
        shard_shape = 1
        for shard_dim in shape[:-1]:
            shard_shape *= shard_dim
        shard_shape = shard_shape // 32
        for [grid_x, grid_y] in itertools.product(range(8, 0, -1), range(8, 0, -1)):
            if shard_shape % (grid_x * grid_y) == 0:
                grid = [grid_x, grid_y]
                break
    assert len(grid) == 2

    return grid, strategy


@pytest.mark.parametrize(
    "shape",
    DRAM_INTERLEAVED_SHAPE_GRIDS,
    ids=[f"{shape}" for shape in DRAM_INTERLEAVED_SHAPE_GRIDS],
)
@pytest.mark.parametrize("op", [abs])
def test_dram_interleaved_shapes(device, shape, op):
    # grid, strategy = get_sharded_layout(shape)
    max_grid = (0, 0)

    # output_memory_config = ttnn.create_sharded_memory_config(
    #     shape=shape,
    #     core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]),
    #     strategy=strategy,
    #     use_height_and_width_as_shard_shape=False,
    # )
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    run_op_test(
        device,
        shape,
        max_grid,
        torch.float16,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        enable_cache=True,
        memory_config=output_memory_config,
    )


ND_LAYOUT_TEST_CASES = [
    ((4, 64, 64), (2, 32, 32), (1, 3)),
    ((8, 64, 64), (2, 32, 32), (3, 3)),
    ((2, 64, 64), (2, 32, 32), (0, 3)),
    ((4, 32, 64), (2, 32, 32), (0, 3)),
    ((4, 64, 32), (2, 32, 32), (0, 3)),
    ((2, 32, 32), (2, 32, 32), (0, 0)),
    # 4D cases
    ((1, 4, 64, 64), (1, 2, 32, 32), (1, 3)),
    ((2, 2, 64, 64), (1, 1, 32, 32), (3, 3)),
]


def create_nd_tensor(device, shape, shard_shape, max_grid):
    torch.manual_seed(0)
    core_ranges = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(*max_grid),
            )
        }
    )

    nd_shard_spec = ttnn.NdShardSpec(
        list(shard_shape),
        core_ranges,
    )
    memory_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=nd_shard_spec,
    )

    torch_tensor = torch.randn(tuple(shape), dtype=torch.float16)
    nd_sharded = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    return nd_sharded


@pytest.mark.skip(reason="See issue #6950")
@pytest.mark.parametrize(
    "shape, shard_shape, max_grid",
    ND_LAYOUT_TEST_CASES,
    ids=[
        f"shape:{shape}-shard:{shard_shape}-grid:{max_grid}"
        for shape, shard_shape, max_grid in ND_LAYOUT_TEST_CASES
    ],
)
def test_nd_ttnn_layout(device, shape, shard_shape, max_grid):
    nd_input_tensor = create_nd_tensor(device, shape, shard_shape, max_grid)
    op_jit = ttnn_jit.jit(debug=True, enable_cache=False)(abs)
    output_tensor = op_jit(nd_input_tensor)

    # We cannot run the TTNN op with an ND tensor as no op supports it, including to_memory_config.
    dram_input_tensor = ttnn.from_torch(
        nd_input_tensor.cpu().to_torch(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    golden_output_tensor = ttnn.abs(dram_input_tensor)
    assert all_close_check(output_tensor, golden_output_tensor)
