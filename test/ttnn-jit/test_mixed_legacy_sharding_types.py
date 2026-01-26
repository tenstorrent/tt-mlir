# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import (
    all_close_check,
    create_sharded_tile_tensor,
)
from op_definitions import add

HEIGHT_WIDTH_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    ((2048, 2048), (7, 7)),
    ((2, 32, 64), (0, 0)),
    ((2, 64, 128), (0, 0)),
    ((32, 64, 2048), (7, 7)),
    ((1, 2, 32, 128), (0, 0)),
]

HEIGHT_BLOCK_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    ((256, 64), (0, 7)),
    ((2048, 256), (7, 7)),
    ((2, 128, 128), (0, 0)),
    ((2, 2, 512, 256), (7, 7)),
]

WIDTH_BLOCK_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    ((64, 256), (7, 0)),
    ((256, 256), (0, 7)),
    ((256, 2048), (7, 7)),
    ((1, 32, 32), (0, 0)),
    ((1, 4, 32, 128), (0, 0)),
    ((2, 4, 32, 2048), (7, 7)),
]


@pytest.mark.parametrize(
    "shape, max_grid",
    HEIGHT_WIDTH_SHARDED_SHAPE_GRIDS,
    ids=[
        f"shape={shape}_grid={max_grid}"
        for shape, max_grid in HEIGHT_WIDTH_SHARDED_SHAPE_GRIDS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("op", [add])
def test_height_width_mixed_legacy_sharding_types(device, shape, max_grid, dtype, op):
    if max_grid == (7, 7):
        pytest.xfail("fails due to d2m generic operand grid mismatch")

    input_tensor_a = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        shard_strategy=ttnn.ShardStrategy.HEIGHT,
    )
    input_tensor_b = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        shard_strategy=ttnn.ShardStrategy.WIDTH,
    )

    op_jit = ttnn_jit.jit(
        debug=True,
    )(op)
    output_tensor = op_jit(input_tensor_a, input_tensor_b)

    golden_output = op(input_tensor_a, input_tensor_b)

    assert all_close_check(output_tensor, golden_output)


@pytest.mark.parametrize(
    "shape, max_grid",
    HEIGHT_BLOCK_SHARDED_SHAPE_GRIDS,
    ids=[
        f"shape={shape}_grid={max_grid}"
        for shape, max_grid in HEIGHT_BLOCK_SHARDED_SHAPE_GRIDS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("op", [add])
def test_height_block_mixed_legacy_sharding_types(device, shape, max_grid, dtype, op):

    if max_grid == (7, 7):
        pytest.xfail("fails due to d2m generic operand grid mismatch")
    input_tensor_a = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        shard_strategy=ttnn.ShardStrategy.HEIGHT,
    )
    input_tensor_b = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        shard_strategy=ttnn.ShardStrategy.BLOCK,
    )

    op_jit = ttnn_jit.jit(
        debug=True,
    )(op)
    output_tensor = op_jit(input_tensor_a, input_tensor_b)

    golden_output = op(input_tensor_a, input_tensor_b)

    assert all_close_check(output_tensor, golden_output)


@pytest.mark.parametrize(
    "shape, max_grid",
    WIDTH_BLOCK_SHARDED_SHAPE_GRIDS,
    ids=[
        f"shape={shape}_grid={max_grid}"
        for shape, max_grid in WIDTH_BLOCK_SHARDED_SHAPE_GRIDS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("op", [add])
def test_width_block_mixed_legacy_sharding_types(device, shape, max_grid, dtype, op):

    if max_grid == (7, 7) or max_grid == (0, 7):
        pytest.xfail("fails due to d2m generic operand grid mismatch")
    input_tensor_a = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        shard_strategy=ttnn.ShardStrategy.WIDTH,
    )
    input_tensor_b = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        shard_strategy=ttnn.ShardStrategy.BLOCK,
    )

    op_jit = ttnn_jit.jit(
        debug=True,
    )(op)
    output_tensor = op_jit(input_tensor_a, input_tensor_b)

    golden_output = op(input_tensor_a, input_tensor_b)

    assert all_close_check(output_tensor, golden_output)
