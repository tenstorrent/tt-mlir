# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest
from op_definitions import *

from utils import (
    run_op_test,
)

# parameters for identity and dram -> l1
SHAPE_GRID_SHARD_STRATEGY = [
    ((1024, 1024), (7, 7), ttnn.ShardStrategy.BLOCK),
    ((1024, 1024), (3, 3), ttnn.ShardStrategy.BLOCK),
    ((256, 64), (0, 7), ttnn.ShardStrategy.HEIGHT),
    ((256, 64), (0, 3), ttnn.ShardStrategy.HEIGHT),
    ((64, 256), (7, 0), ttnn.ShardStrategy.WIDTH),
    ((64, 256), (3, 0), ttnn.ShardStrategy.WIDTH),
]

# parameters for l1 -> l1 resharding
SHAPE_INPUT_GRID_OUTPUT_GRID_SHARD_STRATEGY = [
    ((1024, 1024), (7, 7), (3, 3), ttnn.ShardStrategy.BLOCK),
    ((1024, 1024), (3, 3), (7, 7), ttnn.ShardStrategy.BLOCK),
    ((256, 64), (0, 7), (0, 3), ttnn.ShardStrategy.HEIGHT),
    ((256, 64), (0, 3), (0, 7), ttnn.ShardStrategy.HEIGHT),
    ((64, 256), (7, 0), (3, 0), ttnn.ShardStrategy.WIDTH),
    ((64, 256), (3, 0), (7, 0), ttnn.ShardStrategy.WIDTH),
]

# parameters for l1 -> l1 change shard strategy
SHAPE_INPUT_GRID_OUTPUT_GRID_INPUT_SHARD_STRATEGY_OUTPUT_SHARD_STRATEGY = [
    ((1024, 1024), (7, 7), (0, 7), ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.HEIGHT),
    ((1024, 1024), (7, 7), (7, 0), ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.WIDTH),
    ((512, 256), (0, 7), (7, 7), ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.BLOCK),
    ((512, 512), (0, 7), (7, 0), ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH),
    ((256, 512), (7, 0), (7, 7), ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK),
    ((256, 512), (7, 0), (0, 7), ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.HEIGHT),
]


# test identity (layout -> same layout)
def test_identity_layout_dram(device):
    shape = (1024, 1024)
    max_grid = (7, 7)
    dtype = torch.float16

    output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op=abs,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        memory_config=output_memory_config,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHAPE_GRID_SHARD_STRATEGY,
    ids=[
        f"shape_{shape}_grid_{max_grid}_{shard_strategy.name}"
        for shape, max_grid, shard_strategy in SHAPE_GRID_SHARD_STRATEGY
    ],
)
@pytest.mark.parametrize(
    "op",
    [abs],
    ids=[op.__name__ for op in [abs]],
)
def test_identity_layout_l1(device, op, shape, max_grid, shard_strategy):

    output_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=max_grid[0] + 1, y=max_grid[1] + 1),
        strategy=shard_strategy,
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
        shard_strategy=shard_strategy,
        memory_config=output_memory_config,
    )


# dram interleaved input -> l1 sharded output
@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHAPE_GRID_SHARD_STRATEGY,
    ids=[
        f"shape_{shape}_grid_{max_grid}_{shard_strategy.name}"
        for shape, max_grid, shard_strategy in SHAPE_GRID_SHARD_STRATEGY
    ],
)
@pytest.mark.parametrize(
    "dtype, ttnn_dtype",
    [
        (torch.float32, None),
        (torch.bfloat16, None),
        (torch.bfloat16, ttnn.DataType.BFLOAT8_B),
    ],
    ids=["f32", "bf16", "bfp8"],
)
@pytest.mark.parametrize("op", [abs], ids=[op.__name__ for op in [abs]])
def test_dram_to_sharded_layout(
    device, op, shape, max_grid, shard_strategy, dtype, ttnn_dtype
):

    output_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=max_grid[0] + 1, y=max_grid[1] + 1),
        strategy=shard_strategy,
        use_height_and_width_as_shard_shape=False,
    )

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        ttnn_dtype=ttnn_dtype,
        memory_config=output_memory_config,
    )


# l1 <-> l1 resharding
@pytest.mark.parametrize(
    "shape, input_max_grid, output_max_grid, shard_strategy",
    SHAPE_INPUT_GRID_OUTPUT_GRID_SHARD_STRATEGY,
    ids=[
        f"shape_{shape}_inGrid_{input_max_grid}_outGrid_{output_max_grid}_{shard_strategy.name}"
        for shape, input_max_grid, output_max_grid, shard_strategy in SHAPE_INPUT_GRID_OUTPUT_GRID_SHARD_STRATEGY
    ],
)
@pytest.mark.parametrize(
    "dtype, ttnn_dtype",
    [
        (torch.float32, None),
        (torch.bfloat16, None),
        (torch.bfloat16, ttnn.DataType.BFLOAT8_B),
    ],
    ids=["f32", "bf16", "bfp8"],
)
@pytest.mark.parametrize("op", [abs], ids=[op.__name__ for op in [abs]])
def test_l1_to_l1_resharding(
    device,
    op,
    shape,
    input_max_grid,
    output_max_grid,
    shard_strategy,
    dtype,
    ttnn_dtype,
):

    output_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=output_max_grid[0] + 1, y=output_max_grid[1] + 1),
        strategy=shard_strategy,
        use_height_and_width_as_shard_shape=False,
    )

    run_op_test(
        device,
        shape,
        input_max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=shard_strategy,
        ttnn_dtype=ttnn_dtype,
        memory_config=output_memory_config,
    )


# change shard strategy l1 -> l1
@pytest.mark.parametrize(
    "shape, input_max_grid, output_max_grid, input_shard_strategy, output_shard_strategy",
    SHAPE_INPUT_GRID_OUTPUT_GRID_INPUT_SHARD_STRATEGY_OUTPUT_SHARD_STRATEGY,
    ids=[
        f"shape_{shape}_inGrid_{input_max_grid}_outGrid_{output_max_grid}_{input_shard_strategy.name}_{output_shard_strategy.name}"
        for shape, input_max_grid, output_max_grid, input_shard_strategy, output_shard_strategy in SHAPE_INPUT_GRID_OUTPUT_GRID_INPUT_SHARD_STRATEGY_OUTPUT_SHARD_STRATEGY
    ],
)
@pytest.mark.parametrize(
    "dtype, ttnn_dtype",
    [
        (torch.float32, None),
        (torch.bfloat16, None),
        (torch.bfloat16, ttnn.DataType.BFLOAT8_B),
    ],
    ids=["f32", "bf16", "bfp8"],
)
@pytest.mark.parametrize(
    "op",
    [abs],
    ids=[op.__name__ for op in [abs]],
)
def test_l1_to_l1_change_shard_strategy(
    device,
    op,
    shape,
    input_max_grid,
    output_max_grid,
    input_shard_strategy,
    output_shard_strategy,
    dtype,
    ttnn_dtype,
):

    output_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=output_max_grid[0] + 1, y=output_max_grid[1] + 1),
        strategy=output_shard_strategy,
        use_height_and_width_as_shard_shape=False,
    )

    run_op_test(
        device,
        shape,
        input_max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=input_shard_strategy,
        ttnn_dtype=ttnn_dtype,
        memory_config=output_memory_config,
    )


@pytest.mark.xfail(
    reason="L1 Sharded -> DRAM Interleaved layout transformation fails, see issue #6735",
)
def test_dram_output_layout(device):
    shape = (1024, 1024)
    max_grid = (7, 7)
    dtype = torch.float16

    output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op=abs,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=ttnn.ShardStrategy.BLOCK,
        memory_config=output_memory_config,
    )
