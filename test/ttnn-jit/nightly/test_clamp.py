# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest
from utils import (
    all_close_check,
    create_dram_tensor,
    run_op_test,
)


DRAM_INTERLEAVED_SHAPES = [
    (32, 32),
    (32, 64),
    (64, 32),
    (128, 128),
    (16, 64, 64),
]

L1_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0), ttnn.ShardStrategy.BLOCK),
    ((64, 128), (0, 1), ttnn.ShardStrategy.BLOCK),
    ((256, 32), (3, 1), ttnn.ShardStrategy.HEIGHT),
    ((32, 384), (1, 5), ttnn.ShardStrategy.WIDTH),
    ((2, 32, 384), (1, 5), ttnn.ShardStrategy.WIDTH),
]


# Clamp op wrappers for different parameter styles
def clamp_min_max(input_tensor):
    return ttnn.clamp(input_tensor, min=-7, max=7)


def clamp_min_only(input_tensor):
    return ttnn.clamp(input_tensor, min=-7)


def clamp_max_only(input_tensor):
    return ttnn.clamp(input_tensor, max=7)


def clamp_inf_bounds(input_tensor):
    return ttnn.clamp(input_tensor, min=float("-inf"), max=float("inf"))


def clamp_tensor_bounds(input_tensor, min_tensor, max_tensor):
    return ttnn.clamp(input_tensor, min=min_tensor, max=max_tensor)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "op",
    [
        clamp_min_max,
        clamp_min_only,
        clamp_max_only,
        clamp_inf_bounds,
    ],
)
@pytest.mark.parametrize(
    "shape",
    DRAM_INTERLEAVED_SHAPES,
    ids=[f"{shape}" for shape in DRAM_INTERLEAVED_SHAPES],
)
def test_clamp_scalar_dram(device, shape, dtype, op):
    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        value_range=(-100, 100),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    L1_SHARDED_SHAPE_GRIDS,
    ids=[
        f"shape_{shape}_grid_{grid}_{strategy.name}"
        for shape, grid, strategy in L1_SHARDED_SHAPE_GRIDS
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        clamp_min_max,
        clamp_min_only,
        clamp_max_only,
        clamp_inf_bounds,
    ],
)
def test_clamp_scalar_l1(device, shape, max_grid, shard_strategy, dtype, op):
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=shard_strategy,
        value_range=(-100, 100),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "shape",
    DRAM_INTERLEAVED_SHAPES,
    ids=[f"{shape}" for shape in DRAM_INTERLEAVED_SHAPES],
)
def test_clamp_tensor_bounds_dram(device, shape, dtype):
    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        clamp_tensor_bounds,
        num_inputs=3,
        buffer_type=ttnn.BufferType.DRAM,
        value_range=(-100, 100),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_clamp_tensor_bounds_basic(device, dtype):
    input_tensor = create_dram_tensor(device, (32, 64), dtype, value_range=(-100, 100))
    min_tensor = create_dram_tensor(device, (32, 64), dtype, value_range=(-100, 100))
    max_tensor = create_dram_tensor(device, (32, 64), dtype, value_range=(-100, 100))

    op_jit = ttnn_jit.jit(debug=True)(clamp_tensor_bounds)
    output = op_jit(input_tensor, min_tensor, max_tensor)
    golden = clamp_tensor_bounds(input_tensor, min_tensor, max_tensor)

    assert all_close_check(output, golden)
