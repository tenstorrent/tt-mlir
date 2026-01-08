# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest
from op_definitions import *

from utils import (
    _get_ttnn_op,
    all_close_check,
    memory_configs_equal,
    create_dram_tensor,
    create_sharded_tile_tensor,
    run_op_test,
)

dram_config = ttnn.DRAM_MEMORY_CONFIG

DRAM_SHAPES = [
    (1024, 1024),
]

SHARD_SHAPES_GRIDS = [
    ((1024, 1024), (7, 7), ttnn.ShardStrategy.BLOCK),
]


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHARD_SHAPES_GRIDS,
    ids=[f"{shape}_{strategy.name}" for shape, grid, strategy in SHARD_SHAPES_GRIDS],
)
@pytest.mark.parametrize(
    "dtype, ttnn_dtype",
    [
        (torch.bfloat16, None),
    ],
    ids=["bf16"],
)
@pytest.mark.parametrize(
    "op",
    [
        abs,
    ],
)
@pytest.mark.parametrize("graph_capture", [True])
def test_unary_op_l1(
    device, shape, max_grid, shard_strategy, dtype, ttnn_dtype, op, graph_capture
):
    if op in [log, ceil, floor, sqrt, logical_not] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes for float32")
    if op == reciprocal and (
        ttnn_dtype == ttnn.DataType.BFLOAT8_B or dtype == torch.float32
    ):
        pytest.xfail("reciprocal not supported for bfp8 or f32")

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        graph_capture=graph_capture,
        shard_strategy=shard_strategy,
        ttnn_dtype=ttnn_dtype,
    )


@pytest.mark.parametrize(
    "shape",
    DRAM_SHAPES,
    ids=[f"{shape}" for shape in DRAM_SHAPES],
)
@pytest.mark.parametrize(
    "dtype, ttnn_dtype",
    [
        (torch.bfloat16, None),
    ],
    ids=["bf16"],
)
@pytest.mark.parametrize(
    "op",
    [
        abs,
    ],
)
@pytest.mark.parametrize("graph_capture", [False])
def test_unary_op_dram(device, shape, dtype, ttnn_dtype, op, graph_capture):
    if dtype == torch.float32 and shape == (2048, 2048):
        pytest.skip("Skipping large operation for float32")
    if op in [log, ceil, floor, sqrt, logical_not] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes for float32")

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=graph_capture,
        ttnn_dtype=ttnn_dtype,
    )


@pytest.mark.parametrize("frontend", ["tracing", "graph_capture", "ast"])
def test_layout(device, frontend):
    @ttnn_jit.jit(
        frontend=frontend, debug=True, compile_only=True, memory_config=dram_config
    )
    def abs_jit(x):
        return ttnn.abs(x)

    input_0_torch = torch.randn(512, 512, dtype=torch.bfloat16)
    memory_config = ttnn.create_sharded_memory_config(
        shape=(512, 512),
        core_grid=ttnn.CoreGrid(x=8, y=8),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )
    input_0_ttnn = ttnn.from_torch(
        input_0_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    print(input)

    output_tensor = abs_jit(input_0_ttnn)
    print(output_tensor)
    assert True


@pytest.mark.parametrize("frontend", ["tracing", "graph_capture", "ast"])
def test_layout_double(device, frontend):
    @ttnn_jit.jit(frontend=frontend, debug=True, compile_only=True, memory_config=None)
    def abs_jit(x):
        x = ttnn.abs(x)
        return ttnn.exp(x)

    input_0_torch = torch.randn(512, 512, dtype=torch.bfloat16)
    memory_config = ttnn.create_sharded_memory_config(
        shape=(512, 512),
        core_grid=ttnn.CoreGrid(x=8, y=8),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )
    input_0_ttnn = ttnn.from_torch(
        input_0_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    print(input)

    output_tensor = abs_jit(input_0_ttnn)
    print("hello", type(output_tensor))
    golden_tensor = ttnn.exp(ttnn.abs(input_0_ttnn))
    print(output_tensor)
    assert all_close_check(output_tensor, golden_tensor)
