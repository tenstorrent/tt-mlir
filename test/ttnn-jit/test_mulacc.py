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
SHARDED_SHAPE_GRIDS = (
    [
        (shape, grid, ttnn.ShardStrategy.BLOCK)
        for shape, grid in BLOCK_SHARDED_SHAPE_GRIDS
    ]
    + [
        (shape, grid, ttnn.ShardStrategy.HEIGHT)
        for shape, grid in HEIGHT_SHARDED_SHAPE_GRIDS
    ]
    + [
        (shape, grid, ttnn.ShardStrategy.WIDTH)
        for shape, grid in WIDTH_SHARDED_SHAPE_GRIDS
    ]
)

DRAM_SHAPES = [
    (256, 256),
    (256, 512),
    (512, 512),
    (512, 1024),
    (512, 2048),
    (1024, 1024),
    # other shapes
    (1705, 320),
    (1705, 640),
    (4095, 640),
    (4095, 1280),
    (8190, 640),
    (8190, 1280),
]


def mul_add(input_tensor_a, input_tensor_b, input_tensor_c):
    matmul_result = ttnn.multiply(input_tensor_b, input_tensor_c)
    output = ttnn.add(matmul_result, input_tensor_a)
    return output


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHARDED_SHAPE_GRIDS,
    ids=[
        f"shape_{shape}_grid_{grid}_strategy_{strategy}"
        for shape, grid, strategy in SHARDED_SHAPE_GRIDS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op", [mul_add])
@pytest.mark.parametrize("use_graph_capture", [True, False])
def test_muladd_l1(
    device, shape, max_grid, shard_strategy, dtype, op, use_graph_capture
):
    num_inputs = 3
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs,
        buffer_type=ttnn.BufferType.L1,
        graph_capture=use_graph_capture,
        shard_strategy=shard_strategy,
    )


@pytest.mark.parametrize(
    "shape", DRAM_SHAPES, ids=[f"dram_shape_{s[0]}x{s[1]}" for s in DRAM_SHAPES]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bfloat16"])
@pytest.mark.parametrize("op", [mul_add])
@pytest.mark.parametrize("use_graph_capture", [True, False])
def test_muladd_dram(device, shape, dtype, op, use_graph_capture):
    num_inputs = 3
    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=op,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=use_graph_capture,
    )
