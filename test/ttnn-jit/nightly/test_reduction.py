# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import (
    run_op_test,
)


REDUCTION_SHAPES = [
    # (shape, max_grid, dim)
    # 1x1 grid
    ((32, 32), (0, 0), 1),
    ((128, 128), (0, 0), 0),
    ((64, 128), (0, 0), 1),
    ((128, 64), (0, 0), 0),
    ((128, 128), (0, 0), 1),
    ((256, 256), (0, 0), 0),
    ((128, 1024), (0, 0), 0),
    # grids > 1x1
    ((64, 64), (1, 0), 0),
    ((128, 64), (1, 0), 0),
    ((256, 128), (3, 0), 0),
    ((512, 1024), (7, 0), 0),
    ((64, 32), (0, 1), 1),
    ((512, 512), (0, 3), 1),
    ((384, 96), (0, 5), 1),
    ((1024, 1024), (0, 7), 1),
]

REDUCTION_OPS = [
    ttnn.max,
    ttnn.sum,
    ttnn.mean,
    ttnn.min,
]


@pytest.mark.parametrize("shape, max_grid, dim", REDUCTION_SHAPES)
@pytest.mark.parametrize("op", REDUCTION_OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
def test_reductions(device, shape, max_grid, dim, op, dtype, buffer_type):
    """Test reduction operations (max, sum) with L1 or DRAM config."""

    def reduction_func(input_tensor):
        return op(input_tensor, dim=dim, keepdim=True)

    if op == ttnn.mean:
        pytest.skip(reason="Mean is not currently supported in D2M")

    # L1 block-sharded min has PCC issues for shapes with many tiles on a
    # single core.  The neg→max→neg decomposition produces incorrect results
    # when the intermediate negated tensor is L1 block-sharded with a small
    # grid.  DRAM min and L1 min on larger grids work correctly.
    # TODO(sgholami): Investigate L1 block-sharded min PCC failures.
    if (
        op == ttnn.min
        and buffer_type == ttnn.BufferType.L1
        and max_grid in [(0, 0), (1, 0)]
        and shape not in [(32, 32)]
    ):
        pytest.skip(reason="L1 block-sharded min has PCC issues for small grids")

    shard_strategy = (
        ttnn.ShardStrategy.BLOCK if buffer_type == ttnn.BufferType.L1 else None
    )

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        reduction_func,
        num_inputs=1,
        buffer_type=buffer_type,
        shard_strategy=shard_strategy,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
