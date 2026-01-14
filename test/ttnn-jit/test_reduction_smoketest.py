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


L1_REDUCTION_SHAPES = [
    # (shape, max_grid, dim)
    ((32, 32), (0, 0), 1),
    ((128, 128), (0, 0), 0),
    ((256, 256), (0, 0), 1),
    ((128, 64), (1, 0), 0),
    ((512, 1024), (7, 0), 0),
    ((64, 32), (0, 1), 1),
    ((1024, 1024), (0, 7), 1),
]

DRAM_REDUCTION_SHAPES = [(shape, dim) for shape, _, dim in L1_REDUCTION_SHAPES]

REDUCTION_OPS = [
    ("max", ttnn.max),
    ("sum", ttnn.sum),
    ("mean", ttnn.mean),
    ("min", ttnn.min),
]


# ------------------------------------------------------------
# L1 reduction tests (max, sum)
# ------------------------------------------------------------
@pytest.mark.parametrize(
    "shape, max_grid, dim",
    L1_REDUCTION_SHAPES,
    ids=[f"{shape}_grid{grid}_dim{dim}" for shape, grid, dim in L1_REDUCTION_SHAPES],
)
@pytest.mark.parametrize("op_name, op_func", REDUCTION_OPS)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16],
    ids=["f32", "bf16"],
)
@pytest.mark.skip(
    reason="Reduction ops are not currently supported in ttnn-jit: Issue #6623"
)
def test_reductions_l1(device, shape, max_grid, dim, op_name, op_func, dtype):
    """Test reduction operations (max, sum) with L1 block-sharded config."""

    def reduction_func(input_tensor):
        return op_func(input_tensor, dim=dim, keepdim=True)

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        reduction_func,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=ttnn.ShardStrategy.BLOCK,
    )


# ------------------------------------------------------------
# DRAM reduction tests (max, sum)
# ------------------------------------------------------------
@pytest.mark.parametrize(
    "shape, dim",
    DRAM_REDUCTION_SHAPES,
    ids=[f"{shape}_dim{dim}" for shape, dim in DRAM_REDUCTION_SHAPES],
)
@pytest.mark.parametrize("op_name, op_func", REDUCTION_OPS)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16],
    ids=["f32", "bf16"],
)
@pytest.mark.skip(reason="DRAM reduction ops are not currently supported: Issue #6623")
def test_reductions_dram(device, shape, dim, op_name, op_func, dtype):
    """Test reduction operations (max, sum) with DRAM config."""

    def reduction_func(input_tensor):
        return op_func(input_tensor, dim=dim, keepdim=True)

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        reduction_func,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
    )
