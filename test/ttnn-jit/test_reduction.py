# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import (
    all_close_check,
    memory_configs_equal,
    create_dram_tensor,
    run_op_test,
)

# Testing L1 block-sharded reductions.
# Start with simplest case: 2D tensor, 1x1 grid, reduce dim 0.
# Output will be 1xW which may have sharding issues, but let's see what error we get.

L1_SIMPLE_REDUCTION_SHAPES_GRIDS = [
    # Simplest: single tile, 1x1 grid
    ((32, 32), (0, 0), 0),
    # ((32, 32), (0, 0), 1),
    # 2x2 tiles
    # ((64, 64), (0, 0), 0),
    # ((64, 64), (0, 0), 1),
    # Rectangular shapes (common in real models)
    # ((64, 128), (0, 0), 0),
    # ((64, 128), (0, 0), 1),
    # ((128, 64), (0, 0), 0),
    # ((128, 64), (0, 0), 1),
    # Larger shapes
    # ((128, 128), (0, 0), 0),
    # ((128, 128), (0, 0), 1),
    # ((256, 256), (0, 0), 0),
    # ((256, 256), (0, 0), 1),
    # 3D batch reductions
    # ((4, 64, 64), (0, 0), 0),
    # ((4, 64, 64), (0, 0), 1),
    # ((4, 64, 64), (0, 0), 2),
    # ((8, 128, 128), (0, 0), 0),
    # ((8, 128, 128), (0, 0), 1),
    # ((8, 128, 128), (0, 0), 2),
]


# ------------------------------------------------------------
# Max operation tests (L1 block-sharded, simple 2D)
# ------------------------------------------------------------
@pytest.mark.parametrize("shape, max_grid, dim", L1_SIMPLE_REDUCTION_SHAPES_GRIDS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_max_L1_simple(device, shape, max_grid, dim, dtype):
    """Test max reduction with simplest L1 block-sharded config."""

    def max_func(input_tensor):
        return ttnn.max(input_tensor, dim=dim, keepdim=True)

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        max_func,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        graph_capture=True,
    )


# ------------------------------------------------------------
# Sum operation tests (L1 block-sharded)
# ------------------------------------------------------------
@pytest.mark.parametrize("shape, max_grid, dim", L1_SIMPLE_REDUCTION_SHAPES_GRIDS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sum_L1_simple(device, shape, max_grid, dim, dtype):
    """Test sum reduction with L1 block-sharded config."""

    def sum_func(input_tensor):
        return ttnn.sum(input_tensor, dim=dim, keepdim=True)

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        sum_func,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        graph_capture=True,
    )


# ------------------------------------------------------------
# Min operation tests (L1 block-sharded)
# ------------------------------------------------------------
@pytest.mark.parametrize("shape, max_grid, dim", L1_SIMPLE_REDUCTION_SHAPES_GRIDS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_min_L1_simple(device, shape, max_grid, dim, dtype):
    """Test min reduction with L1 block-sharded config."""

    def min_func(input_tensor):
        return ttnn.min(input_tensor, dim=dim, keepdim=True)

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        min_func,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        graph_capture=True,
    )


# ------------------------------------------------------------
# Mean operation tests (L1 block-sharded)
# ------------------------------------------------------------
@pytest.mark.parametrize("shape, max_grid, dim", L1_SIMPLE_REDUCTION_SHAPES_GRIDS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_mean_L1_simple(device, shape, max_grid, dim, dtype):
    """Test mean reduction with L1 block-sharded config."""

    def mean_func(input_tensor):
        return ttnn.mean(input_tensor, dim=dim, keepdim=True)

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        mean_func,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        graph_capture=True,
    )


# ------------------------------------------------------------
# DRAM Interleaved Tests (for reference/comparison)
# ------------------------------------------------------------


# ------------------------------------------------------------
# Sum operation tests (DRAM)
# ------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,dim",
    [
        # Rank-2 tensors with dim=None
        # ((64, 64), None),
        # Rank-2 tensors with specific dims
        ((64, 64), 0),
        # ((128, 128), 0),
        # ((256, 256), 0),
        # ((64, 64), 1),
        # ((128, 128), 1),
        # ((256, 256), 1),
        # Rank-3 tensors
        # ((4, 64, 64), 0),
        # ((8, 128, 128), 0),
        # ((4, 64, 64), 1),
        # ((8, 128, 128), 1),
        # ((4, 64, 64), 2),
        # ((8, 128, 128), 2),
    ],
)
@pytest.mark.parametrize("keepdim", [True])
@pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.skip(
#    reason="D2M Does not support reduction ops - causes assertion crashes"
# )
def test_sum(device, shape, dim, keepdim, dtype):
    """Test sum reduction operation with various dimensions and keepdim values"""

    def sum_func(input_tensor):
        return ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        sum_func,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=True,
    )


# ------------------------------------------------------------
# Mean operation tests
# ------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,dim",
    [
        # Rank-2 tensors with dim=None
        # ((64, 64), None),
        # Rank-2 tensors with specific dims
        ((64, 64), 0),
        # ((128, 128), 0),
        # ((256, 256), 0),
        # ((64, 64), 1),
        # ((128, 128), 1),
        # ((256, 256), 1),
        # Rank-3 tensors
        # ((4, 64, 64), 0),
        # ((8, 128, 128), 0),
        # ((4, 64, 64), 1),
        # ((8, 128, 128), 1),
        # ((4, 64, 64), 2),
        # ((8, 128, 128), 2),
    ],
)
@pytest.mark.parametrize("keepdim", [True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
# @pytest.mark.skip(
#    reason="D2M Does not support reduction ops - causes assertion crashes"
# )
def test_mean(device, shape, dim, keepdim, dtype):
    """Test mean reduction operation with various dimensions and keepdim values"""

    def mean_func(input_tensor):
        return ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        mean_func,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=True,
    )


# ------------------------------------------------------------
# Max operation tests
# ------------------------------------------------------------
@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.skip(
#    reason="D2M Does not support reduction ops - causes assertion crashes"
# )
def test_max(device, shape, dim, dtype):
    """Test max reduction operation"""

    def max_func(input_tensor):
        return ttnn.max(input_tensor, dim=dim, keepdim=True)

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        max_func,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=True,
    )


# ------------------------------------------------------------
# Min operation tests
# ------------------------------------------------------------
@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.skip(
#    reason="D2M Does not support reduction ops - causes assertion crashes"
# )
def test_min(device, shape, dim, dtype):
    """Test min reduction operation"""

    def min_func(input_tensor):
        return ttnn.min(input_tensor, dim=dim, keepdim=True)

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        min_func,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=True,
    )
