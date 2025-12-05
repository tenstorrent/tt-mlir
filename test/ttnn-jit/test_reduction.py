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

# ------------------------------------------------------------
# Reduction operations
# ------------------------------------------------------------


# ------------------------------------------------------------
# Sum operation tests
# ------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,dim",
    [
        # Rank-2 tensors with dim=None
        ((64, 64), None),
        # Rank-2 tensors with specific dims
        ((64, 64), 0),
        ((128, 128), 0),
        ((256, 256), 0),
        ((64, 64), 1),
        ((128, 128), 1),
        ((256, 256), 1),
        # Rank-3 tensors
        ((4, 64, 64), 0),
        ((8, 128, 128), 0),
        ((4, 64, 64), 1),
        ((8, 128, 128), 1),
        ((4, 64, 64), 2),
        ((8, 128, 128), 2),
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skip(reason="D2M does not support reduction ops")
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
        ((64, 64), None),
        # Rank-2 tensors with specific dims
        ((64, 64), 0),
        ((128, 128), 0),
        ((256, 256), 0),
        ((64, 64), 1),
        ((128, 128), 1),
        ((256, 256), 1),
        # Rank-3 tensors
        ((4, 64, 64), 0),
        ((8, 128, 128), 0),
        ((4, 64, 64), 1),
        ((8, 128, 128), 1),
        ((4, 64, 64), 2),
        ((8, 128, 128), 2),
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skip(reason="D2M Does not support reduction ops")
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
@pytest.mark.skip(reason="D2M Does not support reduction ops")
def test_max(device, shape, dim, dtype):
    """Test max reduction operation"""

    def max_func(input_tensor):
        return ttnn.max(input_tensor, dim=dim)

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
@pytest.mark.skip(reason="D2M Does not support reduction ops")
def test_min(device, shape, dim, dtype):
    """Test min reduction operation"""

    def min_func(input_tensor):
        return ttnn.min(input_tensor, dim=dim)

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
