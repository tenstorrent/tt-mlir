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
# Composite operations
# These operations are composed of multiple primitive operations
# ------------------------------------------------------------


# Test shapes for composite operations
COMPOSITE_SHAPES = [
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]

# ------------------------------------------------------------
# Special functions
# ------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.xfail(reason="D2M Does not support reciprocal op")
def test_digamma_dram(device, shape, dtype):
    """Test digamma function (derivative of log gamma)"""

    def digamma_func(input_tensor):
        return ttnn.digamma(input_tensor)

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        digamma_func,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=True,
    )


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_complex_composite_dram(device, shape, dtype):
    """Test complex composite operation with multiple steps"""

    def complex_op(a, b):
        # (a + b) * (a - b) = a^2 - b^2
        sum_ab = ttnn.add(a, b)
        diff_ab = ttnn.subtract(a, b)
        return ttnn.multiply(sum_ab, diff_ab)

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        complex_op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=True,
    )


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_nested_composite_dram(device, shape, dtype):
    """Test nested composite operations"""

    def nested_op(a, b, c):
        # ((a + b) * c) + a
        sum_ab = ttnn.add(a, b)
        prod_abc = ttnn.multiply(sum_ab, c)
        return ttnn.add(prod_abc, a)

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        nested_op,
        num_inputs=3,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=True,
    )
