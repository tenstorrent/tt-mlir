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

TEST_SHAPE_GRIDS = [
    (32, 32, (0, 0)),
    (64, 64, (0, 0)),
    (256, 256, (7, 7)),
    (2048, 2048, (7, 7)),
    (64, 128, (0, 0)),
]

# ------------------------------------------------------------
# Control flow tests
# ------------------------------------------------------------


def if_else_branch(input_tensor, use_exp=True):
    """Test basic if/else branching."""
    if use_exp:
        output = ttnn.exp(input_tensor)
    else:
        output = ttnn.log(input_tensor)
    return output


def nested_if(input_tensor, mode=0):
    """Test nested if statements."""
    if mode == 0:
        output = ttnn.abs(input_tensor)
    else:
        if mode == 1:
            output = ttnn.exp(input_tensor)
        else:
            output = ttnn.sin(input_tensor)
    return output


def if_with_ops_before_after(input_tensor, apply_exp=True):
    """Test if statement with operations before and after."""
    temp = ttnn.abs(input_tensor)

    if apply_exp:
        temp = ttnn.exp(temp)
    else:
        temp = ttnn.sin(temp)

    output = ttnn.multiply(temp, 0.5)
    return output


def multiple_sequential_ifs(input_tensor, apply_exp=False, apply_cos=False):
    """Test multiple sequential if statements."""
    output = input_tensor

    if apply_exp:
        output = ttnn.exp(output)

    if apply_cos:
        output = ttnn.cos(output)

    return output


def for_loop_simple(input_tensor, iterations=3):
    """Test simple for loop."""
    output = input_tensor
    for _ in range(iterations):
        output = ttnn.add(output, input_tensor)
    return output


def for_loop_with_index(input_tensor, iterations=3):
    """Test for loop with index variable."""
    output = input_tensor
    for i in range(iterations):
        if i == 0:
            output = ttnn.multiply(output, 2.0)
        else:
            output = ttnn.add(output, input_tensor)
    return output


def nested_for_loops(input_tensor, outer=2, inner=2):
    """Test nested for loops."""
    output = input_tensor
    for _ in range(outer):
        for _ in range(inner):
            output = ttnn.multiply(output, 1.1)
    return output


def for_loop_with_if(input_tensor, iterations=4):
    """Test for loop with if statement inside."""
    output = input_tensor
    for i in range(iterations):
        if i % 2 == 0:
            output = ttnn.add(output, input_tensor)
        else:
            output = ttnn.multiply(output, 0.9)
    return output


def if_inside_for_multiple_branches(input_tensor, iterations=3):
    """Test for loop with multiple if branches inside."""
    output = input_tensor
    for i in range(iterations):
        if i == 0:
            output = ttnn.exp(output)
        elif i == 1:
            output = ttnn.sin(output)
        else:
            output = ttnn.cos(output)
    return output


# ------------------------------------------------------------
# Control flow tests
# ------------------------------------------------------------
@pytest.mark.parametrize("h , w, max_grid", TEST_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_exp", [True, False])
def test_if_else_branch(device, h, w, max_grid, dtype, use_exp):
    """Test basic if/else branching with different branches."""

    def op_wrapper(input_tensor):
        return if_else_branch(input_tensor, use_exp=use_exp)

    run_op_test(
        device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1
    )


@pytest.mark.parametrize("h , w, max_grid", TEST_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_nested_if(device, h, w, max_grid, dtype, mode):
    """Test nested if statements with different modes."""

    def op_wrapper(input_tensor):
        return nested_if(input_tensor, mode=mode)

    run_op_test(
        device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1
    )


@pytest.mark.parametrize("h , w, max_grid", TEST_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("apply_exp", [True, False])
def test_if_with_ops_before_after(device, h, w, max_grid, dtype, apply_exp):
    """Test if statement with operations before and after."""

    def op_wrapper(input_tensor):
        return if_with_ops_before_after(input_tensor, apply_exp=apply_exp)

    run_op_test(
        device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1
    )


@pytest.mark.parametrize("h , w, max_grid", TEST_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "apply_exp, apply_cos", [(True, False), (False, True), (True, True)]
)
def test_multiple_sequential_ifs(device, h, w, max_grid, dtype, apply_exp, apply_cos):
    """Test multiple sequential if statements. Note: (False, False) case excluded as it performs no operations."""

    def op_wrapper(input_tensor):
        return multiple_sequential_ifs(
            input_tensor, apply_exp=apply_exp, apply_cos=apply_cos
        )

    run_op_test(
        device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1
    )


@pytest.mark.parametrize("h , w, max_grid", TEST_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("iterations", [1, 2, 3, 5])
def test_for_loop_simple(device, h, w, max_grid, dtype, iterations):
    """Test simple for loop with different iteration counts."""

    def op_wrapper(input_tensor):
        return for_loop_simple(input_tensor, iterations=iterations)

    run_op_test(
        device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1
    )


@pytest.mark.parametrize("h , w, max_grid", TEST_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_for_loop_with_index(device, h, w, max_grid, dtype, iterations):
    """Test for loop with index variable and if statement."""

    def op_wrapper(input_tensor):
        return for_loop_with_index(input_tensor, iterations=iterations)

    run_op_test(
        device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1
    )


@pytest.mark.parametrize("h , w, max_grid", TEST_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("outer, inner", [(2, 2), (2, 3), (3, 2)])
def test_nested_for_loops(device, h, w, max_grid, dtype, outer, inner):
    """Test nested for loops."""

    def op_wrapper(input_tensor):
        return nested_for_loops(input_tensor, outer=outer, inner=inner)

    run_op_test(
        device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1
    )


@pytest.mark.parametrize("h , w, max_grid", TEST_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("iterations", [2, 3, 4])
def test_for_loop_with_if(device, h, w, max_grid, dtype, iterations):
    """Test for loop with if statement inside."""

    def op_wrapper(input_tensor):
        return for_loop_with_if(input_tensor, iterations=iterations)

    run_op_test(
        device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1
    )


@pytest.mark.parametrize("h , w, max_grid", TEST_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("iterations", [3, 4, 5])
def test_if_inside_for_multiple_branches(device, h, w, max_grid, dtype, iterations):
    """Test for loop with multiple if branches inside."""

    def op_wrapper(input_tensor):
        return if_inside_for_multiple_branches(input_tensor, iterations=iterations)

    run_op_test(
        device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1
    )
