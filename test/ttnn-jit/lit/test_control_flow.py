# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

from utils import create_sharded_tile_tensor


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=True)
def if_else_branch(input_tensor, use_exp=True):
    """Test basic if/else branching."""
    if use_exp:
        output = ttnn.exp(input_tensor)
    else:
        output = ttnn.log(input_tensor)
    return output


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=True)
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


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=True)
def if_with_ops_before_after(input_tensor, apply_exp=True):
    """Test if statement with operations before and after."""
    temp = ttnn.abs(input_tensor)

    if apply_exp:
        temp = ttnn.exp(temp)
    else:
        temp = ttnn.sin(temp)

    output = ttnn.multiply(temp, 0.5)
    return output


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=True)
def multiple_sequential_ifs(input_tensor, apply_exp=False, apply_cos=False):
    """Test multiple sequential if statements."""
    output = input_tensor

    if apply_exp:
        output = ttnn.exp(output)

    if apply_cos:
        output = ttnn.cos(output)

    return output


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=True)
def for_loop_simple(input_tensor, iterations=3):
    """Test simple for loop."""
    output = input_tensor
    for _ in range(iterations):
        output = ttnn.add(output, input_tensor)
    return output


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=True)
def for_loop_with_index(input_tensor, iterations=3):
    """Test for loop with index variable."""
    output = input_tensor
    for i in range(iterations):
        if i == 0:
            output = ttnn.multiply(output, 2.0)
        else:
            output = ttnn.add(output, input_tensor)
    return output


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=True)
def nested_for_loops(input_tensor, outer=2, inner=2):
    """Test nested for loops."""
    output = input_tensor
    for _ in range(outer):
        for _ in range(inner):
            output = ttnn.multiply(output, 1.1)
    return output


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=True)
def for_loop_with_if(input_tensor, iterations=4):
    """Test for loop with if statement inside."""
    output = input_tensor
    for i in range(iterations):
        if i % 2 == 0:
            output = ttnn.add(output, input_tensor)
        else:
            output = ttnn.multiply(output, 0.9)
    return output


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=True)
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


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    input_tensor_a_l1 = create_sharded_tile_tensor(
        device, (32, 32), (0, 0), torch.bfloat16
    )

    _ = if_else_branch(input_tensor_a_l1)
    _ = if_else_branch(input_tensor_a_l1, use_exp=False)

    _ = nested_if(input_tensor_a_l1)
    _ = nested_if(input_tensor_a_l1, mode=1)

    _ = if_with_ops_before_after(input_tensor_a_l1)
    _ = if_with_ops_before_after(input_tensor_a_l1, apply_exp=False)

    _ = multiple_sequential_ifs(input_tensor_a_l1)
    _ = multiple_sequential_ifs(input_tensor_a_l1, apply_exp=True, apply_cos=True)

    _ = for_loop_simple(input_tensor_a_l1)
    _ = for_loop_simple(input_tensor_a_l1, iterations=5)

    _ = for_loop_with_index(input_tensor_a_l1)
    _ = for_loop_with_index(input_tensor_a_l1, iterations=5)

    _ = nested_for_loops(input_tensor_a_l1)
    _ = nested_for_loops(input_tensor_a_l1, outer=3, inner=3)

    _ = for_loop_with_if(input_tensor_a_l1)
    _ = for_loop_with_if(input_tensor_a_l1, iterations=5)

    _ = if_inside_for_multiple_branches(input_tensor_a_l1)
    _ = if_inside_for_multiple_branches(input_tensor_a_l1, iterations=5)

    ttnn.close_device(device)
