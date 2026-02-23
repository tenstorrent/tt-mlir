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
    pcc_check,
)


def binary_tree_add(in0, in1, in2, in3):
    """Binary tree of adds: add(add(in0, in1), add(in2, in3))"""
    left = ttnn.add(in0, in1)
    right = ttnn.add(in2, in3)
    return ttnn.add(left, right)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024)],
    ids=["1024x1024"],
)
@pytest.mark.parametrize(
    "dtype, ttnn_dtype",
    [
        (torch.bfloat16, None),
    ],
    ids=["bf16"],
)
def test_binary_tree_dram(device, shape, dtype, ttnn_dtype):
    """Test a binary tree of adds with all inputs in DRAM."""
    in0 = create_dram_tensor(device, shape, dtype, ttnn_dtype=ttnn_dtype)
    in1 = create_dram_tensor(device, shape, dtype, ttnn_dtype=ttnn_dtype)
    in2 = create_dram_tensor(device, shape, dtype, ttnn_dtype=ttnn_dtype)
    in3 = create_dram_tensor(device, shape, dtype, ttnn_dtype=ttnn_dtype)

    compiled_op = ttnn_jit.jit(
        debug=True,
        compile_only=False,
    )(binary_tree_add)

    output = compiled_op(in0, in1, in2, in3)
    golden_output = binary_tree_add(in0, in1, in2, in3)

    assert all_close_check(output, golden_output)
    passed, pcc = pcc_check(output, golden_output)
    assert passed, f"PCC check failed: {pcc} < 0.99"
