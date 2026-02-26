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
    get_expected_block_sharded_memory_config,
    create_dram_tensor,
    create_sharded_tile_tensor,
    run_op_test,
)

# TTNN JIT test for gpt-oss subgraph, see https://github.com/tenstorrent/tt-mlir/issues/7107.

r"""
%416 (input)        %25 (gate_up_proj weight)
             \                /
              \              /
         +------------------+
         |  %417 = matmul   |  4x32x2880 x 4x2880x5760 -> 4x32x5760
         +------------------+
                  |
                  |  + %17 (bias)
                  v
         +------------------+
         |  %418 = add      |  4x32x5760
         +------------------+
              /          \
    odd indices           even indices
    (step=2,start=1)      (step=2,start=0)
            /                  \
   +-----------------+   +-----------------+
   | %419 = slice    |   | %422 = slice    |
   +-----------------+   +-----------------+
           |                     |
-------------subgraph starts here----------------
           v                     v
   +-----------------+   +-----------------+
   | %420 = clamp    |   | %423 = clamp    |
   |  [-7, 7]        |   |  [-inf, 7]      |
   +-----------------+   +-----------------+
           |                  /     \
           |  + %30 (scalar) /       \  * %2 (scale)
           v                /         \
   +-----------------+     |    +-----------------+
   | %421 = add      |     |    | %424 = multiply |
   +-----------------+     |    +-----------------+
           |               |            |
           |               |            v
           |               |    +-----------------+
           |               |    | %425 = sigmoid  |
           |               |    +-----------------+
           |               |            |
           |                \          /
           |                 \        /
           |            +-----------------+
           |            | %426 = multiply |  x * sigmoid(x) = SiLU
           |            +-----------------+
            \                /
             \              /
         +------------------+
         | %427 = multiply  |  gate * SiLU(up)
         +------------------+
                  |
                  v
              (output)

---------------subgraph ends here-----------------
"""


@ttnn_jit.jit(debug=True, compile_only=False)
def gptoss_subgraph(left_input, right_input):
    left_clamp = ttnn.clamp(left_input, min=-7.0, max=7.0)
    right_clamp = ttnn.clamp(right_input, min=float("-inf"), max=7)

    left_out = ttnn.add(left_clamp, 1.0)

    right_temp = ttnn.multiply(right_clamp, 2.0)
    right_temp = ttnn.sigmoid(right_temp)
    right_out = ttnn.multiply(right_clamp, right_temp)

    out = ttnn.multiply(left_out, right_out)
    return out


def test_oss_subgraph(device):

    left_input = create_dram_tensor(device, (4, 32, 3072), torch.float16)
    right_input = create_dram_tensor(device, (4, 32, 3072), torch.float16)
    print("left input:", left_input)
    print("right input:", right_input)
    output = gptoss_subgraph(left_input, right_input)
    print("Output tensor:", output)

    assert output.shape == (4, 32, 3072)


@ttnn_jit.jit(debug=True)
def rank_3_unary(input):
    return ttnn.exp(input)


# shape (4, 16, 3072) failing:
""" FAILED test/ttnn-jit/test_gptoss_subgraph.py::test_rank_3 - RuntimeError: TT_FATAL @ /localdev/brapanan/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/tensor/spec/tensor_spec.cpp:69: num_shards_along_height <= shard_grid.y
info:
Number of shards along height 4 must not exceed number of rows 2 for row major orientation! """


@pytest.mark.parametrize(
    "shape", [(4, 16, 3072), (2, 32, 3072)], ids=["(4,16,3072)", "(2,32,3072)"]
)
def test_rank_3(device, shape):
    input = create_dram_tensor(device, shape, torch.float16)
    output = rank_3_unary(input)
    assert output.shape == shape


@ttnn_jit.jit(debug=True)
def fusion_example(input_a, input_b):
    out = ttnn.add(input_a, input_b)
    out = ttnn.multiply(out, 2.0)
    return out


def test_fusion_example(device):
    input_a = create_dram_tensor(device, (4, 32, 3072), torch.float16)
    input_b = create_dram_tensor(device, (4, 32, 3072), torch.float16)
    output = fusion_example(input_a, input_b)
    assert output.shape == (4, 32, 3072)
