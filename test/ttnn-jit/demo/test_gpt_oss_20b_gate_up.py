# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import create_dram_tensor, all_close_check

r"""
GPT-OSS gate up proj activation subgraph:

%416 (input)        %25 (gate_up_proj weight)
             \                /
              \              /
         +------------------+
         |  %417 = matmul   |
         +------------------+
                  |
                  |  + %17 (bias)
                  v
         +------------------+
         |  %418 = add      |
         +------------------+
              /          \
    odd indices           even indices
    (step=2,start=1)      (step=2,start=0)
            /                  \
   +-----------------+   +-----------------+
   | %419 = slice    |   | %422 = slice    |
   +-----------------+   +-----------------+
           |                     |
------------JIT'ed subgraph starts here-------------
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

------------JIT'ed subgraph ends here-------------
"""


@ttnn_jit.jit(debug=True, extra_pipeline_options="enable-elementwise-fusion=true")
def gptoss_gateup_subgraph(
    left_input,
    right_input,
    left_min,
    left_max,
    right_min,
    right_max,
    bias_value,
    alpha_value,
):

    left_clamp = ttnn.clamp(left_input, min=left_min, max=left_max)
    left_out = ttnn.add(left_clamp, bias_value)

    right_clamp = ttnn.clamp(right_input, min=right_min, max=right_max)
    right_temp = ttnn.multiply(right_clamp, alpha_value)
    right_temp = ttnn.sigmoid(right_temp)
    right_out = ttnn.multiply(right_clamp, right_temp)

    out = ttnn.multiply(left_out, right_out)
    return out


def gptoss_gateup_subgraph_not_jit(
    left_input,
    right_input,
    left_min,
    left_max,
    right_min,
    right_max,
    bias_value,
    alpha_value,
):
    left_clamp = ttnn.clamp(left_input, min=left_min, max=left_max)
    left_out = ttnn.add(left_clamp, bias_value)

    right_clamp = ttnn.clamp(right_input, min=right_min, max=right_max)
    right_temp = ttnn.multiply(right_clamp, alpha_value)
    right_temp = ttnn.sigmoid(right_temp)
    right_out = ttnn.multiply(right_clamp, right_temp)

    out = ttnn.multiply(left_out, right_out)
    return out


def test_oss_gateup_subgraph(device):
    shape = (2, 32, 3072)
    dtype = torch.bfloat16
    limit = 7.0
    bias = 1.0
    alpha = 1.702

    left_input = create_dram_tensor(device, shape, dtype)
    right_input = create_dram_tensor(device, shape, dtype)

    left_min = create_dram_tensor(
        device, shape, dtype, input_transform=lambda t: t.fill_(-limit)
    )
    left_max = create_dram_tensor(
        device, shape, dtype, input_transform=lambda t: t.fill_(limit)
    )
    right_min = create_dram_tensor(
        device, shape, dtype, input_transform=lambda t: t.fill_(float("-inf"))
    )
    right_max = create_dram_tensor(
        device, shape, dtype, input_transform=lambda t: t.fill_(limit)
    )
    bias_value = create_dram_tensor(
        device, shape, dtype, input_transform=lambda t: t.fill_(bias)
    )
    alpha_value = create_dram_tensor(
        device, shape, dtype, input_transform=lambda t: t.fill_(alpha)
    )

    output = gptoss_gateup_subgraph(
        left_input,
        right_input,
        left_min,
        left_max,
        right_min,
        right_max,
        bias_value,
        alpha_value,
    )
    output_not_jit = gptoss_gateup_subgraph_not_jit(
        left_input,
        right_input,
        left_min,
        left_max,
        right_min,
        right_max,
        bias_value,
        alpha_value,
    )
    assert all_close_check(output, output_not_jit)
