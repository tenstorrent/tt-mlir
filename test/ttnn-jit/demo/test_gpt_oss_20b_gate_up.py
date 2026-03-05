# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import create_dram_tensor, all_close_check


@ttnn_jit.jit(debug=True)
def gptoss_gateup_subgraph(
    left_input,
    right_input,
    left_min,
    left_max,
    right_min,
    right_max,
    add_value,
    mul_value,
):

    left_clamp = ttnn.clamp(left_input, min=left_min, max=left_max)
    left_out = ttnn.add(left_clamp, add_value)

    right_clamp = ttnn.clamp(right_input, min=right_min, max=right_max)
    right_temp = ttnn.multiply(right_clamp, mul_value)
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
    add_value,
    mul_value,
):
    left_clamp = ttnn.clamp(left_input, min=left_min, max=left_max)
    left_out = ttnn.add(left_clamp, add_value)

    right_clamp = ttnn.clamp(right_input, min=right_min, max=right_max)
    right_temp = ttnn.multiply(right_clamp, mul_value)
    right_temp = ttnn.sigmoid(right_temp)
    right_out = ttnn.multiply(right_clamp, right_temp)

    out = ttnn.multiply(left_out, right_out)
    return out


def test_oss_gateup_subgraph(device):
    shape = (2, 32, 3072)
    dtype = torch.float32

    left_input = create_dram_tensor(device, shape, dtype)
    right_input = create_dram_tensor(device, shape, dtype)

    left_min = create_dram_tensor(
        device, shape, dtype, input_transform=lambda t: t.fill_(-7.0)
    )
    left_max = create_dram_tensor(
        device, shape, dtype, input_transform=lambda t: t.fill_(7.0)
    )
    right_min = create_dram_tensor(
        device, shape, dtype, input_transform=lambda t: t.fill_(float("-inf"))
    )
    right_max = create_dram_tensor(
        device, shape, dtype, input_transform=lambda t: t.fill_(7.0)
    )
    add_value = create_dram_tensor(
        device, shape, dtype, input_transform=lambda t: t.fill_(1.0)
    )
    mul_value = create_dram_tensor(
        device, shape, dtype, input_transform=lambda t: t.fill_(2.0)
    )

    output = gptoss_gateup_subgraph(
        left_input,
        right_input,
        left_min,
        left_max,
        right_min,
        right_max,
        add_value,
        mul_value,
    )
    output_not_jit = gptoss_gateup_subgraph_not_jit(
        left_input,
        right_input,
        left_min,
        left_max,
        right_min,
        right_max,
        add_value,
        mul_value,
    )
    assert all_close_check(output, output_not_jit)
