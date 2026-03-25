# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Perf benchmark for the GPT-OSS gate_up projection activation subgraph.
# Compares JIT-compiled vs plain TTNN execution to measure JIT overhead/speedup
# on a realistic multi-op fusion subgraph.

import ttnn
import ttnn_jit
import torch

import pytest

from utils import create_dram_tensor


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


@pytest.mark.parametrize(
    "op",
    [gptoss_gateup_subgraph],
    ids=["gptoss_gateup_subgraph"],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [ttnn.DataType.BFLOAT16],
    ids=["bf16"],
)
@pytest.mark.parametrize(
    "memory_config_id",
    ["dram_interleaved"],
)
@pytest.mark.parametrize(
    "jit_enabled",
    [True, False],
)
def test_subgraph_perf(
    op,
    ttnn_dtype,
    memory_config_id,
    jit_enabled,
    perf_device,
):
    device = perf_device
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

    if jit_enabled:
        function_to_test = ttnn_jit.jit(enable_cache=True)(op)
    else:
        function_to_test = op

    output = function_to_test(
        left_input,
        right_input,
        left_min,
        left_max,
        right_min,
        right_max,
        bias_value,
        alpha_value,
    )
    assert output is not None
