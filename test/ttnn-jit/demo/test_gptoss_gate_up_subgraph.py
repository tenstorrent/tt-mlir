# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import (
    create_dram_tensor,
    create_sharded_tile_tensor,
    all_close_check,
    pcc_check,
)

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


@ttnn_jit.jit(debug=True)
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

    print("output", output)
    print("output_not_jit", output_not_jit)

    pcc_passed, pcc = pcc_check(output, output_not_jit)

    out_torch = output.cpu().to_torch()
    ref_torch = output_not_jit.cpu().to_torch()

    TILE_H, TILE_W = 32, 32
    flat_h = shape[0] * shape[1]
    flat_w = shape[2]
    out_2d = out_torch.reshape(flat_h, flat_w)
    ref_2d = ref_torch.reshape(flat_h, flat_w)

    tile_rows = flat_h // TILE_H
    tile_cols = flat_w // TILE_W
    all_close = True
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            out_tile = out_2d[
                tr * TILE_H : (tr + 1) * TILE_H, tc * TILE_W : (tc + 1) * TILE_W
            ]
            ref_tile = ref_2d[
                tr * TILE_H : (tr + 1) * TILE_H, tc * TILE_W : (tc + 1) * TILE_W
            ]
            if not torch.allclose(out_tile, ref_tile, atol=1e-1, rtol=1e-1):
                max_diff = (out_tile - ref_tile).abs().max().item()
                print(f"tile [{tr},{tc}] FAILED  max_diff={max_diff:.6f}")
                all_close = False

    print(f"PCC check passed: {pcc_passed} with pcc={pcc}")
    assert all_close, "Some tiles failed"


""" def test_fusion(device):
    shape = (256, 256)
    dtype = torch.bfloat16

    def graph_fn(x, y):
        left_a = ttnn.exp(x)

        #right_a = ttnn.exp(y)
        #right_b = ttnn.abs(right_a)
        #right_c = ttnn.add(y, right_b)

        #out = ttnn.multiply(left_a, right_a)

        simple_out = ttnn.multiply(left_a, y)
        return simple_out

    input_a = create_sharded_tile_tensor(device, shape, (7, 7), dtype, shard_strategy=ttnn.ShardStrategy.BLOCK)
    input_b = create_sharded_tile_tensor(device, shape, (7, 7), dtype, shard_strategy=ttnn.ShardStrategy.BLOCK)

    compiled_op = ttnn_jit.jit(debug=True)(graph_fn)
    output = compiled_op(input_a, input_b)

    output_not_jit = graph_fn(input_a, input_b)

    print("output", output)
    print("output_not_jit", output_not_jit)
    assert all_close_check(output, output_not_jit) """


def test_diamond_fusion(device):
    shape = (512, 512)
    dtype = torch.bfloat16

    def graph_fn(x):
        b = ttnn.exp(x)
        d = ttnn.add(b, b)
        return d

    input = create_dram_tensor(device, shape, dtype)

    compiled_op = ttnn_jit.jit(debug=True)(graph_fn)
    output = compiled_op(input)
    output_not_jit = graph_fn(input)

    print("output", output)
    print("output_not_jit", output_not_jit)

    passed, pcc = pcc_check(output, output_not_jit)
    assert passed, f"PCC check failed: {pcc} < 0.99"

    out_torch = output.cpu().to_torch()
    ref_torch = output_not_jit.cpu().to_torch()

    TILE_H, TILE_W = 32, 32
    tile_rows = shape[0] // TILE_H
    tile_cols = shape[1] // TILE_W
    all_close = True
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            out_tile = out_torch[
                tr * TILE_H : (tr + 1) * TILE_H, tc * TILE_W : (tc + 1) * TILE_W
            ]
            ref_tile = ref_torch[
                tr * TILE_H : (tr + 1) * TILE_H, tc * TILE_W : (tc + 1) * TILE_W
            ]
            if not torch.allclose(out_tile, ref_tile, atol=1e-1, rtol=1e-1):
                max_diff = (out_tile - ref_tile).abs().max().item()
                print(f"tile [{tr},{tc}] FAILED  max_diff={max_diff:.6f}")
                all_close = False

    assert all_close, "Some tiles failed"
