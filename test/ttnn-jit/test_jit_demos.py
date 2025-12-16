# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import all_close_check


def test_cosh_demo(device):
    @ttnn_jit.jit(debug=True)
    def cosh(input_tensor):
        e_pos_x = ttnn.exp(input_tensor)
        e_neg_x = ttnn.exp(ttnn.neg(input_tensor))
        nr_term = ttnn.add(e_pos_x, e_neg_x)
        output = ttnn.multiply(nr_term, 0.5)
        return output

    def model(input_0, input_1):
        x = cosh(input_0)
        y = ttnn.exp(input_1)
        return ttnn.matmul(x, y)

    def cosh_not_jit(input_tensor):
        e_pos_x = ttnn.exp(input_tensor)
        e_neg_x = ttnn.exp(ttnn.neg(input_tensor))
        nr_term = ttnn.add(e_pos_x, e_neg_x)
        output = ttnn.multiply(nr_term, 0.5)
        return output

    def model_not_jit(input_0, input_1):
        x = cosh_not_jit(input_0)
        y = ttnn.exp(input_1)
        return ttnn.matmul(x, y)

    input_0_torch = torch.randn(512, 512, dtype=torch.bfloat16)
    input_1_torch = torch.randn(512, 512, dtype=torch.bfloat16)
    memory_config = ttnn.create_sharded_memory_config(
        shape=(512, 512),
        core_grid=ttnn.CoreGrid(x=8, y=8),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )
    input_0_ttnn = ttnn.from_torch(
        input_0_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    input_1_ttnn = ttnn.from_torch(
        input_1_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = model(input_0_ttnn, input_1_ttnn)
    output_tensor_not_jit = model_not_jit(input_0_ttnn, input_1_ttnn)
    assert all_close_check(output_tensor, output_tensor_not_jit)
