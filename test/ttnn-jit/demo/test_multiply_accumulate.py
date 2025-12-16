# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import all_close_check


def test_mul_add_demo(device):
    @ttnn_jit.jit(debug=True)
    def mul_add(input_tensor_a, input_tensor_b, input_tensor_c):
        matmul_result = ttnn.multiply(input_tensor_b, input_tensor_c)
        output = ttnn.add(matmul_result, input_tensor_a)
        return output

    def mul_add_not_jit(input_tensor_a, input_tensor_b, input_tensor_c):
        matmul_result = ttnn.multiply(input_tensor_b, input_tensor_c)
        output = ttnn.add(matmul_result, input_tensor_a)
        return output

    input_a_torch = torch.randn(512, 512, dtype=torch.bfloat16)
    input_b_torch = torch.randn(512, 512, dtype=torch.bfloat16)
    input_c_torch = torch.randn(512, 512, dtype=torch.bfloat16)

    memory_config = ttnn.create_sharded_memory_config(
        shape=(512, 512),
        core_grid=ttnn.CoreGrid(x=8, y=8),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )
    input_a_ttnn = ttnn.from_torch(
        input_a_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    input_b_ttnn = ttnn.from_torch(
        input_b_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    input_c_ttnn = ttnn.from_torch(
        input_c_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )

    output_tensor = mul_add(input_a_ttnn, input_b_ttnn, input_c_ttnn)
    output_tensor_not_jit = mul_add_not_jit(input_a_ttnn, input_b_ttnn, input_c_ttnn)
    assert all_close_check(output_tensor, output_tensor_not_jit)
