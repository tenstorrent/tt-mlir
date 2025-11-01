# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn_jit.api import jit

from utils import create_sharded_tile_tensor, create_dram_tensor


def abs(input_tensor):
    return ttnn.abs(input_tensor)


def test_program_cache(device):
    """
    Test program caching behavior for ttnn-jit.

    Cache should hit when:
    - Same operation, same parameters, only data differs

    Cache should miss when:
    - Different JIT parameters (e.g., max_grid)
    - Different tensor properties (memory config, data type, shape)
    """
    h1, w1 = 256, 256
    h2, w2 = 512, 512

    # Create operations with different JIT parameters
    op_single_core = jit(max_grid=(0, 0))(abs)
    op_full_grid = jit(max_grid=(7, 7))(abs)

    assert op_single_core.num_entries == 0, "No entries should be in the cache"
    assert op_full_grid.num_entries == 0, "No entries should be in the cache"

    tensor_single_h1_w1_bf16_0 = create_sharded_tile_tensor(
        device, h1, w1, (0, 0), torch.bfloat16
    )
    tensor_single_h1_w1_bf16_1 = create_sharded_tile_tensor(
        device, h1, w1, (0, 0), torch.bfloat16
    )
    tensor_single_h1_w1_fp32 = create_sharded_tile_tensor(
        device, h1, w1, (0, 0), torch.float32
    )
    tensor_single_h2_w2_bf16 = create_sharded_tile_tensor(
        device, h2, w2, (0, 0), torch.bfloat16
    )

    tensor_full_h1_w1_bf16_0 = create_sharded_tile_tensor(
        device, h1, w1, (7, 7), torch.bfloat16
    )
    tensor_full_h1_w1_bf16_1 = create_sharded_tile_tensor(
        device, h1, w1, (7, 7), torch.bfloat16
    )

    tensor_dram_h1_w1_bf16_0 = create_dram_tensor(device, h1, w1, torch.bfloat16)
    tensor_dram_h1_w1_bf16_1 = create_dram_tensor(device, h1, w1, torch.bfloat16)

    # Test 1: Cache miss - first compilation (op_single_core + L1 single + h1,w1 + bf16)
    output = op_single_core(tensor_single_h1_w1_bf16_0)
    assert op_single_core.num_entries == 1, "First call should be a cache miss"
    # ttnn.deallocate(tensor_single_h1_w1_bf16_0)
    ttnn.deallocate(output)

    # Test 2: Cache HIT - same tensor should be a cache hit
    output = op_single_core(tensor_single_h1_w1_bf16_0)
    assert op_single_core.num_entries == 1, "Same tensor should be a cache hit"
    ttnn.deallocate(tensor_single_h1_w1_bf16_0)
    ttnn.deallocate(output)

    # Test 3: Cache HIT - same op, same tensor properties, different data
    output = op_single_core(tensor_single_h1_w1_bf16_1)
    assert (
        op_single_core.num_entries == 1
    ), "Same op with same tensor metadata but different data should be a cache hit"
    ttnn.deallocate(tensor_single_h1_w1_bf16_1)
    ttnn.deallocate(output)

    # Test 4: Cache miss - completely different op with it's own cache
    output = op_full_grid(tensor_full_h1_w1_bf16_0)
    assert op_full_grid.num_entries == 1, "New op, should be cache miss"
    ttnn.deallocate(tensor_full_h1_w1_bf16_0)
    ttnn.deallocate(output)

    # Test 5: Cache HIT - op_full_grid with same properties, different data
    output = op_full_grid(tensor_full_h1_w1_bf16_1)
    assert (
        op_full_grid.num_entries == 1
    ), "Same op with same tensor metadata but different data should be a cache hit"
    ttnn.deallocate(tensor_full_h1_w1_bf16_1)
    ttnn.deallocate(output)

    # Test 6: Cache miss - different dtype (fp32)
    output = op_single_core(tensor_single_h1_w1_fp32)
    assert op_single_core.num_entries == 2, "Different dtype should be cache miss"
    ttnn.deallocate(tensor_single_h1_w1_fp32)
    ttnn.deallocate(output)

    # Test 7: Cache miss - different shape (h2, w2)
    output = op_single_core(tensor_single_h2_w2_bf16)
    assert (
        op_single_core.num_entries == 3
    ), "Different tensor shape should be cache miss"
    ttnn.deallocate(tensor_single_h2_w2_bf16)
    ttnn.deallocate(output)

    # Test 8: Cache miss - different memory config (DRAM))
    output = op_single_core(tensor_dram_h1_w1_bf16_0)
    assert (
        op_single_core.num_entries == 4
    ), "Different memory config (L1 vs DRAM) should be cache miss"
    ttnn.deallocate(tensor_dram_h1_w1_bf16_0)
    ttnn.deallocate(output)

    # Test 9: Cache HIT - DRAM tensor with same properties, different data
    output = op_single_core(tensor_dram_h1_w1_bf16_1)
    assert (
        op_single_core.num_entries == 4
    ), "DRAM tensor with same properties should be a cache hit"
    ttnn.deallocate(tensor_dram_h1_w1_bf16_1)
    ttnn.deallocate(output)
