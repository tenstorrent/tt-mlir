# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn_jit.api import jit, _cache

from utils import _create_sharded_tile_tensor, _create_dram_tensor


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
    op_single_core = jit(backend="ttnn", max_grid=(0, 0))(abs)
    op_full_grid = jit(backend="ttnn", max_grid=(7, 7))(abs)

    tensor_single_h1_w1_bf16_0 = _create_sharded_tile_tensor(
        device, h1, w1, (0, 0), torch.bfloat16
    )
    tensor_single_h1_w1_bf16_1 = _create_sharded_tile_tensor(
        device, h1, w1, (0, 0), torch.bfloat16
    )
    tensor_single_h1_w1_fp32 = _create_sharded_tile_tensor(
        device, h1, w1, (0, 0), torch.float32
    )
    tensor_single_h2_w2_bf16 = _create_sharded_tile_tensor(
        device, h2, w2, (0, 0), torch.bfloat16
    )

    tensor_full_h1_w1_bf16_0 = _create_sharded_tile_tensor(
        device, h1, w1, (7, 7), torch.bfloat16
    )
    tensor_full_h1_w1_bf16_1 = _create_sharded_tile_tensor(
        device, h1, w1, (7, 7), torch.bfloat16
    )

    tensor_dram_h1_w1_bf16_0 = _create_dram_tensor(device, h1, w1, torch.bfloat16)
    tensor_dram_h1_w1_bf16_1 = _create_dram_tensor(device, h1, w1, torch.bfloat16)

    # Test 1: Cache miss - first compilation (op_single_core + L1 single + h1,w1 + bf16)
    output = op_single_core(tensor_single_h1_w1_bf16_0)
    assert _cache.cache_hits() == 0, "First call should be a cache miss"
    ttnn.deallocate(tensor_single_h1_w1_bf16_0)
    ttnn.deallocate(output)

    # Test 2: Cache HIT - same op, same tensor properties, different data
    output = op_single_core(tensor_single_h1_w1_bf16_1)
    assert (
        _cache.cache_hits() == 1
    ), "Same op with same tensor properties but different data should be a cache hit"
    ttnn.deallocate(tensor_single_h1_w1_bf16_1)
    ttnn.deallocate(output)

    # Test 3: Cache miss - different max_grid (op_full_grid + L1 full + h1,w1 + bf16)
    output = op_full_grid(tensor_full_h1_w1_bf16_0)
    assert _cache.cache_hits() == 1, "Different max_grid should cause cache miss"
    ttnn.deallocate(tensor_full_h1_w1_bf16_0)
    ttnn.deallocate(output)

    # Test 4: Cache HIT - op_full_grid with same properties, different data
    output = op_full_grid(tensor_full_h1_w1_bf16_1)
    assert (
        _cache.cache_hits() == 2
    ), "op_full_grid with same tensor properties should be a cache hit"
    ttnn.deallocate(tensor_full_h1_w1_bf16_1)
    ttnn.deallocate(output)

    # Test 5: Cache miss - different dtype (op_single_core + L1 single + h1,w1 + fp32)
    output = op_single_core(tensor_single_h1_w1_fp32)
    assert _cache.cache_hits() == 2, "Different dtype should cause cache miss"
    ttnn.deallocate(tensor_single_h1_w1_fp32)
    ttnn.deallocate(output)

    # Test 6: Cache miss - different shape (op_single_core + L1 single + h2,w2 + bf16)
    output = op_single_core(tensor_single_h2_w2_bf16)
    assert _cache.cache_hits() == 2, "Different tensor shape should cause cache miss"
    ttnn.deallocate(tensor_single_h2_w2_bf16)
    ttnn.deallocate(output)

    # Test 7: Cache miss - different memory config (op_single_core + DRAM + h1,w1 + bf16)
    output = op_single_core(tensor_dram_h1_w1_bf16_0)
    assert (
        _cache.cache_hits() == 2
    ), "Different memory config (L1 vs DRAM) should cause cache miss"
    ttnn.deallocate(tensor_dram_h1_w1_bf16_0)
    ttnn.deallocate(output)

    # Test 8: Cache HIT - DRAM tensor with same properties, different data
    output = op_single_core(tensor_dram_h1_w1_bf16_1)
    assert (
        _cache.cache_hits() == 3
    ), "DRAM tensor with same properties should be a cache hit"
    ttnn.deallocate(tensor_dram_h1_w1_bf16_1)
    ttnn.deallocate(output)
