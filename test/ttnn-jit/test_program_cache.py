# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import random

from ttnn_jit.api import jit

from utils import create_sharded_tile_tensor, create_dram_tensor, all_close_check


def abs(input_tensor):
    return ttnn.abs(input_tensor)


def exp(input_tensor):
    return ttnn.exp(input_tensor)


def cos(input_tensor):
    return ttnn.cos(input_tensor)


def test_jit_cache(device):
    """
    Test JitCache

    JitCache should hit when:
    - Same operation, same parameters, only data differs

    Cache should miss when:
    - Different JIT parameters (e.g., max_grid)
    - Different tensor properties (memory config, data type, shape)
    """
    h1, w1 = 256, 256
    h2, w2 = 320, 320

    # Create operations with different JIT parameters
    op_single_core = jit(max_grid=(0, 0), enable_cache=True)(abs)
    op_full_grid = jit(max_grid=(7, 7), enable_cache=True)(abs)

    assert op_single_core.num_entries == 0, "No entries should be in the cache"
    assert op_full_grid.num_entries == 0, "No entries should be in the cache"

    # Test 1: Cache miss - first compilation (op_single_core + L1 single + h1,w1 + bf16)
    tensor_single_h1_w1_bf16_0 = create_sharded_tile_tensor(
        device, h1, w1, (0, 0), torch.bfloat16
    )
    output = op_single_core(tensor_single_h1_w1_bf16_0)
    assert op_single_core.num_entries == 1, "First call should be a cache miss"
    golden = ttnn.abs(tensor_single_h1_w1_bf16_0)
    assert all_close_check(output, golden, debug=False)
    ttnn.deallocate(golden)
    ttnn.deallocate(output)

    # Test 2: Cache HIT - same tensor should be a cache hit
    output = op_single_core(tensor_single_h1_w1_bf16_0)
    assert op_single_core.num_entries == 1, "Same tensor should be a cache hit"
    golden = ttnn.abs(tensor_single_h1_w1_bf16_0)
    assert all_close_check(output, golden, debug=False)
    ttnn.deallocate(tensor_single_h1_w1_bf16_0)
    ttnn.deallocate(golden)
    ttnn.deallocate(output)

    # Test 3: Cache HIT - same op, same tensor properties, different data
    tensor_single_h1_w1_bf16_1 = create_sharded_tile_tensor(
        device, h1, w1, (0, 0), torch.bfloat16
    )
    output = op_single_core(tensor_single_h1_w1_bf16_1)
    assert (
        op_single_core.num_entries == 1
    ), "Same op with same tensor metadata but different data should be a cache hit"
    golden = ttnn.abs(tensor_single_h1_w1_bf16_1)
    assert all_close_check(output, golden, debug=False)
    ttnn.deallocate(tensor_single_h1_w1_bf16_1)
    ttnn.deallocate(golden)
    ttnn.deallocate(output)

    # Test 4: Cache miss - completely different op with it's own cache
    tensor_full_h1_w1_bf16_0 = create_sharded_tile_tensor(
        device, h1, w1, (7, 7), torch.bfloat16
    )
    output = op_full_grid(tensor_full_h1_w1_bf16_0)
    assert op_full_grid.num_entries == 1, "New op, should be cache miss"
    golden = ttnn.abs(tensor_full_h1_w1_bf16_0)
    assert all_close_check(output, golden, debug=False)
    ttnn.deallocate(tensor_full_h1_w1_bf16_0)
    ttnn.deallocate(golden)
    ttnn.deallocate(output)

    # Test 5: Cache HIT - op_full_grid with same properties, different data
    tensor_full_h1_w1_bf16_1 = create_sharded_tile_tensor(
        device, h1, w1, (7, 7), torch.bfloat16
    )
    output = op_full_grid(tensor_full_h1_w1_bf16_1)
    assert (
        op_full_grid.num_entries == 1
    ), "Same op with same tensor metadata but different data should be a cache hit"
    golden = ttnn.abs(tensor_full_h1_w1_bf16_1)
    assert all_close_check(output, golden, debug=False)
    ttnn.deallocate(tensor_full_h1_w1_bf16_1)
    ttnn.deallocate(golden)
    ttnn.deallocate(output)

    # Test 6: Cache miss - different dtype (fp32)
    tensor_single_h1_w1_fp32 = create_sharded_tile_tensor(
        device, h1, w1, (0, 0), torch.float32
    )
    output = op_single_core(tensor_single_h1_w1_fp32)
    assert op_single_core.num_entries == 2, "Different dtype should be cache miss"
    golden = ttnn.abs(tensor_single_h1_w1_fp32)
    assert all_close_check(output, golden, debug=False)
    ttnn.deallocate(tensor_single_h1_w1_fp32)
    ttnn.deallocate(golden)
    ttnn.deallocate(output)

    # Test 7: Cache miss - different shape (h2, w2)
    tensor_single_h2_w2_bf16 = create_sharded_tile_tensor(
        device, h2, w2, (0, 0), torch.bfloat16
    )
    output = op_single_core(tensor_single_h2_w2_bf16)
    assert (
        op_single_core.num_entries == 3
    ), "Different tensor shape should be cache miss"
    golden = ttnn.abs(tensor_single_h2_w2_bf16)
    assert all_close_check(output, golden, debug=False)
    ttnn.deallocate(tensor_single_h2_w2_bf16)
    ttnn.deallocate(golden)
    ttnn.deallocate(output)

    # Test 8: Cache miss - different memory config (DRAM))
    tensor_dram_h1_w1_bf16_0 = create_dram_tensor(device, h1, w1, torch.bfloat16)
    output = op_single_core(tensor_dram_h1_w1_bf16_0)
    assert (
        op_single_core.num_entries == 4
    ), "Different memory config (L1 vs DRAM) should be cache miss"
    golden = ttnn.abs(tensor_dram_h1_w1_bf16_0)
    assert all_close_check(output, golden, debug=False)
    ttnn.deallocate(tensor_dram_h1_w1_bf16_0)
    ttnn.deallocate(golden)
    ttnn.deallocate(output)

    # Test 9: Cache HIT - DRAM tensor with same properties, different data
    tensor_dram_h1_w1_bf16_1 = create_dram_tensor(device, h1, w1, torch.bfloat16)
    output = op_single_core(tensor_dram_h1_w1_bf16_1)
    assert (
        op_single_core.num_entries == 4
    ), "DRAM tensor with same properties should be a cache hit"
    golden = ttnn.abs(tensor_dram_h1_w1_bf16_1)
    assert all_close_check(output, golden, debug=False)
    ttnn.deallocate(tensor_dram_h1_w1_bf16_1)
    ttnn.deallocate(golden)
    ttnn.deallocate(output)


def test_program_cache_hits(device):
    """
    Testing ProgramDescCache for correct overrides when cache hits. On cache hit, should override rt, ct, crt args.
    Note: as of now, ProgramDescCache will always miss until override_runtime_arguments is implemented in ttnn.generic op.

    This test will randomize tensor buffer addresses, and run jit'ed ops for the same shapes. The JitCache will always hit,
    but the ProgramDescCache will miss (due to reasons above). Run it many times to ensure it's not a fluke.
    """
    random.seed(0)
    tensor_full_h1_w1_bf16_0 = create_sharded_tile_tensor(
        device, 512, 512, (7, 7), torch.bfloat16
    )
    jit_abs = jit(enable_cache=True)(abs)
    jit_exp = jit(enable_cache=True)(exp)
    jit_cos = jit(enable_cache=True)(cos)
    golden_map = {
        jit_abs: ttnn.abs,
        jit_exp: ttnn.exp,
        jit_cos: ttnn.cos,
    }

    output = jit_abs(tensor_full_h1_w1_bf16_0)
    output = jit_exp(tensor_full_h1_w1_bf16_0)
    output = jit_cos(tensor_full_h1_w1_bf16_0)
    ttnn.deallocate(tensor_full_h1_w1_bf16_0)
    assert jit_abs.num_entries == 1, "New op is cache miss"
    assert jit_exp.num_entries == 1, "New op is cache miss"
    assert jit_cos.num_entries == 1, "New op is cache miss"

    for i in range(500):
        # Allocate random number of random dummy tensors to randomize tensor buffer addresses.
        dummy_shapes = [(256, 256), (512, 512)]
        num_dummies = random.randint(1, 10)
        dummies = []

        for _ in range(num_dummies):
            h, w = random.choice(dummy_shapes)
            dummy = create_sharded_tile_tensor(device, h, w, (7, 7), torch.bfloat16)
            dummies.append(dummy)

        input_tensor = create_sharded_tile_tensor(
            device, 512, 512, (7, 7), torch.bfloat16
        )
        for dummy in dummies:
            ttnn.deallocate(dummy)

        jit_op, golden_op = random.choice(list(golden_map.items()))
        output = jit_op(input_tensor)
        golden = golden_op(input_tensor)
        assert all_close_check(output, golden, debug=False)
        assert (
            jit_op.num_entries == 1
        ), "Same tensor metadata but different data is JitCache hit"
        ttnn.deallocate(input_tensor)
        ttnn.deallocate(golden)
        ttnn.deallocate(output)
