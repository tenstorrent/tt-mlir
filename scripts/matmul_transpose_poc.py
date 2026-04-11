#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PoC: Compare original vs transposed matmul approaches for precision.

Run individual tests for tracy profiling:
  pytest scripts/matmul_transpose_poc.py -k "original_interleaved_bf8" -s
  pytest scripts/matmul_transpose_poc.py -k "original_width_mcast_bf8" -s
  pytest scripts/matmul_transpose_poc.py -k "transposed_interleaved_bf8" -s
  pytest scripts/matmul_transpose_poc.py -k "transposed_height_sharded_bf8" -s
  pytest scripts/matmul_transpose_poc.py -k "transposed_l1_activation_bf8" -s

Run all:
  pytest scripts/matmul_transpose_poc.py -s
"""

import pytest
import torch
import ttnn


M, K, N = 32, 4096, 7168


def pcc(expected, actual):
    e = expected.flatten().float()
    a = actual.flatten().float()
    return torch.corrcoef(torch.stack([e, a]))[0, 1].item()


def max_abs_error(expected, actual):
    return (expected.float() - actual.float()).abs().max().item()


def mean_abs_error(expected, actual):
    return (expected.float() - actual.float()).abs().mean().item()


def report(name, result, gold):
    p = pcc(gold, result)
    mae = mean_abs_error(gold, result)
    max_ae = max_abs_error(gold, result)
    print(f"\n  {name}")
    print(f"  PCC={p:.8f}  MAE={mae:.6f}  MaxAE={max_ae:.4f}")
    return p


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def golden():
    torch.manual_seed(42)
    A = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    B = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    gold = A.float() @ B.float()
    B_T = B.transpose(-2, -1).contiguous()
    return A, B, B_T, gold


def make_compute_config(device, hifi4=False):
    if hifi4:
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


# =====================================================================
# ORIGINAL: A[32x4096] x B[4096x7168] = C[32x7168]
# =====================================================================


def test_original_interleaved_bf8(device, golden):
    """Original matmul, DRAM interleaved, bf16 activation x bf8 weight, HiFi2."""
    A, B, _, gold = golden
    dram = ttnn.DRAM_MEMORY_CONFIG
    a_tt = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)
    b_tt = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b, memory_config=dram)

    out = ttnn.matmul(a_tt, b_tt, dtype=ttnn.bfloat16, compute_kernel_config=make_compute_config(device))
    result = ttnn.to_torch(out)
    p = report("Original interleaved, bf16 x bf8, HiFi2", result, gold)
    assert p > 0.99


def test_original_interleaved_bf16(device, golden):
    """Original matmul, DRAM interleaved, bf16 x bf16, HiFi4."""
    A, B, _, gold = golden
    dram = ttnn.DRAM_MEMORY_CONFIG
    a_tt = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)
    b_tt = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)

    out = ttnn.matmul(a_tt, b_tt, dtype=ttnn.bfloat16, compute_kernel_config=make_compute_config(device, hifi4=True))
    result = ttnn.to_torch(out)
    p = report("Original interleaved, bf16 x bf16, HiFi4", result, gold)
    assert p > 0.99


def test_original_width_mcast_bf8(device, golden):
    """Original matmul with explicit 1D mcast width config (matches MLIR). bf16 x bf8."""
    A, B, _, gold = golden
    dram = ttnn.DRAM_MEMORY_CONFIG
    a_tt = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)
    b_tt = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b, memory_config=dram)

    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 7),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=4,
        out_block_h=1,
        out_block_w=4,
        per_core_M=1,
        per_core_N=4,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    out = ttnn.matmul(a_tt, b_tt, program_config=cfg,
                      dtype=ttnn.bfloat16, compute_kernel_config=make_compute_config(device))
    result = ttnn.to_torch(out)
    p = report("Original 1D mcast width, bf16 x bf8, HiFi2", result, gold)
    assert p > 0.99


def test_original_width_mcast_bf16(device, golden):
    """Original matmul with explicit 1D mcast width config. bf16 x bf16."""
    A, B, _, gold = golden
    dram = ttnn.DRAM_MEMORY_CONFIG
    a_tt = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)
    b_tt = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)

    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 7),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=4,
        out_block_h=1,
        out_block_w=4,
        per_core_M=1,
        per_core_N=4,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    out = ttnn.matmul(a_tt, b_tt, program_config=cfg,
                      dtype=ttnn.bfloat16, compute_kernel_config=make_compute_config(device, hifi4=True))
    result = ttnn.to_torch(out)
    p = report("Original 1D mcast width, bf16 x bf16, HiFi4", result, gold)
    assert p > 0.99


# =====================================================================
# TRANSPOSED: B^T[7168x4096] x ttnn.transpose(A)[4096x32] = C^T[7168x32]
#             then ttnn.transpose(C^T) -> C[32x7168]
#
# B^T from const_eval (pre-transposed weight).
# A^T via ttnn.transpose on device (runtime graph op).
# C via ttnn.transpose on device (runtime graph op).
# =====================================================================


def test_transposed_interleaved_bf8(device, golden):
    """Transposed matmul, DRAM interleaved, bf8 weight x bf16 activation, HiFi2."""
    A, _, B_T, gold = golden
    dram = ttnn.DRAM_MEMORY_CONFIG
    a_tt = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)
    bt_tt = ttnn.from_torch(B_T, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b, memory_config=dram)

    at_tt = ttnn.transpose(a_tt, -2, -1)
    out = ttnn.matmul(bt_tt, at_tt, dtype=ttnn.bfloat16, compute_kernel_config=make_compute_config(device))
    out = ttnn.transpose(out, -2, -1)

    result = ttnn.to_torch(out)
    p = report("Transposed interleaved, bf8 x bf16, HiFi2", result, gold)
    assert p > 0.99


def test_transposed_interleaved_bf16(device, golden):
    """Transposed matmul, DRAM interleaved, bf16 x bf16, HiFi4."""
    A, _, B_T, gold = golden
    dram = ttnn.DRAM_MEMORY_CONFIG
    a_tt = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)
    bt_tt = ttnn.from_torch(B_T, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)

    at_tt = ttnn.transpose(a_tt, -2, -1)
    out = ttnn.matmul(bt_tt, at_tt, dtype=ttnn.bfloat16,
                      compute_kernel_config=make_compute_config(device, hifi4=True))
    out = ttnn.transpose(out, -2, -1)

    result = ttnn.to_torch(out)
    p = report("Transposed interleaved, bf16 x bf16, HiFi4", result, gold)
    assert p > 0.99


def test_transposed_height_sharded_bf8(device, golden):
    """Transposed matmul, height-sharded B^T in L1 (bf8), bf16 activation, HiFi2.

    B^T (weight) height-sharded in L1: 56 cores, [128, 4096] per core (~540KB in bf8).
    A^T (activation) multicast via mcast_in0=False.
    """
    A, _, B_T, gold = golden
    dram = ttnn.DRAM_MEMORY_CONFIG
    a_tt = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)
    bt_tt = ttnn.from_torch(B_T, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b, memory_config=dram)

    at_tt = ttnn.transpose(a_tt, -2, -1)

    num_cores = 56
    shard_h = N // num_cores  # 128
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}
    )
    shard_spec = ttnn.ShardSpec(shard_grid, [shard_h, K], ttnn.ShardOrientation.ROW_MAJOR)
    height_shard_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec
    )
    bt_sharded = ttnn.to_memory_config(bt_tt, height_shard_cfg)

    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 7),
        in0_block_w=2,
        out_subblock_h=4,
        out_subblock_w=1,
        out_block_h=4,
        out_block_w=1,
        per_core_M=4,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )

    out = ttnn.matmul(bt_sharded, at_tt, program_config=cfg,
                      dtype=ttnn.bfloat16, compute_kernel_config=make_compute_config(device))
    out = ttnn.transpose(out, -2, -1)

    result = ttnn.to_torch(out)
    p = report("Transposed height-sharded, bf8 x bf16, HiFi2", result, gold)
    assert p > 0.99


def test_transposed_height_sharded_bf16(device, golden):
    """Transposed matmul, height-sharded B^T in L1 (bf16), HiFi4.

    bf16 shard [128, 4096] = 1MB/core — tight for L1, may OOM.
    """
    A, _, B_T, gold = golden
    dram = ttnn.DRAM_MEMORY_CONFIG
    a_tt = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)
    bt_tt = ttnn.from_torch(B_T, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)

    at_tt = ttnn.transpose(a_tt, -2, -1)

    num_cores = 56
    shard_h = N // num_cores
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}
    )
    shard_spec = ttnn.ShardSpec(shard_grid, [shard_h, K], ttnn.ShardOrientation.ROW_MAJOR)
    height_shard_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec
    )
    bt_sharded = ttnn.to_memory_config(bt_tt, height_shard_cfg)

    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 7),
        in0_block_w=2,
        out_subblock_h=4,
        out_subblock_w=1,
        out_block_h=4,
        out_block_w=1,
        per_core_M=4,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )

    out = ttnn.matmul(bt_sharded, at_tt, program_config=cfg,
                      dtype=ttnn.bfloat16, compute_kernel_config=make_compute_config(device, hifi4=True))
    out = ttnn.transpose(out, -2, -1)

    result = ttnn.to_torch(out)
    p = report("Transposed height-sharded, bf16 x bf16, HiFi4", result, gold)
    assert p > 0.99


def test_transpose_a_hs_activation_bf8(device, golden):
    """Original A*B via transpose_a: A^T height-sharded in L1, B in DRAM.

    A^T [4096x32] height-sharded across 32 cores (8x4):
      shard [128, 32] = 8KB/core — trivially fits in L1.
    transpose_a=True -> matmul sees [32x4096].
    B [4096x7168] in DRAM (original weight, no pre-transpose needed).
    Result = [32x7168] = C directly, no transpose-back needed.

    Tests whether height-sharded input improves precision over
    the width-sharded original, while keeping activation in L1.
    """
    A, B, _, gold = golden
    dram = ttnn.DRAM_MEMORY_CONFIG
    a_tt = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)
    b_tt = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b, memory_config=dram)

    # Transpose activation on device, then height-shard in L1
    at_tt = ttnn.transpose(a_tt, -2, -1)

    num_cores = 32
    shard_h = K // num_cores  # 4096/32 = 128
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}
    )
    shard_spec = ttnn.ShardSpec(shard_grid, [shard_h, M], ttnn.ShardOrientation.ROW_MAJOR)
    hs_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec
    )
    at_sharded = ttnn.to_memory_config(at_tt, hs_cfg)

    out = ttnn.matmul(at_sharded, b_tt, transpose_a=True,
                      dtype=ttnn.bfloat16, compute_kernel_config=make_compute_config(device))
    result = ttnn.to_torch(out)
    p = report("transpose_a HS activation, bf8 weight, HiFi2", result, gold)
    assert p > 0.99


def test_transpose_a_hs_activation_bf16(device, golden):
    """Same as above but bf16 weight, HiFi4."""
    A, B, _, gold = golden
    dram = ttnn.DRAM_MEMORY_CONFIG
    a_tt = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)
    b_tt = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)

    at_tt = ttnn.transpose(a_tt, -2, -1)

    num_cores = 32
    shard_h = K // num_cores
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}
    )
    shard_spec = ttnn.ShardSpec(shard_grid, [shard_h, M], ttnn.ShardOrientation.ROW_MAJOR)
    hs_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec
    )
    at_sharded = ttnn.to_memory_config(at_tt, hs_cfg)

    out = ttnn.matmul(at_sharded, b_tt, transpose_a=True,
                      dtype=ttnn.bfloat16, compute_kernel_config=make_compute_config(device, hifi4=True))
    result = ttnn.to_torch(out)
    p = report("transpose_a HS activation, bf16 weight, HiFi4", result, gold)
    assert p > 0.99


def test_transposed_l1_activation_bf8(device, golden):
    """Transposed matmul: B^T (weight) in DRAM, A^T (activation) in L1.

    Simulates realistic compiler placement: weight stays in DRAM,
    small activation A^T [4096x32] (~256KB in bf16) moves to L1.
    B^T [7168x4096] stays in DRAM interleaved.
    A^T placed in L1 interleaved for faster multicast.
    """
    A, _, B_T, gold = golden
    dram = ttnn.DRAM_MEMORY_CONFIG
    l1 = ttnn.L1_MEMORY_CONFIG
    a_tt = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)
    bt_tt = ttnn.from_torch(B_T, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b, memory_config=dram)

    # Transpose activation on device, then move to L1
    at_tt = ttnn.transpose(a_tt, -2, -1)
    at_l1 = ttnn.to_memory_config(at_tt, l1)

    out = ttnn.matmul(bt_tt, at_l1, dtype=ttnn.bfloat16, compute_kernel_config=make_compute_config(device))
    out = ttnn.transpose(out, -2, -1)

    result = ttnn.to_torch(out)
    p = report("Transposed DRAM weight + L1 activation, bf8 x bf16, HiFi2", result, gold)
    assert p > 0.99


def test_transposed_l1_activation_bf16(device, golden):
    """Transposed matmul: B^T (weight bf16) in DRAM, A^T (activation) in L1, HiFi4."""
    A, _, B_T, gold = golden
    dram = ttnn.DRAM_MEMORY_CONFIG
    l1 = ttnn.L1_MEMORY_CONFIG
    a_tt = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)
    bt_tt = ttnn.from_torch(B_T, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram)

    at_tt = ttnn.transpose(a_tt, -2, -1)
    at_l1 = ttnn.to_memory_config(at_tt, l1)

    out = ttnn.matmul(bt_tt, at_l1, dtype=ttnn.bfloat16,
                      compute_kernel_config=make_compute_config(device, hifi4=True))
    out = ttnn.transpose(out, -2, -1)

    result = ttnn.to_torch(out)
    p = report("Transposed DRAM weight + L1 activation, bf16 x bf16, HiFi4", result, gold)
    assert p > 0.99
