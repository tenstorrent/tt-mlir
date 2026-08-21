"""Fidelity sweep over the fleet's DRAM-sharded matmul shapes: is LoFi faster, and is
it still correct?

MVMULs per tile-MAC = replay_buf_len(16 for full tiles) * fidelity_loops
(llk_math_matmul.h:339,:434), so LoFi issues half of HiFi2's and a quarter of HiFi4's.
The question this answers is whether the resulting speedup survives a numerics check.

Golden is a float32 torch matmul of the *original* tensors, so the reported PCC folds in
both the bfp8/bfp4 weight quantization and the fidelity loss.  Compare LoFi's PCC against
HiFi2's on the same row: HiFi2 is the accuracy the shipped configs already accept, so the
delta is what dropping to LoFi actually costs.

  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<...>/generated/profiler
  pytest -s ds-perf/test_ds_fidelity_sweep.py -k "<shape> and LoFi"
"""
import pytest, torch, ttnn

TILE, BANKS = 32, 8

# (id, model, role, K, N, weight dtype, in0_cores, in0_block_w) -- in0_block_w is what
# tt-mlir emits today, except q3b_down_w43 which is the best legal point (isolated only).
SHAPES = [
    ("q3b_down",       "qwen_2_5_3b",  "down",    11008,  2048, "bfp8",  8,  1),
    ("q3b_down_w43",   "qwen_2_5_3b",  "down",    11008,  2048, "bfp8",  8, 43),
    ("q3b_gateup",     "qwen_2_5_3b",  "gate/up",  2048, 11008, "bfp8",  8,  4),
    ("q3b_oproj",      "qwen_2_5_3b",  "o_proj",   2048,  2048, "bfp8",  8,  8),
    ("l1b_down",       "llama_3_2_1b", "down",     8192,  2048, "bfp8",  8, 32),
    ("l1b_gateup",     "llama_3_2_1b", "gate/up",  2048,  8192, "bfp8",  8,  8),
    ("q8b_down",       "qwen_3_8b",    "down",    12288,  4096, "bfp8",  8, 16),
    ("q8b_gateup",     "qwen_3_8b",    "gate/up",  4096, 12288, "bfp8",  8,  4),
    ("f7b_down",       "falcon3_7b",   "down",    23040,  3072, "bfp8",  8, 18),
    ("f7b_gateup",     "falcon3_7b",   "gate/up",  3072, 23040, "bfp8",  8,  2),
    ("l8b_down",       "llama_3_1_8b", "down",    14336,  4096, "bfp8",  8, 14),
    ("l8b_gateup",     "llama_3_1_8b", "gate/up",  4096, 14336, "bfp4",  8,  8),
]
FID = [("LoFi", ttnn.MathFidelity.LoFi, 1), ("HiFi2", ttnn.MathFidelity.HiFi2, 2)]
DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def _rect(device, n):
    cg = device.compute_with_storage_grid_size()
    assert n <= cg.x * cg.y, f"{n} cores > {cg.x}x{cg.y} grid"
    full, rem = divmod(n, cg.x)
    rs = []
    if full:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cg.x - 1, full - 1)))
    if rem:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(rs)


def _pcc(golden, actual):
    g = golden.flatten().to(torch.float32)
    a = actual.flatten().to(torch.float32)
    if torch.allclose(g, a):
        return 1.0
    gm, am = g - g.mean(), a - a.mean()
    denom = gm.norm() * am.norm()
    return float("nan") if denom == 0 else float((gm @ am) / denom)


@pytest.mark.parametrize("fid_name,fid,loops", FID, ids=[f[0] for f in FID])
@pytest.mark.parametrize("shape", SHAPES, ids=[s[0] for s in SHAPES])
def test_fidelity(device, shape, fid_name, fid, loops):
    sid, model, role, K, N, wdt, in0_cores, w = shape
    M = 32
    kt, nt = K // TILE, -(-N // TILE)
    sn = -(-nt // BANKS)
    assert kt % in0_cores == 0 and (kt // in0_cores) % w == 0, "illegal blocking"
    print(f"\n[{sid} {fid_name}] {model} {role} {K}x{N} {wdt} "
          f"w={w} shard_n={sn} MVMULs/tile-MAC={16*loops}")

    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K).bfloat16()
    b = torch.randn(1, 1, K, N).bfloat16()
    golden = torch.matmul(a.to(torch.float32), b.to(torch.float32))

    a_t = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(_rect(device, in0_cores), [M, K // in0_cores],
                           ttnn.ShardOrientation.ROW_MAJOR)))
    b_t = ttnn.from_torch(
        b, dtype=DT[wdt], layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
            ttnn.ShardSpec(_rect(device, BANKS), [K, sn * TILE],
                           ttnn.ShardOrientation.ROW_MAJOR)))
    kw = dict(
        program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=w, per_core_M=1, per_core_N=sn, fused_activation=None),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(_rect(device, BANKS), [M, sn * TILE],
                           ttnn.ShardOrientation.ROW_MAJOR)),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=fid, packer_l1_acc=True))

    out = ttnn.linear(a_t, b_t, **kw)
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out)[..., :N]
    pcc = _pcc(golden, got)
    rel = float((got.to(torch.float32) - golden).abs().mean() / golden.abs().mean())
    for _ in range(4):                      # repeats for the profiler to median over
        out = ttnn.linear(a_t, b_t, **kw)
    ttnn.synchronize_device(device)
    print(f"[{sid} {fid_name}] PCC={pcc:.6f}  mean_rel_err={rel:.5f}")
    ttnn.deallocate(out); ttnn.deallocate(a_t); ttnn.deallocate(b_t)
