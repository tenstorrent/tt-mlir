"""Is LoFi accurate enough for the bfp8 DS matmuls that ship at HiFi2?

PCC against an fp32 golden is dominated by bfp8 weight quantization, so it can hide the
fidelity signal.  This reports four things per shape:

  pcc_lofi / pcc_hifi2  - each fidelity against the fp32 golden
  pcc_vs_each_other     - LoFi's output against HiFi2's, i.e. do they agree
  pcc_bf16_floor        - the fp32 golden merely rounded to bf16, the output dtype.
                          Any PCC at or above this floor is indistinguishable from
                          the rounding the op has to do anyway.

Also reports max and mean relative error, which PCC can mask.
"""
import pytest, torch, ttnn

TILE, BANKS = 32, 8
# bfp8 shapes that tt-mlir ships at HiFi2 (buildComputeConfig's non-bfp4 branch)
SHAPES = [
    ("f7b_down",   "falcon3_7b",   "down",    23040,  3072, 18),
    ("q8b_down",   "qwen_3_8b",    "down",    12288,  4096, 16),
    ("l8b_down",   "llama_3_1_8b", "down",    14336,  4096, 14),
    ("q3b_gateup", "qwen_2_5_3b",  "gate/up",  2048, 11008,  4),
    ("l1b_down",   "llama_3_2_1b", "down",     8192,  2048, 32),
    ("q3b_oproj",  "qwen_2_5_3b",  "o_proj",   2048,  2048,  8),
]


def _rect(device, n):
    cg = device.compute_with_storage_grid_size()
    full, rem = divmod(n, cg.x)
    rs = []
    if full:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cg.x - 1, full - 1)))
    if rem:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(rs)


def _pcc(x, y):
    x = x.flatten().to(torch.float64); y = y.flatten().to(torch.float64)
    xm, ym = x - x.mean(), y - y.mean()
    d = xm.norm() * ym.norm()
    return float("nan") if d == 0 else float((xm @ ym) / d)


def _run(device, K, N, w, fid):
    M = 32
    kt, nt = K // TILE, -(-N // TILE)
    sn = -(-nt // BANKS)
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K).bfloat16()
    b = torch.randn(1, 1, K, N).bfloat16()
    a_t = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1, ttnn.ShardSpec(_rect(device, BANKS), [M, K // BANKS],
                ttnn.ShardOrientation.ROW_MAJOR)))
    b_t = ttnn.from_torch(b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM, ttnn.ShardSpec(_rect(device, BANKS), [K, sn * TILE],
                ttnn.ShardOrientation.ROW_MAJOR)))
    out = ttnn.linear(a_t, b_t,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=w, per_core_M=1, per_core_N=sn, fused_activation=None),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1, ttnn.ShardSpec(_rect(device, BANKS), [M, sn * TILE],
                ttnn.ShardOrientation.ROW_MAJOR)),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=fid, packer_l1_acc=True))
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out)[..., :N].to(torch.float32)
    ttnn.deallocate(out); ttnn.deallocate(a_t); ttnn.deallocate(b_t)
    return got, torch.matmul(a.to(torch.float32), b.to(torch.float32))


@pytest.mark.parametrize("shape", SHAPES, ids=[s[0] for s in SHAPES])
def test_lofi_accuracy(device, shape):
    sid, model, role, K, N, w = shape
    lo, golden = _run(device, K, N, w, ttnn.MathFidelity.LoFi)
    hi, _ = _run(device, K, N, w, ttnn.MathFidelity.HiFi2)
    g = golden.squeeze()
    floor = g.to(torch.bfloat16).to(torch.float32)     # output-dtype rounding alone
    rel = lambda x: float((x.squeeze() - g).abs().mean() / g.abs().mean())
    mx = lambda x: float((x.squeeze() - g).abs().max() / g.abs().max())
    print(f"\n[{sid}] {model} {role} {K}x{N} bfp8 w={w}")
    print(f"  pcc vs fp32 golden   LoFi {_pcc(g, lo):.7f}   HiFi2 {_pcc(g, hi):.7f}")
    print(f"  pcc bf16-round floor      {_pcc(g, floor):.7f}   <- rounding the golden to bf16")
    print(f"  pcc LoFi vs HiFi2         {_pcc(hi, lo):.7f}")
    print(f"  mean rel err         LoFi {rel(lo):.6f}   HiFi2 {rel(hi):.6f}   "
          f"bf16 floor {rel(floor):.6f}")
    print(f"  max  rel err         LoFi {mx(lo):.6f}   HiFi2 {mx(hi):.6f}   "
          f"bf16 floor {mx(floor):.6f}")
