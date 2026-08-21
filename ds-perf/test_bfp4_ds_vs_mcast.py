"""The bfp4 gate/up shape is the only fleet shape tt-mlir already gives LoFi
(buildComputeConfig's bfp4 branch), so it is the one place DS and 1D multicast can be
compared with no fidelity handicap on either side.

DS is pinned to the 8 DRAM-bank cores; production multicast for this shape uses an 11x9
grid with in0_block_w=2, per_core_N=5.  Both are run here at explicit LoFi.
"""
import pytest, torch, ttnn

TILE, BANKS = 32, 8
K, N = 4096, 14336          # llama_3_1_8b gate/up, bfp4 weights
# (id, kernel, cores, in0_block_w, weight dtype)
# bfp8 has 1088 B/tile against bfp4's 576, so the in1 CB doubles and w=8 is expected
# to overrun L1 -- w=4 is the largest that fits, which is what tt-mlir picks for
# comparable bfp8 shapes.
CASES = [
    ("ds_c8_w8",        "ds",    8,  8, "bfp4"),
    ("mcast_c99_w2",    "mcast", 99, 2, "bfp4"),   # what the no-DS compile emits
    ("mcast_c99_w4",    "mcast", 99, 4, "bfp4"),
    ("mcast_c64_w8",    "mcast", 64, 8, "bfp4"),
    ("mcast_c32_w8",    "mcast", 32, 8, "bfp4"),
    ("mcast_c8_w8",     "mcast", 8,  8, "bfp4"),   # matched cores against DS
    ("ds_c8_w8_bfp8",   "ds",    8,  8, "bfp8"),   # expected to overrun L1
    ("ds_c8_w4_bfp8",   "ds",    8,  4, "bfp8"),
    ("ds_c8_w2_bfp8",   "ds",    8,  2, "bfp8"),
    ("mcast_c99_w2_bfp8",   "mcast", 99, 2, "bfp8"),
    ("mcast_c8_w4_bfp8",    "mcast", 8,  4, "bfp8"),
    # multicast bfp8 was only ever run at the production w=2; at 99 cores per_core_N
    # is 5 tiles so the in1 CB is tiny and w can go much higher.
    ("mcast_c99_w4_bfp8",   "mcast", 99, 4, "bfp8"),
    ("mcast_c99_w8_bfp8",   "mcast", 99, 8, "bfp8"),
    ("mcast_c99_w16_bfp8",  "mcast", 99, 16, "bfp8"),
    ("mcast_c99_w32_bfp8",  "mcast", 99, 32, "bfp8"),
    ("mcast_c64_w16_bfp8",  "mcast", 64, 16, "bfp8"),
    # and the same courtesy for bfp4, in case production w=2 was not its best either
    ("mcast_c99_w8_bfp4",   "mcast", 99, 8, "bfp4"),
    ("mcast_c99_w16_bfp4",  "mcast", 99, 16, "bfp4"),
    # control for "is bfp8 slower only because w must be smaller?": run bfp4 at the
    # same w bfp8 is forced to.  If bfp4 at w=4 still matches bfp4 at w=8, then w is
    # not what separates the dtypes -- bytes moved are.
    ("ds_c8_w4_bfp4",   "ds", 8, 4, "bfp4"),
    ("ds_c8_w2_bfp4",   "ds", 8, 2, "bfp4"),
]
DTYPE = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}
BPE = {"bfp4": 0.5625, "bfp8": 1.0625}


def _rect(device, n):
    cg = device.compute_with_storage_grid_size()
    assert n <= cg.x * cg.y
    full, rem = divmod(n, cg.x)
    rs = []
    if full:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cg.x - 1, full - 1)))
    if rem:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(rs)


def _pcc(g, a):
    g = g.flatten().to(torch.float32); a = a.flatten().to(torch.float32)
    gm, am = g - g.mean(), a - a.mean()
    d = gm.norm() * am.norm()
    return float("nan") if d == 0 else float((gm @ am) / d)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_bfp4(device, case):
    sid, kind, cores, w, wdt = case
    M = 32
    kt, nt = K // TILE, N // TILE
    cg = device.compute_with_storage_grid_size()
    assert kt % w == 0
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K).bfloat16()
    b = torch.randn(1, 1, K, N).bfloat16()
    golden = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True)

    if kind == "ds":
        sn = -(-nt // BANKS)
        a_t = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1, ttnn.ShardSpec(_rect(device, cores), [M, K // cores],
                    ttnn.ShardOrientation.ROW_MAJOR)))
        b_t = ttnn.from_torch(b, dtype=DTYPE[wdt], layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM, ttnn.ShardSpec(_rect(device, BANKS), [K, sn * TILE],
                    ttnn.ShardOrientation.ROW_MAJOR)))
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=w, per_core_M=1, per_core_N=sn, fused_activation=None)
        out_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(_rect(device, BANKS), [M, sn * TILE], ttnn.ShardOrientation.ROW_MAJOR))
        print(f"\n[{sid}] DS  cores={cores} w={w} per_core_N={sn} {wdt} LoFi "
              f"in1CB={w*sn*(1088 if wdt=='bfp8' else 576)*3/1024:.0f} KB")
    else:
        pcn = -(-nt // cores)
        a_t = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b_t = ttnn.from_torch(b, dtype=DTYPE[wdt], layout=ttnn.TILE_LAYOUT, device=device,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(cg.x, max(1, -(-cores // cg.x))),
            in0_block_w=w, out_subblock_h=1, out_subblock_w=1,
            per_core_M=1, per_core_N=pcn,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        out_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(_rect(device, cores), [M, pcn * TILE], ttnn.ShardOrientation.ROW_MAJOR))
        print(f"\n[{sid}] mcast cores={cores} w={w} per_core_N={pcn} {wdt} LoFi")

    kw = dict(program_config=pc, memory_config=out_mc, dtype=ttnn.bfloat16,
              compute_kernel_config=ckc)
    out = ttnn.linear(a_t, b_t, **kw)
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out)[..., :N]
    print(f"[{sid}] PCC={_pcc(golden, got):.6f}")
    for _ in range(4):
        out = ttnn.linear(a_t, b_t, **kw)
    ttnn.synchronize_device(device)
    ttnn.deallocate(out); ttnn.deallocate(a_t); ttnn.deallocate(b_t)
