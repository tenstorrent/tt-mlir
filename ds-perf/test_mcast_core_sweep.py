"""How many cores does a decode matmul need before DRAM, not compute, is the limit?

Same weight shape and dtype as the DS tests, but on
MatmulMultiCoreReuseMultiCast1DProgramConfig with the compute grid swept. DS is pinned
to the 8 DRAM-bank-adjacent cores and cannot be swept, so this is the only way to
separate "too few cores" from "not enough DRAM bandwidth".

Read device time the same way as test_ds_matmul_isolated.py (TT_METAL_DEVICE_PROFILER=1,
per-core spans, max over cores).
"""
import pytest, torch, ttnn

TILE = 32
# (K, N) from falcon3_7b down -- the shape where DS did best, so the comparison is fair
K, N = 23040, 3072
CORES = [8, 12, 16, 24, 32, 48, 96]   # divisors of N-tiles (96)


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


# Matched-blocking comparison: multicast on the same 8 cores as DS, at the same
# in0_block_w, so the kernel is the only difference between the two measurements.
MATCHED_W = [18, 9, 6, 3, 2]      # divisors of kPerCore=90 that fit L1 at per_core_N=12


@pytest.mark.parametrize("w", MATCHED_W, ids=[f"m8w{w}" for w in MATCHED_W])
def test_mcast_8cores_matched_w(device, w):
    """1D multicast pinned to 8 cores, in0_block_w swept over the DS-legal values."""
    M, cores = 32, 8
    kt, nt = K // TILE, N // TILE
    per_core_n = nt // cores
    assert kt % w == 0
    cg = device.compute_with_storage_grid_size()
    print(f"\n[mcast8 w={w}] per_core_N={per_core_n} in1CB={w*per_core_n*1088*3/1024:.0f} KB")
    torch.manual_seed(0)
    a_t = ttnn.from_torch(torch.randn(1, 1, M, K).bfloat16(), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b_t = ttnn.from_torch(torch.randn(1, 1, K, N).bfloat16(), dtype=ttnn.bfloat8_b,
                          layout=ttnn.TILE_LAYOUT, device=device,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cg.x, 1),
        in0_block_w=w, out_subblock_h=1, out_subblock_w=1,
        per_core_M=1, per_core_N=per_core_n,
        fuse_batch=True, fused_activation=None, mcast_in0=True)
    out_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(_rect(device, cores), [M, per_core_n * TILE],
                       ttnn.ShardOrientation.ROW_MAJOR))
    kw = dict(program_config=pc, memory_config=out_mc, dtype=ttnn.bfloat16,
              compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                  math_fidelity=ttnn.MathFidelity.HiFi2, packer_l1_acc=True))
    out = ttnn.linear(a_t, b_t, **kw)
    ttnn.synchronize_device(device)
    for _ in range(4):
        out = ttnn.linear(a_t, b_t, **kw)
    ttnn.synchronize_device(device)
    print(f"[mcast8 w={w}] LEGAL")
    ttnn.deallocate(out); ttnn.deallocate(a_t); ttnn.deallocate(b_t)


@pytest.mark.parametrize("cores", CORES, ids=[f"c{c}" for c in CORES])
def test_mcast_cores(device, cores):
    M = 32
    kt, nt = K // TILE, N // TILE
    if nt % cores:
        pytest.skip(f"{nt} N-tiles not divisible by {cores} cores")
    per_core_n = nt // cores
    cg = device.compute_with_storage_grid_size()
    # in0_block_w: divide K-tiles, keep the in1 block inside L1
    in0_block_w = next((w for w in (8, 4, 2, 1)
                        if kt % w == 0 and w * per_core_n * 1088 * 3 < 1_200_000), 1)
    print(f"\n[mcast c{cores}] K={K} N={N} cores={cores} per_core_N={per_core_n} "
          f"in0_block_w={in0_block_w}")

    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K).bfloat16()
    b = torch.randn(1, 1, K, N).bfloat16()
    a_t = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b_t = ttnn.from_torch(b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cg.x, max(1, -(-cores // cg.x))),
        in0_block_w=in0_block_w, out_subblock_h=1, out_subblock_w=1,
        per_core_M=1, per_core_N=per_core_n,
        fuse_batch=True, fused_activation=None, mcast_in0=True)
    out_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(_rect(device, cores), [M, per_core_n * TILE],
                       ttnn.ShardOrientation.ROW_MAJOR))
    kw = dict(program_config=pc, memory_config=out_mc, dtype=ttnn.bfloat16,
              compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                  math_fidelity=ttnn.MathFidelity.HiFi2, packer_l1_acc=True))
    out = ttnn.linear(a_t, b_t, **kw)
    ttnn.synchronize_device(device)
    for _ in range(4):
        out = ttnn.linear(a_t, b_t, **kw)
    ttnn.synchronize_device(device)
    print(f"[mcast c{cores}] LEGAL")
    ttnn.deallocate(out); ttnn.deallocate(a_t); ttnn.deallocate(b_t)
