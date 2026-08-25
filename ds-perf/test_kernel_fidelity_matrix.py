"""Separate kernel choice from math fidelity: DS and 1D multicast, each at LoFi and HiFi2.

The shipped configs conflate the two. `buildComputeConfig` attaches an explicit compute
config to DS matmuls only, so a DS matmul runs the fidelity that function picks while its
multicast counterpart takes ttnn's default (LoFi). Any DS-vs-multicast timing therefore
measures kernel *and* fidelity at once, and MVMULs per tile-MAC scale with fidelity
(replay_buf_len * fidelity_loops), so the confound is worth ~1.5x on a shape whose issue
slots bind.

This runs the 2x2 per shape so each factor can be read on its own:

    DS   @ LoFi     DS   @ HiFi2
    mcast@ LoFi     mcast@ HiFi2

Each kernel is given its own best blocking rather than a matched in0_block_w -- DS is
pinned to the bank-adjacent cores and multicast is not, so "best config per kernel at
equal fidelity" is the comparison that decides which kernel to emit.

Drive it with ds-perf/run_kernel_fidelity_matrix.sh, which runs one config per process
under the device profiler and takes durations per core (DEVICE KERNEL DURATION is corrupt
for multi-core ops on this card -- see ds-perf/README.md).
"""
import pytest
import torch
import ttnn

BANKS = 8
TILE = 32
M = 32
BYTES_PER_ELEM = {ttnn.bfloat8_b: 1.0625, ttnn.bfloat4_b: 0.5625, ttnn.bfloat16: 2.0}
SPEC_GBPS = 512.0  # p150 DRAM

FID = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2}

# (shape id, K, N, dtype, DS in0_block_w, multicast core counts to try)
#   f7b_down  -- the shape DS does best on, and the one with an existing core sweep
#   q3b_down  -- prime kPerCore (43), so DS has only in0_block_w 1 or 43
#   q3b_oproj -- small square weight; the shape that lost under DS on n150
SHAPES = [
    ("f7b_down", 23040, 3072, ttnn.bfloat8_b, 18, (32, 96)),
    ("q3b_down", 11008, 2048, ttnn.bfloat8_b, 43, (32, 64)),
    ("q3b_oproj", 2048, 2048, ttnn.bfloat8_b, 8, (32, 64)),
]

CASES = []
for sid, K, N, dt, ds_w, mc_cores in SHAPES:
    for fid in FID:
        CASES.append((f"{sid}__ds_w{ds_w}__{fid}", sid, K, N, dt, "ds", ds_w, fid))
        for c in mc_cores:
            CASES.append((f"{sid}__mcast_c{c}__{fid}", sid, K, N, dt, "mcast", c, fid))


def _rect(device, n):
    """Row-wrapped CoreRangeSet covering exactly n cores."""
    cg = device.compute_with_storage_grid_size()
    assert n <= cg.x * cg.y, f"{n} cores exceeds the {cg.x}x{cg.y} grid"
    full, rem = divmod(n, cg.x)
    rs = []
    if full:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cg.x - 1, full - 1)))
    if rem:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(rs)


def _l1_sharded(grid, shape):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shape, ttnn.ShardOrientation.ROW_MAJOR))


def _run(device, kw, a_t, b_t, reps=5):
    out = ttnn.linear(a_t, b_t, **kw)
    ttnn.synchronize_device(device)
    for _ in range(reps - 1):
        out = ttnn.linear(a_t, b_t, **kw)
    ttnn.synchronize_device(device)
    return out


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_kernel_fidelity(device, case):
    name, sid, K, N, wdtype, kernel, knob, fid = case
    kt, nt = K // TILE, N // TILE
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K).bfloat16()
    b = torch.randn(1, 1, K, N).bfloat16()
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=FID[fid], packer_l1_acc=True)

    if kernel == "ds":
        in0_block_w = knob
        k_per_core = kt // BANKS
        assert kt % BANKS == 0, f"{kt} K-tiles not divisible by {BANKS} banks"
        assert k_per_core % in0_block_w == 0, (
            f"in0_block_w={in0_block_w} does not divide {k_per_core} K-tiles/core")
        shard_n = -(-nt // BANKS)
        a_t = ttnn.from_torch(
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=_l1_sharded(_rect(device, BANKS), [M, K // BANKS]))
        b_t = ttnn.from_torch(
            b, dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                ttnn.ShardSpec(_rect(device, BANKS), [K, shard_n * TILE],
                               ttnn.ShardOrientation.ROW_MAJOR)))
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w, per_core_M=1, per_core_N=shard_n,
            fused_activation=None)
        out_mc = _l1_sharded(_rect(device, BANKS), [M, shard_n * TILE])
        print(f"\n[{name}] DS banks={BANKS} k/core={k_per_core} w={in0_block_w} "
              f"per_core_N={shard_n} fidelity={fid}")
    else:
        cores = knob
        if nt % cores:
            pytest.skip(f"{nt} N-tiles not divisible by {cores} cores")
        per_core_n = nt // cores
        cg = device.compute_with_storage_grid_size()
        in0_block_w = next((w for w in (8, 4, 2, 1)
                            if kt % w == 0 and w * per_core_n * 1088 * 3 < 1_200_000), 1)
        a_t = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b_t = ttnn.from_torch(b, dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(cg.x, max(1, -(-cores // cg.x))),
            in0_block_w=in0_block_w, out_subblock_h=1, out_subblock_w=1,
            per_core_M=1, per_core_N=per_core_n,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        out_mc = _l1_sharded(_rect(device, cores), [M, per_core_n * TILE])
        print(f"\n[{name}] mcast cores={cores} per_core_N={per_core_n} "
              f"w={in0_block_w} fidelity={fid}")

    kw = dict(program_config=pc, memory_config=out_mc, dtype=ttnn.bfloat16,
              compute_kernel_config=ckc)
    out = _run(device, kw, a_t, b_t)
    wb = K * N * BYTES_PER_ELEM[wdtype]
    print(f"[{name}] LEGAL. weight {wb/1e6:.2f} MB, "
          f"{wb/SPEC_GBPS/1e3:.1f} us at {SPEC_GBPS:.0f} GB/s")
    ttnn.deallocate(out)
    ttnn.deallocate(a_t)
    ttnn.deallocate(b_t)
