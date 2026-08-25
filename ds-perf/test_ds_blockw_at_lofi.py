"""Where does small in0_block_w actually start costing, once fidelity is matched?

`kMinBlockWidthFraction` declines DS when the fitted in0_block_w falls below
kPerCore/fraction. The threshold was calibrated against timings taken with DS on HiFi2,
which charged DS an issue-slot tax that multicast never paid, so the boundary it was
drawn at is not trustworthy on its own.

in0_block_w sets the per-bank read burst (burst = in0_block_w * shard_n tiles), so
shrinking it should cost DRAM efficiency rather than compute -- and unlike the fidelity
tax, that cost should survive at LoFi. This sweeps in0_block_w at both fidelities so the
two effects can be told apart and the cut can be read off the LoFi curve.

Drive it with ds-perf/run_ds_blockw_at_lofi.sh.
"""
import pytest
import torch
import ttnn

BANKS = 8
TILE = 32
M = 32
BPE = 1.0625  # bfloat8_b
SPEC_GBPS = 512.0  # p150 DRAM

FID = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2}

# (shape id, K, N, in0_block_w values -- divisors of kPerCore that fit L1)
#   f7b_down  kPerCore 90: a wide ladder, so the fall-off has resolution
#   q3b_down  kPerCore 43 is prime: only 43 or the collapse to 1, nothing between
SHAPES = [
    ("f7b_down", 23040, 3072, (18, 9, 6, 3, 2, 1)),
    ("q3b_down", 11008, 2048, (43, 1)),
]

CASES = [(f"{sid}__w{w}__{fid}", sid, K, N, w, fid)
         for sid, K, N, ws in SHAPES for w in ws for fid in FID]


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


def _l1_sharded(grid, shape):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shape, ttnn.ShardOrientation.ROW_MAJOR))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_ds_blockw(device, case):
    name, sid, K, N, w, fid = case
    kt, nt = K // TILE, N // TILE
    k_per_core = kt // BANKS
    assert k_per_core % w == 0, f"w={w} does not divide kPerCore={k_per_core}"
    shard_n = -(-nt // BANKS)
    burst_tiles = w * shard_n
    in1cb_kb = burst_tiles * 1088 * 3 / 1024
    if in1cb_kb > 1224:
        pytest.skip(f"in1CB {in1cb_kb:.0f} KB exceeds the p150 ceiling")

    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K).bfloat16()
    b = torch.randn(1, 1, K, N).bfloat16()
    a_t = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=_l1_sharded(_rect(device, BANKS), [M, K // BANKS]))
    b_t = ttnn.from_torch(
        b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
            ttnn.ShardSpec(_rect(device, BANKS), [K, shard_n * TILE],
                           ttnn.ShardOrientation.ROW_MAJOR)))
    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=w, per_core_M=1, per_core_N=shard_n, fused_activation=None)
    kw = dict(program_config=pc,
              memory_config=_l1_sharded(_rect(device, BANKS), [M, shard_n * TILE]),
              dtype=ttnn.bfloat16,
              compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                  math_fidelity=FID[fid], packer_l1_acc=True))
    print(f"\n[{name}] kPerCore={k_per_core} w={w} ratio={k_per_core//w} "
          f"burst={burst_tiles} tiles = {burst_tiles*1088/1024:.0f} KB "
          f"in1CB={in1cb_kb:.0f} KB fidelity={fid}")
    out = ttnn.linear(a_t, b_t, **kw)
    ttnn.synchronize_device(device)
    for _ in range(4):
        out = ttnn.linear(a_t, b_t, **kw)
    ttnn.synchronize_device(device)
    wb = K * N * BPE
    print(f"[{name}] LEGAL. weight {wb/1e6:.2f} MB, "
          f"{wb/SPEC_GBPS/1e3:.1f} us at {SPEC_GBPS:.0f} GB/s")
    ttnn.deallocate(out)
    ttnn.deallocate(a_t)
    ttnn.deallocate(b_t)
