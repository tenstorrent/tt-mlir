"""Isolated DS matmul: one ttnn.linear with an explicitly built
MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig, nothing else on device.

Weight is DRAM width-sharded over all 8 banks (never varied -- the DS kernel puts one
reader on the worker core adjacent to each bank). Activation is L1 width-sharded over
`in0_cores`. Sweeps in0_block_w, which sets the per-bank read burst
(burst = in0_block_w * shard_n tiles).

Legality check only (fast -- configs that overrun L1 fail here):
  pytest -s ds-perf/test_ds_matmul_isolated.py

For device time, drive the profiler directly and post-process per core.  Do NOT use
`python -m tracy`: it imports websockets, which the ttmlir toolchain venv lacks.  And do
NOT read DEVICE KERNEL DURATION from ops_perf_results.csv -- it is corrupt for every
multi-core op on this card (see ds-perf/README.md).  One process per config, so each
device log holds a single matmul:

  export TT_METAL_DEVICE_PROFILER=1
  export TT_METAL_PROFILER_DIR=<build>/python_packages/ttrt/runtime/generated/profiler
  pytest -s ds-perf/test_ds_matmul_isolated.py -k <case>
  # then take, per op, the max over cores of that core's own kernel span

Isolation matters: three of these configs allocate an in1 circular buffer too large to
coexist with a full model's L1 buffers, and are reachable only in a test like this.  The
in1CB ceiling on p150 sits between 1148 and 1224 KB.
"""
import pytest, torch, ttnn

BANKS = 8
TILE = 32
BYTES_PER_ELEM = {ttnn.bfloat8_b: 1.0625, ttnn.bfloat4_b: 0.5625, ttnn.bfloat16: 2.0}
SPEC_GBPS = 512.0            # p150 DRAM, matches BH_DRAM_BANDWIDTH_GB_PER_SEC in tt-metal

# (id, K, N, in0_cores, in0_block_w, weight dtype) -- the configs the MLIR harness
# could not build, plus each shape's best legal point as an in-test control.
CASES = [
    ("q3b_down_c8_w43",     11008,  2048,  8, 43, ttnn.bfloat8_b),
    ("q3b_down_c43_w8",     11008,  2048, 43,  8, ttnn.bfloat8_b),
    ("q3b_gateup_c8_w8",     2048, 11008,  8,  8, ttnn.bfloat8_b),
    ("q8b_down_c16_w24",    12288,  4096, 16, 24, ttnn.bfloat8_b),
    ("q8b_down_c8_w16",     12288,  4096,  8, 16, ttnn.bfloat8_b),
    ("q8b_gateup_c16_w8",    4096, 12288, 16,  8, ttnn.bfloat8_b),
    ("q8b_gateup_c8_w4",     4096, 12288,  8,  4, ttnn.bfloat8_b),
    ("f7b_down_c24_w30",    23040,  3072, 24, 30, ttnn.bfloat8_b),
    ("f7b_down_c8_w18",     23040,  3072,  8, 18, ttnn.bfloat8_b),
    ("f7b_gateup_c24_w4",    3072, 23040, 24,  4, ttnn.bfloat8_b),
    ("f7b_gateup_c8_w2",     3072, 23040,  8,  2, ttnn.bfloat8_b),
]


def _grid(device, n):
    """Row-wrapped CoreRangeSet covering exactly n cores of the compute grid."""
    cg = device.compute_with_storage_grid_size()
    assert n <= cg.x * cg.y, f"{n} cores exceeds the {cg.x}x{cg.y} compute grid"
    full, rem = divmod(n, cg.x)
    rs = []
    if full:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cg.x - 1, full - 1)))
    if rem:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(rs)


def _sharded(shape, grid, layout, dtype=None):
    return ttnn.MemoryConfig(
        layout, ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shape, ttnn.ShardOrientation.ROW_MAJOR))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_ds_matmul(device, case):
    name, K, N, in0_cores, in0_block_w, wdtype = case
    M = 32
    kt, nt = K // TILE, N // TILE
    assert kt % in0_cores == 0, f"{kt} K-tiles not divisible by {in0_cores} in0 cores"
    k_per_core = kt // in0_cores
    assert k_per_core % in0_block_w == 0, (
        f"in0_block_w={in0_block_w} does not divide {k_per_core} K-tiles/core")
    shard_n = -(-nt // BANKS)                      # per-bank weight shard width, in tiles
    burst_tiles = in0_block_w * shard_n
    tile_bytes = 1088 if wdtype == ttnn.bfloat8_b else 576
    print(f"\n[{name}] K={K} N={N} in0_cores={in0_cores} k/core={k_per_core} "
          f"in0_block_w={in0_block_w} shard_n={shard_n} "
          f"burst={burst_tiles} tiles = {burst_tiles*tile_bytes/1024:.0f} KB  "
          f"in1CB={in0_block_w*shard_n*tile_bytes*3/1024:.0f} KB")

    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K).bfloat16()
    b = torch.randn(1, 1, K, N).bfloat16()

    # activation: L1 width-sharded across in0_cores
    a_t = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=_sharded([M, K // in0_cores], _grid(device, in0_cores),
                               ttnn.TensorMemoryLayout.WIDTH_SHARDED))
    # weight: DRAM width-sharded over all 8 banks
    b_t = ttnn.from_torch(
        b, dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
            ttnn.ShardSpec(_grid(device, BANKS), [K, shard_n * TILE],
                           ttnn.ShardOrientation.ROW_MAJOR)))

    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w, per_core_M=1, per_core_N=shard_n, fused_activation=None)
    out_mc = _sharded([M, shard_n * TILE], _grid(device, BANKS),
                      ttnn.TensorMemoryLayout.WIDTH_SHARDED)

    out = ttnn.linear(a_t, b_t, program_config=pc, memory_config=out_mc,
                      dtype=ttnn.bfloat16,
                      compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                          math_fidelity=ttnn.MathFidelity.HiFi2, packer_l1_acc=True))
    ttnn.synchronize_device(device)
    # a few more so tracy has repeats to median over, program cache warm after the first
    for _ in range(4):
        out = ttnn.linear(a_t, b_t, program_config=pc, memory_config=out_mc,
                          dtype=ttnn.bfloat16,
                          compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                              math_fidelity=ttnn.MathFidelity.HiFi2, packer_l1_acc=True))
    ttnn.synchronize_device(device)

    weight_bytes = K * N * BYTES_PER_ELEM[wdtype]
    print(f"[{name}] LEGAL. weight {weight_bytes/1e6:.2f} MB -- "
          f"{weight_bytes/SPEC_GBPS/1e3:.1f} us at the part's {SPEC_GBPS:.0f} GB/s")
    ttnn.deallocate(out); ttnn.deallocate(a_t); ttnn.deallocate(b_t)
