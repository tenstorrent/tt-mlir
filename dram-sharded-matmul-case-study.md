# Case study: `DRAMShardedProgramConfig` brings no gain on Qwen2.5-3B decode

> **Scope note (2026-08-18).** The verdict below now has fleet-wide support
> ([`dram-sharding-fleet-case-study.md`](dram-sharding-fleet-case-study.md)): across 10 measured
> Blackhole decode models DS is slower on nine (1.09x-1.88x) and neutral on one; no model comes
> out ahead, though 4 of 32 individual shapes (all qkv) do gain 3.6%-11.2% under DS.
> But the **mechanism** below is Qwen2.5-3B-specific. 3B is the only model whose `kTiles/8` is
> prime, and that trap explains only why it is 1.88x instead of ~1.25x. The general reason DS
> loses is a ceiling: the DS kernel saturates near 320 GB/s while 1D multicast reaches ~390 on
> the same shapes, including on models whose `in0_block_w` is perfectly healthy.

Measured on silicon, 2026-08-17. Blackhole, single chip (1350 MHz, 10x11 worker grid,
8 DRAM banks). tt-mlir `ae60283368`, tt-metal `f1f4ff75`, `TT_RUNTIME_ENABLE_PERF_TRACE=ON`.
Workload: `ttnn_qwen_2_5_3b_bs32_isl128` decode graphs dumped from tt-xla, one compiled with
DRAM-sharded matmul (`runc802`) and one without (`runf475`).

## Verdict

On the shape that dominates this model, **the DRAM-sharded matmul path is slower than plain
interleaved 1D multicast at every legal parameterization** — not just at the parameters
tt-mlir currently picks.

| down_proj `32x11008 @ 11008x2048` bfp8 | time | bandwidth |
|---|---|---|
| DS as tt-mlir ships it (`in0_block_w=1`) | 235.6 us | 101.7 GB/s |
| DS, best of an exhaustive legal sweep (`in0 on 43 cores, w=8`) | **85.6 us** | 279.7 GB/s |
| Interleaved 1D multicast, same shape | **65.6 us** | 365.3 GB/s |

Whole traced decode step, full graphs: **DS 21.96 ms vs no-DS 13.88 ms** (1.58x), ceiling
45.5 vs 72.0 iterations/s. Matmul carries essentially all of it: 16.61 ms vs 8.84 ms across
the same 181 operations. The one-time const-eval prologue is also ~2x worse under DS
(43.75 vs 21.38 ms) because the DRAM-sharded weights cost more to prepare.

Tuning the DS config recovers most of the self-inflicted loss (235.6 -> 85.6 us, worth
5.43 ms per decode iteration) but still lands **1.30x behind simply not using it**.

## Method, and why these numbers are trustworthy

`ops_perf_results.csv` is unusable for this on this card: `DEVICE KERNEL DURATION` is computed
as `max(ZONE_END across cores) - min(ZONE_START across cores)`, and some cores' cycle counters
are zeroed at device init while others are not, so **every multi-core op** inherits a constant
~1.4288e13-cycle (~2.9 h) offset. Core count predicts it exactly: 1-core ops 308/308 sane,
16-core ops 0/144 sane. Buffer overflow and `TT_METAL_PROFILER_SYNC=1` were both ruled out.

All timings here are therefore recomputed **per core** from `.logs/profile_log_device.csv`
(each core's own `last kernel ZONE_END - first ZONE_START`, then max across cores), which never
mixes two clocks. Validation of that method:

- Op counts reconstruct the architecture exactly: 181 matmuls = 36 layers x 5 + lm_head,
  36 SdpaDecode, 72 RotaryEmbedding, 72 PagedUpdateCache, 73 LayerNorm.
- Single-op microbenchmarks of the down_proj shape reproduce both full-graph endpoints to
  within 0.5%: DS baseline 235.6 vs 236.3 us, multicast 65.6 vs 65.9 us.
- lm_head and fused QKV are on 1D multicast in **both** compiles and reproduce at 1.00x and
  1.01x across separate runs, bounding run-to-run noise under 1%.

## The parameter space is exhausted, not under-explored

`in0_block_w` is the only parameter that moves the number. In0 core count matters *only*
because it sets which `in0_block_w` values are legal — metal enforces
`(shard_width_tiles) % in0_block_w == 0`
(`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:1263`).

Measured, 5 loops each:

| in0 cores | K-tiles/core | K-steps | `in0_block_w` | time |
|---|---|---|---|---|
| 8 | 43 | 43 | 1 | 235.6 us |
| 43 | 8 | 8 | 1 | 236.1 us |
| 2 | 172 | 43 | 4 | 98.1 us |
| 86 | 4 | 1 | 4 | 98.0 us |
| 1 | 344 | 43 | 8 | 85.8 us |
| 43 | 8 | 1 | 8 | 85.6 us |

Pairs at 1-vs-43 cores and 2-vs-86 cores agree within 0.5 us. Bandwidth tracks the DRAM read
burst (`w` tiles), not parallelism or K-step count: `w=1` -> ~101 GB/s, `w=4` -> ~244,
`w=8` -> ~280.

Note this curve is specific to this shape's 23.95 MB of weights. The fleet study found op size is
a secondary factor worth 6-14%, so a 68 KB burst on a 4.5 MB matmul reaches only ~240 GB/s rather
than the ~280 seen here. Do not read the saturation point off this sweep alone.

Enumerating every legal in0 core count (divisors of 344 within a 110-core grid) against
tt-mlir's own CB budget model (`MatmulProgramConfig.cpp:313-397`), the reachable `w` set is
`{1, 2, 4, 8, 43}`. `w=16` and `w=32` divide no reachable `kPerCore`. `w=43` is the only value
with a bigger burst and it is **not usable**: at 8 in0 cores its `in1CB` alone is 1096.5K
(total 1418.5K), and metal rejects it on hardware with

```
Statically allocated circular buffers in program 8 clash with L1 buffers on core range [0-0 - 7-9].
L1 buffer allocated at 1417216 and static circular buffer region ends at 1459456
```

At every other core count `w=43` exceeds the budget outright. So **`w=8` -> 85.6 us is the
ceiling of this kernel on this shape**, and it is below the multicast alternative.

Scope note: only down_proj benefits from tuning. `in1CB` scales with the per-bank weight width
(43 tiles for gate/up), so gate/up's `w=8` costs 1122K and is over budget at any core count —
it already sits at its maximum of 4. o_proj already sits at its maximum of 8.

## Is `DRAMShardedProgramConfig` what tt-metal's hand-tuned version uses?

Partly — and the answer cuts against tt-mlir less than expected.

**It is used.** `models/demos/blackhole/qwen36/tt/tp_common.py` builds
`ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` for a hand-tuned Blackhole Qwen.
So the config is not abandoned legacy code.

**But metal's own heuristic hits the identical weakness on this shape.** `tp_common.py` derives
the activation grid per-op rather than fixing it:

```python
rows, cols = _find_grid(k_tiles)            # target 32 cores, but max_r = max_c = 8
num_cores = rows * cols
k_tiles_per_core = k_tiles // num_cores
in0_block_w = _find_largest_divisor(k_tiles_per_core)   # max_div = 8
per_core_N = n_tiles // num_cores if n_tiles >= num_cores else 1
```

Replaying it:

| shape | k_tiles | grid | cores | k/core | `in0_block_w` | `per_core_N` |
|---|---|---|---|---|---|---|
| down_proj | 344 | 1x8 | 8 | 43 | **1** | 8 |
| gate/up | 64 | 4x8 | 32 | 2 | 2 | 10 |
| o_proj | 64 | 4x8 | 32 | 2 | 2 | 2 |

`_find_grid(344)` wants 43 cores (closest divisor to its target of 32) but rejects it because
`cols <= 8`, falling back to 8 cores. 43 is prime, so `_find_largest_divisor(43, max_div=8)`
returns 1. **Metal's hand-tuned Blackhole Qwen therefore also lands on `in0_block_w=1` for
down_proj.** tt-mlir's hardcoded `kNumIn0Cores = 8` (`MatmulRules.cpp:41`) is not a divergence
from the reference; it reproduces the reference's blind spot. The 43-core arrangement that
measures 85.6 us is unreachable by *either* heuristic, though it is valid on hardware (verified
by direct execution, using row-wrapped core placement rather than a single 8-wide row).

**The high-performance path is a different kernel.** `models/demos/llama3_70b_galaxy/tt/qwen_model_config.py`
reaches its big matmuls with `gather_in0=True` and
`num_global_cb_receivers = 2 if prefetch else 1` — the ring/GlobalCircularBuffer path on
`MatmulMultiCoreReuseMultiCast1DProgramConfig`, not `DRAMShardedProgramConfig`.

## Where the DRAM-sharding gain actually lives

`global_cb` appears **29 times** in
`matmul_multicore_reuse_mcast_1d_program_factory.cpp` and **0 times** in the DRAM-sharded
factory. The GlobalCircularBuffer is a property of the 1D multicast kernel; the DRAM-sharded
program config gets no GCB support at all. "Making DRAM sharding fast" means *feeding a
gather_in0 1D ring matmul from DRAM-sharded weights through a GCB* — a different kernel that
happens to read the same weight layout.

**tt-mlir cannot express this today.** `gather_in0` hard-requires a GCB: setting
`gather_in0 = true` with `num_global_cb_receivers = 0` fails at runtime with

```
TT_FATAL: MatmulMultiCoreReuseMultiCast1DProgramConfig: Num global CB receivers must be greater than 0
```

verified by building three ring variants (r8 width-sharded, r8 interleaved, r4 width-sharded)
and running all three. The TTNN attribute already carries `gather_in0`,
`num_global_cb_receivers` and `hop_cores`, but there is no way to construct the GCB object they
refer to, so the fields can only ever be emitted as `false`/`0`.

Missing from the MLIR representation:

- the `GlobalCircularBuffer` itself (sender->receiver core mapping, per-receiver FIFO size,
  L1 vs L1_SMALL)
- DRAM-bank-to-receiver mappings
- GCB FIFO size / page size
- tensor-prefetcher `start` / `queue` / `stop` lifecycle (or the combined
  `tensor_prefetcher_matmul`)
- receiver-contiguous (`NdShardSpec`) DRAM weight layout, distinct from today's
  width-sharded-over-banks

The geometry an implementation must satisfy is fully specified by
`validate_dram_sender_global_cb_gather_in0_geometry` (`matmul_device_operation.cpp:894`):
`num_recv == num_senders * recv_per_bank == ring_size` (the in0 grid),
`weight_K_tiles % ring_size == 0`, `weight_N_tiles % num_senders == 0`,
`per_bank_N % recv_per_bank == 0`, `per_core_N == per_bank_N / recv_per_bank`, and bank *b*'s
receivers pinned to ring positions `[b*recv, (b+1)*recv)`. For down_proj with 8 banks that
admits exactly one point: **ring_size 8, 8 senders, 1 receiver/bank, `per_core_N = 8`**.

### Hardware caveat, scoped correctly

The **DRISC/DRAM-sender** flavour is unavailable on this card: `IsTensorPrefetcherSupported`
is just `hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)`
(`tensor_prefetcher_manager.cpp:1183`) and returns false here. That takes out the tensor
prefetcher *and* every geometry-validated constructor, since
`create_global_circular_buffer_for_matmul_1d`, `_recv_contig` and `_with_dram_senders` all
build DRAM-sender GCBs.

It does **not** rule out GCB itself. The worker-sender variant
(`ttnn.create_global_circular_buffer`) predates the DRISC work and the matmul still accepts it
— `matmul_device_operation.cpp:1792` branches on `sender_core_type(global_cb) == DRAM` to
choose a validator. That path is untested here: it has no geometry cross-validation, and
metal's own comment (line 891) notes the validator exists to catch "silent-hang configs where
the matmul reads more in1 pages than the prefetcher pushes", so a hand-rolled mapping risks
hanging the device. Worth a guarded prototype, not a blind one.

## Recommendations

1. **For this model today, turn DS off.** 13.88 ms vs 21.96 ms per decode step. No DS
   configuration tt-mlir can emit beats that.
2. **Make `kNumIn0Cores` per-op anyway** — worth 5.43 ms/iteration (down_proj 236.3 -> 85.6 us,
   step 21.96 -> ~16.5 ms, ceiling 45.5 -> ~60 it/s). Feed the *existing* `in0BlockW` search
   `numIn0Cores = 43` and it selects `w=8` unaided; no change to that loop. It cannot be a
   constant bump: the gate at `MatmulRules.cpp:219` requires
   `(K/kTileSize) % kNumIn0Cores == 0`, and 43 would decline the K=2048 shapes
   (`64 % 43 != 0`). It must be a search over divisors of `kTiles`, picking the one whose
   `kPerCore` admits the largest CB-affordable `in0BlockW`. Note this is a consolation prize,
   not a path to parity.
3. **The real work is GCB + `gather_in0` representation**, not DS parameter tuning. Prototype
   worker-sender GCB on the down_proj shape first to confirm it clears 65.6 us before
   committing the multi-layer compiler change.
4. **Latent compile-failure risk.** For down_proj at `w=43` tt-mlir's CB model computes 1418.5K
   against ~1427K usable — it believes the config fits by ~9 KB, while metal rejects that exact
   config with a 41 KB L1 clash. Whether it is reachable depends on what the caller passes as
   `l1Available`, which has not been traced.

## Reproduction

```bash
export TT_METAL_PROFILER_TRACE_TRACKING=1        # else post-processing dies on traced graphs
ttmlir-opt --ttnn-common-to-runtime-pipeline graph.mlir -o runtime.mlir
ttmlir-translate --ttnn-to-flatbuffer runtime.mlir -o graph.ttnn
ttrt perf graph.ttnn --loops 1 --trace-region-size 268435456 \
    --enable-program-cache --ignore-version
python -O -c "from tracy.process_ops_logs import process_ops; process_ops(None,None,False)"
python percore_perf.py  --device-log .logs/profile_log_device.csv --ops-data .logs/tracy_ops_data.csv
python matmul_detail.py --percore percore_ops.csv --report reports/ops_perf_results.csv
```

Four flags are load-bearing and three of them are silently wrong by default:
`--enable-program-cache` (ttrt registers bool args with `action="store_true"` and drops the
registered `default=True`, `run.py:1249`), `--trace-region-size` (`run.py:588` assigns it
unconditionally so the default 0 beats metal's own default), `TT_METAL_PROFILER_TRACE_TRACKING`
(metal emits trace END unguarded but BEGIN/REPLAY only under this flag, so post-processing
dies with `KeyError: 0`), and `python -O` for post-processing
(`process_ops_logs.py:606` aborts the whole report when any op lacks device data — 26 of 3557
here).

Scripts: `gen_downproj_tests.py` (single-op config sweep generator),
`run_downproj_sweep.sh`, `percore_perf.py` (epoch-safe durations),
`matmul_detail.py` (per-matmul join with shapes and program configs).
