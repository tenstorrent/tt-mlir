# Softmax profiling: ttnn vs d2m (fused / unfused)

Goal: compare a numerically-stable softmax on a **single Tensix core** across three
implementations, profile where the time goes (device-side, per op), and build out
reusable device-profiling tooling.

## The three cases (all stable softmax, dim=1, fp32, single core)

| case | what it is | how it's built |
|---|---|---|
| **ttnn** | ttnn's native `ttnn.softmax` | `ttir.softmax` → `ttir-to-ttnn-backend-pipeline` → flatbuffer |
| **d2m fused** | one `d2m.generic` containing tilize + softmax math + untilize | hand-authored MLIR in `mlir/d2m_fused_*.mlir` → d2m backend pipeline → flatbuffer |
| **d2m unfused** | the natural d2m-jit lowering: **4 generics** (zeros, tilize, compute, untilize) | the `probe.py` DSL kernel (no hand-authoring) |

Sizes: `1x1` (32x32, 1 tile), `2x2` (64x64, 4 tiles), `3x3` (96x96, 9 tiles).

### FAIRNESS NOTE (important — and now resolved with device data)
- Both sides run **numerically-stable** softmax (max-subtract, 2 reductions). The default
  `ttir.softmax`→ttnn lowering emits `numericStable = false` — we flip it to `true`
  (`mlir/ttnn_softmax_*.mlir` already have it). Do not benchmark without this.
- Both run on a single core. ttnn keeps the tensor DRAM-interleaved; d2m uses L1-sharded.
- The original worry was "ttnn doesn't tilize/untilize/tile_fill but d2m fused does."
  **Device profiling shows that's not what happens:** ttnn's compiled program *also*
  decomposes into `Tilize → Softmax → Untilize` (three separate device ops). So the
  apples-to-apples "just the softmax math" number is **ttnn `SoftmaxDeviceOperation`** vs
  **d2m unfused compute generic** — both with tilize/untilize excluded (see table below).
  `tile_fill` (reduction scalers) lives inside d2m's compute generic but is part of the
  reduction; ttnn's softmax kernel fills scalers internally too, so it is not an unfairness.

## Device-side profiling — STATUS: VERIFIED WORKING (2026-06-09)

The handoff's "device profiling hangs" was **misdiagnosed**. The real failure was a
**system-desc mismatch**, not a hang. `ttrt perf` works cleanly. Root cause and recipe:

- `ttrt perf` sets `TT_METAL_DEVICE_PROFILER=1`, which **reserves a 64-byte profiler buffer
  in L1**, bumping `l1_unreserved_base` 103776 → 103840. A flatbuffer compiled *without* the
  profiler embeds the smaller base, so (a) it fails ttrt's system-desc compatibility check,
  and (b) its compile-time L1 addresses can overlap the profiler reservation.
- Fixes, per case:
  - **d2m unfused** (d2m-jit, captures the live device desc): generate it **under**
    `TT_METAL_DEVICE_PROFILER=1` so addresses sit above the profiler buffer. Then it
    profiles cleanly with no overrides. (See `gen_cases.sh` / commands below.)
  - **ttnn**: run perf with `--ignore-version`. Safe — ttnn allocates L1 dynamically at
    runtime, so the embedded base is only used for the (cosmetic) compatibility check.
  - **d2m fused**: run perf with `--ignore-version`. Safe — the hand-authored source already
    carries `l1_unreserved_base=103840` (== profiler device); the only desc diffs are in
    DRAM base / bank-mapping / capabilities, which a pure-L1 single-core run never touches.
  - Always pass `--fabric-config disabled` (these are single-chip; `fabric_1d` reserves
    extra L1/DRAM for nothing).
- Host-side Tracy (`capture-release`) and device CSV (`profile_log_device.csv`,
  `ops_perf_results.csv`) are both produced. `-DPROFILE_KERNEL=1` is compiled into the
  kernels, confirming device instrumentation is live.

Note: the eth `idle_erisc` fw-overflow risk flagged in the prior handoff did **not**
materialize for these single-core single-chip cases.

## Device kernel duration results (ns, single core, profiler-enabled)

Per-op `DEVICE KERNEL DURATION [ns]` from `ttrt perf` (`devperf.py`):

| | tilize | softmax math | untilize | zeros/init | **total** |
|---|---|---|---|---|---|
| **ttnn 1x1** | 3082 | **5614** | 4142 | — | 12838 |
| **ttnn 2x2** | 3516 | **6779** | 5147 | — | 15442 |
| **ttnn 3x3** | 4097 | **7704** | 5808 | — | 17609 |
| **d2m-unfused 1x1** | 737 | **10118** | 923 | 555 | 12333 |
| **d2m-unfused 2x2** | 1127 | **26244** | 1368 | 926 | 29665 |
| **d2m-unfused 3x3** | 1803 | **51072** | 2110 | 1605 | 56590 |
| **d2m-fused 1x1** | ←──── all bundled in one generic ────→ | | | | 11545 |
| **d2m-fused 2x2** | | | | | 28042 |
| **d2m-fused 3x3** | | | | | 54640 |

### The fair comparison: softmax math only (tilize/untilize excluded both sides)

| tiles | ttnn `Softmax` | d2m compute generic | d2m / ttnn |
|---|---|---|---|
| 1 (1x1) | 5614 | 10118 | 1.8× |
| 4 (2x2) | 6779 | 26244 | 3.9× |
| 9 (3x3) | 7704 | 51072 | 6.6× |

## Findings

1. **At 1 tile, total device kernel time is ~equal (~12 µs) across all three.** The huge
   wall-clock gap is almost entirely **host dispatch/launch overhead**, not device compute —
   ~98–99% of submit→wait for every case (panel A). Current reproduced warm wall-clock
   (`bench.py` and `plot_perf.py` agree): ttnn ~730 µs, d2m-fused ~2030 µs, d2m-unfused
   ~5300 µs; subtract the ~12–57 µs device kernel time and the rest is host. (The prior
   handoff's 68/469/1271 µs were a lighter-loaded box; these numbers are host-Python-bound
   and load-sensitive — the *ordering and the host-bound conclusion* are stable, the absolute
   µs are not. The device-kernel numbers, from tracy, are deterministic and are the real
   signal.) d2m's per-submit host cost scales with the number of generics (~270 µs each),
   which is why unfused (4) ≫ fused (1) ≫ ttnn.

2. **The real compute signal is in how the softmax math scales on one core.** ttnn's softmax
   kernel scales sub-linearly (5.6→7.7 µs from 1→9 tiles); d2m's compute generic scales
   super-linearly (10→51 µs). So the fair "just the math" gap widens from **1.8× → 6.6×**.
   ttnn's single-core multi-tile reduction is far more efficient — this is where d2m should
   focus, not on dispatch or on tilize/untilize.

3. **d2m's tilize/untilize are actually *cheaper* than ttnn's** (0.7–2 µs vs 3–6 µs), because
   d2m runs L1-sharded while the ttnn case is DRAM-interleaved. Excluding tilize/untilize
   (the "fair" view) therefore *removes a d2m advantage* and isolates the compute deficit.

4. **Fusion helps dispatch, not compute.** d2m-fused vs d2m-unfused total device time is
   within noise (e.g. 54.6 vs 56.6 µs at 3x3); fusion mainly removes 3 extra kernel launches
   (the host-side win). The compute generic dominates either way.

## Root cause of the compute gap: 2D-loop fusion vs DST-register fusion (VALIDATED)

Hypothesis (Arsen): ttnn fuses at the **DST/tile-register** level (chain math ops on a tile
while it's in registers), whereas d2m fuses **whole 2D loops** (one loop nest per op, each
materializing its intermediate back to L1). Validated two ways.

### Quantitative — marginal device-kernel cost per *added* tile (slope, 1→9 tiles)

Using `probe.py {ident,exp,softmax}` (compute generic only; `ident` = load→store, no math):

| compute kernel | 1 tile | 4 tiles | 9 tiles | **ns / extra tile** |
|---|---|---|---|---|
| identity copy (no math) | 563 | 967 | 1651 | **~136** |
| exp (1 math op) | 3410 | 11915 | 25561 | **~2770** |
| softmax (full chain) | 10136 | 26206 | 51011 | **~5100** |
| ttnn softmax (contrast) | 5614 | 6779 | 7704 | **~261** |

A bare tile-loop with no math is ~136 ns/tile, so the per-tile cost is **all in the math
passes**: d2m pays ~5100 ns/tile for softmax vs ttnn's ~261 ns/tile (**~20×**), and even a
single `exp` adds ~2770 ns/tile.

### Structural — the generated compute kernels

**d2m** (`ttmlir_main_compute_kernel5`, the unfused softmax-math generic) is **7 separate
loop nests**, one per tile op, each ending in `pack_tile` to a distinct L1 circular buffer,
with `experimental::unpack_stall_on_pack()` — a hard pack→L1→unpack RAW barrier — between
*every* stage:

```
fill max-scaler → cb1 | reduce_max → cb3 | bcast+sub → cb5 | exp → cb4 |
fill sum-scaler → cb0 | reduce_sum → cb6 | bcast+div → cb_out
```

Each loop does `tile_regs_acquire(); <math>; tile_regs_commit(); pack_tile(...); tile_regs_release()`
on **one tile at a time**, the DST register is released between ops (never holds an
intermediate), and HW is re-init'd per op (`reduce_init`/`reduce_uninit` even wrap each
individual `reduce_tile`). So each tile is read-from-L1 / computed / written-to-L1 ~7 times,
fully serialized.

**ttnn** (`ttnn/.../normalization/softmax/device/kernels/attention/compute/softmax.cpp`) does
**4 passes** and chains in registers: `(x − max)` and `exp` run inside *one*
`tile_regs_acquire/commit` block with no L1 round-trip between them; the `recip` is fused as
a post-op of the sum-reduce; it processes `ndst` tiles per acquire (block batching); and it
uses `exp_tile<EXP_APPROX>` (approximate exp) vs d2m's full-precision `exp_tile`.

### Decomposing the d2m softmax compute: structure vs exp-fidelity
`(exp − ident)` isolates the exp op; `(softmax − exp)` isolates the rest of the chain
(reduces/sub/div/fills). Both are large; **structure alone dwarfs ttnn**:

| | copy floor | exp (fidelity lever) | other math (structure lever) | ttnn total math |
|---|---|---|---|---|
| 1x1 | 563 | 2847 | 6726 | 5614 |
| 2x2 | 967 | 10948 | 14277 | 6779 |
| 3x3 | 1651 | 23910 | 25450 | 7704 |

So there are **two** levers: ~half the cost is the full-precision `exp_tile` (ttnn uses
`EXP_APPROX`), and the other half is the materializing per-op loop structure — and that
"other math" *by itself* (25450 ns at 9 tiles) already exceeds ttnn's *entire* softmax
(7704 ns). Even a free exp would leave d2m structurally ~3× slower.

See **`softmax_perf.png`** (generated by `plot_perf.py`) — 4 panels: (A) wall-clock =
device + host-dispatch, (B) math-only scaling, (C) marginal ns/tile (~20×), (D) this
decomposition.

### What this means for d2m
The gap is **not** dispatch, tilize/untilize, or tile_fill — it's that the
`d2m.generic` → TTKernel lowering emits one materializing loop per tile op with a serializing
L1 barrier between them. To close it: fuse consecutive elementwise tile ops into a single
`tile_regs_acquire` region (keep intermediates in DST), batch `ndst` tiles per acquire, drop
the per-op `reduce_init`/reconfig where the config is unchanged, and consider `EXP_APPROX`.
NOTE: this is single-core / multi-*tile* scaling; true multi-*core* (grid > 1x1) scaling is a
separate axis not measured here.

## Files
- `mlir/d2m_fused_{1x1,2x2,3x3,4x4}.mlir` — hand-authored fused single-generic sources
- `mlir/ttnn_softmax_{1x1,2x2,3x3}.mlir` — ttnn-lowered sources (numericStable already true)
- `gen_cases.sh` — regenerates ttnn + d2m-fused flatbuffers into `./_fb/`
- `probe.py <mode> <N>` — the d2m-jit DSL path (`softmax`/`exp`/`ident`); set
  `D2M_JIT_SAVE_FLATBUFFER_PATH` to also emit a flatbuffer
- `devperf.py` — **device-profiling sweep**: runs `ttrt perf` on every case and tabulates
  per-op `DEVICE KERNEL DURATION`. This is the main tool.
- `plot_perf.py` — builds **`softmax_perf.png`** (4 panels) from the device CSVs + live
  warm wall-clock: dispatch-vs-device, math scaling, marginal ns/tile, compute decomposition.
- `bench.py` / `vbench.py` — host wall-clock (submit→wait) latency + PCC-vs-torch
- `softmax_perf.png` — the device-profiling figure (this run)
- `softmax_trendline.png` — the older wall-clock plot

## How to run (exact)
```bash
cd <repo>            # tt-mlir root
source env/activate
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys   # ttrt query --save-artifacts first

# 1) ttnn + d2m-fused flatbuffers
bash tools/scripts/d2m_pipeclean/softmax_profiling/gen_cases.sh

# 2) d2m-unfused flatbuffers — MUST be generated under the profiler so L1 addresses
#    sit above the profiler buffer (otherwise ttrt perf mismatches / risks overlap):
for n in 1 2 3; do
  TT_METAL_DEVICE_PROFILER=1 \
  D2M_JIT_SAVE_FLATBUFFER_PATH=$(pwd)/_fb/d2m_unfused_${n}x${n}_prof.ttm \
    python3 tools/scripts/d2m_pipeclean/softmax_profiling/probe.py softmax $n
done

# 3) device-profiling sweep -> per-op device kernel durations + summary table
python3 tools/scripts/d2m_pipeclean/softmax_profiling/devperf.py

# (correctness/host-latency for one case, no profiler)
python3 tools/scripts/d2m_pipeclean/softmax_profiling/vbench.py _fb/d2m_fused_2x2.ttm 64
```

Per-case device CSVs land in `perf_artifacts/<binary>/perf/ops_perf_results.csv`.

### Reproduce the root-cause validation
```bash
# marginal-per-tile microbench: generate ident/exp/softmax under the profiler, then perf each
for mode in ident exp softmax; do for n in 1 2 3; do
  TT_METAL_DEVICE_PROFILER=1 D2M_JIT_SAVE_FLATBUFFER_PATH=$(pwd)/_fb/d2m_${mode}_${n}x${n}_prof.ttm \
    python3 tools/scripts/d2m_pipeclean/softmax_profiling/probe.py $mode $n
  ttrt perf _fb/d2m_${mode}_${n}x${n}_prof.ttm --fabric-config disabled   # compute generic = 3rd / largest op
done; done

# dump the generated compute kernels to /tmp to read the loop structure
ttrt perf _fb/d2m_softmax_2x2_prof.ttm --fabric-config disabled --dump-kernels
#   /tmp/ttmlir_main_compute_kernel5_*.cpp  -> the 7-loop softmax math
# ttnn contrast: third_party/tt-metal/.../normalization/softmax/device/kernels/attention/compute/softmax.cpp
```

## Next steps (not done here)
- Per-stage device zones (`DeviceZoneScopedN`) inside the **fused** generic, to split its
  single bundled number into tilize / math / untilize without relying on the unfused split.
- Host-side split of `submit()` vs `wait()` to quantify the dispatch overhead from finding 1.
- Investigate the d2m single-core reduction kernel's super-linear scaling (finding 2).
