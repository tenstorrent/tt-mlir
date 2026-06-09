# Softmax profiling: ttnn vs d2m (fused / unfused)

Goal: compare a numerically-stable softmax on a **single Tensix core** across three
implementations, profile where the time goes, and build out device-side profiling
tooling we can reuse.

## The three cases (all stable softmax, dim=1, fp32, 1x1 core grid)

| case | what it is | how it's built |
|---|---|---|
| **ttnn** | ttnn's native `ttnn.softmax` (1 program) | `ttir.softmax` → `ttir-to-ttnn-backend-pipeline` → flatbuffer |
| **d2m fused** | one `d2m.generic` containing tilize + softmax math + untilize | hand-authored MLIR in `mlir/d2m_fused_*.mlir` → d2m backend pipeline → flatbuffer |
| **d2m unfused** | the natural d2m-jit lowering: **4 generics** (zeros, tilize, compute, untilize) | the `probe.py` DSL kernel (no hand-authoring) |

Sizes: `1x1` (32x32), `2x2` (64x64), `3x3` (96x96) tiles.

### FAIRNESS NOTE (important)
- Both sides must run **numerically-stable** softmax (max-subtract, 2 reductions). The
  default `ttir.softmax`→ttnn lowering emits `numericStable = false` (1 reduction, no
  max-sub) — we flip it to `true` (see `gen_cases.sh`). Do not benchmark without this.
- Both run on a single core (1x1 grid). ttnn keeps the tensor DRAM-interleaved; d2m uses
  L1-sharded. Note this when comparing.

## Current results (wall-clock, warm host submit->wait, µs)

| | 1x1 | 2x2 | 3x3 |
|---|---|---|---|
| ttnn | 68 | 69 | 63 |
| d2m fused | 469 | 483 | 469 |
| d2m unfused (4 generics) | 1271 | 1343 | 1399 |

Overhead decomposition (fit latency = fixed + slope*tiles): **at these sizes everything is
~100% fixed per-submit overhead; compute is in the noise.** So the gap is dispatch/launch,
NOT compute-kernel efficiency, at 1-9 tiles. See `softmax_trendline.png`.
**This is exactly why we need device-side profiling** — to separate real on-device kernel
time from host/runtime artifacts (data return, layout conversion, program-cache effects).

Open question (4x4 / 16 tiles): the **d2m unfused** path *hangs* at 4x4 on a single core
(launch-level: workers never receive a kernel, `k_ids:0`); the **d2m fused** 4x4 case
(`mlir/d2m_fused_4x4.mlir`) is the in-progress check for whether the fused path avoids it.

## Files
- `mlir/d2m_fused_{1x1,2x2,3x3,4x4}.mlir` — hand-authored fused single-generic sources
- `mlir/ttnn_softmax_{1x1,2x2,3x3}.mlir` — ttnn-lowered sources (numericStable already true)
- `gen_cases.sh` — regenerates all flatbuffers from the MLIR sources
- `bench.py` — warm submit->wait latency for a list of flatbuffers
- `vbench.py <flatbuffer> <S>` — PCC-vs-torch + latency for one flatbuffer
- `probe.py <mode> <N>` — the d2m-jit DSL path (`softmax`/`exp`/`ident`), runs on device, prints PCC
- `softmax_trendline.png` — the plot

## How to run (exact)
```bash
cd <repo>            # tt-mlir root
source env/activate
# device reset if wedged:
python -m tt_smi -r

# 1) regenerate all flatbuffers (ttnn + d2m fused) into ./_fb/
bash tools/scripts/d2m_pipeclean/softmax_profiling/gen_cases.sh

# 2) correctness + latency for one case
python3 tools/scripts/d2m_pipeclean/softmax_profiling/vbench.py _fb/d2m_fused_2x2.ttm 64

# 3) the unfused (d2m-jit) path, prints PCC directly
python3 tools/scripts/d2m_pipeclean/softmax_profiling/probe.py softmax 2
```

## Device-side profiling — STATUS & the actual task

What I tried (2026-06-09) and found:
- Rebuilt with `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON` (→ tt-metal `ENABLE_TRACY=ON`). Build is fine.
- **Host-side Tracy works:** running with `capture-release` attached produced a ~30 KB
  `.tracy` trace (host zones: dispatch/submit/etc.). Useful for host-side breakdown.
- **Device-side profiling is BLOCKED on this setup:** with the profiler build, the device
  **program launch hangs** — for BOTH ttnn and d2m, with and without
  `TT_METAL_DEVICE_PROFILER=1`, on a clean reset device (verified with `timeout -s KILL`,
  so not zombie contention). No `profile_log_device.csv` is produced. The profiler build
  bakes device-side instrumentation into the dispatch/eth firmware, which overflows the eth
  `idle_erisc` code region — the SAME failure family as `TT_METAL_WATCHER` (see memory note
  `sdpa-4x4-eth-fw-overflow`): firmware bundle 19.7.0's `idle_erisc` is already ~at the
  limit, and any extra instrumentation tips it over → device wedge at launch.
- Because of this, the profiler build also breaks NORMAL device runs. **This branch's build
  is configured with PERF_TRACE OFF** (working device); flip it ON only to reproduce the
  hang / continue the investigation.

Reproduce the device-profiler hang:
```bash
cmake -B build -DTT_RUNTIME_ENABLE_PERF_TRACE=ON && cmake --build build
python -m tt_smi -r
TT_METAL_DEVICE_PROFILER=1 python3 .../vbench.py _fb/ttnn_softmax_1x1.ttnn 32   # hangs at launch
```

Tasks for Arsen:
1. **Unblock device profiling** — the eth `idle_erisc` overflow is the wall. Options to try:
   (a) a firmware bundle where `idle_erisc` fits with instrumentation (the system warns 19.7.0
   is "newer than latest fully tested 19.5.0"); (b) build the eth dispatch kernel `-Os`;
   (c) a dispatch-core config that doesn't put dispatch on eth (single-chip / no fabric).
2. Once it launches, get **per-op device duration** (Tracy device zones / `profile_log_device.csv`)
   for the 3 cases — the apples-to-apples number that separates on-device kernel time from host
   dispatch. NOTE from this work: at 1–9 tiles the wall-clock gap is ~100% fixed per-submit
   overhead (compute is in the noise — see overhead decomposition above), so device-op timing
   will likely show negligible compute; the interesting signal is in the **dispatch/launch path**.
3. **Host-side profiling that needs no device instrumentation** (works today): run with
   `capture-release` attached and inspect host zones; also split host `submit()` vs `wait()`
   to separate host-enqueue from device-round-trip.
4. What device instrumentation we could add to d2m-generated kernels (`DeviceZoneScopedN`) for
   per-stage timing once the eth-firmware blocker is resolved.
