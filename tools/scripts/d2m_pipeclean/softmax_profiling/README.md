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

## Device-side profiling (the actual task)
Build is configured with `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON` (→ tt-metal `ENABLE_TRACY=ON`).
Tracy `capture-release` lives at
`third_party/tt-metal/src/tt-metal/build_Release/tools/profiler/bin/capture-release`.

Things to investigate / build out:
1. Per-op **device** duration (Tracy device zones) for each of the 3 cases — isolate
   on-device kernel time from host dispatch. This is the number that makes the comparison
   apples-to-apples.
2. Whether enabling the profiler changes the eth/idle-erisc firmware size (the watcher does
   — it overflows `idle_erisc`); confirm the profiler build still launches programs.
3. Split host `submit()` vs `wait()` time (host enqueue vs device round-trip) — a quick
   device-profiler-free signal; see notes in this file's git history.
4. What instrumentation we can add to d2m-generated kernels (DeviceZoneScopedN) for future
   per-stage timing.
