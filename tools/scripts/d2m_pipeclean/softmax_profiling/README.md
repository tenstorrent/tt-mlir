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

## Device-side profiling — STATUS (UNVERIFIED) & the task

Honest state (2026-06-09): I tried to get device-side profiling working but **could not get a
clean result, and my attempts were contaminated** — so treat the conclusions below as
UNVERIFIED, not as "Tracy is blocked."

What I actually observed:
- Rebuilt with `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON` (→ tt-metal `ENABLE_TRACY=ON`). Build is fine.
- **Host-side Tracy works:** `capture-release` attached connected and produced a ~30 KB `.tracy`
  trace. So Tracy host instrumentation is functional.
- Device runs under the profiler **hung** and produced no `profile_log_device.csv` — BUT the
  test environment was polluted by **stale processes holding the `CHIP_IN_USE_0_PCIe` lock**
  (orphaned python from `timeout`'d runs; `timeout` SIGTERM doesn't kill device-blocked python).
  A hung run that holds the chip lock makes EVERY subsequent run hang waiting for the lock —
  which is indistinguishable from a "profiler hang" unless you check. I did not isolate cleanly,
  so **I cannot claim device profiling is blocked.** It may well work in a clean process.

Toggling `ENABLE_TRACY` does NOT recompile tt-metal (compile commands identical on/off), so the
libs are not the differentiator. The branch ships with PERF_TRACE OFF.

### FIRST task for Arsen: re-test device profiling CLEANLY
1. Kill any stale chip-lock holders BEFORE every device run:
   `for p in $(ps -eo pid,cmd | grep -E "[v]bench.py|[p]robe.py|[c]apture-release" | awk "{print \$1}"); do kill -9 $p; done`
   then `python -m tt_smi -r`. Use `timeout -s KILL` (not plain `timeout`) so runs never orphan.
   If a run hangs, check `ps`/the log for `Waiting for lock 'CHIP_IN_USE...'` — that's a stale
   process, NOT a real device/profiler hang.
2. Rebuild `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON`, then on a CLEAN device run one case with
   `TT_METAL_DEVICE_PROFILER=1` and look for `.../generated/profiler/.logs/profile_log_device.csv`.
   Determine for real whether device profiling works or hangs.

### Possible (not confirmed) risk
The eth `idle_erisc` firmware on bundle 19.7.0 is near its code-size limit (the `TT_METAL_WATCHER`
hits an overflow — see memory note `sdpa-4x4-eth-fw-overflow`). IF device profiling genuinely
hangs after the clean re-test above, this is the prime suspect; mitigations: a firmware bundle
where it fits (19.7.0 warns it's newer than the tested 19.5.0), build the eth dispatch kernel
`-Os`, or a no-eth-dispatch (single-chip) config.

### Then
3. Per-op **device duration** for the 3 cases (the apples-to-apples number isolating on-device
   kernel time from host dispatch). NOTE: at 1–9 tiles the wall-clock gap is ~100% fixed
   per-submit overhead (see overhead decomposition above) — so the interesting signal is the
   **dispatch/launch path**, not compute.
4. **Host-side profiling** (works today, no device instrumentation): `capture-release` host
   zones; split host `submit()` vs `wait()` (host-enqueue vs device-round-trip).
5. Device instrumentation for d2m kernels (`DeviceZoneScopedN`) for per-stage timing.
