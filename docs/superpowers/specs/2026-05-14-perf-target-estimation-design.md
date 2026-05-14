# Perf Target Estimation in `TTNNCollectPerfMetrics`

Single-chip, first-cut performance ceiling estimator that lives inside the existing TTNN perf metrics pass and emits its result alongside the existing JSON report.

## Goal

Produce a per-graph estimate of the theoretical top performance achievable on one Wormhole B0 (n150) chip, computed from:

- Total parameter footprint (weights + constants).
- KV cache footprint, if any.
- DRAM bandwidth roofline.
- Compute roofline (matmul/conv-only FLOPs).
- A simple decision rule for DRAM-bound vs compute-bound.
- A top perf estimate in samples/sec or iterations/sec.

Compare the result against `tt-metal/models/README.md` published targets to see how close a naive roofline gets.

## Where it lives

`lib/Dialect/TTNN/Transforms/TTNNCollectPerfMetrics.cpp` already walks the forward function once and writes a JSON report. We add an additional pass over function arguments + ops to compute the new section, and merge it into the same JSON under a new `perf_targets` key.

Reasons to extend rather than create a new pass:
- Same code already iterates the right function (`isForwardDeviceFunc` or `isTraceMainFunc`).
- Same output file, so downstream consumers (e.g. tt-xla's `aggregate_ttnn_perf_metrics`) only need to handle one file.
- Sharing the gate (`ttnn-perf-metrics-enabled`) keeps the pipeline simple.

## Inputs we already have in IR

- `SystemDescAttr` is attached to the module — `ttcore::getCurrentScopeSystemDesc(module)` returns it. We use `chipDescs[0]` and warn if there is more than one chip (single-chip scope).
- Function arguments are tagged with `ttcore.argument_type` (Input / Parameter / Constant / Default) — see `ttmlir/Dialect/TTCore/IR/Utils.h` (`getFunctionArgumentType`, `isInputArgumentType`, etc.).
- KV cache args are tagged with the `ttcore.kv_cache` unit attribute, populated by `TTIRInferKVCacheArgumentTypes`. `isKVCacheArgument(funcOp, argIndex)` returns true on those.
- Tensor element sizes come from `ttcore::getElementSizeBytes(Type elementType)` which handles both raw scalar element types and the tiled `TileType`.

## Hardware constants

`ChipDescAttr` does not include DRAM bandwidth or peak FLOPS. We hardcode by arch (matches the values published in `models/demos/deepseek_v3/reference/ttnn.md` and the Wormhole architecture docs):

| Arch | DRAM BW | Peak BF16 FLOPS (LoFi) | Notes |
|---|---|---|---|
| WormholeB0 | 288 GB/s | 4 TFLOPS/engine × grid (8×8 = 64 cores by default) ≈ 256 TFLOPS | Use `chipDesc.grid` to pick up actual worker grid. |
| Blackhole | 512 GB/s | placeholder, refined later | Out of scope for first cut. |

For LLM decode we know the math fidelity is almost always LoFi or HiFi2 (BFP8 weights), so reporting both LoFi and HiFi2 peak FLOPS in the output makes the result easy to interpret without baking in a single guess.

## What we compute

All quantities are produced once per forward function and serialized under `perf_targets`.

1. **Total params** — count elements of every function arg whose `ttcore.argument_type` is **not** `Input` (so Parameter + Constant + Default), excluding KV cache args.
2. **Total param memory** — sum `numElements × elementSizeBytes` over those args. Memory uses the actual element type that survives the TTNN pipeline (so BFP8 layouts naturally get 1 byte / element via `TileType::getSizeBytes()`).
3. **KV cache memory** — sum `numElements × elementSizeBytes` over args where `isKVCacheArgument` returns true. Reported separately and added on top.
4. **DRAM bandwidth** — hardcoded per arch.
5. **DRAM roofline time (s)** — `(param_memory + kv_cache_memory) / dram_bandwidth`.
6. **Total compute FLOPs** — walked from forward ops. For each matmul-class op we count `2 × output_volume × inner_dim`:
   - `ttnn.matmul` / `ttnn.linear`: `2 × M × N × K` where K is the inner dim of `a` (respect `transpose_a`).
   - `ttnn.sparse_matmul`: same shape but multiplied by `nnz / E` when provided.
   - `ttnn.conv2d`: `2 × N × H_out × W_out × C_in × C_out × K_H × K_W / groups`. H_out/W_out come from the explicit attrs on the op.
   - Anything else: ignored. Higher-level ops like `ttnn.scaled_dot_product_attention` decompose into matmuls during lowering, so by the time TTNNCollectPerfMetrics runs the matmuls inside are already visible as their own ops.
7. **Peak FLOPS** — hardcoded per arch, reported per math fidelity.
8. **Compute roofline time (s)** — `total_flops / peak_flops` (we use LoFi for the headline number).
9. **DRAM-bound vs compute-bound** — `dram_bound iff dram_time > compute_time`.
10. **Top perf time** — `2 × max(dram_time, compute_time)`. The 2× factor reflects observed utilization on real workloads. Implemented in a single named lambda `topPerfTime(bound, dramTime, computeTime)` so we can tweak the formula in one place.
11. **Top perf estimate** — `1 / top_perf_time`, units of samples/sec (or iterations/sec).

## Output schema

Added inside the existing JSON:

```json
"perf_targets": {
  "arch": "wormhole_b0",
  "chip_count_in_system_desc": 1,
  "single_chip_assumption": true,
  "dram_bandwidth_bytes_per_sec": 288000000000,
  "peak_flops": {
    "lofi": 256000000000000,
    "hifi2": 128000000000000,
    "hifi3": 85333333333333,
    "hifi4": 64000000000000
  },
  "params": {
    "count": 1235814400,
    "memory_bytes": 1235814400,
    "memory_gb": 1.151
  },
  "kv_cache": {
    "memory_bytes": 16777216,
    "memory_gb": 0.0156
  },
  "compute": {
    "total_flops": 2471628800,
    "breakdown": {
      "matmul": 2400000000,
      "linear": 71628800,
      "conv2d": 0
    }
  },
  "roofline": {
    "dram_time_sec": 0.00434,
    "compute_time_sec_lofi": 0.0000097,
    "bound": "dram",
    "top_perf_time_sec": 0.00868,
    "top_perf_samples_per_sec": 115.2
  }
}
```

## tt-xla consumption

`tests/benchmark/utils.py:aggregate_ttnn_perf_metrics` already reads each per-graph JSON and stitches a `config` block onto the benchmark result. We extend it to also forward the new fields (param count, param memory, KV cache memory, total FLOPs, top perf estimate). The new fields are optional so older JSON files keep working.

## Single-chip safety

If `chipDescs.size() > 1` we emit one warning per pass run and continue with `chipDescs[0]`. This matches the user requirement: even on N300 we are using a single chip.

## Out of scope (for this first cut)

- Multi-chip / tensor-parallel rooflines.
- Activation memory (focus is the weight read in decode).
- Real FLOP counting for non-matmul ops.
- A configurable utilization factor (the 2× is hardcoded but in one place).
- Blackhole tuning beyond placeholder constants.

## Validation plan

1. Build tt-mlir with `-DTTMLIR_ENABLE_RUNTIME=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache`.
2. Run a representative subset of single-chip benchmarks in tt-xla (`test_llama_3_2_1b`, `test_llama_3_2_3b`, `test_llama_3_1_8b`, `test_mistral_7b`, `test_qwen_2_5_7b`, `test_falcon3_7b`, plus smaller ones for sanity).
3. Pull the `perf_targets` from each per-graph JSON and build a comparison table: our estimate vs `tt-metal/models/README.md` target vs current measured perf.

## Decisions made autonomously

- Use `chipDescs[0]` and warn on multi-chip. Doesn't require multi-chip aggregation.
- Output FLOPS for all four math fidelities so the JSON consumer can pick the relevant one — saves having to rerun for HiFi2 vs LoFi.
- Place the 2× utilization factor in `topPerfTime()` so it can be tuned without hunting through the file.
- Defer activation memory; for the LLM workloads we are validating, decode is dominated by weight + KV cache reads, which matches the published targets.
