# Falcon3-7B chisel accuracy debug — n150 attempt

**Issue:** tenstorrent/tt-inference-server#4752 — Falcon3-7B-Instruct bf16 weights+KV
scores ~5 pts *lower* on ifeval than bfp8 despite higher precision. Lead:
op-by-op-compare the bf16 vs bfp8 decode graphs with **chisel** to find where the
bf16 KV/weight path diverges.

**Machine used:** single **Wormhole n150** (issue targets **Blackhole p150**).
**Branch:** `dgolubovic/chisel-falcon3-debug`.
**Pins:** tt-mlir HEAD `6b2d83bbee` (issue used `3abca428`); tt-metal `f1f4ff75…`
(matches issue exactly).

---

## ▶ Continue here on p150

Everything needed is committed on this branch: the chisel runner/analyzer, the
cpu-hoist stripper, the compute-op probe, and the IR dumps under
`falcon3_chisel/ir/`. On the p150 machine:

```bash
git checkout dgolubovic/chisel-falcon3-debug
cmake --build build                       # if -lnsl fails, see "Build wall 2" below
source env/activate
export LD_LIBRARY_PATH="$PWD/third_party/tt-metal/src/tt-metal/build_Debug/lib:\
$PWD/build/runtime/lib:$LD_LIBRARY_PATH"
ttrt query --save-artifacts
export SYSTEM_DESC_PATH="$PWD/ttrt-artifacts/system_desc.ttsys"

# bf16 graph, op-by-op PCC
python3 falcon3_chisel/strip_cpu_hoist.py \
  falcon3_chisel/ir/notrace/falcon3_7b_bf16_1layer_notrace_decode_ttnn.mlir \
  falcon3_chisel/bf16_1layer_stripped.mlir
python3 falcon3_chisel/run_chisel.py falcon3_chisel/bf16_1layer_stripped.mlir \
  -o falcon3_chisel/bf16.chisel.jsonl
python3 falcon3_chisel/analyze_chisel.py falcon3_chisel/bf16.chisel.jsonl

# same for the bfp8 baseline, then diff the per-op PCC tables
python3 falcon3_chisel/strip_cpu_hoist.py \
  falcon3_chisel/ir/notrace/falcon3_7b_baseline_bfp8_1layer_notrace_decode_ttnn.mlir \
  falcon3_chisel/bfp8_1layer_stripped.mlir
python3 falcon3_chisel/run_chisel.py falcon3_chisel/bfp8_1layer_stripped.mlir \
  -o falcon3_chisel/bfp8.chisel.jsonl
python3 falcon3_chisel/analyze_chisel.py falcon3_chisel/bfp8.chisel.jsonl
```

Focus the comparison on the attention/KV path (see "Prioritized suspects"): the bf16
compute ops (matmuls, rms_norm) are already confirmed clean on n150, so the divergence
is expected in **SDPA-decode / paged_update_cache** with bf16 K/V. On p150 the KV cache
and the `<11,10>` SDPA grid fit, so these graphs should run as-is.

For a real end-to-end model PCC, drive the traced decode graph with realistic
page-table/position inputs instead of the random indices `run_chisel.py` generates
(random inputs still give valid *isolated* per-op PCC, but not a meaningful e2e number).

---

## TL;DR

- The chisel harness is **built and verified working on n150** (builder test passes;
  produces isolated + accumulated per-op PCC).
- I could **not** run the *decode graphs from the issue* on n150. They are compiled
  artifacts specialized for **Blackhole**, and hit two hard walls on Wormhole:
  1. **DRAM capacity** — the two full-32K-context paged KV caches are **3.42 GB each**;
     their on-device layout-conversion peak is **6.85 GB**, and with weights already
     resident the 2nd cache OOMs on n150 (~12.8 GB DRAM). *Clean `TT_FATAL: Out of
     Memory` — see evidence below.*
  2. **Blackhole core grid baked into the IR** — SDPA `compute_with_storage_grid_size
     = <11,10>` and the L1-sharded layouts use core ranges up to column 10 (11 wide),
     exceeding n150's 8×8 grid.
- These are exactly why the issue specifies p150. **Op-by-op device validation of the
  attention/KV path needs the p150 you're setting up.**
- What I *could* run on n150 (compute ops, no KV cache / no blackhole sharding) gives a
  useful negative result: **the bf16 matmuls and rms_norm are numerically clean
  (PCC > 0.999).** The bf16 gap is therefore very likely in the **attention/KV-cache
  path**, not the GEMMs — which matches the issue's primary lead.
- **e2e model PCC: not measurable on n150** (the decode graph can't execute here).

---

## What was done (process log)

### 1. Build (three environment walls cleared)
- **tt-metal submodule fetch failed** (`upload-pack: not our ref` for umd/tracy).
  Fixed by checking out the pinned `f1f4ff75…` and updating submodules **offline**
  (all needed commits were already local); the ExternalProject git-update then
  short-circuits (HEAD == target).
- **Linker: `unable to find library -lnsl`** (no `libnsl-dev`; only `libnsl.so.1`
  present). Fixed with a shim symlink `falcon3_chisel/libshim/libnsl.so ->
  /lib/x86_64-linux-gnu/libnsl.so.1` and `LIBRARY_PATH`.
- Remaining build "failures" are only **gtest test-discovery** steps (they run test
  exes at configure time without `libTTMLIRRuntime.so` on the path) — harmless; all
  libraries + Python packages (`chisel`, `ttmlir`, `golden`, `_ttmlir_runtime`) build
  and import.
- Runtime libs live in `third_party/tt-metal/src/tt-metal/build_Debug/lib` and
  `build/runtime/lib` — must be on `LD_LIBRARY_PATH`.

### 2. Chisel harness understood + runner written
Chisel installs runtime debug hooks (`bind`/`session`) that fire per TTNN op while a
flatbuffer executes, computing a **torch golden per op** and recording PCC:
- **isolated** = golden from that op's actual device inputs (per-op fidelity),
- **accumulated** = golden chained from graph inputs (cumulative divergence).

Tools written (in `falcon3_chisel/`):
- `run_chisel.py` — translate a TTNN `.mlir` → flatbuffer, open device, bind a chisel
  session, generate inputs, submit, dump `*.chisel.jsonl`.
- `analyze_chisel.py` — rank problematic ops by worst PCC / non-pass status.
- `strip_cpu_hoist.py` — see below.
- `probe_compute_ops.py` — builder-driven chisel probe of the compute ops.

Verified end-to-end on n150: `test_chisel_records_one_layer_nn` **passes**.

### 3. IR preprocessing
The decode graphs carry a CPU-hoisted const-eval (`ttcore.cpu_module`, `enable_const_eval=True`)
containing a TTIR op (`ttir.reshape`), which `ttmlir-translate --ttnn-to-flatbuffer`
cannot parse (TTIR dialect not registered). Since it is a numeric no-op (reshape
128→1×1×1×128), `strip_cpu_hoist.py` rewrites it to an on-device `ttnn.reshape` and
removes the cpu module. After stripping, **translation to flatbuffer succeeds.**

### 4. Execution attempt on n150 → the two walls (with evidence)

Translation OK; on submit, `to_device` of inputs fails. Isolating input-by-input:

```
TT_FATAL: Out of Memory: Not enough space to allocate 6845104128 B DRAM buffer
across 12 banks ... (allocated: 709919552 B/bank, free: 360853632 B/bank)
  at BankManager::allocate_buffer  <- allocate_device_buffer  <- MeshTensor::allocate_on_device
```
Triggered converting **arg9 (2nd KV cache)** at ~7.7 GB cumulative resident.

Program input inventory (single decoder layer, decode):
```
arg5  embed_tokens         131072x3072 bf16   805 MB
arg6  kv_cache_1           52224x4x32x256 bf16  3.42 GB   (page table 32x1024 => 32768 ctx)
arg9  kv_cache_0           52224x4x32x256 bf16  3.42 GB
arg12 gate_up_proj          46080x3072 bf16    283 MB
arg11 down_proj              3072x23040 bf16    142 MB
... (other weights/inputs small)          TOTAL ~8.13 GB raw, peak >12.8 GB on device
```

Blackhole grid baked in (would block attention even if memory fit):
```
ttnn.paged_scaled_dot_product_attention_decode ... program_config =
  #ttnn.sdpa_program_config<compute_with_storage_grid_size = <11, 10>, ...>   # n150 is 8x8
#ttnn_layout22 ... core_ranges = <[core_range<(0,0),(10,1)>, core_range<(0,2),(9,2)>]>  # col 10 > 7
```

### 5. What ran on n150 (compute ops)
Builder-constructed Falcon3-shaped ops in bf16, validated by chisel (no KV cache, no
blackhole sharding):

| op (bf16)          | shapes                         | isolated PCC | accumulated PCC |
|--------------------|--------------------------------|-------------:|----------------:|
| qkv_proj matmul    | (32×3072)·(5120×3072)ᵀ         | 0.999985     | 0.999985        |
| o_proj matmul      | (32×3072)·(3072×3072)ᵀ         | 0.999985     | 0.999985        |
| gate_up_proj matmul| (32×3072)·(46080×3072)ᵀ        | 0.999985     | 0.999985        |
| down_proj matmul   | (32×23040)·(3072×23040)ᵀ       | 0.999319     | 0.999319        |
| rms_norm           | (32×3072), eps 1e-6            | 0.999998     | 0.999998        |

All clean. `down_proj` is the lowest (largest K=23040 contraction) but still >0.999.
→ **The bf16 GEMMs/norm are not the divergence source.**

---

## Static bf16-vs-bfp8 differential (the op-by-op comparison, done on IR)

Op structure is identical; only dtype/conversions differ:

| aspect            | bfp8 baseline                        | bf16 config                    |
|-------------------|--------------------------------------|--------------------------------|
| matmul weights    | `bfp_bf8` tiles                      | `bf16`                         |
| KV cache dtype    | `bfp_bf8` tiles                      | `bf16`                         |
| SDPA K/V read     | `bfp_bf8`                            | `bf16`                         |
| extra conversions | more `typecast`/`to_device`/`from_device` | fewer                     |
| rms_norm cfg      | hifi4 + fp32_dest_acc (both)         | hifi4 + fp32_dest_acc (both)   |
| SDPA / matmul fidelity | **no explicit `math_fidelity`** — TTNN default | same |

Notable: **SDPA-decode and the matmuls carry no explicit `math_fidelity`/`fp32_dest_acc`
in the IR** — they run at the TTNN default. The issue's global hifi4/fp32 knobs land on
`rms_norm` but not on these ops, consistent with "the knobs don't account for the gap."

---

## Prioritized suspects for chisel on p150

Given the compute ops are clean on n150, focus chisel (isolated + accumulated PCC,
bf16 graph vs bfp8 graph, op-by-op) on the attention/KV path:

1. **`ttnn.paged_scaled_dot_product_attention_decode`** with **bf16** K/V — top suspect.
   Compare isolated PCC bf16 vs bfp8; check default math_fidelity for the bf16 KV read.
2. **`ttnn.paged_update_cache`** into the **bf16** cache — verify the bf16 cache
   write/round-trip matches golden (accumulated mode across the two updates).
3. **RoPE region** (`sin`/`cos`/`slice_static`/`multiply`/`subtract`/`add`/`concat`) —
   the "index-arithmetic" area flagged in the issue (cf. tt-xla #5116); watch accumulated
   PCC drift here.
4. **`ttnn.nlp_concat_heads_decode`** — no chisel golden registered (will show
   `no_golden`/`golden_promoted`); add a golden if it needs validation.

Chisel goldens are registered for matmul, rms_norm, SDPA-decode, paged SDPA-decode,
paged_update_cache, embedding, silu, and all elementwise ops used here.

---

## How to reproduce on p150

```bash
source env/activate
export LD_LIBRARY_PATH="$PWD/third_party/tt-metal/src/tt-metal/build_Debug/lib:\
$PWD/build/runtime/lib:$PWD/falcon3_chisel/libshim:$LD_LIBRARY_PATH"
export SYSTEM_DESC_PATH="$PWD/ttrt-artifacts/system_desc.ttsys"

# 1. strip cpu-hoist so ttnn-to-flatbuffer can parse
python3 falcon3_chisel/strip_cpu_hoist.py \
  falcon3_chisel/ir/notrace/falcon3_7b_bf16_1layer_notrace_decode_ttnn.mlir \
  falcon3_chisel/bf16_1layer_stripped.mlir

# 2. run chisel (op-by-op PCC) — on p150 the KV cache + <11,10> grid fit
python3 falcon3_chisel/run_chisel.py falcon3_chisel/bf16_1layer_stripped.mlir \
  -o falcon3_chisel/bf16_1layer.chisel.jsonl

# 3. rank problematic ops
python3 falcon3_chisel/analyze_chisel.py falcon3_chisel/bf16_1layer.chisel.jsonl

# repeat for the bfp8 graph and diff the per-op PCC tables.
```

Notes for p150:
- The full-model (non-1layer) graphs are the true target; the 1-layer graphs share the
  same full-context KV cache so start there.
- For end-to-end model PCC, run the traced decode graph (`@trace_24_main`) with the real
  page-table/positions rather than the random indices `run_chisel.py` generates.
- `run_chisel.py` uses random inputs (valid small ints for token/page indices). Isolated
  PCC is meaningful regardless; **accumulated PCC and e2e PCC need realistic inputs.**
