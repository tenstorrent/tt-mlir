# opt-2 accuracy failure on llama-3.1-70B qb2 (2×2 mesh) — investigation

> ## CURRENT VERDICT (end of 2026-07-27 session) — read this first
>
> **Symptom:** opt-2 (optimization_level=2, i.e. greedy optimizer + `memoryLayoutAnalysis`)
> collapses llama-70B qb2 **2×2** 80-layer decode PCC to **0.624**; the opt<2 workaround gives
> **0.9917**. Depth-compounding — at 3 layers opt-1 ≡ opt-2 (≈0.978), so fixes must be judged
> at ~80 layers.
>
> **Root cause: NOT yet pinpointed, but the space is heavily narrowed.** Confirmed at 80L:
> | hypothesis | test | verdict |
> |---|---|---|
> | matmul math fidelity (LoFi/HiFi2) | HiFi2 → 0.626 | **RULED OUT** |
> | matmul fp32 dest-accumulation | fp32 on qkv/o_proj/down_proj → 0.641 | **RULED OUT** |
> | matmul output sharding | force DRAM-interleaved → 0.638 | **RULED OUT** (matmuls fully exonerated) |
> | phantom cores (the #5738 norm-bug class) | static scan | **RULED OUT** (ubiquitous in *both* meshes; only bbox-reducing op, the norm, is clean) |
> | all_reduce / CCL | IR diff | **RULED OUT** (identical opt-1 vs opt-2 @2×2) |
> | QKV-split / rotary layout | IR diff | **RULED OUT** (mesh-driven, shared by working opt<2@2×2) |
>
> ⇒ It is a **structural** issue in opt-2's sharding of the **non-matmul residual/elementwise
> stream** (sharded add/multiply/silu + inserted reshards) at **low TP (2×2/TP=2)**, NOT a
> precision problem. It is **not Blackhole-specific** — opt-2 works on the same board at TP=4
> (qb2 default `(1,4)`); it tracks the TP factor. Galaxy runs level-1, so it is not an opt-2
> datapoint. Exact culprit op still to be found via emit-iterate bisection, which is currently
> **blocked by two EmitPy codegen bugs** on the 2×2 mesh (see
> `codegen_emitpy_mesh_issues_5738.md`).
>
> **On the norm's core grid (`1×8` vs `8×8`):** under opt-2 the fused `distributed_rms_norm`
> lands on a **clean full-bbox 1×8 (8 cores)** — bit-identical output (1.0), *not* the bug —
> whereas the opt<2 workaround hand-places **8×8 (64 cores)**. opt-2 can't reach 64 because
> `LegalTensorLayoutAnalysis`'s canonical **row-major** placement can only form a full-bbox
> *even* divisor of 128 up to 8 cores on the 11-wide BH grid; reaching 64 needs
> explicit-rectangular placement. This is a **throughput** refinement, not a correctness issue
> (details in `distributed_rms_norm_opt2_migration.md`, "Remaining perf note").
>
> ### ⚠️ Reading guide — SUPERSEDED sections below
> This doc accreted across sessions. Everything **above** and the sections dated
> **(2026-07-27)** / titled **"… RULED OUT"** are current. The sections **"Finding:
> bf16-accumulation drift … down_proj"**, **"The clincher — it is PRECISION"**, **"Why opt-1 ✓
> … tp2×2 ✗" (K-dependent)**, **"LEADING HYPOTHESIS: large-K bf16 …"**, **"fp32-acc
> experiment"**, and **"Fix"** below are an **earlier hypothesis (matmul K-accumulation
> precision) that the 80-layer experiments above DISPROVED** — kept only as a historical
> record. Do not act on them.

## Decode-graph config diff (opt-1 vs opt-2, 2×2, per op type — only differences)

From `emit_opt{1,2}/graph_1/main.py` (decode graph, verified mesh=2×2). SDPA-decode
itself is **INTERLEAVED DRAM in BOTH** — opt-2 only adds redundant no-op reshards to the
same layout, so the earlier "SDPA-decode batch-axis sharding" lead is **dead** for this
graph. The true difference is that opt-2 shards essentially the whole residual stream:

| op | opt-1 | opt-2 |
|---|---|---|
| matmul (×16) | INTERLEAVED/DRAM | **WIDTH_SHARDED/L1** |
| add (residual, ×8) | INTERLEAVED/DRAM | **BLOCK_SHARDED ×2 / WIDTH_SHARDED ×6 /L1** |
| multiply (×4) | INTERLEAVED/DRAM | BLOCK/WIDTH_SHARDED/L1 |
| silu (×3) | INTERLEAVED/DRAM | WIDTH_SHARDED/L1 |
| cos/sin | INTERLEAVED/DRAM | WIDTH_SHARDED/L1 |
| embedding / rotary_embedding / paged_update_cache / slice | INTERLEAVED/DRAM | INTERLEAVED/**L1** |
| repeat (GQA kv) / where (mask) / max / eq | INTERLEAVED/DRAM | **HEIGHT_SHARDED/L1** |

So opt-1 = all-interleaved-DRAM; opt-2 = aggressively L1-sharded. The compounding decode
error lives somewhere in this sharding (matmul kernels and/or the sharded residual
`add`/elementwise), but the static diff cannot isolate *which* op without per-layer
localization at 80L. **Fidelity is NOT among the differences** (both emit
`compute_kernel_config=None`; the LoFi/HiFi2 gap is a runtime default, already ruled out).

## 1×4 (works) vs 2×2 (fails) — the discriminating diff

opt-2 works at 1×4 (tp) but collapses at 2×2 — so the culprit is a **2×2-specific** choice.
Cheap codegen diff of the decode graph (opt-2, 3-layer) surfaces these structural deltas
(strongest leads, decode/attention-path, mesh-driven):

- **`nlp_create_qkv_heads_decode`**: present at 1×4 (HEIGHT_SHARDED), **ABSENT at 2×2** —
  2×2 splits QKV heads via a generic reshape/slice path instead of the fused decode op.
- **`rotary_embedding`**: `INTERLEAVED/L1` at 2×2 vs `HEIGHT_SHARDED/L1` at 1×4.
- **`matmul`**: 2×2 decode matmuls have **NO program config** (→ tt-metal HiFi2 default);
  1×4 matmuls **do** (→ LoFi). This *re-confirms fidelity is irrelevant* — the failing 2×2
  config already runs the higher-fidelity path.

(Both emits above held `QB2_NORM=shard` constant, so the 2×2-vs-1×4 diff is norm-consistent.
The 2×2 emit with *default* norm segfaults during decode-graph export — noted, not chased,
since the non-export 80L default run completes at 0.624.)

### Pinpointed attention-path delta (2×2 vs 1×4)

- **QKV head split BEFORE attention differs:**
  - 1×4 (works): `ttnn.experimental.nlp_create_qkv_heads_decode` (HEIGHT_SHARDED, batch/head-aware fused op).
  - 2×2 (fails): **generic `reshape`** → rotary → interleaved DRAM → SDPA-decode. No fused create-heads op (grep: 0 vs 3).
  - `nlp_concat_heads_decode` (AFTER attention) is present in BOTH — only the pre-attention split differs.
- **rotary_embedding** output: 2×2 `INTERLEAVED/L1` vs 1×4 `HEIGHT_SHARDED/L1`.

This is a decode-specific, 2×2-only difference in the batch/head-layout region — consistent
with the earlier #5738 note localizing decode-PCC to batch-axis handling of the attention
path. Plausible mechanism: the generic reshape-based head split mis-orders batch↔head when
batch is sharded across the 2×2 data-parallel axis, injecting a small per-layer bias that
compounds over 80 layers (invisible at 3L where opt-1≈opt-2≈0.978).

## Ranked candidates (all still require an ~80L silicon run to confirm)

1. **Pre-attention QKV head split** (generic reshape at 2×2 vs fused `nlp_create_qkv_heads_decode`
   at 1×4) — strongest: decode-specific, 2×2-only, batch/head-layout, matches prior note.
2. **rotary_embedding layout** (interleaved vs height-sharded at 2×2) — adjacent to #1, test together.
3. **Pervasive residual-stream L1 sharding** (matmul/add/multiply/silu WIDTH/BLOCK-sharded vs
   opt-1 interleaved) — the broad difference; would compound but not obviously 2×2-specific.

## What "opt-1 vs opt-2" actually is + is it Blackhole-specific? (2026-07-27)

`optimization_level` (tt-xla compile option) maps (`TTNNPipelines.h:599-617`):
- level ≥1: greedy optimizer ON. level ≥2: **memoryLayoutAnalysisEnabled** ON (the aggressive
  L1 sharding). So **opt-1 = greedy w/o mem-layout sharding; opt-2 = greedy + mem-layout sharding.**
  The bug is in the level-2 mem-layout sharding.

**Galaxy (WH, 4×8) does NOT use opt-2** — it hard-codes `optimization_level=1` (`test_llms.py:1182`).
So Galaxy ≈ qb2-opt-1 (sharding off); its passing says nothing about opt-2.

**Not Blackhole-specific — it's the TP factor (per-device K):** no arch branches exist anywhere in
the test/loader/pipeline (confirmed). opt-2 (level 2) **works on Blackhole at TP=4** — the qb2
*default* mesh is `(1,4)`=TP4 with `optimization_level=2`, and it passes; only `QB2_MESH=2d`=`(2,2)`
=TP2 fails. Same board, same opt level, only TP 4→2. Failure scales with per-device down_proj
K = intermediate(28672)/TP:

| config | opt lvl | TP | down_proj K/dev | result |
|---|---|---|---|---|
| qb2 2×2 (`QB2_MESH=2d`) | 2 | 2 | 14336 | fails 0.62 |
| qb2 1×4 (default `tp`) | 2 | 4 | 7168 | works |
| Galaxy 4×8 | 1 | 8 | 3584 | works |

Galaxy dodges the bug twice (mem-layout off AND TP=8). Caveat: no one runs level-2 + TP=2 on WH,
so a WH-vs-BH threshold difference isn't strictly excluded — the fp32-acc run is the SW-fixability
decider. HF config: hidden=8192, intermediate=28672, heads=64, kv_heads=8, layers=80.

## PRECISION RULED OUT (both levers tested at 80L)

| config | 80L decode | vs baseline 0.624 |
|---|---|---|
| HiFi2 (fidelity) | 0.626 | no change |
| fp32_dest_acc on qkv/o_proj/down_proj | 0.641 | +0.017, marginal |

**Neither matmul-precision lever fixes it.** The bug is NOT matmul arithmetic precision.

### Structural bisection: matmul-output sharding — ALSO RULED OUT

Forced ALL matmul outputs to DRAM-interleaved (opt-1's layout; `getOutputHints` drops sharded
candidates), kept opt-2's other sharding. Verified in emit (matmul outputs → INTERLEAVED/DRAM
6/6). 80L decode = **0.638** ≈ baseline 0.624. **Matmuls are fully exonerated** (both precision
AND output sharding). The culprit is opt-2's **sharded residual/elementwise ops** (add, multiply,
silu → BLOCK/WIDTH-sharded L1) and/or the **reshards** (`to_memory_config`) it inserts.

Working hypothesis (structural, not precision): a sharded elementwise op or reshard has a shard
spec at TP=2 with **phantom cores / padding mismatch** (like the original #5738 fused-norm bug),
reading uninitialized/garbage L1 → tiny systematic per-layer error → compounds over 80 layers.
This is 2×2-specific because per-device shapes differ from TP=4, and precision-agnostic — exactly
the observed signature. **Pinpointing needs per-op golden localization** (DebugHooks post_op +
get_op_output_tensor, TT_RUNTIME_DEBUG=ON build) — the matmul-bisection avenue is exhausted.

### Phantom-core hypothesis — RULED OUT (static)

Scanned every sharded op's CoreRangeSet in the 2×2 decode graph for core-count ≠ bbox-area.
Phantom cores are **ubiquitous in BOTH** 2×2 (69 instances) and 1×4 (76), almost all in
**elementwise** ops (add/matmul/silu) where each core is independent → harmless. The only
bbox-reducing op, `fused_rms_minimal` (RMS norm), appears in **neither** list → clean rectangle
in both (the shipped #5738 fix holds). So phantom cores are not the discriminator.

Also checked: per-device batch/DP handling isn't visible in the IR (batch is a mesh-level
concept; per-device decode tensors are `[1,1,1,X]` identically in both meshes).

### Next bisection: force elementwise ops interleaved (RUNNING)

matmul-interleaved didn't recover → test the residual/elementwise stream. Suppressed the base
`OpRuleBook::getOutputHints` sharded fallback (OpRuleBook.cpp) so default-rulebook ops (add,
multiply, silu, …) go non-sharded while matmul/attention keep opt-2 layouts. If 80L decode
recovers → elementwise sharding is the culprit; else → attention-path ops or reshards.

## Emit-iterate approach (edit emitted code, reload, bisect — no rebuild)

`TTXLA_CODEGEN_EXPORT_DIR` emits `main.py`; `TTXLA_CODEGEN_LOAD_DIR` reloads the (edited)
emit instead of compiling (both handled in `pjrt_implementation/src/api/compile_options.cc`).
So per-op layouts can be flipped in `main.py` and re-run in minutes (reload skips the slow
optimizer compile) while the benchmark still computes PCC. Plan: bisect the minimal
opt-1→opt-2 layout change that breaks PCC.

Layer count: **3 is useless** (opt-1≡opt-2≈0.978 there — measured; and a 3-layer per-op
compare misleadingly fingered matmul, since ruled out). Need enough layers to compound;
using N=40 (prefill also collapses: 0.42 @80L).

Tooling snags hit + workarounds:
- **Decode-graph export segfaults at scale** (`graph_1/main.py` not written at 40L; works at
  3L). Pre-existing codegen-export bug, not the accuracy bug. → iterate on **prefill** (also
  collapses) via `--pcc-prefill`.
- **Load hits** `Can't convert tensor on MeshShape([2,2]) to row-major … supply a mesh
  composer` in `cpu_hoisted_const_eval`→`to_torch`. Const-eval is compile-time constant
  folding (accuracy-neutral). → re-emit with `QB2_CONST_EVAL=0` (llm_benchmark.py:535).

## Candidate #1 (QKV split) — ALSO RULED OUT

opt-1@2×2 and opt-2@2×2 **both** use the generic reshape QKV split (0 fused
`nlp_create_qkv_heads_decode`, 12 concat, 18 rotary — identical). The fused-op / rotary-layout
difference is purely **mesh**-driven (1×4 vs 2×2), shared by opt-1 which works. Not the bug.

Also ruled out as the opt-1-vs-opt-2@2×2 difference (static): **all_reduce** (identical:
interleaved DRAM, Ring, cluster_axis=0 in both) and **matmul kernel type** (both meshes use
`MatmulMultiCoreReuseMultiCast1DProgramConfig`; matmuls DO carry program configs at 2×2 — the
earlier "no PC at 2×2" was a census-window artifact).

## [⚠️ SUPERSEDED — disproven by 80L tests; see CURRENT VERDICT at top] LEADING HYPOTHESIS (post-triangulation): large-K bf16 accumulation in the 1D sharded matmul

The bug = intersection of "opt-2-specific" AND "2×2-specific". opt-2's residual-stream matmuls
use the **1D-multicast sharded kernel** (blocked bf16 dest-accumulation) at both meshes; opt-1
uses the interleaved MultiCore kernel. Per-device matmul K is `intermediate/TP`:
- 1×4 (TP=4): K = intermediate/4 → small → opt-2 works.
- 2×2 (TP=2): K = intermediate/2 → **2× larger** → more bf16 accumulation rounding (worst on
  large-K `down_proj`/`o_proj`), a tiny per-layer bias that compounds over 80 layers.
- opt-1@2×2: same large K but the interleaved kernel accumulates differently → works.

**Predicted fix:** `fp32_dest_acc=true` on the sharded matmuls (accumulate in fp32).

### fp32-acc experiment (implemented, 80L running)

Implementation (uncommitted, tree):
- `MatmulProgramConfig.cpp getMaxSubblockSize`: return `maxSubblockSize/2` (=4) when the op has
  no compute config, so the emitted 1D program config is sized for the fp32 DST budget.
- `MatmulRules.cpp applyOpSpecificAttrs`: set `compute_config = {HiFi2, fp32_dest_acc=true,
  packer_l1_acc=true}` on program-config'd matmuls — **gated to output N ≤ 8192**.
- The N-gate is required: blanket fp32 **OOMs L1** on N-heavy matmuls (lm_head N=vocab,
  gate/up-proj N=intermediate/TP=14336) — those aren't the large-K case and run once / aren't
  reduction heavy. Gate (N≤8192) keeps fp32 on down_proj (N=hidden=8192, K=intermediate/TP),
  o_proj (N=8192, K=hidden/TP), qkv (N=10240/TP) — the reduction-heavy compounding injectors.
  Verified: 1-layer emit → exactly 3 matmuls get fp32 (qkv/o_proj/down_proj), gate/up/lm_head skipped.

Verified: 1-layer emit shows 3 matmuls with `fp32_dest_acc_en=true`, subblocks ≤4, no OOM.
3-layer smoke (no export) runs clean: prefill 0.932 / decode 0.979 (3L doesn't diverge — just
confirms it runs). 80L opt-2@2×2 with fp32 is the decisive test (baseline 0.624 → target ~0.99).

Side note: codegen **export** of the 2×2 decode graph segfaults (`graph_1` never written) —
pre-existing, independent of fp32 (graph_0 exports fine; the non-export run path is unaffected).

### Production fix design (if the fp32 test passes)

The current change is a **test hack** (two coupled pieces): `MatmulProgramConfig.cpp:28-33`
returns subblock 4 when compute_config is null (pre-assuming fp32), and `MatmulRules.cpp`
`applyOpSpecificAttrs` sets fp32 gated on output N≤8192. These must stay consistent or diverge.

Clean version (from pipeline map): `TTNNSetComputeKernelConfig` **already runs before** the
analysis (`TTNNPipelines.cpp:466` < `:489`), and already has the right hook —
`applyLargeInnerDimBf16MatmulConfig` (`TTNNSetComputeKernelConfig.cpp:19-45`) sets
`fp32_dest_acc=true + packer_l1_acc=true` for **bf16 matmuls with inner dim K > 50000**
(vocab-backward fix). down_proj's per-device K is only 14336, so it's missed.

**Fix = add a lower K-tier** to that pre-analysis selector so reduction-heavy matmuls get fp32:
- Keys on **K (per-device inner dim)** — the exact hypothesis variable — so it auto-scopes to
  low-TP: down_proj K=14336 (TP=2) caught, K=7168 (TP=4) not (TP=4 already works).
- Threshold must sit in **(8192, 14336]** (e.g. ~12000): excludes gate/up_proj (K=hidden=8192,
  column-parallel so K not divided by TP) — which is exactly why the **N-gate/OOM problem
  disappears** (the N-heavy matmuls have small K, so a K-gate skips them for free).
- Runs before analysis → `getComputeConfigAttr()` is non-null at `MatmulProgramConfig.cpp:268`
  → subblocks size to 4 via the real fp32 branch → **delete both hacks**.
- Blast radius: it's a global bf16-matmul threshold; lowering it changes any model with
  8192<K≤~12000. Needs a lit test combining fp32 + `matmul_multi_core_reuse_multi_cast_1d`
  (no existing coverage) and a perf check. Open question the test answers first: does down_proj
  alone need fp32, or o_proj/qkv too (current hack covers all 3 via N≤8192)?

## RULED OUT (confirmed)

- **Matmul math fidelity (LoFi vs HiFi2).** 80L decode with forced HiFi2 = 0.626 ≈ baseline
  0.624. And the *failing* 2×2 decode matmuls carry **no** program config → already the
  HiFi2 default, while the *working* 1×4 matmuls carry one → LoFi. Fidelity is not it.
- **RMS norm** (#5738 fix shipped; prior finding).
- **3-layer prefill op-by-op analysis** — measured in a regime with no opt-1/opt-2 divergence.

## Recommended next experiment

Force opt-2 at 2×2 to use `nlp_create_qkv_heads_decode` for the pre-attention split (match
1×4), or equivalently make the decode attention QKV-prep layout match 1×4, then run llama
80L 2×2 and check decode PCC vs 0.624→0.99. One ~40-min run per attempt.

## Fidelity difference (DEAD END — kept as mechanism notes)

`matmul_device_operation.cpp:2683`:
```cpp
increase_fidelity = !has_program_config && !has_user_grid && !are_inputs_low_precision_df;
math_fidelity     = increase_fidelity ? MathFidelity::HiFi2 : MathFidelity::LoFi;
// default_fp32_acc = is_float_32 (output FLOAT32) -> false for bf16 output (both opt1 & opt2)
```
- **opt-1 matmuls: `program_config=None`** → `increase_fidelity=true` → **HiFi2**.
- **opt-2 matmuls: carry an optimizer `program_config`** → `increase_fidelity=false` → **LoFi**.

Both emit `compute_kernel_config=None` and both use bf16 accumulation; **the only default
difference is fidelity (HiFi2 vs LoFi)**, and it is triggered *solely by the presence of
the program config*. LoFi truncates more mantissa bits per multiply; over the deep
bf16/bf4 residual stream (worst at `down_proj`'s large K, ×2 at 2-way TP) this is the
drift. **Confirmed:** injecting HiFi2 into opt-2's matmuls recovers qkv/all_reduce toward
opt-1 (a HiFi4 test earlier *overshot* past opt-1's HiFi2, which is why fidelity looked
"ruled out" — that was a mis-test).

## Symptom

llama-3.1-70B, qb2 (Blackhole), 2×2 mesh, bf16 activations + **bf4** gate/up-proj
weights. Decode PCC (gated at 0.94):

| layers | prefill PCC | decode PCC |
|---|---|---|
| 1  | 0.9989 | 0.9993 |
| 3  | 0.918  | 0.9787 |
| 80 | 0.385  | 0.624  |

opt-1 (the hand-rolled workaround path) with the **same** bf4 weights stays ~0.99 at
80 layers. So the regression is caused by **opt-2's layout choices**, and it
**accumulates per layer** (near-perfect at 1 layer, collapses by 80).

## Ruled out

| hypothesis | how ruled out |
|---|---|
| `distributed_rms_norm` (the #5738 op) | 80-layer opt-2 with the fused norm (0.624), norm replicated (0.6242), and norm force-decomposed with sharding unchanged (0.6256) are **identical**; and prefill (which contains **zero** `distributed_rms_norm`) is equally broken (0.385). |
| matmul **sharding** | forcing matmul DRAM-interleaved at opt-2 (3-layer) lifted prefill only 0.918→0.939 and left **decode unchanged** (0.9787→0.9782). Minor contributor, not the cause. |
| ~~matmul math fidelity~~ (this was the culprit — see "Exact root") | *Superseded.* An early test forced **HiFi4** and saw it diverge *more* — but that overshot past opt-1's actual **HiFi2**. The real difference **is** fidelity: opt-2 runs **LoFi**, opt-1 **HiFi2**. Forcing opt-2 to HiFi2 recovers it. |
| SDPA-decode | IR-identical between opt-1/opt-2 (same DRAM-interleaved layout, bf8 KV, scale, causal). |
| all_reduce config | IR-identical (sum / ring / bf16-interleaved). |

## Method: op-by-op value comparison (opt-1 vs opt-2)

Because opt level only changes *on-device* layout (not the mesh distribution), each
per-device shard holds the same logical content in opt-1 and opt-2, and
`ttnn.to_torch` normalizes away layout — so a per-op **value** comparison is valid,
and opt-1 (≈reference, PCC 0.99) is the baseline.

1. Emit opt-1 and opt-2 codegen (`TTXLA_CODEGEN_EXPORT_DIR`, `QB2_TRACE=0`) for the
   3-layer model → runnable `graph_0/main.py` (prefill) with captured inputs.
2. Instrument each `main.py`: after every compute op (`matmul`, `add`, `multiply`,
   `all_reduce`, `all_gather`, `rms_norm_pre_all_gather`, `mean`, `rsqrt`, `silu`,
   `sdpa`, …) insert a dump of `ttnn.to_torch(shard)` per device. (Scripts:
   `instrument.py`, `compare.py`; drivers `driver5.py`/`driver6.py`.)
3. Run each standalone with a **pure-ttnn driver** (bypasses torch-xla, which the
   codegen-*load* path can't execute here — it throws `GraphInputMatcher` in a
   cpu-hoisted const-eval). Cache ops are no-op'd (prefill writes but never reads the
   KV cache) and SDPA inputs tilized — both applied identically to opt-1/opt-2, so
   the comparison stays valid.
4. Compare compute ops aligned by variable name (`ttnn_matmul_5` etc. — stable across
   opt levels since compute ops come from the same TTIR; reshards differ and are
   skipped).

**Compare prefill, not decode:** decode's inputs (KV cache + hidden state) come from
prefill, which already diverges — so a decode diff is confounded. Prefill has the
same real inputs for both.

## [⚠️ SUPERSEDED — disproven by 80L tests; see CURRENT VERDICT at top] Finding: bf16-accumulation drift in the sharded matmuls, dominated by `down_proj`

Per-op *injected* error (drop each op causes beyond its input), opt-1 vs opt-2 prefill,
default fidelity. Only a few ops inject; the rest carry:

```
op (role)                  per-layer injection      clean?
matmul down_proj (MLP out)  L0 -0.00039, L1 -0.00061  NO  <- dominant, grows with depth
all_reduce (qkv TP-reduce)  L0 -0.00045               NO
matmul qkv (attn in)        -0.00012 / -0.00015 / -0.00024  NO (small)
matmul gate / up            0.0                       YES (1.00000)
matmul o_proj               0.0                       YES (1.00000)
all_gather, MLP all_reduce  ~0.0                      YES
rms_norm / mean / rsqrt / add / silu (elementwise)  ~0.0  YES (until input drifts)
```

- **`down_proj` (MLP output matmul) is the dominant injector, every layer**, and its
  injection **grows with depth** (a sharded matmul amplifies an already-drifted input),
  so the damage is ~80 accumulated `down_proj` injections carried by the residual adds.
- **`gate_proj`/`up_proj`/`o_proj` are bit-identical** — opt-2 shards them so their
  kernels round identically to opt-1.

### [⚠️ SUPERSEDED — this "it is PRECISION" conclusion was DISPROVEN at 80L; see CURRENT VERDICT] The clincher — it is PRECISION, not a wrong-sharding bug

`o_proj` (clean, 1.00000) and `down_proj` (injects −0.00039) have **byte-identical**
configs: `BLOCK_SHARDED`, `11×9=99` cores, shard `[64,384]`, out_subblock_w=6,
per_core_N=12, ROW_MAJOR. Same sharding type, same tiling, same grid — **opposite
behavior**. The only difference is the **reduction depth K** (down_proj K = intermediate
≈ 3.5× o_proj K = hidden). Same sharding + more K ⇒ more bf16 accumulation steps ⇒ more
rounding. If it were a wrong-sharding bug, identical configs would behave identically.

## Sharding-type audit (is it a second phantom-core bug like the norm? No)

Checked every op type opt-2 changes for a wrong sharding (phantom cores / misaligned
shards / wrong orientation):

| op | opt-2 sharding | phantom? | verdict |
|---|---|---|---|
| layer matmuls (qkv/o_proj/gate/up/down) | BLOCK_SHARDED `11×9` | no — full rectangle | correct; injection is K-dependent precision |
| lm_head | WIDTH_SHARDED 106-core | yes (`11×9+7`) | output-parallel, output PCC fine (0.9989) — not the culprit |
| add / multiply / silu | width/block-sharded | yes (2–4 ranges) | **benign** — elementwise is layout-invariant (no reduction); all ~1.0 |
| all_reduce / all_gather | interleaved / sharded | some multi-range | reduces over **devices**, not bbox; IR-identical to opt-1 → clean |
| distributed_rms_norm | WIDTH_SHARDED `1×8` | no — full-bbox | bit-identical (1.0) |

**Why this is unlike the #5738 norm bug:** that was a *correctness* failure — the fused
norm kernel **reduces over its shard-grid bounding box**, so phantom cores fed
**uninitialized L1** into the reduction → PCC ≈ 0 (catastrophic). Here the **only
bbox-reducing op is the norm**, and opt-2 gives it a clean full-bbox `1×8`, so that
failure mode cannot recur. The phantom/multi-range placements that *do* exist are all on
**non-bbox-reducing** ops (elementwise, output-parallel lm_head), so they are harmless.
The opt-2 loss is a *precision* (accumulation) problem, ~1e-4/matmul, not a ~PCC-0
correctness bug.

## [⚠️ SUPERSEDED — the K-dependent-precision explanation below was disproven at 80L; the TP-factor observation is right but the *mechanism* is structural, not precision] Why opt-1 ✓, tp4 (1×4) ✓, but tp2x2 (2×2) + opt-2 ✗

The drift is (opt-2 sharded matmul kernel) × (per-device reduction depth K), and K
depends on the tensor-parallel degree set by the mesh:

- **opt-1 (any mesh):** matmuls are DRAM-interleaved, `program_config=None` (auto-picked
  accurate kernels) → no sharded-kernel drift.
- **opt-2 tp4 (1×4):** 4-way TP → row-parallel matmuls (`down_proj`, `o_proj`) reduce
  over **K/4** per device → small bf16 error.
- **opt-2 tp2x2 (2×2):** only 2-way TP → those matmuls reduce over **K/2** per device —
  **2× the accumulation depth** → ~2× the per-matmul rounding, compounded over 80 layers
  → the collapse.

So only tp2x2+opt-2 hits both the sharded kernel *and* the large per-device K.

## lm_head (not a hotspot)

opt-2 config: `WIDTH_SHARDED` L1, 106 cores (`11×9+7` phantom), `MatmulMultiCoreReuseMultiCast1D`,
K=4096, shard `[576,608]` (opt-1: DRAM-interleaved). It is a **single** final projection
that **carries** the accumulated hidden-state drift into the logits — its own injection
is ~0. Its true output PCC equals the benchmark logits PCC (0.9989 @1L → ~0.38 @80L
prefill). (A per-device op-by-op comparison spuriously showed ~0.12 for lm_head because
opt-1 interleaves the vocab output while opt-2 width-shards it across the 4 devices — the
per-device shards aren't comparable; the 0.9989 logits PCC is authoritative.)

## [⚠️ SUPERSEDED — this fp32/HiFi2-based fix was DISPROVEN at 80L (HiFi2 0.626, fp32 0.641); see CURRENT VERDICT at top] Fix

**Primary fix — set matmul fidelity to HiFi2 when the optimizer attaches a program
config.** The root is that a program-config'd matmul defaults to LoFi. So the optimizer
(or the compute-config pass) must set `math_fidelity = HiFi2` on any matmul it gives a
`program_config`, matching what tt-metal uses for the no-config path. This is a *pure
compute-config change* — HiFi2 does **not** shrink the DST register budget, so it does
**not** conflict with the existing `out_subblock` sizes (unlike fp32-acc). Confirmed to
recover the fidelity-dominated ops:
```
op            LoFi (opt-2)   HiFi2
matmul_0 qkv   0.99988        0.99996
all_reduce_0   0.99943        0.99981
matmul_4 down  0.99961        0.99966  (residual — see below)
```

**Secondary — fp32 dest-accumulation for the large-K `down_proj` residual.** Even at
HiFi2, `down_proj` still injects ~−0.00034 (its large-K block-sharded accumulation
differs from opt-1's interleaved kernel). fp32 dest-acc addresses this, but it halves the
DST register budget, so the optimizer must size `out_subblock_h*out_subblock_w ≤ 4`
(`MatmulProgramConfig::getMaxSubblockSize` already does this **iff** the compute config is
known at program-config-generation time). Since `TTNNSetComputeKernelConfig` currently
runs **after** the optimizer (`TTNNPipelines.cpp:~459` vs optimizer ~`111-183`), a late
fp32 flag conflicts with the already-chosen subblock (reproduced: `out_subblock_w=5/6`
fails `matmul_device_operation.cpp:566` under fp32). So fp32-acc requires the compute
config to be decided **before** the optimizer.

**Where to implement:** in the matmul rulebook / `MatmulProgramConfig` generation, when
emitting a `MatmulProgramConfig`, also emit a `DeviceComputeKernelConfig` with
`math_fidelity=HiFi2` (and, for the fp32-acc variant, `fp32_dest_acc_en=true` + size the
subblock ≤4). This keeps the compute config consistent with the program config it's paired
with, at generation time.

**Validation plan:** apply HiFi2 (then +fp32-acc), run llama 1→3→80-layer opt-2 on qb2,
confirm decode PCC recovers toward opt-1's ~0.99. HiFi2 alone should recover most
(qkv/all_reduce fully, down_proj partially); +fp32-acc should close the down_proj residual.
Op-by-op-vs-opt-1 localizes; end-to-end decode PCC is the acceptance test.

### Alternative (fallback)

**Less aggressive sharding for this model** — opt-1 (interleaved) is accurate; opt-2's
aggressive L1 sharding of the TP matmul→all_reduce path is what introduces the bf16 drift.
A precision-/depth-aware layout policy that keeps that path closer to interleaved for deep
bf4 models would recover accuracy at some throughput cost. This is a fallback if fp32-acc
proves insufficient at 80 layers.

## Reproduction assets (local, in /home/mvasiljevic)

- Emits (instrumented `main.py` + `driver*.py`, `QB2_TRACE=0`):
  `emit_opt{1,2}` (3-layer), `e2_opt{1,2}` (2-layer), `e1_opt{1,2}` (1-layer, reaches lm_head).
- Scripts: `instrument.py` (insert per-op `_cbop_dump`), `analyze_layers.py` (per-op PCC +
  injected error + layer/role), `compare.py`, `mm_configs.sh` (dump matmul configs),
  `driver5.py` (baseline), `driver_hifi2.py` (HiFi2 test), `driver6.py` (fp32 test).
- Dumps: `dumps1_opt{1,2}` (1-layer), `dumps1_opt2_hifi2` (HiFi2), per-op per-shard tensors.
- Key evidence: HiFi2 recovers qkv/all_reduce; o_proj vs down_proj identical config /
  different K; tt-metal `matmul_device_operation.cpp:2683` (LoFi-when-program-config).
