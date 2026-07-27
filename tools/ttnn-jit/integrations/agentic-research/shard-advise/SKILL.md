---
name: shard-advise
description: "Get a good-enough L1 sharding strategy for a pre-written ttnn model (or a single block) from the tt-mlir compiler, instead of hand-deriving it. The greedy optimizer decides which tensors to L1-shard, how, on what grid, plus the matmul program config for its pick; use it to skip the mechanical sharding search during model bringup / porting, apply the result as a baseline, then tune the rest and re-query specific blocks as needed. Produces a structured report.json (per-op layout, program_config, reshards, L1 spill). Not for perf profiling and not for dtype-precision decisions."
---

# Shard Advise

## Mission

Get a good-enough L1 sharding strategy for a pre-written model **without spending
your own reasoning budget rediscovering it**. Deriving which tensors to shard,
how, and on what grid is exactly the mechanical search tt-mlir's greedy L1
optimizer already does — so ask the advisor instead of exploring it yourself.

Give it the model (or one block). It traces the ttnn function, runs the greedy
optimizer, and hands back, per op, the layout to use and the matmul program
config for it (`report.json`). Apply that as your sharding baseline, then spend
your effort on the parts the advisor doesn't cover — dtype precision, the
DRAM-sharded-weight strategy, kernel configs (see Scope) — rather than on the
sharding the compiler can hand you for free.

Intended loop:

1. **Get a baseline** — run the advisor on the model → apply the reported
   layouts + program configs. Cheap, fast, compiler-validated.
2. **Tune what's left** — profile, then adjust the axes the advisor doesn't own.
3. **Re-query a piece** — changed one block, or want a fresh strategy for just
   the MLP / attention? Point `advise_decoder.py` at that piece and ask again.

## When to use

- You have a pre-written (or freshly ported) model and need a sharding strategy
  to start from — don't hand-derive it, ask the advisor first.
- You want a per-op layout + program-config map for a block you didn't write.
- You changed a block and want the compiler's current best layout for it.

It is a fast baseline, not the last word: it reasons about L1 layout + the
program config for its pick (see Scope). Do **not** use it for perf numbers
(profile instead) or to decide tensor dtype precision (bf16 vs bfp8/bfp4).

## Setup (once per shell)

```bash
source .agents/skills/shard-advise/scripts/bootstrap.sh
```

`bootstrap.sh` activates the pre-built tt-mlir advisor env (from
`$TTMLIR_ADVISOR_HOME`) and ensures `SYSTEM_DESC_PATH` is set. If it reports the
advisor env is missing, that is one-time operator setup — see the integration
README; do not try to build tt-mlir from inside an experiment.

## Run it

Point the `advise_decoder.py` capture target at the experiment's decoder, then
run the advisor in a **fresh process** and read `report.json` — never scrape
stdout:

```bash
# edit scripts/advise_decoder.py: set MODEL_DIR / config / layer to the experiment
ttnn-advise capture .agents/skills/shard-advise/scripts/advise_decoder.py:decode \
    --out /tmp/shard-advice 2>/dev/null

python -c "import json; d=json.load(open('/tmp/shard-advice/report.json')); \
  print('\n'.join(f\"{o['index']:>3} {o['op']:<45} {o['layout']}\" for o in d['ops']))"
```

Or, if a TTIR `.mlir` dump already exists (no device needed):

```bash
ttnn-advise mlir path/to/model.ttir.mlir --out /tmp/shard-advice 2>/dev/null
```

## Read the result

`/tmp/shard-advice/report.json`:
- `ops[]`: `{index, op, layout}` — e.g. `l1/width_sharded/1x64 cores=(0,0)-(7,7)`
- `reshards[]`: `{kind, producer, consumer, from, to, output_revert}`
- `spill`: `{ran, total_spills}` — near-zero is healthy
- `total_ops`, `final_choices`, `artifacts{...}`

Also written: `report.txt` (human-readable), `final_ir.mlir` (authoritative TTNN
IR), `pipeline.log` (captured native output, for debugging only).

**Apply it as the baseline:** `ops[].layout` + `ops[].program_config` are the
strategy to write onto each op's `memory_config=` / `program_config=`. Typically
the advisor width-shards the L1-resident projections across the grid with a
1d-multicast matmul config and keeps SDPA-decode / KV cache in DRAM — take that
as given rather than re-deriving it. If the model already sets something and the
advisor disagrees, prefer the advisor's pick unless you have a measured reason
not to (then it becomes a tuning question, step 2).

## Scope — do not over-read

The advisor advises L1 layout / sharding **and** the matmul **program config**
the optimizer picks for that strategy (e.g. `matmul_multi_core_reuse_multi_cast_1d
@8x8`, in each op's `program_config`). It faithfully traces the dtypes the model
already chose (bfp4/bfp8 weights included), so layout reasoning uses the real
footprint — but it does not *recommend* a dtype change.

It **does** now pick the **DRAM-sharded-weight** matmul strategy when that wins
(`matmul_multi_core_reuse_multi_cast_dram_sharded`), but only if the capture lets
it — see "Capture preconditions" below. It does not tune **compute-kernel
configs** (hifi2/hifi4) and does not *recommend* a dtype change. Comparing to a
hand-tuned model, expect agreement on the layout skeleton and the matmul
strategy, and gaps on precision and compute-kernel config.

## Capture preconditions — get these wrong and the advice is silently narrower

The advisor is deterministic: same capture, same build, byte-identical
`final_ir.mlir`. So a "missing" recommendation is never flakiness — it means the
capture never gave the optimizer the option. DRAM-sharding is gated on
properties of the *traced graph*, so the capture has to be faithful:

| requirement | why | if you get it wrong |
| --- | --- | --- |
| weights **bfp4/bfp8**, tiled, DRAM-interleaved | *policy, not a kernel limit* — bf16 DS runs at PCC 1.0000, but DS streams the weights so bf16 moves 2x bfp8's bytes | no DS candidate; override with `--pipeline-options allow-bf16-dram-sharded-matmul=true` |
| **any batch** whose activation is one tile row (M <= 32) | tt-metal's DS kernel currently takes an in0 height of exactly one tile | M > 32 is offered anyway and refused by tt-metal with *"currently only support in0 tensor height of tile height"* |
| K/32 divisible by *some* in0-core count | the contraction dim is split across the in0 cores; the count is now chosen from K, not fixed at 8 | effectively never rejects |
| matmul or linear (**bias is fine**), weight `[K,N]` or `[1,1,K,N]` | a genuinely batched weight (per-expert MoE) is not a DS shape | no DS candidate |

**Match the shipped precision.** The most common trap: a capture builds weights
in bf16 "because dtype is not an advisor decision". That was true before
DRAM-sharding; dtype is now *the* gate. A bf16 capture of a model that ships BFP4
reports 0 DRAM-sharded matmuls and looks like a considered verdict.

**Check the report before drawing conclusions.** `report.txt` /
`report.json.dram_sharding` now say, per matmul, whether DS was *considered* and
which gate rejected it:

```
=== DRAM-sharded matmuls: 0 of 5 (0 considered) ===
  [0] ttnn.linear [4096, 6144] bf16  -> no: weight dtype is bf16, DS needs bfp_bf4/bfp_bf8 ...
  NOTE: DRAM-sharding was never even a candidate here. ...
```

`dram_sharded_considered == 0` means *fix the capture*, not "the model does not
want DRAM sharding".

**Host-only captures:** if the capture avoids silicon (stub device, host-resident
tensors), do **not** build real BFP weights — the host BFP packing path
initializes the cluster, and every op-model query then dies with
`Watcher server is unavailable, and the target is not a mock device`. Keep the
bf16 host buffer and present the shipped dtype to the tracer instead; the tracer
reads only `.shape`/`.dtype` off a captured weight. See the `_BfpView` proxy in
`forge_experiments/qb2-experiments/dram-sharded-advisor/scripts/` (agentic-research).

**Advice is not a measurement, and it is not a correctness check.** A DRAM-sharded
recommendation means the config is legal and the optimizer preferred it. It does
**not** mean it is faster — the optimizer has no cost model and prefers DS
whenever it is legal. It also does **not** mean it is numerically safe: the
advisor never reasons about numerics. GPT-OSS is the cautionary case — DRAM-sharded
attention there is latency-competitive but fails the sliding-window boundary
(BF16 pos-130 PCC 0.913), so enabling the bf16 switch on that model makes the
advisor propose exactly the config known to break it. Sweep for perf **and** PCC
before shipping anything from this report.

## Gotchas

- **Fresh process per run** — the optimizer's device context is process-global.
- **Missing op = loud, actionable failure.** By default the advisor traces the
  model straight into the TTNN dialect. A ttnn op with no tracer handler fails
  with `ttnn.<op> has no direct-TTNN handler yet`, naming exactly what to add —
  that is a bounded per-op task in tt-mlir, not a dead end. Report the op rather
  than working around it. (For a model that hits one, `--tracer interception`
  routes through the older TTIR path as a stopgap.)
- **`tensor.memory_config()` during tracing** — some optimized decoders branch on
  it to skip a redundant move; the analysis tracer refuses it by design
  (`memory_config is unknown during analysis`). Subclass the decoder for capture
  and make those moves unconditional; the optimizer folds away what it does not
  need. Committed examples: the Qwen3-32B and Falcon3-7B capture scripts.
- **ttnn version skew** — the advisor traces against tt-mlir's ttnn, not the
  experiment's tt-metal branch; diverged op signatures surface as the same
  loud trace failure.
- Read `report.json`; the CLI keeps stdout to a 5-line summary and routes all
  pipeline/device logging to `pipeline.log`.
