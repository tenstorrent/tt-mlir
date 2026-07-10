---
name: shard-advise
description: "Get the tt-mlir compiler's L1 memory-layout / sharding recommendation for a ttnn decode (or prefill) block and diff it against the layout the model hand-wrote. Use during model bringup / layout tuning when a decoder-layer, attention, or MLP block runs below its lower bound and you want a compiler-informed reference for which tensors should be L1-sharded (and how) versus DRAM, plus the matmul program config the optimizer picks for that strategy. Produces a structured report.json (per-op layout, program_config, reshards, L1 spill). Not for perf profiling and not for dtype-precision decisions."
---

# Shard Advise

## Mission

Turn "is this block laid out well?" into a concrete, compiler-derived answer.
The advisor traces a ttnn function, runs tt-mlir's greedy L1 optimizer, and
reports the layout it would choose per op. Diff that against the `memory_config=`
the model hand-wrote: agreement is a positive signal; a divergence (e.g. the
optimizer width-shards a projection the model left DRAM-interleaved) is a lead.

This is a **reference/second-opinion** tool, not ground truth — it reasons only
about L1 layout (see Scope). Treat divergences as hypotheses to verify, the same
way `investigate-results` treats agent explanations.

## When to use

- A decode block runs below the decoder-layer lower bound and you suspect layout.
- You changed a block's sharding and want to know if the compiler agrees.
- You want a per-op layout map of a model you didn't write.

Do **not** use it for perf numbers (profile instead) or to decide tensor dtype
precision (bf16 vs bfp8/bfp4 — out of scope).

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

**The diff that matters:** for each projection/norm/mul, compare `ops[].layout`
to the `memory_config=` the model passed. The advisor width-shards L1-resident
projections across the grid and keeps SDPA-decode / KV cache in DRAM; if the
model differs, ask why.

## Scope — do not over-read

The advisor advises L1 layout / sharding **and** the matmul **program config**
the optimizer picks for that strategy (e.g. `matmul_multi_core_reuse_multi_cast_1d
@8x8`, in each op's `program_config`). It faithfully traces the dtypes the model
already chose (bfp4/bfp8 weights included), so layout reasoning uses the real
footprint — but it does not *recommend* a dtype change.

It does **not** pick the **DRAM-sharded-weight** matmul strategy (a distinct
optimizer feature landing soon; once chosen its program config surfaces the same
way) or tune **compute-kernel configs** (hifi2/hifi4). Comparing to a hand-tuned
model, expect agreement on the layout skeleton + chosen-strategy program config,
and gaps on precision and the DRAM-sharded-weight strategy.

## Gotchas

- **Fresh process per run** — the optimizer's device context is process-global.
- **ttnn version skew** — the advisor traces against tt-mlir's ttnn, not the
  experiment's tt-metal branch. If tracing fails on a ttnn op, that op's tracer
  handler needs aligning (bounded work in tt-mlir); report it rather than
  working around it.
- Read `report.json`; the CLI keeps stdout to a 5-line summary and routes all
  pipeline/device logging to `pipeline.log`.
