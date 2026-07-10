# agentic-research integration: `shard-advise` skill

Drop-in package that adds the tt-mlir **L1 shard advisor** as an agent skill for
the [`agentic-research`](https://github.com/tenstorrent/agentic-research) repo.
It lets a bringup agent ask *"what L1 layout would the tt-mlir optimizer pick for
this decode block?"* and diff that against the `memory_config=` choices the model
hand-wrote — surfacing gaps like "you left this DRAM-interleaved; the optimizer
would width-shard it across the grid."

## What's here

```
shard-advise/
  SKILL.md              # the skill (codex harness format)
  scripts/
    bootstrap.sh        # locate + activate a pre-built tt-mlir advisor env
    advise_decoder.py   # capture target: wraps OptimizedDecoder -> make_inputs
```

## Where it goes

Copy `shard-advise/` into the agentic-research repo at:

```
.agents/skills/shard-advise/
```

(alongside `investigate-results`, `experiment-review`, etc.)

## One-time operator setup (NOT per experiment)

The advisor is a **tt-mlir** tool; building it is heavy and done once, out of
band — not by the per-experiment agent. Provision a tt-mlir checkout built with:

```bash
cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTTMLIR_ENABLE_OPMODEL=ON \
  -DTTMLIR_ENABLE_TTNN_JIT=ON \
  -DTTMLIR_ENABLE_RUNTIME=ON -DTTMLIR_ENABLE_STABLEHLO=ON
cmake --build build          # installs ttnn_jit + the `ttnn-advise` CLI into the venv
```

Then export its location so `bootstrap.sh` can find it:

```bash
export TTMLIR_ADVISOR_HOME=/path/to/tt-mlir     # checkout with the build above
```

`bootstrap.sh` activates that env and ensures a system descriptor exists.

## The one real risk: ttnn version skew

The advisor traces by **executing** ttnn, so the model must run against the
advisor's ttnn (tt-mlir's pinned tt-metal SHA), which differs from the agents'
`agentic-research/fast-models-fast` branch. If op signatures / vocabulary
diverge, tracing breaks.

**De-risk before relying on it:** run `advise_decoder.py` on one experiment's
`optimized_decoder.py`. If it traces, integration is real. If it fails on a
specific ttnn op, that op's tracer handler needs aligning — bounded, per-op work
in `tools/ttnn-jit/_src/interception_tracer.py`.

## Scope

The advisor reasons about L1 memory layout / sharding, and for the sharding
strategy it picks it also reports the matmul **program config** the optimizer
generated and backend-validated (e.g. `matmul_multi_core_reuse_multi_cast_1d
@8x8`), surfaced per op in `report.json`. It *does* faithfully trace the input
dtypes the model already chose (bfp4/bfp8 weights included), so layout reasoning
uses the true footprint.

What it does **not** do: recommend a dtype change (bf16→bfp8/bfp4 stays the
model's call), pick the expert's DRAM-sharded-weight matmul strategy (a distinct
optimizer feature that's landing soon — once chosen, its program config surfaces
through the same path), or tune compute-kernel configs.
