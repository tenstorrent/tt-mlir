---
name: shard-advise
description: Get L1 sharding advice for a ttnn graph via the ttnn-jit shard advisor. Use when the user wants per-op TTNN memory-layout / sharding recommendations for a decoder layer, attention/MLP block, or any ttnn function — either from an existing TTIR .mlir file or by tracing a Python ttnn function. Produces a structured report.json (ops, layouts, reshards, L1 spill). Not for perf profiling (use diagnose-tt-perf) or for inspecting an already-compiled flatbuffer (use analyze-sharding).
---

# Instructions

Use the `ttnn-advise` CLI to get per-op L1 layout / sharding advice from the
greedy optimizer. `capture` traces the ttnn function straight into the **TTNN
dialect** (a ttnn-framework model *is* TTNN) and runs the greedy optimizer with
no lowering; `mlir` reads an existing TTIR file. Either way it writes a
browsable artifact set.

**Always run it in a fresh process (a Bash call), and read `report.json` for the
result — do not scrape stdout.** The tool prints only a 5-line summary to stdout;
all pipeline/device logging goes to `pipeline.log` and stderr.

## Setup (once per shell)

```bash
source env/activate
export SYSTEM_DESC_PATH="$(pwd)/ttrt-artifacts/system_desc.ttsys"   # required
# capture mode also needs (tracing opens a real ttnn device):
export LIBRARY_PATH="$(pwd)/.local/libnsl-shim:$LIBRARY_PATH"
export PYTHONPATH="$TT_METAL_HOME/tools:$PYTHONPATH"
```

If `SYSTEM_DESC_PATH` is missing: `ttrt query --save-artifacts` first.

## Two modes

### `mlir` — advise on an existing TTIR file (preferred; no device)
Fast, deterministic, needs only `SYSTEM_DESC_PATH` (the optimizer queries a mock
device built from it). Use whenever a `.ttir.mlir` dump already exists.

```bash
ttnn-advise mlir path/to/model.ttir.mlir --out /tmp/advice 2>/dev/null
```

### `capture` — trace a Python ttnn function on device, then advise
Use when there's no TTIR yet, only a runnable ttnn function. The target module
**must** define both the function and `make_inputs(device)` returning its args:

```python
# target.py
import ttnn
def decode(x, ...): ...                 # the ttnn fn to advise
def make_inputs(device):                # returns the ttnn.Tensor args, on device
    return (mk(device, ...), ...)
```

```bash
ttnn-advise capture target.py:decode --out /tmp/advice 2>/dev/null
```

`capture` traces directly to TTNN by default (`--tracer ttnn`). A ttnn op with
no tracer handler fails loudly with `ttnn.<op> has no direct-TTNN handler yet` —
a genuine add-op signal, not a dead end. `--tracer interception` routes through
the older TTIR path as a stopgap for such ops.

Common flags (both modes): `--opt-level N` (default 2), `--out DIR`.

## Reading the result

The out dir contains:
- **`report.json`** — the machine-readable result. Read this:
  - `ops[]`: `{index, op, layout}` e.g. `l1/width_sharded/1x64 cores=(0,0)-(7,7)`
  - `reshards[]`: `{kind, producer, consumer, from, to, output_revert}`
  - `spill`: `{ran, total_spills}` (near-zero = healthy)
  - `total_ops`, `final_choices`, `artifacts{...}`
- `report.txt` — human-readable op layouts + reshards + decision-trace rationale.
- `final_ir.mlir` — authoritative final TTNN IR (every layout + reshard explicit).
- `pipeline.log` — captured native pipeline/device output (only for debugging).

```bash
python -c "import json; d=json.load(open('/tmp/advice/report.json')); \
  print('\n'.join(f\"{o['index']:>3} {o['op']:<45} {o['layout']}\" for o in d['ops']))"
```

## Scope — what the advisor does and does NOT recommend

It reasons about **L1 memory layout / sharding**, and for the sharding strategy
it picks it also reports the matmul **program config** the optimizer generated
and backend-validated (e.g. `matmul_multi_core_reuse_multi_cast_1d @8x8`, in each
op's `program_config`). It is faithful-to-source on dtypes: it traces the model's
bf16/bfp8_b/bfp4_b as written and uses the real footprint, but does not *recommend*
a dtype change.

What it does **not** do (and its silence is not a signal about these): pick the
**DRAM-sharded-weight** matmul strategy (a distinct optimizer feature landing
soon — once chosen, its program config surfaces through the same path), or tune
**compute-kernel configs** (hifi2/hifi4). Comparing to a hand-tuned model, expect
agreement on the layout skeleton + the chosen-strategy program config, and gaps
on precision and the DRAM-sharded-weight strategy.

## Gotchas

- **Fresh process per run** — the optimizer's device context is process-global;
  never run two advisories in one process.
- **Rebuild reaches the advisor only via a full build.** The CLI runs the
  pipeline through the Python bindings (`libTTMLIRCompiler.so`), a different
  artifact from `ttmlir-opt`. After changing optimizer/pass/rule-book code, run
  `cmake --build build` (relink + reinstall) — building only `ttmlir-opt` will
  silently run the OLD code in the advisor.
- If `capture` fails with `ttnn.<op> has no direct-TTNN handler yet`, add that
  handler in `tools/ttnn-jit/_src/ttnn_emit_tracer.py` (bounded per-op work), or
  use `--tracer interception` for that model in the meantime.
