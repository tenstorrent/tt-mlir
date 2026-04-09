---
name: run-ops-mlir-snippets
description: >
  Compile and optionally execute every func.func in an ops.mlir-style snippet file
  (or every .mlir file in a directory) using `run_ops_mlir_snippets.py`. Use when the
  user wants to compile or run TTIR op snippets on device, test ops.mlir files, or
  check which ops compile/execute successfully.
---

# Run ops.mlir snippets (compile + execute)

Given an `ops.mlir`-style file (a module containing one `func.func` per unique TTIR op
configuration), compile each function to TTMetal (or TTNN) and optionally execute on
device.

The input can be:
- **A single `.mlir` file** (e.g. `ops.mlir`)
- **A directory** of `.mlir` files -- processes every `*.mlir` in it. Each file gets
  its own report. **Caution**: only point a directory at folders that contain ops-style
  snippet files, not raw/preprocessed model IR.

The driver script is `tools/scripts/run_ops_mlir_snippets.py`.

## Prerequisites

```bash
source env/activate
ttrt query --save-artifacts                    # creates system descriptor
export SYSTEM_DESC_PATH="$(pwd)/ttrt-artifacts/system_desc.ttsys"
```

## Basic usage

Single file:

```bash
python tools/scripts/run_ops_mlir_snippets.py path/to/ops.mlir
```

Multiple files (per-file reports + combined report at common parent):

```bash
python tools/scripts/run_ops_mlir_snippets.py path/to/*/ops.mlir
```

Directory (processes every `*.mlir` in the dir):

```bash
python tools/scripts/run_ops_mlir_snippets.py path/to/dir/
```

This compiles **and** executes every snippet. Each function is wrapped in its own
module, compiled via `compile_ttir_module_to_flatbuffer`, and run with `execute_fb`.
In directory mode, the device is opened once and shared across all files.

## Flags

| Flag | Effect |
|------|--------|
| `--skip-exec` | Compile only; do not open a device or run |
| `--target {ttmetal,ttnn}` | Compile target (default: `ttmetal`) |
| `--sys-desc PATH` | Override `SYSTEM_DESC_PATH` |
| `--output-root DIR` | Root for artifact dirs (default: `.`) |
| `--save-artifacts` | Keep flatbuffers / compiled MLIR under the artifact dir |
| `--print-ir` | Print compiled MLIR to stdout |
| `--fail-fast` | Stop on first compile or execution failure |
| `--disable-eth-dispatch` | Same as pytest `--disable-eth-dispatch` |

## Common workflows

### Compile-only triage (no device needed)

Use `--skip-exec` to find which ops fail at compile time without requiring hardware:

```bash
python tools/scripts/run_ops_mlir_snippets.py path/to/ops.mlir --skip-exec
```

### Multi-graph model directory

After running the **ttir-model-op-analysis** skill on a multi-graph directory like
`vllm_opt/`, each graph gets its own subdirectory with an `ops.mlir`. Pass all of
them in one command:

```bash
python tools/scripts/run_ops_mlir_snippets.py vllm_opt/*/ops.mlir --skip-exec
```

This writes `ops-run-report.txt` next to each `ops.mlir`, plus a combined
`ops-run-report.txt` at the common parent with per-file summaries and all
failures in one place:

```
vllm_opt/
  ops-run-report.txt            # combined report across all graphs
  graph1/
    ops.mlir
    ops-run-report.txt          # compile results for graph1
  graph2/
    ops.mlir
    ops-run-report.txt          # compile results for graph2
```

**Important**: pass the specific `ops.mlir` files, not the subdirectories. The
subdirectories also contain `preprocessed.mlir` (the full model graph), which is
not a snippet file and will produce a useless failure report if the runner tries
to process it.

### Fail-fast to find the first broken op

```bash
python tools/scripts/run_ops_mlir_snippets.py path/to/ops.mlir --fail-fast
```

### Save artifacts for debugging

```bash
python tools/scripts/run_ops_mlir_snippets.py path/to/ops.mlir \
    --save-artifacts --output-root /tmp/snippets --print-ir
```

Artifacts land in `<output-root>/ops_mlir_snippets/<filename>/<func_name>/<target>/`.

## Report

The script writes a **`<stem>-run-report.txt`** in the same directory as each input
`.mlir` file (e.g. `ops-run-report.txt` for `ops.mlir`). In directory mode, each file
gets its own report. The report has three sections:

1. **Summary** at top -- target, mode, pass/fail counts at a glance.
2. **Per-op table** -- one row per function showing compile (and execute) status.
3. **Failure details** -- numbered list with the Python exception **and**
   the captured MLIR diagnostics (L1 memory exceeded, missing parser, etc.).

Example (compile-only):

```
target:  ttmetal
input:   /path/to/ops.mlir
mode:    compile-only
total:   50 ops

  compile: 47/50 passed, 3 failed

────────────────────────────────────────────────────────────────────────

  func_name   compile
  ──────────  ───────
  softmax_0   ok
  matmul_0    FAILED
  reshape_0   FAILED
  ...

────────────────────────────────────────────────────────────────────────

  Failure details (3)

  [1] matmul_0 — compile FAILED
      exception: Failed to run pass manager
      diagnostics:
        can't find feasible allocation because all 8 var(s) are bound
        error: 'func.func' op required L1 memory usage 3309568 exceeds
               memory capacity 1395424 (usable space is [103712, 1499136))

  [2] reshape_0 — compile FAILED
      exception: No parser found for opview <class '...ReshapeOp'>
```

With `--skip-exec`, the execute column is omitted. Diagnostics are captured from
C-level stderr so MLIR allocator errors, verification failures, etc. appear
in the report even though the Python exception only says "Failed to run pass manager".

## Interpreting stdout

The script also prints a banner per snippet to stdout:

```
============================================================
Snippet: ops.mlir/softmax_0
============================================================
  compile: ok
  execute: ok
```

On failure you'll see `compile: FAILED: <error>` or `execute: FAILED: <error>`.
At the end: either `all N snippet(s) succeeded across M file(s)` or
`N snippet(s) failed across M file(s)`.

## Error handling

- **Compile failures** skip to the next snippet (unless `--fail-fast`).
- **Execution failures** close and re-open the device before continuing, so one hang
  doesn't block the rest of the run.
- If a snippet causes a device hang that persists across re-open, use `--skip-exec` to
  isolate compile issues, then test individual snippets by extracting the function into
  its own file.

## Generating ops.mlir

If you don't already have an `ops.mlir`, see the **ttir-model-op-analysis** skill which
produces one from a model's TTIR dump via `tools/scripts/ttir_model_op_inventory.py`.
