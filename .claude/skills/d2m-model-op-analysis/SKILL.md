---
name: ttir-model-op-analysis
description: >
  Given a `.mlir` file (or a directory of `.mlir` files) with TTIR ops, run the same TTIR
  normalization passes as `TTIRToTTMetalFrontendPipeline` before D2M, then produce per-file
  outputs: `preprocessed.mlir`, `ttir-op-report.txt` (op counts from normalized IR),
  and `ops.mlir` (one func per unique op configuration, golden-style). Optional: per-pass IR dumps.
---

# TTIR model op analysis (post-frontend-normalization)

Given a `.mlir` file (or a directory of `.mlir` files, e.g. a multi-graph model)
containing TTIR ops (from vLLM, torch-mlir, or tt-forge), normalize it the way the
compiler does before D2M, then inventory which TTIR ops (and shapes) appear. **Do not**
open `TTIRToD2M.cpp`, grep `populateTTIRToD2MPatterns`, or produce "D2M coverage"
unless the user explicitly asks for that comparison.

The input can be:
- **A single `.mlir` file** (e.g. `model.mlir`)
- **A directory** of `.mlir` files (e.g. `vllm_opt/` containing `graph1.mlir`,
  `graph2.mlir`). Each file is processed independently with per-graph reports.

## Step 1: Run the frontend TTIR normalization pipeline

These passes match `createTTIRToTTMetalFrontendPipeline` in
`lib/Dialect/TTMetal/Pipelines/TTMetalPipelines.cpp`:

| # | Pass flag | Short name (for filenames) |
|---|-----------|---------------------------|
| 1 | `--canonicalize` | `canonicalize` |
| 2 | `--ttir-predicate-type-alignment` | `predicate-type-alignment` |
| 3 | `--ttir-element-type-normalization` | `element-type-normalization` |
| 4 | `--ttir-to-ttir-decomposition` | `decomposition` |
| 5 | `--ttir-explicate-tms` | `explicate-tms` |
| 6 | `--ttir-erase-inverse-ops` | `erase-inverse-ops` |
| 7 | `--ttir-move-reshape-to-constant` | `move-reshape-to-constant` |
| 8 | `--ttir-fold-constant-reshape-broadcast` | `fold-constant-reshape-broadcast` |
| 9 | `--ttir-implicit-broadcast-fold` | `implicit-broadcast-fold` |

### Single-file mode

Create `<basename>/` next to the input (basename = filename without `.mlir`):

```
<basename>/
  preprocessed.mlir              # IR after all passes above
  ttir-op-report.txt             # counts + sorted mnemonic list (see below)
  ops.mlir                       # one func per unique TTIR op configuration
```

```bash
source env/activate
INPUT="<input.mlir>"
BASENAME="$(basename "$INPUT" .mlir)"
mkdir -p "$BASENAME"

ttmlir-opt \
  --canonicalize \
  --ttir-predicate-type-alignment \
  --ttir-element-type-normalization \
  --ttir-to-ttir-decomposition \
  --ttir-explicate-tms \
  --ttir-erase-inverse-ops \
  --ttir-move-reshape-to-constant \
  --ttir-fold-constant-reshape-broadcast \
  --ttir-implicit-broadcast-fold \
  -o "$BASENAME/preprocessed.mlir" \
  "$INPUT"
```

### Directory mode (multi-graph models)

When the input is a directory like `vllm_opt/` containing `graph1.mlir`,
`graph2.mlir`, etc., loop over each file and create a subdirectory per graph.
Each subdirectory uses clean names -- the folder provides the graph context:

```
vllm_opt/
  graph1.mlir                        # original input
  graph2.mlir
  ttir-op-report.txt                 # combined report across all graphs
  graph1/                            # per-graph output folder
    preprocessed.mlir
    ttir-op-report.txt
    ops.mlir
  graph2/
    preprocessed.mlir
    ttir-op-report.txt
    ops.mlir
```

```bash
source env/activate
INPUT_DIR="<dir>"
for f in "$INPUT_DIR"/*.mlir; do
  STEM="$(basename "$f" .mlir)"
  mkdir -p "$INPUT_DIR/$STEM"
  ttmlir-opt \
    --canonicalize \
    --ttir-predicate-type-alignment \
    --ttir-element-type-normalization \
    --ttir-to-ttir-decomposition \
    --ttir-explicate-tms \
    --ttir-erase-inverse-ops \
    --ttir-move-reshape-to-constant \
    --ttir-fold-constant-reshape-broadcast \
    --ttir-implicit-broadcast-fold \
    -o "$INPUT_DIR/$STEM/preprocessed.mlir" \
    "$f" || { echo "FAILED on $f"; continue; }
done
```

#### `ttir-op-report.txt` and `ops.mlir`

**Do not hand-roll parsing.** From the repo root (any Python 3.12+; no venv import deps).
The script accepts multiple paths, so pass all preprocessed files in one command:

```bash
python tools/scripts/model_breakdown/ttir_model_op_inventory.py "$INPUT_DIR"/*/preprocessed.mlir
```

This writes `ttir-op-report.txt` and `ops.mlir` next to each `preprocessed.mlir`,
plus a combined `ttir-op-report.txt` at the common parent directory with per-file
summary and merged mnemonic counts.

Optional flags: `--report PATH`, `--ops PATH` (single-file only), `-v`.

The script reports: total `"ttir.*"` instances, per-mnemonic counts (with optional percentages),
distinct mnemonic count, and distinct op configurations (SSA-normalized op lines). **`ops.mlir`**
follows the same shape as `test/python/golden/mlir_snippets/models/qwen3_4b/ops.mlir`: one
`func.func` per unique configuration (`<mnemonic>_<index>`), args `%arg0`, ..., single op + `return`.
For multi-result ops such as `ttir.sort` and `ttir.topk`, the generated op must use MLIR
multi-result binding syntax (`%0:2 = ...`) and return projected results (`return %0#0, %0#1 :
...`). If you see `operation defines 2 results but was provided 1 to bind`, regenerate with
`tools/scripts/model_breakdown/ttir_model_op_inventory.py` instead of hand-editing every snippet.

`ttir.full` and `ttir.constant` are **excluded** from `ops.mlir` unless their result is directly
returned by the module -- they almost always just produce values consumed by other ops, and are
already inlined as const producers inside those ops' test functions. The report still counts them.

It assumes **one generic TTIR op per line** (normal `ttmlir-opt` output). If needed, sanity-check
parse with `ttmlir-opt` on `ops.mlir`.

### Optional: dump IR after each pass

Only if the user asks. Same pass loop as above; write
`<basename>/<basename>.<short-name>.mlir` for each stage. Final stage is the source for
`ttir-op-report.txt` and `ops.mlir` (same as `*.implicit-broadcast-fold.mlir` when you ran
all nine).

```bash
source env/activate
INPUT="<input.mlir>"
BASENAME="$(basename "$INPUT" .mlir)"
mkdir -p "$BASENAME"
PREV="$INPUT"
declare -a PASSES=(
  "--canonicalize|canonicalize"
  "--ttir-predicate-type-alignment|predicate-type-alignment"
  "--ttir-element-type-normalization|element-type-normalization"
  "--ttir-to-ttir-decomposition|decomposition"
  "--ttir-explicate-tms|explicate-tms"
  "--ttir-erase-inverse-ops|erase-inverse-ops"
  "--ttir-move-reshape-to-constant|move-reshape-to-constant"
  "--ttir-fold-constant-reshape-broadcast|fold-constant-reshape-broadcast"
  "--ttir-implicit-broadcast-fold|implicit-broadcast-fold"
)
for entry in "${PASSES[@]}"; do
  IFS='|' read -r FLAG SHORTNAME <<< "$entry"
  OUT="$BASENAME/$BASENAME.$SHORTNAME.mlir"
  ttmlir-opt $FLAG -o "$OUT" "$PREV" || { echo "FAILED at: $FLAG"; break; }
  PREV="$OUT"
done
```

**NOTE:** If a pass fails, check `include/ttmlir/Dialect/TTIR/Transforms/Passes.td` for the
registered flag. Some setups need
`--ttcore-register-device="system-desc-path=..."` before other passes.

## Step 2: Fill the report and `ops.mlir`

Run **`tools/scripts/model_breakdown/ttir_model_op_inventory.py`** on each `preprocessed.mlir` file
(single-file mode). See above.

## Next: compile/run the snippets

To compile or execute the generated `ops.mlir` on device, see the
**run-ops-mlir-snippets** skill.
