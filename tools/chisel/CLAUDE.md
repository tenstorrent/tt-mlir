# Chisel — Differential Debugging Tool

Chisel is a differential debugging tool for tt-mlir that performs op-by-op
comparison between golden (CPU/PyTorch) and device (TT hardware) execution at
the TTNN dialect level.

## Architecture

Chisel is a **passive observer** — it does not drive compilation or execution.
It hooks into TTRT binary execution via `DebugHooks` callbacks (preProgram,
postProgram, preOp, postOp) and compares each TTNN op's device output against
a CPU golden reference.

Key design decisions:
- **Single TTNN module** for both golden and device (no TTIR/TTNN correlation)
- **Hierarchical state model**: `ChiselContext → BinaryState → ProgramState`
  — each level owns appropriate state (global tensor pool, per-binary MLIR/IRModule,
  per-program golden pool and op iterator)
- **Singleton `ChiselContext`** because `DebugHooks` callbacks are plain
  functions that need shared state
- **Iterator-based op tracking** — `ProgramState.op_iter` advances with each
  preOp/postOp callback, no manual index or reset needed
- **Reuses `tools/golden/GOLDEN_MAPPINGS`** for TTNN op golden implementations
- **MLIR from flatbuffer** — reads TTNN MLIR string from `TTNNBinary.mlir.source`
  and parses it in `BinaryState` on first encounter
- **Library only** — no CLI entry point; caller registers callbacks and drives
  TTRT execution

## Module Layout

```
tools/chisel/chisel/
├── __init__.py        # Package init
├── context.py         # ChiselContext, BinaryState, ProgramState
├── callbacks.py       # 4 callback functions for DebugHooks
├── executor.py        # Golden execution function (TTNN ops on CPU via PyTorch)
├── tensors.py         # TensorPool (stores GoldenMapTensor directly)
├── ops.py             # IRModule wrapper, hash_location, op input/output extraction
├── report.py          # CSV report writer
└── utils.py           # Location parsing, dtype maps, runtime tensor conversion
```

## Data Flow

For each program during TTRT execution:
1. **preProgram**: Get/create `BinaryState` (parse MLIR if new binary) and
   `ProgramState`. Reset op iterator. Copy
   `global_tensor_pool` → program's golden pool. Copy program input tensors
   from device → golden pool (via `get_program_input_ids`).
2. For each TTNN op:
   - **preOp**: `next(op_iter)`, handle skip-mode input stashing
   - **HW executes op**
   - **postOp**: Capture device output, execute golden via `GOLDEN_MAPPINGS`,
     compare (PCC, abs error, rel error), write CSV row
3. **postProgram**: Copy program's golden pool → `global_tensor_pool`,
   finalize report section.

## Key Dependencies

- `tools/golden/mapping.py` — `GOLDEN_MAPPINGS` dict, `get_golden_function()`, `GoldenMapTensor`
- `tools/golden/metrics.py` — Unified PCC/atol/rtol comparison (shared with builder and ttrt)
- `ttrt.runtime.DebugHooks` — callback registration point
- `TTNNBinary.mlir` — TTNN MLIR text stored in flatbuffer (`include/ttmlir/Target/TTNN/binary.fbs`), accessed via `ttrt.binary`

## Design Docs

See `tools/chisel/docs/` for detailed architecture and feature overview
documentation.
