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
- **Two-phase rollout**: PR 1 delivers per-op isolation testing (slim
  `ChiselContext` with `ir_module`, `op_iter`, stashed inputs, 2 callbacks).
  PR 2 adds the full hierarchical state model and cross-op golden chaining.
- **Hierarchical state model (PR 2+)**: `ChiselContext → BinaryState → ProgramState`
  — each level owns appropriate state (global tensor pool, per-binary MLIR/IRModule,
  per-program golden pool and op iterator)
- **Singleton `ChiselContext`** because `DebugHooks` callbacks are plain
  functions that need shared state
- **Iterator-based op tracking** — `op_iter` advances with each
  preOp/postOp callback, no manual index or reset needed
- **Reuses `tools/golden/GOLDEN_MAPPINGS`** for TTNN op golden implementations
- **MLIR from flatbuffer** — reads TTNN MLIR string from `TTNNBinary.mlir.source`
  and parses it (in slim context for PR 1, in `BinaryState` for PR 2+)
- **Library only** — no CLI entry point; caller registers callbacks and drives
  TTRT execution

## Module Layout

**PR 1 (Single Op Isolation):**
```
tools/chisel/chisel/
├── __init__.py        # Package init
├── context.py         # Slim ChiselContext (ir_module, op_iter, stashed inputs)
├── callbacks.py       # preOp/postOp only (2 callbacks)
├── executor.py        # execute_golden(op, ir_module, inputs: dict)
├── ops.py             # IRModule wrapper, op input/output extraction
└── utils.py           # Dtype maps, runtime tensor conversion
```

**PR 2+ (Single Program Flow and beyond):**
```
tools/chisel/chisel/
├── __init__.py        # Package init
├── context.py         # Full ChiselContext/BinaryState/ProgramState hierarchy
├── callbacks.py       # 4 callback functions for DebugHooks
├── executor.py        # Golden execution + pool-aware wrapper
├── tensors.py         # TensorPool (stores GoldenMapTensor directly)
├── ops.py             # IRModule wrapper, op input/output extraction
├── report.py          # CSV report writer (PR 3)
└── utils.py           # Dtype maps, runtime tensor conversion
```

## Data Flow

**PR 1 — Isolation Mode (per-op, no chaining):**

For each TTNN op during TTRT execution:
1. **preOp**: `next(op_iter)`, copy device input tensors to host, stash
2. **HW executes op**
3. **postOp**: Execute golden with stashed device inputs via `GOLDEN_MAPPINGS`,
   capture device output, compare (PCC, abs error, rel error), log to stdout,
   discard golden output

**PR 2+ — Program Flow Mode (cross-op chaining):**

For each program during TTRT execution:
1. **preProgram**: Get/create `BinaryState` (parse MLIR if new binary) and
   `ProgramState`. Reset op iterator. Copy
   `global_tensor_pool` → program's golden pool. Copy program input tensors
   from device → golden pool (via `get_program_input_ids`).
2. For each TTNN op:
   - **preOp**: `next(op_iter)`, handle skip-mode input stashing
   - **HW executes op**
   - **postOp**: Execute golden with inputs from golden pool (previous golden
     outputs), store golden output in pool, capture device output, compare
     (PCC, abs error, rel error), write CSV row
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
