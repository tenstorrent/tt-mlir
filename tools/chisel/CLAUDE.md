# Chisel — Differential Debugging Tool

Chisel is a differential debugging tool for tt-mlir that performs op-by-op
comparison between golden (CPU/PyTorch) and device (TT hardware) execution at
the TTNN dialect level.

## Architecture

Chisel is a **passive observer** — it does not drive compilation or execution.
It hooks into TTRT binary execution via `DebugHooks` preop/postop callbacks and
compares each TTNN op's device output against a CPU golden reference.

Key design decisions:
- **Single TTNN module** for both golden and device (no TTIR/TTNN correlation)
- **Singleton `ChiselContext`** because `DebugHooks` callbacks are plain
  functions that need shared state
- **Reuses `tools/golden/GOLDEN_MAPPINGS`** for TTNN op golden implementations
- **No compilation step** — receives a pre-compiled TTNN module
- **Library only** — no CLI entry point; caller registers callbacks and drives
  TTRT execution

## Module Layout

```
tools/chisel/chisel/
├── __init__.py        # Package init
├── context.py         # ChiselContext singleton (central orchestrator)
├── callbacks.py       # preop/postop callback functions for DebugHooks
├── executor.py        # GoldenExecutor (TTNN ops on CPU via PyTorch)
├── registry.py        # TTNN op tracking and tensor registration
├── tensors.py         # TensorPool and TensorValue
├── ops.py             # IRModule wrapper, hash_location, op input/output extraction
├── report.py          # CSV report writer
└── utils.py           # Location parsing, dtype maps, runtime tensor conversion
```

## Data Flow

For each TTNN op during TTRT execution:
1. **preop**: Capture device input tensors, copy to golden tensor pool
2. **HW executes op**
3. **postop**: Capture device output, look up op in `GOLDEN_MAPPINGS`, execute
   golden function on CPU, compare (PCC, abs error, rel error), write CSV row

## Key Dependencies

- `tools/golden/mapping.py` — `GOLDEN_MAPPINGS` dict, `get_golden_function()`, `GoldenMapTensor`
- `tools/golden/metrics.py` — Unified PCC/atol/rtol comparison (shared with builder and ttrt)
- `ttrt.runtime.DebugHooks` — callback registration point

## Design Docs

See `tools/chisel/docs/` for detailed architecture and feature overview
documentation.
