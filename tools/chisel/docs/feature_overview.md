# Chisel: Feature Overview

## What is Chisel?

Chisel is a **differential debugging tool** for TT-MLIR that performs op-by-op
comparison between **golden** (CPU reference) and **device** (TT hardware)
execution — both at the **TTNN dialect level**.

For every TTNN operation executed on hardware, Chisel replays the same operation
on CPU using PyTorch-based golden reference implementations and computes
numerical accuracy metrics. This enables developers to pinpoint exactly which
hardware operation introduces numerical divergence.

## Key Capabilities

- **Op-by-op comparison**: Every TTNN op executed on device is independently
  compared against its golden CPU counterpart.
- **Accuracy metrics**: Per-op computation of:
  - **PCC** (Pearson Correlation Coefficient)
  - **Absolute error** (max absolute difference)
  - **Relative error** (max relative difference)
- **CSV reporting**: Structured per-op report with operation names, locations,
  input/output tensor info, and all accuracy metrics.
- **Tensor caching**: Optional disk-based caching of golden and device tensors
  for post-mortem analysis.
- **Callback-driven**: Integrates non-invasively into any TTRT execution flow
  via `DebugHooks` preop/postop callbacks — no separate CLI or execution
  pipeline required.

## How It Works

Chisel operates as a **passive observer** during TTRT binary execution:

1. **Initialize** a `ChiselContext` singleton with the TTNN MLIR module and
   output configuration.
2. **Register** Chisel's preop/postop callback functions with
   `ttrt.runtime.DebugHooks.get()`.
3. **Execute** the TTNN flatbuffer binary through TTRT as usual.
4. For each TTNN op, Chisel's callbacks automatically:
   - Capture device input/output tensors
   - Replay the op on CPU using golden reference implementations from
     `tools/golden/`
   - Compute and record accuracy metrics
5. **Inspect** the generated CSV report.

### Usage Example

```python
from chisel.context import ChiselContext
from chisel.callbacks import chisel_pre_op_callback, chisel_post_op_callback
import ttrt.runtime

# Initialize the singleton context
ctx = ChiselContext(
    ttnn_module=module,
    output_dir=Path("./chisel_output"),
    report_path=Path("./chisel_report.csv"),
)

# Register callbacks
debug_hooks = ttrt.runtime.DebugHooks.get(
    chisel_pre_op_callback,
    chisel_post_op_callback,
)

# Execute the binary — Chisel observes via callbacks
# ... run through ttrt as normal ...

# Cleanup
ChiselContext.reset_instance()
```

## What Changed From the Previous Chisel

| Aspect | Old Chisel | New Chisel |
|--------|-----------|------------|
| Location | `runtime/tools/chisel/` | `tools/chisel/` |
| Comparison level | TTIR (golden) vs TTNN (device) | TTNN (golden) vs TTNN (device) |
| Entry point | CLI via `main.py` with argparse | Library only — callback functions |
| Compilation | Chisel ran its own TTIR-to-TTNN pass pipeline | None — receives pre-compiled TTNN module |
| Execution | Chisel drove TTRT execution via `RtApi` | Passive — observes via callbacks |
| Context pattern | Single-use object created in `main()` | Singleton accessed by callbacks |
| Golden executor | Custom PyTorch mappings for TTIR ops | Reuses `tools/golden/GOLDEN_MAPPINGS` for TTNN ops |
| Packaging | `setup.py` with `pip install -e` | CMake `declare_mlir_python_sources()` |

### Why TTNN-Level Comparison?

The old approach compared TTIR (high-level) ops against TTNN (low-level) device
ops. This required a complex Registry to correlate ops across two different IR
representations, handle op fusion mismatches, and merge groups where TTIR ops
had no direct TTNN counterpart.

By comparing at the same TTNN level, the architecture is significantly
simplified:
- **One IR module** instead of two
- **Direct 1:1 op correspondence** — no cross-dialect correlation needed
- **No fusion mismatch handling** — both golden and device see the same ops
- **Reuse of existing golden mappings** from `tools/golden/`
