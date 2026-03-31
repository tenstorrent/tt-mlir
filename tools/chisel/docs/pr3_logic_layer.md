# PR 3: Orchestration

## Goal

Add the `ChiselContext` singleton that wires all components together, plus
the `callbacks.py` module with plain functions compatible with `DebugHooks.get()`,
and the utility/metrics modules (`utils.py`, `metrics.py`) that are consumed by
the orchestration layer. After this PR, Chisel is fully functional as a
library — callers can register callbacks and observe TTNN binary execution.

## Files

### New Files

| File | Description |
|------|-------------|
| `tools/chisel/chisel/context.py` | `ChiselContext` singleton — central orchestrator |
| `tools/chisel/chisel/callbacks.py` | `chisel_pre_op_callback`, `chisel_post_op_callback` |
| `tools/chisel/chisel/utils.py` | Location parsing, dtype maps, runtime tensor conversion |

### Modified Files

| File | Change |
|------|--------|
| `tools/chisel/chisel/__init__.py` | Export `ChiselContext`, callback functions |
| `tools/chisel/CMakeLists.txt` | Add new source files |

## Implementation Details

### `utils.py`

Consolidated utility module with three concerns:

**Location parsing:**
- `parse_op_location(op_location: str) -> Tuple[int, int]` — parse runtime location string to tuple

**Dtype maps:**
- `ttir_dtype_maps: Dict` — TTIR element types to PyTorch dtypes
- `ttrt_dtype_maps: Dict` — TTRT runtime tensor types to PyTorch dtypes

**Runtime utilities:**
- `get_torch_tensor(tensor: RtTensor) -> torch.Tensor` — convert runtime tensor to PyTorch
- `debug_wrap(*, debug: bool = False)` — decorator factory for pdb integration

### Metrics — imported from `tools/golden/metrics.py`

Chisel does **not** have a local `metrics.py`. All comparison functions are
imported from the unified `golden.metrics` module (created in PR 0c):

```python
from golden.metrics import compute_pcc, compute_atol, compute_rtol
```

This module provides:
- `compute_pcc(golden, calculated)` — Pearson Correlation Coefficient (pure torch)
- `compute_atol(golden, calculated)` — maximum absolute difference
- `compute_rtol(golden, calculated)` — maximum relative error
- `compute_metrics(golden, calculated)` — full comparison dict (pcc, atol, rtol, allclose, mae, rmse, cosine_similarity)

See [PR 0c](pr0c_unified_metrics.md) for the full API and implementation details.

### Porting Notes for `utils.py`

**From `runtime/tools/chisel/chisel/utils/location.py`:**
- **Port:** `parse_op_location()` — used in context.py preop/postop to parse
  runtime location strings
- `hash_location()` and `UNKNOWN_LOCATION` are already inlined in `ops.py` (PR 1)

**From `runtime/tools/chisel/chisel/utils/runtime_utils.py`:**
- **Port as-is:** `ttir_dtype_maps`, `ttrt_dtype_maps`, `get_torch_tensor()`
- `update_device_tensor()` is NOT ported here — it belongs in context.py
  where it's used during preop callback

**From `runtime/tools/chisel/chisel/utils/debug.py`:**
- **Port as-is:** `debug_wrap()` decorator

### `context.py`

```python
class ChiselContext:
    _instance: Optional["ChiselContext"] = None

    def __init__(
        self,
        output_dir: Path,
        report_path: Path,
        main_fn: str = "main",
        caching: bool = True,
    ):
        ChiselContext._instance = self

        # MLIR module initialized lazily on first preop callback —
        # the binary (containing TTNNBinary.mlir.source) is available there.
        self.device_ir_module: IRModule | None = None

        # Dual tensor pools — golden (CPU) and device (hardware)
        self.golden_tensor_pool = TensorPool(caching=caching, output_dir=output_dir / "golden")
        self.device_tensor_pool = TensorPool(caching=caching, output_dir=output_dir / "device")

        # Single-module registry
        self.registry = Registry(module=self.device_ir_module)
        self.registry.load_all_ops()

        # Golden executor reusing GOLDEN_MAPPINGS
        self.executor = GoldenExecutor(self.registry, self.golden_tensor_pool)

        # Reporting
        self.report = ReportWriter(report_path, self.device_ir_module.get_asm_state())

    @classmethod
    def get_instance(cls) -> "ChiselContext":
        if cls._instance is None:
            raise RuntimeError("ChiselContext not initialized")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def preop(self, binary, program_context, op_context) -> None:
        """
        Called before each TTNN op executes on device.

        1. Get op location via tt_runtime.runtime.get_op_loc_info(op_context)
        2. Capture device input tensors from op_context
        3. Store in device_tensor_pool
        4. Copy inputs to golden_tensor_pool for golden execution
        """

    def postop(self, binary, program_context, op_context) -> None:
        """
        Called after each TTNN op executes on device.

        1. Capture device output tensor from op_context
        2. Store in device_tensor_pool
        3. Look up TTNN op in registry by location
        4. Execute golden function via GoldenExecutor
        5. Compare golden vs device output (PCC, abs_err, rel_err)
        6. Write row to CSV report
        """
```

### `callbacks.py`

Thin module with two plain functions compatible with `DebugHooks.get()`:

```python
from chisel.context import ChiselContext

def chisel_pre_op_callback(binary, program_context, op_context):
    ChiselContext.get_instance().preop(binary, program_context, op_context)

def chisel_post_op_callback(binary, program_context, op_context):
    ChiselContext.get_instance().postop(binary, program_context, op_context)
```

These use the same `(binary, program_context, op_context)` signature as
builder's own callbacks (see `builder_runtime.py:584-601`), making them a
drop-in replacement.

### `__init__.py` exports

```python
from chisel.context import ChiselContext
from chisel.callbacks import chisel_pre_op_callback, chisel_post_op_callback
```

## Porting Notes for `context.py`

### From `runtime/tools/chisel/chisel/core/context.py`

The old ChiselContext is the most heavily modified file. Here's what changes:

**Remove entirely:**
- `ttir_module` parameter — no golden TTIR module
- `self.golden_ir_module` / `self.modules[ExecutionType.GOLDEN]` — single module only
- `compare_outputs()` — comparison now happens inline in `postop()`
- `get_corresponding_tensors()` — no cross-module tensor mapping needed
- `skip_group()` — no group skipping logic
- `function_argument_bridge()` — handled differently in callback flow
- `run()` — Chisel is passive, caller drives execution
- `bind_callbacks()` — caller registers callbacks directly
- `setup_ttrt()` — Chisel doesn't own runtime setup
- `load_inputs_from_disk()` / `generate_random_inputs()` — inputs captured from
  runtime in preop callback, not generated upfront

**Add new:**
- Singleton pattern (`_instance`, `get_instance()`, `reset_instance()`)
- `preop()` method — captures device inputs, copies to golden pool
- `postop()` method — captures device output, runs golden, compares, reports

**Adapt from old `preop()`/`postop()` in old context:**
The old context has these methods but they work with dual modules and the
Registry's `should_compare()`. The new versions are simpler:
- Old `preop()` checked `program_context`, extracted tensor refs, updated device
  pool, and handled function arguments. New `preop()` does the same but without
  ExecutionType branching.
- Old `postop()` called `compare_outputs()` which used the Registry to find
  corresponding golden/device tensors. New `postop()` directly runs the golden
  executor and compares.

**Runtime API usage (same as old):**
- `tt_runtime.runtime.get_op_loc_info(op_context)` — get operation location string
- `tt_runtime.runtime.get_op_output_tensor(op_context, program_context)` — get device output
- `tt_runtime.runtime.get_op_debug_str(op_context)` — get debug string
- `tt_runtime.runtime.get_op_input_refs()` — get input tensor references
- `tt_runtime.runtime.get_op_output_ref()` — get output tensor reference

### `callbacks.py` — NEW

This is a new file. The old chisel registered callbacks via `bind_callbacks()`
on ChiselContext, which called `DebugHooks.get(self.preop, self.postop)`.

The new design separates callbacks into their own module because:
- `DebugHooks.get()` expects plain functions, not bound methods
- The singleton pattern lets callbacks access context without closure capture
- It mirrors the builder's pattern (`pre_op_get_callback_fn`, `post_op_get_callback_fn`)

## Test Plan

### `test_utils.py`
- `test_parse_op_location()` — known location strings yield expected `(line, col)` tuples
- `test_dtype_maps()` — verify all expected dtype mappings exist and are valid torch dtypes
- `test_get_torch_tensor()` — mock runtime tensor, verify conversion to torch.Tensor

**Test dependencies:** `torch`, `ttrt.runtime` (or mocks). Tested here alongside
runtime because these utilities need runtime types to exercise meaningfully.

### `test_context.py`

> **Note:** Metrics tests live in `test/python/golden/test_metrics.py` and are
> covered by PR 0c. No metrics tests in this PR.

**Singleton lifecycle tests (no hardware needed):**
- `test_singleton_not_initialized()` — `get_instance()` raises `RuntimeError`
  before any construction
- `test_singleton_construction()` — construct with mocked TTNN module,
  `get_instance()` returns same object
- `test_singleton_reset()` — call `reset_instance()`, `get_instance()` raises again
- `test_singleton_reinitialization()` — construct → reset → construct new →
  `get_instance()` returns new object

**Component wiring tests (mock MLIR module):**
- `test_components_initialized()` — after construction, verify `device_ir_module`,
  `golden_tensor_pool`, `device_tensor_pool`, `registry`, `executor`, `report`
  are all non-None
- `test_registry_loaded()` — verify `registry.load_all_ops()` was called during init

### `test_callbacks.py`

- `test_pre_op_delegates_to_context()` — mock `ChiselContext.get_instance()`,
  call `chisel_pre_op_callback(binary, prog_ctx, op_ctx)`, verify
  `context.preop(binary, prog_ctx, op_ctx)` was called
- `test_post_op_delegates_to_context()` — same for postop
- `test_callback_without_context_raises()` — call callback without initializing
  context, verify `RuntimeError` from `get_instance()`

**Test dependencies:** `unittest.mock` for mocking context and runtime objects.
No hardware or MLIR needed for callback delegation tests.

## Dependencies

- **PR 2** — `registry.py`, `executor.py`, `report.py`
- **PR 1** — `tensors.py`, `ops.py`
- **PR 0a** — DebugHooks refactor must land before this PR. This is where
  Python callbacks get registered and invoked via `DebugHooks.get()`. Without
  the fix, callback copies cause segfaults when called from tt-xla without GIL.
- **PR 0c** — Unified metrics in `tools/golden/metrics.py`. This PR's
  `context.py` imports `compute_pcc`, `compute_atol`, `compute_rtol` from
  `golden.metrics` instead of a local `metrics.py`.
