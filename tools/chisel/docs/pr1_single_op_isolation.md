# PR 1: Single Op Isolation Testing

## Goal

Deliver op-level isolation testing — each TTNN op is tested independently.
`preOp` copies device input tensors to host, `postOp` runs the golden function
via `GOLDEN_MAPPINGS` and compares against the device output. Golden outputs
are discarded after comparison — no cross-op tensor chaining.

This PR proves the core golden-vs-device comparison loop works for individual
ops without requiring program-level state or tensor persistence.

**Not included:** TensorPool, cross-op golden chaining, program-level callbacks
(`preProgram`/`postProgram`), ReportWriter (CSV), disk caching, cross-program
tensor sharing, skip mode, builder integration.

## Files

### New Files

| File | Description |
|------|-------------|
| `tools/chisel/CMakeLists.txt` | CMake packaging using `declare_mlir_python_sources` |
| `tools/chisel/chisel/__init__.py` | Package init + exports |
| `tools/chisel/chisel/ops.py` | `IRModule` wrapper, `get_op_inputs()`, `get_op_outputs()` |
| `tools/chisel/chisel/executor.py` | `execute_golden()` — CPU replay of a single TTNN op |
| `tools/chisel/chisel/utils.py` | Dtype maps, runtime tensor conversion |
| `tools/chisel/chisel/callbacks.py` | `preOp`/`postOp` only (2 callbacks) |
| `tools/chisel/chisel/context.py` | Slim `ChiselContext` — ir_module, op_iter, stashed inputs |

### Modified Files

| File | Change |
|------|--------|
| `tools/CMakeLists.txt` | Add `add_subdirectory(chisel)` under the Python bindings guard |

## Implementation Details

### `CMakeLists.txt`

Follow the `tools/builder/CMakeLists.txt` and `tools/golden/CMakeLists.txt` pattern
(`declare_mlir_python_sources` to register files, then `add_mlir_python_modules`
to copy them into the build/install tree so the package is importable):

```cmake
include(AddMLIRPython)

declare_mlir_python_sources(ChiselSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    chisel/__init__.py
    chisel/ops.py
    chisel/executor.py
    chisel/context.py
    chisel/callbacks.py
    chisel/utils.py
)

add_mlir_python_modules(ChiselPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/chisel"
  INSTALL_PREFIX "python_packages/chisel"
  DECLARED_SOURCES ChiselSources
)
```

Add to `tools/CMakeLists.txt`:
```cmake
if(TTMLIR_ENABLE_BINDINGS_PYTHON AND MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(builder)
  add_subdirectory(golden)
  add_subdirectory(chisel)   # <-- new
endif()
```

### `ops.py`

**Utility functions:**

```python
@cache
def get_op_outputs(op: Operation) -> list:
    """Extract tensor-like outputs (results with shape and element_type)."""

@cache
def get_op_inputs(op: Operation) -> list:
    """Extract tensor-like inputs (operands with shape and element_type)."""
```

**`IRModule`** — wraps an MLIR Module with caching and traversal. Accepts an
MLIR source string and parses it internally:

```python
class IRModule:
    def __init__(
        self,
        mlir_source: str,
        functions: List[str],
        current_function_name: str | None = None,
        ignored_ops: List[str] = [],
    ):
        # Parse the MLIR source string into a Module
        self.context = Context()
        self.module = Module.parse(mlir_source, self.context)
        ...

    def get_function(self) -> Operation: ...
    def get_function_inputs(self) -> List[BlockArgument]: ...
    def get_function_ops(self) -> List[Operation]: ...
    def get_asm_state(self) -> AsmState: ...

```

### `executor.py`

Standalone function that executes a single TTNN operation on CPU using
`GOLDEN_MAPPINGS`. Takes a plain dict of input tensors (not a TensorPool) and
returns the golden output without storing it anywhere.

```python
def execute_golden(op: Operation, ir_module: IRModule, inputs: dict) -> Any:
    """
    Execute a TTNN op on CPU via GOLDEN_MAPPINGS.

    1. Look up op type in GOLDEN_MAPPINGS via get_golden_function()
    2. If not found, raise RuntimeError (fail hard)
    3. Retrieve input tensors from the provided inputs dict
    4. Call golden function with PyTorch tensors
    5. Return result (caller decides whether to store or discard)
    """
```

**Fail-hard behavior:** If `get_golden_function(type(op))` returns `None`, the
function raises `RuntimeError(f"No golden implementation for {type(op).__name__}")`.

**Key dependency:** `tools/golden/mapping.py` — use `get_golden_function(type(op))`
to look up the golden callable.

**Note:** In PR 2, a pool-aware wrapper is added that pulls inputs from
`TensorPool` and stores outputs back. The core `execute_golden()` signature
remains unchanged.

### `utils.py`

Consolidated utility module:

**Dtype maps:**
- `mlir_dtype_maps: Dict` — MLIR element types to PyTorch dtypes
- `ttrt_dtype_maps: Dict` — TTRT runtime tensor types to PyTorch dtypes

**Runtime utilities:**
- `get_torch_tensor(tensor: RtTensor) -> torch.Tensor` — convert runtime tensor to PyTorch
- `debug_wrap(*, debug: bool = False)` — decorator factory for pdb integration

### `context.py`

Slim `ChiselContext` singleton — holds only what's needed for per-op isolation
testing. No `BinaryState`/`ProgramState` hierarchy (added in PR 2).

```python
class ChiselContext:
    _instance: Optional["ChiselContext"] = None

    def __init__(self, ir_module: IRModule):
        ChiselContext._instance = self
        self.ir_module = ir_module
        self.op_iter: Iterator = iter(ir_module.get_function_ops())
        self._current_op: Operation | None = None
        self._stashed_inputs: dict | None = None

    @classmethod
    def get_instance(cls) -> "ChiselContext":
        if cls._instance is None:
            raise RuntimeError("ChiselContext not initialized")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None
```

**Fields:**
- `ir_module` — parsed MLIR module (created once at init)
- `op_iter` — iterator over MLIR ops, advances with each preOp callback
- `_current_op` — the op being processed (set in preOp, used in postOp)
- `_stashed_inputs` — device input tensors copied in preOp, consumed in postOp

### `callbacks.py`

Two plain functions compatible with `DebugHooks` op-level callbacks.
No program-level callbacks in this PR.

```python
from chisel.context import ChiselContext

def chisel_pre_op_callback(binary, program_context, op_context):
    """
    1. Advance op_iter to get current MLIR op
    2. Copy device input tensors to host
    3. Stash inputs in ctx._stashed_inputs for postOp
    """
    ctx = ChiselContext.get_instance()
    ctx._current_op = next(ctx.op_iter)

    # Copy device inputs to host
    ctx._stashed_inputs = {}
    for i, inp in enumerate(get_op_inputs(ctx._current_op)):
        ref = tt_runtime.runtime.get_op_input_refs(op_context)[i]
        device_tensor = tt_runtime.runtime.retrieve_tensor(ref, program_context)
        ctx._stashed_inputs[get_ssa_name(inp)] = get_torch_tensor(device_tensor)

def chisel_post_op_callback(binary, program_context, op_context):
    """
    1. Run golden function with stashed inputs
    2. Capture device output tensor
    3. Compare golden vs device (PCC, abs_err, rel_err)
    4. Log metrics to stdout/logger
    5. Discard golden output (no pool storage)
    """
    ctx = ChiselContext.get_instance()

    # Execute golden
    golden_result = execute_golden(
        ctx._current_op, ctx.ir_module, ctx._stashed_inputs
    )

    # Capture device output
    device_output = tt_runtime.runtime.get_op_output_tensor(
        op_context, program_context
    )
    device_torch = get_torch_tensor(device_output)

    # Compare
    pcc = compute_pcc(golden_result, device_torch)
    atol = compute_atol(golden_result, device_torch)
    rtol = compute_rtol(golden_result, device_torch)

    # Log
    op_name = type(ctx._current_op).__name__
    logger.info(f"{op_name}: PCC={pcc:.6f}, atol={atol:.6e}, rtol={rtol:.6e}")

    # Discard — no pool storage
    ctx._stashed_inputs = None
```

### `__init__.py` exports

```python
from chisel.context import ChiselContext
from chisel.callbacks import (
    chisel_pre_op_callback,
    chisel_post_op_callback,
)
```

### Metrics — imported from `tools/golden/metrics.py`
<!-- TODO: Lets just copy metrics from builder into tools/golden/metrics rewriten in torch and not numpy. -->

Chisel does **not** have a local `metrics.py`. All comparison functions are
imported from the unified `golden.metrics` module (created in PR 0c):

```python
from golden.metrics import compute_pcc, compute_atol, compute_rtol
```

## Porting Notes

### `ops.py` from `runtime/tools/chisel/chisel/core/ops.py`

**`get_op_outputs()` and `get_op_inputs()`:**
- **Port as-is** — pure MLIR operation introspection, no ExecutionType dependency

**`IRModule` changes:**
- **Remove** `execution_type: ExecutionType` constructor parameter and attribute
- **Remove** `self.execution_type` — used only in `__repr__` and passed to AsmState
- **Keep as-is:** `get_function()`, `get_function_inputs()`, `get_function_ops()`,
  `get_asm_state()`
- The `ignored_ops` parameter stays — useful for skipping `ttnn.deallocate` and
  similar non-compute ops

### `executor.py` — NEW (does not port from old `golden_executor.py`)

The old `GoldenExecutor` class at `runtime/tools/chisel/chisel/core/golden_executor.py`
executed TTIR ops with custom golden functions and had extensive special-case
handling for TTIR-specific ops (`ttir.empty`, `func.return`, `ttir.dot_general`,
`ttir.broadcast`, `ttir.pad`, `ttir.permute`).

**Write fresh as a standalone function** because:
- The old executor targets TTIR ops; the new function targets TTNN ops
- TTNN ops in `GOLDEN_MAPPINGS` use a different calling convention
  (they accept `GoldenMapTensor` objects from `tools/golden/`)
- The old special-case handling for TTIR ops doesn't apply
- A class adds no value — a standalone function takes `op`, `ir_module`, and
  `inputs` as arguments

**Difference from PR 2:** In this PR, `execute_golden()` takes a plain `dict`
of inputs (captured from device in preOp) and returns the result without storing
it. In PR 2, a pool-aware wrapper pulls inputs from `TensorPool` and stores
outputs back.

### `context.py` — NEW (slim version)

The slim `ChiselContext` in this PR holds only:
- `ir_module` — the parsed MLIR module
- `op_iter` — iterator advancing through ops
- `_stashed_inputs` — temporary storage between preOp and postOp
- `_current_op` — the op being processed

In PR 2, this is expanded to the full `ChiselContext`/`BinaryState`/`ProgramState`
hierarchy with `TensorPool` for golden tensor persistence.

### `callbacks.py` — NEW (2 callbacks, not 4)

Only `preOp` and `postOp` in this PR. Program-level callbacks
(`preProgram`/`postProgram`) are added in PR 2.

**Callback flow per op:**
1. **preOp**: advance `op_iter`, copy device input tensors to host, stash
2. **HW executes op** (outside our control)
3. **postOp**: run golden with stashed inputs, capture device output, compare, log

Each op is self-contained — no dependency on previous ops' golden outputs.

### `utils.py` porting

**From `runtime/tools/chisel/chisel/utils/runtime_utils.py`:**
- **Port and rename:** `ttir_dtype_maps` → `mlir_dtype_maps`, `ttrt_dtype_maps`, `get_torch_tensor()`

**From `runtime/tools/chisel/chisel/utils/debug.py`:**
- **Port as-is:** `debug_wrap()` decorator

## Test Plan

### `test_ops.py`
- `test_ir_module_creation()` — parse a small TTNN MLIR module string, create IRModule
- `test_get_function()` — verify `get_function()` returns the expected function op
- `test_get_function_ops()` — verify operations are listed in correct order
- `test_get_op_inputs_outputs()` — verify tensor-like operand/result extraction
- `test_ignored_ops()` — verify ops in `ignored_ops` list are filtered

**Test dependencies:** `ttmlir` Python bindings for MLIR module parsing.

**Example test fixture:**
```python
SIMPLE_TTNN_MODULE = """
module {
  func.func @main(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttnn.abs"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
"""
```

### `test_executor.py`
- `test_execute_golden_abs()` — provide input tensor dict,
  call `execute_golden()` with `ttnn.AbsOp`, verify output matches `torch.abs(input)`
- `test_execute_golden_add()` — two-input op, verify output matches `torch.add(a, b)`
- `test_unmapped_op_raises()` — mock an op type not in GOLDEN_MAPPINGS,
  verify `RuntimeError` is raised
- `test_result_returned_not_stored()` — verify function returns result without
  side effects (no pool mutation)

**Test dependencies:** `torch`, `ttmlir` bindings, `tools/golden/mapping.py`.

### `test_utils.py`
- `test_dtype_maps()` — verify all expected dtype mappings exist and are valid torch dtypes
- `test_get_torch_tensor()` — mock runtime tensor, verify conversion to torch.Tensor

**Test dependencies:** `torch`, `ttrt.runtime` (or mocks).

### `test_context.py`

**Singleton lifecycle tests (no hardware needed):**
- `test_singleton_not_initialized()` — `get_instance()` raises `RuntimeError`
  before any construction
- `test_singleton_construction()` — construct with mocked IRModule,
  `get_instance()` returns same object
- `test_singleton_reset()` — call `reset_instance()`, `get_instance()` raises again
- `test_op_iter_advances()` — create ChiselContext with known ops, call
  `next(ctx.op_iter)` repeatedly, verify correct op sequence
- `test_stashed_inputs_lifecycle()` — set `_stashed_inputs`, verify accessible,
  set to None, verify cleared

### `test_callbacks.py`

- `test_pre_op_advances_op_iter()` — mock runtime, call `chisel_pre_op_callback`,
  verify `ctx._current_op` is set and `_stashed_inputs` is populated
- `test_post_op_runs_golden_and_compares()` — mock runtime and executor,
  call `chisel_post_op_callback`, verify golden execution and comparison
- `test_post_op_clears_stash()` — verify `_stashed_inputs` is None after postOp
- `test_callback_without_context_raises()` — call callback without initializing
  context, verify `RuntimeError` from `get_instance()`

**Test dependencies:** `unittest.mock` for mocking context and runtime objects.

## Dependencies

- **PR 0a-1** — GIL-Safety Fix (callbacks must not be copied)
- **PR 0a-2a** — Named Callback API (register preOp/postOp by name)
- **PR 0c** — Unified Metrics (`compute_pcc`, `compute_atol`, `compute_rtol`)

Does **not** require:
- PR 0a-2b (program-level hooks) — no preProgram/postProgram in this PR
- PR 0a-3 (introspection bindings) — no program_index or input_refs queries

This is the first chisel PR — no dependency on other chisel PRs.
