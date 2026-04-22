# PR 1: Single Op Isolation Testing + Builder Integration

## Goal

Deliver op-level isolation testing — each TTNN op is tested independently.
`preOp` copies device input tensors to host, `postOp` runs the golden function
via `GOLDEN_MAPPINGS` and compares against the device output. Golden outputs
are discarded after comparison — no cross-op tensor chaining.

This PR also includes builder integration (previously PR 5): the `enable_chisel`
parameter in `execute_fb()`, the `chisel.bind()`/`chisel.unbind()` lifecycle,
and API forwarding through `compile_and_execute_ttnn()`.

This PR proves the core golden-vs-device comparison loop works for individual
ops through the real builder pipeline, without requiring program-level state or
tensor persistence.

**Not included:** TensorPool, cross-op golden chaining, program-level callbacks
(`preProgram`/`postProgram`), ReportWriter (CSV), disk caching, cross-program
tensor sharing, skip mode.

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
| `tools/chisel/chisel/bind.py` | `bind()` creates ChiselContext + registers DebugHooks, `unbind()` resets singleton |

### Modified Files

| File | Change |
|------|--------|
| `tools/CMakeLists.txt` | Add `add_subdirectory(chisel)` under the Python bindings guard |
| `tools/builder/base/builder_runtime.py` | Add `enable_chisel: bool = False` to `execute_fb()`. Mutual exclusivity check. `chisel.bind()` call when enabled. |
| `tools/builder/base/builder_apis.py` | Add `enable_chisel: bool = False` to `compile_and_execute_ttnn()` and `_compile_and_execute()`, forward to `execute_fb()` |
| `test/python/golden/conftest.py` | Add `--enable-chisel` pytest option forwarded to `compile_and_execute_ttnn` |

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
    ):
        # Parse the MLIR source string into a Module
        self.context = Context()
        self.module = Module.parse(mlir_source, self.context)
        ...

    def get_function(self, function_name: str) -> Operation: ...
    def get_function_inputs(self, function_name: str) -> List[BlockArgument]: ...
    def get_function_ops(self, function_name: str) -> List[Operation]: ...
    def get_asm_state(self, function_name: str) -> AsmState: ...

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
        self.op_iter: Iterator = iter(ir_module.get_function_ops(function_name))
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

### `bind.py`

One-call setup and teardown for builder integration:

```python
def bind():
    """Initialize ChiselContext and register op callbacks with DebugHooks."""
    import _ttmlir_runtime as tt_runtime

    ChiselContext()
    tt_runtime.runtime.DebugHooks.get(
        chisel_pre_op_callback,
        chisel_post_op_callback,
    )


def unbind():
    """Tear down ChiselContext singleton. Safe to call even if bind() was not called."""
    ChiselContext.reset_instance()
```

### Builder Integration

**`execute_fb()` in `builder_runtime.py` — new parameter:**

```python
def execute_fb(
    compiled_bin,
    # ... existing params ...
    enable_chisel: bool = False,
):
```

**Mutual exclusivity check** (early in function body):

```python
if enable_chisel and enable_intermediate_verification:
    raise ValueError(
        "enable_chisel and enable_intermediate_verification are mutually "
        "exclusive. Use one or the other."
    )
```

**Callback registration** (where `DebugHooks.get()` is called):

```python
if enable_chisel:
    import chisel
    chisel.bind()
elif verify_intermediates or dump_memory:
    tt_runtime.runtime.DebugHooks.get(
        pre_op_get_callback_fn(callback_runtime_config),
        post_op_get_callback_fn(callback_runtime_config),
    )
```

**`compile_and_execute_ttnn()` and `_compile_and_execute()` in `builder_apis.py`**
receive the same `enable_chisel: bool = False` parameter and forward it to
`execute_fb()`.

**`--enable-chisel` pytest option** in `test/python/golden/conftest.py` adds
`enable_chisel=True` to the kwargs passed to `compile_and_execute_ttnn()`.

### `__init__.py` exports

```python
from chisel.context import ChiselContext
from chisel.callbacks import (
    chisel_pre_op_callback,
    chisel_post_op_callback,
)
from chisel.bind import bind, unbind
```

### Metrics — `tools/golden/metrics.py`

Created as part of this PR. Pure-torch implementations of `compute_pcc`,
`compute_atol`, `compute_rtol` ported from the old chisel metrics at
`runtime/tools/chisel/chisel/utils/metrics.py`. This removes the PR 0c
dependency.

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
- **Remove** `current_function_name` state — all methods (`get_function()`,
  `get_function_inputs()`, `get_function_ops()`, `get_asm_state()`) take
  `function_name: str` as an explicit parameter instead

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

Unit tests are deferred to PR 1.5. This PR focuses on integration tests that
exercise chisel through the real builder pipeline on device.

See [TestingPlan.md](TestingPlan.md) for the full testing strategy.

### Golden Test Suite (primary)

Run the existing golden test suite with `--enable-chisel`:

```bash
pytest test/python/golden/test_ttnn_ops.py --enable-chisel
```

This exercises chisel against every op the golden tests cover — no new test
code required. Validates that callbacks fire, IRModule parsing succeeds for
compiler-generated MLIR, `execute_golden()` finds golden functions, and op
iterator stays in sync.

### Builder Integration Tests (requires device)

| Test | Description |
|------|-------------|
| `test_chisel_sigmoid` | Single unary op via `compile_and_execute_ttnn(enable_chisel=True)`. Verify chisel log contains `ttnn.sigmoid` with PCC > 0.99. |
| `test_chisel_relu` | Numerically exact unary op. Verify PCC = 1.0. |
| `test_chisel_add` | Binary op. Verify two inputs stashed correctly. |
| `test_chisel_matmul` | Binary op with shape change. Verify chisel handles output shape ≠ input shape. |
| `test_chisel_add_relu` | Two-op chain. Verify chisel logs 2 ops in correct order with independent PCC. |
| `test_chisel_matmul_softmax` | Shape-changing op + attribute op. Verify op\_iter advances correctly. |

### Mutual Exclusivity Test

| Test | Description |
|------|-------------|
| `test_chisel_and_verify_raises` | `execute_fb(enable_chisel=True, enable_intermediate_verification=True)` raises `ValueError` |

## Dependencies

None — PR 1 uses existing runtime APIs only.

**Runtime APIs used (all currently available):**
- `DebugHooks.get(pre_op, post_op)` — registers preOp/postOp callbacks
- `get_op_input_refs()` — retrieves input tensor references in preOp
- `get_op_output_ref()` — retrieves output tensor reference in postOp
- `retrieve_tensor_from_pool()` — converts tensor ref to host tensor
- `unregister_hooks()` — clears registered callbacks (called by builder's
  `finally` block)

**Why no runtime PRs needed:**
- `bind.py` uses `DebugHooks.get(pre, post)` directly (not a named callback API)
- Builder integration uses `elif` (mutual exclusivity), so multi-client
  callbacks (PR 0a-2a) are not needed
- `tools/golden/metrics.py` is created within this PR (not a separate
  prerequisite)
- GIL-safety (PR 0a-1) is a performance optimization, not a correctness
  requirement for single-threaded Python callbacks

Does **not** require:
- PR 0a-1 (GIL-safety fix) — not a correctness issue for PR 1's use case
- PR 0a-2a (named callback API) — mutual exclusivity in builder eliminates
  the need for multi-client callbacks
- PR 0a-2b (program-level hooks) — no preProgram/postProgram in this PR
- PR 0a-3 (introspection bindings) — no program_index or input_refs queries

Builder integration (previously PR 5) is included in this PR. Modified builder
files (`builder_runtime.py`, `builder_apis.py`) are part of this PR's scope.

This is the first chisel PR — no dependency on other chisel or runtime PRs.
