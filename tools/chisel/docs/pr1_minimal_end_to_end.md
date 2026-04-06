# PR 1: Minimal End-to-End Chisel

## Goal

Deliver the complete chisel package — all modules, full callback flow — runnable
against a single-program binary. Golden vs device comparison metrics are logged
to stdout/logger. This PR proves the entire architecture works end-to-end.

**Not included:** ReportWriter (CSV), disk caching, global tensor pool
cross-program/cross-binary sharing, skip mode.

## Files

### New Files

| File | Description |
|------|-------------|
| `tools/chisel/CMakeLists.txt` | CMake packaging using `declare_mlir_python_sources` |
| `tools/chisel/chisel/__init__.py` | Package init + exports |
| `tools/chisel/chisel/tensors.py` | `TensorPool` (stores `GoldenMapTensor` directly, no disk caching) |
| `tools/chisel/chisel/ops.py` | `IRModule` wrapper, `get_op_inputs()`, `get_op_outputs()` |
| `tools/chisel/chisel/executor.py` | `execute_golden()` — CPU replay of TTNN ops via `GOLDEN_MAPPINGS` |
| `tools/chisel/chisel/context.py` | `ChiselContext` singleton, `BinaryState`, `ProgramState` |
| `tools/chisel/chisel/callbacks.py` | 4 callback functions for `DebugHooks` |
| `tools/chisel/chisel/utils.py` | Dtype maps, runtime tensor conversion |

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
    chisel/tensors.py
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

### `tensors.py`

**`TensorPool`** — dict subclass mapping keys to `GoldenMapTensor` directly.
No disk caching in this PR (added in PR 2).

- `GoldenMapTensor` (from `tools/golden/mapping.py`) already provides all
  needed tensor semantics (sharding, `__torch_function__` for shard-wise ops,
  dtype conversion via `golden_map_tensor_as_torch_tensors()`)
- Device tensors are ephemeral — captured from the runtime API in each callback
  and consumed immediately, so they don't need pool storage
- The old `TensorValue.snapshot`/`working` split is unnecessary because golden
  ops don't mutate inputs (they return new `GoldenMapTensor` instances)

```python
class TensorPool(dict):
    """Dict mapping SSA name (or globalId) -> GoldenMapTensor."""

    def __init__(self):
        super().__init__()
```

### `executor.py`

Standalone function that executes TTNN operations on CPU using
`GOLDEN_MAPPINGS`. For each TTNN op encountered during device execution, the
function replays it with PyTorch.

```python
def execute_golden(op: Operation, ir_module: IRModule, tensor_pool: TensorPool) -> Any:
    """
    Execute a TTNN op on CPU via GOLDEN_MAPPINGS.

    1. Look up op type in GOLDEN_MAPPINGS via get_golden_function()
    2. If not found, raise RuntimeError (fail hard)
    3. Retrieve input tensors from the per-program golden_tensor_pool
    4. Call golden function with PyTorch tensors
    5. Store result in golden_tensor_pool
    6. Return result
    """
```

**Fail-hard behavior:** If `get_golden_function(type(op))` returns `None`, the
function raises `RuntimeError(f"No golden implementation for {type(op).__name__}")`.

**Key dependency:** `tools/golden/mapping.py` — use `get_golden_function(type(op))`
to look up the golden callable.

### `utils.py`

Consolidated utility module:

**Dtype maps:**
- `ttir_dtype_maps: Dict` — TTIR element types to PyTorch dtypes
- `ttrt_dtype_maps: Dict` — TTRT runtime tensor types to PyTorch dtypes

**Runtime utilities:**
- `get_torch_tensor(tensor: RtTensor) -> torch.Tensor` — convert runtime tensor to PyTorch
- `debug_wrap(*, debug: bool = False)` — decorator factory for pdb integration

### `context.py`

Contains three classes that form the hierarchical state model. In this PR,
`postop()` logs comparison metrics to stdout/logger instead of writing CSV.
No `global_tensor_pool` cross-program copying yet (added in PR 2).

```python
class ChiselContext:
    _instance: Optional["ChiselContext"] = None

    def __init__(self, output_dir: Path):
        ChiselContext._instance = self
        self.binaries: Dict[int, BinaryState] = {}
        self.current_binary: BinaryState | None = None
        self.current_program: ProgramState | None = None

    @classmethod
    def get_instance(cls) -> "ChiselContext":
        if cls._instance is None:
            raise RuntimeError("ChiselContext not initialized")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def preprogram(self, binary, program_context) -> None:
        """
        1. Get or create BinaryState for binary.id
           - If new: parse MLIR
        2. Get or create ProgramState for program_index
        3. program.reset_for_new_execution()
        4. Copy program input tensors from device -> golden pool
           (via get_program_input_ids)
        5. Set current_binary and current_program
        """

    def postprogram(self, binary, program_context) -> None:
        """
        1. Log program-level summary (total ops, ops with low PCC)
        2. Clear current_binary and current_program
        """

    def preop(self, binary, program_context, op_context) -> None:
        """
        1. op = next(self.current_program.op_iter)
        """

    def postop(self, binary, program_context, op_context) -> None:
        """
        1. Capture device output tensor
        2. Execute golden function via execute_golden()
        3. Compare golden vs device (PCC, abs_err, rel_err)
        4. Log metrics to stdout/logger
        """


class BinaryState:
    def __init__(self, binary):
        self.ir_module = IRModule(mlir_source=binary.mlir.source)
        self.programs: Dict[int, ProgramState] = {}

    def get_or_create_program(self, program_index) -> "ProgramState":
        if program_index not in self.programs:
            self.programs[program_index] = ProgramState(
                program_index, self.ir_module
            )
        return self.programs[program_index]


class ProgramState:
    def __init__(self, program_index: int, ir_module: IRModule):
        self.program_index = program_index
        self.golden_tensor_pool = TensorPool()
        self.ops: List[OpInfo] = [...]  # ordered from ir_module for this program
        self.op_iter: Iterator[OpInfo] = iter(self.ops)

    def reset_for_new_execution(self) -> None:
        self.op_iter = iter(self.ops)
        # golden_tensor_pool is NOT cleared
```

### `callbacks.py`

Thin module with four plain functions compatible with `DebugHooks`:

```python
from chisel.context import ChiselContext

def chisel_pre_program_callback(binary, program_context):
    ChiselContext.get_instance().preprogram(binary, program_context)

def chisel_post_program_callback(binary, program_context):
    ChiselContext.get_instance().postprogram(binary, program_context)

def chisel_pre_op_callback(binary, program_context, op_context):
    ChiselContext.get_instance().preop(binary, program_context, op_context)

def chisel_post_op_callback(binary, program_context, op_context):
    ChiselContext.get_instance().postop(binary, program_context, op_context)
```

Two signatures:
- **Program-level**: `(binary, program_context)` — same `(Binary, CallbackContext)` types
- **Op-level**: `(binary, program_context, op_context)` — same signature as builder's own callbacks

### `__init__.py` exports

```python
from chisel.context import ChiselContext
from chisel.callbacks import (
    chisel_pre_program_callback,
    chisel_post_program_callback,
    chisel_pre_op_callback,
    chisel_post_op_callback,
)
```

### Metrics — imported from `tools/golden/metrics.py`

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

### `tensors.py` from `runtime/tools/chisel/chisel/core/tensors.py`

**Simplified — drop `TensorValue` and `DeviceHandle` entirely:**
- **Remove** `TensorValue` class — pool stores `GoldenMapTensor` directly
- **Remove** `DeviceHandle` — device tensor read/write stays inline in callbacks
- **Keep** `TensorPool(dict)` — but without disk caching (added in PR 2)
- In the new design, `golden_tensor_pool` (CPU tensors) is the only
  `TensorPool` on `ProgramState`. Device tensors are ephemeral.

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
- The integration with `TensorPool` is simpler without dual pools
- A class adds no value — `ir_module` and `tensor_pool` live on
  `BinaryState`/`ProgramState` already; a standalone function takes them
  as arguments

### `context.py` — redesign from `runtime/tools/chisel/chisel/core/context.py`

The old ChiselContext is the most heavily modified file. The new version is a
complete redesign with a hierarchical state model.

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
- `_op_index` — replaced by `ProgramState.op_iter`
- `_current_binary_id` — replaced by `binaries` dict + `preProgram` callback
- `_check_program_transition()` — not needed with explicit callbacks
- `_reset_for_new_program()` — replaced by `ProgramState.reset_for_new_execution()`

**Add new:**
- Singleton pattern (`_instance`, `get_instance()`, `reset_instance()`)
- `BinaryState` class — per-binary state (IRModule, programs dict)
- `ProgramState` class — per-program state (golden pool, op_iter)
- `preprogram()` / `postprogram()` methods — program-level callbacks
- `preop()` / `postop()` methods — op-level callbacks

**Adapt from old `preop()`/`postop()`:**
The old context has these methods but they work with dual modules. The new
versions are simpler:
- Old `preop()` checked `program_context`, extracted tensor refs, updated device
  pool, and handled function arguments. New `preop()` does the same but without
  ExecutionType branching and uses `next(op_iter)` instead of `_op_index`.
- Old `postop()` called `compare_outputs()` which used the Registry to find
  corresponding golden/device tensors. New `postop()` directly runs the
  `execute_golden()` and compares.

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
- Four callbacks instead of two (program-level + op-level)

### `utils.py` porting

**From `runtime/tools/chisel/chisel/utils/runtime_utils.py`:**
- **Port as-is:** `ttir_dtype_maps`, `ttrt_dtype_maps`, `get_torch_tensor()`

**From `runtime/tools/chisel/chisel/utils/debug.py`:**
- **Port as-is:** `debug_wrap()` decorator

## Test Plan

### `test_tensors.py`
- `test_tensor_pool_insert_retrieve()` — insert `GoldenMapTensor`, retrieve by key
- `test_tensor_pool_is_dict()` — verify TensorPool behaves as dict (keys, values, items, len)

**Test dependencies:** `torch` only.

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
- `test_execute_golden_abs()` — pre-populate golden_tensor_pool with input tensor,
  call `execute_golden()` with `ttnn.AbsOp`, verify output matches `torch.abs(input)`
- `test_execute_golden_add()` — two-input op, verify output matches `torch.add(a, b)`
- `test_unmapped_op_raises()` — mock an op type not in GOLDEN_MAPPINGS,
  verify `RuntimeError` is raised
- `test_result_stored_in_pool()` — after execution, verify result is in tensor_pool

**Test dependencies:** `torch`, `ttmlir` bindings, `tools/golden/mapping.py`.

### `test_utils.py`
- `test_dtype_maps()` — verify all expected dtype mappings exist and are valid torch dtypes
- `test_get_torch_tensor()` — mock runtime tensor, verify conversion to torch.Tensor

**Test dependencies:** `torch`, `ttrt.runtime` (or mocks).

### `test_context.py`

**Singleton lifecycle tests (no hardware needed):**
- `test_singleton_not_initialized()` — `get_instance()` raises `RuntimeError`
  before any construction
- `test_singleton_construction()` — construct with mocked deps,
  `get_instance()` returns same object
- `test_singleton_reset()` — call `reset_instance()`, `get_instance()` raises again
- `test_singleton_reinitialization()` — construct -> reset -> construct new ->
  `get_instance()` returns new object

**BinaryState tests:**
- `test_binary_state_creation()` — mock binary with `mlir.source`, verify
  `IRModule` is created
- `test_get_or_create_program()` — first call creates `ProgramState`, second
  call returns the same instance
- `test_multiple_programs()` — create programs 0 and 1, verify they have
  independent golden pools

**ProgramState tests:**
- `test_program_state_creation()` — verify golden pool and `op_iter` are initialized
- `test_reset_for_new_execution()` — call `reset_for_new_execution()`, verify
  `op_iter` is reset. Verify golden pool is preserved.
- `test_op_iter_advances()` — create ProgramState with known ops, call
  `next(op_iter)` repeatedly, verify correct op sequence
- `test_op_iter_reset()` — exhaust iter, reset, verify starts from beginning

**PreProgram/PostProgram flow tests:**
- `test_preprogram_creates_binary_state()` — call `preprogram()` with mock
  binary, verify `BinaryState` created in `ctx.binaries`
- `test_preprogram_creates_program_state()` — verify `ProgramState` created
  in binary's `programs` dict

### `test_callbacks.py`

- `test_pre_program_delegates_to_context()` — mock `ChiselContext.get_instance()`,
  call `chisel_pre_program_callback(binary, prog_ctx)`, verify
  `context.preprogram(binary, prog_ctx)` was called
- `test_post_program_delegates_to_context()` — same for `postprogram`
- `test_pre_op_delegates_to_context()` — same for `preop`
- `test_post_op_delegates_to_context()` — same for `postop`
- `test_callback_without_context_raises()` — call callback without initializing
  context, verify `RuntimeError` from `get_instance()`

**Test dependencies:** `unittest.mock` for mocking context and runtime objects.

## Dependencies

- **PRs 0a-1 through 0a-3** — All runtime DebugHooks PRs must land before
  this PR. PR 0a-1 fixes GIL safety, PR 0a-2a adds named callback API,
  PR 0a-2b adds program-level hooks, and PR 0a-3 adds introspection bindings
  (`get_program_index`, `get_program_input/output_refs`, `Binary.id`,
  `Tensor.global_id`).
- **PR 0c** — Unified metrics in `tools/golden/metrics.py`.

This is the first chisel PR — no dependency on other chisel PRs.
