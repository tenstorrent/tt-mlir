# PR 3: Orchestration

## Goal

Add the `ChiselContext` singleton with its hierarchical state model
(`BinaryState`, `ProgramState`), the `callbacks.py` module with four callback
functions compatible with `DebugHooks`, and the utility module (`utils.py`).
After this PR, Chisel is fully functional as a library — callers can register
callbacks and observe TTNN binary execution.

## Files

### New Files

| File | Description |
|------|-------------|
| `tools/chisel/chisel/context.py` | `ChiselContext` singleton, `BinaryState`, `ProgramState` |
| `tools/chisel/chisel/callbacks.py` | 4 callback functions: `preProgram`, `postProgram`, `preOp`, `postOp` |
| `tools/chisel/chisel/utils.py` | Location parsing, dtype maps, runtime tensor conversion |

### Modified Files

| File | Change |
|------|--------|
| `tools/chisel/chisel/__init__.py` | Export `ChiselContext`, all 4 callback functions |
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

Contains three classes that form the hierarchical state model:

```python
class ChiselContext:
    _instance: Optional["ChiselContext"] = None

    def __init__(
        self,
        output_dir: Path,
        report_base_path: Path,
        caching: bool = True,
    ):
        ChiselContext._instance = self

        # Cross-binary/cross-program golden tensor sharing
        # Keyed by Tensor::globalId
        self.global_tensor_pool = TensorPool(caching=caching, output_dir=output_dir / "global")

        # State per binary — created lazily on first preProgram
        self.binaries: Dict[int, BinaryState] = {}

        # Set by preProgram, used by preOp/postOp
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
        Called once at the start of each program execution.

        1. Get or create BinaryState for binary.id
           - If new: parse MLIR
        2. Get or create ProgramState for program_index
        3. program.reset_for_new_execution()
        4. Copy global_tensor_pool → program.golden_tensor_pool
        5. Copy program input tensors from device → golden pool
           (via get_program_input_ids, for inputs not already in pool)
        6. Set current_binary and current_program
        7. Start new report section
        """

    def postprogram(self, binary, program_context) -> None:
        """
        Called once at the end of each program execution.

        1. Copy program.golden_tensor_pool → global_tensor_pool
        2. Finalize report section
        """

    def preop(self, binary, program_context, op_context) -> None:
        """
        Called before each TTNN op executes on device.

        1. op = next(self.current_program.op_iter)
        2. If skip mode: stash inputs before device op runs
        """

    def postop(self, binary, program_context, op_context) -> None:
        """
        Called after each TTNN op executes on device.

        1. Capture device output tensor
        2. Execute golden function via execute_golden()
        3. Compare golden vs device (PCC, abs_err, rel_err)
        4. Write row to CSV report
        """


class BinaryState:
    def __init__(self, binary, caching: bool = True, output_dir: Path = None):
        # Extract TTNN MLIR from binary.mlir.source, parse module
        self.ir_module = IRModule(mlir_source=binary.mlir.source)
        self.programs: Dict[int, ProgramState] = {}
        self.report = ReportWriter(...)

    def get_or_create_program(self, program_index) -> "ProgramState":
        if program_index not in self.programs:
            self.programs[program_index] = ProgramState(
                program_index, self.ir_module
            )
        return self.programs[program_index]


class ProgramState:
    def __init__(self, program_index: int, ir_module: IRModule):
        self.program_index = program_index
        self.golden_tensor_pool = TensorPool(...)
        self.ops: List[OpInfo] = [...]  # ordered from ir_module for this program
        self.op_iter: Iterator[OpInfo] = iter(self.ops)
        self._skip_stash: dict[str, Tensor] | None = None

    def reset_for_new_execution(self) -> None:
        """Called by preProgram on each program execution."""
        self.op_iter = iter(self.ops)
        self._skip_stash = None
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

## Porting Notes for `context.py`

### From `runtime/tools/chisel/chisel/core/context.py`

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
- `global_tensor_pool` — cross-binary/cross-program golden tensor sharing

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
- `test_singleton_construction()` — construct with mocked deps,
  `get_instance()` returns same object
- `test_singleton_reset()` — call `reset_instance()`, `get_instance()` raises again
- `test_singleton_reinitialization()` — construct → reset → construct new →
  `get_instance()` returns new object

**BinaryState tests:**
- `test_binary_state_creation()` — mock binary with `mlir.source`, verify
  `IRModule` is created
- `test_get_or_create_program()` — first call creates `ProgramState`, second
  call returns the same instance
- `test_multiple_programs()` — create programs 0 and 1, verify they have
  independent golden pools

**ProgramState tests:**
- `test_program_state_creation()` — verify golden pool
  and `op_iter` are initialized
- `test_reset_for_new_execution()` — call `reset_for_new_execution()`, verify
  `op_iter` is reset and `_skip_stash` is None. Verify golden pool is preserved.
- `test_op_iter_advances()` — create ProgramState with known ops, call
  `next(op_iter)` repeatedly, verify correct op sequence
- `test_op_iter_reset()` — exhaust iter, reset, verify starts from beginning

**PreProgram/PostProgram flow tests:**
- `test_preprogram_creates_binary_state()` — call `preprogram()` with mock
  binary, verify `BinaryState` created in `ctx.binaries`
- `test_preprogram_creates_program_state()` — verify `ProgramState` created
  in binary's `programs` dict
- `test_preprogram_copies_global_to_program_pool()` — pre-populate
  `global_tensor_pool`, call `preprogram()`, verify matching tensors appear
  in `program.golden_tensor_pool`
- `test_postprogram_copies_program_to_global_pool()` — populate
  `program.golden_tensor_pool`, call `postprogram()`, verify entries appear
  in `global_tensor_pool`

### `test_callbacks.py`

- `test_pre_program_delegates_to_context()` — mock `ChiselContext.get_instance()`,
  call `chisel_pre_program_callback(binary, prog_ctx)`, verify
  `context.preprogram(binary, prog_ctx)` was called
- `test_post_program_delegates_to_context()` — same for `postprogram`
- `test_pre_op_delegates_to_context()` — mock `ChiselContext.get_instance()`,
  call `chisel_pre_op_callback(binary, prog_ctx, op_ctx)`, verify
  `context.preop(binary, prog_ctx, op_ctx)` was called
- `test_post_op_delegates_to_context()` — same for `postop`
- `test_callback_without_context_raises()` — call callback without initializing
  context, verify `RuntimeError` from `get_instance()`

**Test dependencies:** `unittest.mock` for mocking context and runtime objects.
No hardware or MLIR needed for callback delegation tests.

## Dependencies

- **PR 2** — `executor.py`, `report.py`
- **PR 1** — `tensors.py`, `ops.py`
- **PR 0a** — DebugHooks refactor must land before this PR. This is where
  Python callbacks get registered and invoked via `DebugHooks`. Without the
  fix, callback copies cause segfaults when called from tt-xla without GIL.
  Also exposes `Binary.id` property and program-level callbacks.
- **PR 0c** — Unified metrics in `tools/golden/metrics.py`. This PR's
  `context.py` imports `compute_pcc`, `compute_atol`, `compute_rtol` from
  `golden.metrics` instead of a local `metrics.py`.
