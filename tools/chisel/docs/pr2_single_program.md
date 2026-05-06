# PR 2: Single Program Flow

## Goal

Add cross-op golden tensor chaining within a program. Golden outputs persist in
`TensorPool` and feed into subsequent ops, allowing the golden execution path to
diverge naturally from the device path. Introduces the full
`ChiselContext`/`BinaryState`/`ProgramState` hierarchy and program-level
callbacks (`preProgram`/`postProgram`).

After this PR, Chisel can track how numerical errors accumulate across a
full single-program binary — not just per-op isolation.

**Not included:** ReportWriter (CSV), disk caching, global tensor pool
cross-program/cross-binary sharing, skip mode.

## Files

### New Files

| File | Description |
|------|-------------|
| `tools/chisel/chisel/tensors.py` | `TensorPool` (stores `GoldenMapTensor` directly, no disk caching) |

### Modified Files

| File | Change |
|------|--------|
| `tools/chisel/chisel/context.py` | Expand slim `ChiselContext` to full hierarchy: `ChiselContext` → `BinaryState` → `ProgramState` |
| `tools/chisel/chisel/callbacks.py` | Add `preProgram`/`postProgram` callbacks (4 total), update `preOp`/`postOp` to use `ProgramState`'s golden pool |
| `tools/chisel/chisel/executor.py` | Add pool-aware execution path (pull inputs from pool, store outputs) |
| `tools/chisel/chisel/__init__.py` | Update exports (4 callbacks, `ChiselContext`) |
| `tools/chisel/CMakeLists.txt` | Add `chisel/tensors.py` to sources |

## Key Difference from PR 1

In PR 1 (isolation mode), each op's golden inputs come from **device** — the
preOp callback copies device tensors to host. This means each op is tested
against the device's actual inputs, independently.

In PR 2 (program flow mode), golden inputs come from the **TensorPool** — the
previous op's golden output. This means if op N produces slightly wrong results
on device, the golden path for op N+1 uses the correct golden output from op N,
not the device's wrong output. The golden and device paths diverge naturally,
revealing how errors compound.

## Implementation Details

### `tensors.py`

**`TensorPool`** — dict subclass mapping keys to `GoldenMapTensor` directly.
No disk caching in this PR (added in PR 3).

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

### `context.py` — Expanded Hierarchy

Replaces the slim `ChiselContext` from PR 1 with the full three-level hierarchy.

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
           - If new: parse MLIR from binary.mlir.source
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
        1. Execute golden function via execute_golden()
           - Inputs come from golden_tensor_pool (previous golden outputs)
        2. Store golden result in golden_tensor_pool
        3. Capture device output tensor
        4. Compare golden vs device (PCC, abs_err, rel_err)
        5. Log metrics to stdout/logger
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

### `executor.py` — Pool-Aware Path

The core `execute_golden(op, ir_module, inputs: dict)` from PR 1 remains
unchanged. A new wrapper adds pool integration:

```python
def execute_golden_from_pool(
    op: Operation, ir_module: IRModule, tensor_pool: TensorPool
) -> Any:
    """
    Pool-aware golden execution:
    1. Retrieve input tensors from tensor_pool by SSA name
    2. Call execute_golden(op, ir_module, inputs)
    3. Store result in tensor_pool by output SSA name
    4. Return result
    """
    inputs = {get_ssa_name(inp): tensor_pool[get_ssa_name(inp)]
              for inp in get_op_inputs(op)}
    result = execute_golden(op, ir_module, inputs)
    tensor_pool[get_ssa_name(get_op_outputs(op)[0])] = result
    return result
```

### `callbacks.py` — 4 Callbacks

Expand from 2 (PR 1) to 4 callbacks. The program-level callbacks manage
`BinaryState`/`ProgramState` lifecycle. The op-level callbacks now use the
golden pool instead of device-stashed inputs.

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

## Porting Notes

### `tensors.py` from `runtime/tools/chisel/chisel/core/tensors.py`

**Simplified — drop `TensorValue` and `DeviceHandle` entirely:**
- **Remove** `TensorValue` class — pool stores `GoldenMapTensor` directly
- **Remove** `DeviceHandle` — device tensor read/write stays inline in callbacks
- **Keep** `TensorPool(dict)` — but without disk caching (added in PR 3)
- In the new design, `golden_tensor_pool` (CPU tensors) is the only
  `TensorPool` on `ProgramState`. Device tensors are ephemeral.

### `context.py` — redesign from `runtime/tools/chisel/chisel/core/context.py`

The old ChiselContext is the most heavily modified file. The new version is a
complete redesign with a hierarchical state model.

**Remove entirely:**
- `ttir_module` parameter — no golden TTIR module
- `self.golden_ir_module` / `self.modules[ExecutionType.GOLDEN]` — single module only
- `compare_outputs()` — comparison now happens inline in `postop()`
- `get_corresponding_tensors()` — no cross-module tensor mapping needed
- `skip_group()` — no group skipping logic (added in PR 4)
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
  `execute_golden_from_pool()` and compares.

**Runtime API usage (same as old):**
- `tt_runtime.runtime.get_op_loc_info(op_context)` — get operation location string
- `tt_runtime.runtime.get_op_output_tensor(op_context, program_context)` — get device output
- `tt_runtime.runtime.get_op_debug_str(op_context)` — get debug string
- `tt_runtime.runtime.get_op_input_refs()` — get input tensor references
- `tt_runtime.runtime.get_op_output_ref()` — get output tensor reference

### `callbacks.py` — expand from 2 to 4

Add `chisel_pre_program_callback` and `chisel_post_program_callback`. Update
`chisel_pre_op_callback` to no longer stash device inputs (inputs now come from
pool). Update `chisel_post_op_callback` to store golden outputs in pool.

## Test Plan

### `test_tensors.py`
- `test_tensor_pool_insert_retrieve()` — insert `GoldenMapTensor`, retrieve by key
- `test_tensor_pool_is_dict()` — verify TensorPool behaves as dict (keys, values, items, len)

**Test dependencies:** `torch` only.

### `test_context.py`

**Full hierarchy lifecycle tests:**
- `test_singleton_not_initialized()` — `get_instance()` raises `RuntimeError`
- `test_singleton_construction()` — construct, `get_instance()` returns same object
- `test_singleton_reset()` — call `reset_instance()`, `get_instance()` raises again

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

**PreProgram/PostProgram flow tests:**
- `test_preprogram_creates_binary_state()` — call `preprogram()` with mock
  binary, verify `BinaryState` created in `ctx.binaries`
- `test_preprogram_creates_program_state()` — verify `ProgramState` created
  in binary's `programs` dict
- `test_postop_stores_golden_in_pool()` — verify golden output is stored in
  `ProgramState.golden_tensor_pool` after `postop()`

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

- **PR 1** — Single Op Isolation Testing (ops.py, executor.py, utils.py, CMakeLists)
- **PR 0a-2b** — Program-Level Hooks (preProgram/postProgram callbacks)
- **PR 0a-3** — Introspection Bindings (`get_program_index`,
  `get_program_input/output_refs`, `Binary.id`, `Tensor.global_id`)
- **PR 0c** — Unified Metrics (if not already required by PR 1)
