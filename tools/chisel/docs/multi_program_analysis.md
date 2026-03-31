# Multi-Program Execution Analysis

## Problem Statement

The new Chisel implementation assumes a single program execution per
`ChiselContext` lifetime. In practice, TTRT runs multiple programs back-to-back
from the same binary (e.g., forward + backward passes) or re-runs the same
program (training loops). Three issues arise:

1. **Stale TensorRefs**: Each `submit()` creates a fresh `ProgramContext` with a
   new tensor pool. `TensorRef` objects cached in `DeviceHandle` from program N
   point to deallocated memory in program N+1.
2. **No transition detection**: Chisel ignores the `binary` callback parameter —
   no mechanism to detect when a new program starts.
3. **Golden tensor sharing**: Forward and backward passes share weights. If
   program 0 computes golden outputs, program 1 should reuse them as golden
   inputs rather than recomputing.

## Background: How TTRT Handles Multiple Programs

Each call to `submit(device, binary, program_index, inputs)` creates an
ephemeral `ProgramContext` containing a fresh `ProgramTensorPool`,
`GlobalSemaphorePool`, and `DylibManager`. After execution, the
`ProgramContext` is destroyed (`runtime/lib/ttnn/program_executor.cpp:142-156`).

The device handle persists across submissions. Device memory (DRAM, L1) persists
unless explicitly deallocated. The `Binary` object carries a unique
auto-incrementing `binaryId`.

### What changes between consecutive `submit()` calls

| Component | Persists | Notes |
|-----------|----------|-------|
| Device handle | Yes | Same device across all submissions |
| Device memory (DRAM, L1) | Yes | Must be explicitly deallocated |
| Program cache | Yes | Caches compiled programs |
| ProgramContext / tensor pool | **No** | Fresh per-submission, destroyed after |
| Intermediate tensors | **No** | Cleared between runs |
| Global semaphores | **No** | Pooled fresh per execution |

### How callbacks receive context

Callback signature: `(Binary, CallbackContext, OpContext)`

- `Binary` wraps the flatbuffer binary. Has a `binaryId` in C++ (not yet exposed
  to Python).
- `CallbackContext` wraps a `ProgramContext*` — opaque in Python, no methods
  exposed.
- `OpContext` wraps the flatbuffer operation descriptor.
- Callbacks are registered once globally via `DebugHooks.get()` and fire for
  ALL program executions.

### Binary vs Program

A **Binary** (`TTNNBinary`) is a serialized flatbuffer file containing one or
more programs, a system descriptor, and optionally the original MLIR source
(`include/ttmlir/Target/TTNN/binary.fbs`):

```
TTNNBinary {
  version, schema_hash, ttmlir_git_hash,
  system_desc,
  mlir,
  programs: [Program]    // one or more programs
}
```

A **Program** is a named sequence of operations within a binary, corresponding
to a single MLIR function (e.g. `forward`, `backward`, or a const-eval helper).
Each program has its own inputs, outputs, and operation list
(`include/ttmlir/Target/TTNN/program.fbs:151-161`):

```
Program {
  name,                  // e.g. "forward", "backward"
  inputs: [TensorRef],
  outputs: [TensorRef],
  operations: [Operation],
  private: bool,         // true for internal-only (e.g. hoisted const-eval)
  mesh_shape
}
```

A single binary can contain multiple programs — for example, a training binary
might have a forward pass (program 0), a backward pass (program 1), and a
const-eval function (program 2, marked private). Each call to
`submit(device, binary, program_index, inputs)` executes exactly one program.

### Runtime Identifiers

The runtime provides several identifiers for tracking execution across programs
and binaries:

| Identifier | Type | Source | Scope |
|---|---|---|---|
| `Binary::binaryId` | `static atomic<uint64_t>` | `runtime/lib/binary.cpp:56-58` | Unique per `Binary` construction, process-wide |
| `Tensor::globalId` | `static atomic<uint64_t>` | `runtime/lib/common/types.cpp:192-194` | Unique per `Tensor` construction, process-wide |
| Program index | `uint32_t` | `program_index` arg to `submit()` | Positional within a binary (0, 1, 2...) |

Programs have **no process-wide counter** of their own. To uniquely identify a
program execution across all binaries, use the composite key
`(binary.id, program_index)`.

**`Tensor::globalId` vs `TensorRef.global_id`**: The flatbuffer-level
`TensorRef.global_id` is assigned at compile time and scoped per-binary — it
serves as an index into `ProgramTensorPool`. The runtime-level
`Tensor::globalId` is assigned at tensor construction via a static atomic
counter and is unique across all binaries, programs, and executors within a
process. `ProgramExecutor` stores **pointers** to the original input `Tensor`
objects (`program_executor.cpp:126-127`), not copies, so `Tensor::globalId`
survives across program and binary boundaries.

## Problem 1: Stale TensorRefs

### Root Cause

`DeviceHandle` (in `tensors.py`) stores a `TensorRef` and an optional cached
`Tensor` object. Both are tied to the `ProgramContext` that was live when they
were created. When the next `submit()` runs, the old `ProgramContext` is
destroyed — the `TensorRef` now points to deallocated memory.

In the current old chisel (`runtime/tools/chisel/chisel/core/tensors.py:40-50`),
`TensorValue.update_tensor_in_pool()` caches the `Tensor` object and reuses it:

```python
def update_tensor_in_pool(self, program_context):
    if self.tensor is None:
        self.tensor = retrieve_tensor_from_pool(program_context, self.tensor_ref)
    update_device_tensor(program_context, self.tensor_ref, self.tensor, self.execution_data)
```

This would crash or produce garbage if `self.tensor_ref` / `self.tensor` are
from a previous program's destroyed `ProgramContext`.

### Solution: Clear device pool on program transition

On detecting a new program, call `device_tensor_pool.clear()`. This drops all
`TensorValue` objects and their `DeviceHandle` references. New `TensorRef`s
from the new `ProgramContext` will be captured in `preop()` as usual.

**Key invariant**: Never carry device `TensorValue` objects across program
boundaries.

## Problem 2: Detecting Program Transitions

### Root Cause

Chisel's `preop(binary, program_context, op_context)` receives `binary` but
ignores it. There is no signal from TTRT saying "a new program is starting" —
callbacks are called per-op, uniformly.

### Prerequisite: Expose `Binary.id` to Python

`Binary.id()` exists in C++ (`runtime/include/tt/runtime/types.h:347`) but is
NOT exposed to Python (`runtime/python/binary/binary.cpp:29-49`). The fix is
one line:

```cpp
.def_prop_ro("id", &tt::runtime::Binary::id)
```

The ID is a global atomic counter starting at 0, incremented per `Binary`
construction (`runtime/lib/binary.cpp:56-58`):

```cpp
std::uint64_t Binary::nextBinaryId() {
  static std::atomic<uint64_t> id{0};
  return id.fetch_add(1, std::memory_order_relaxed);
}
```

Properties:
- **Not a content hash** — same flatbuffer loaded twice gets different IDs
- **Process-scoped** — resets to 0 on process restart
- **Monotonically increasing** — never reused within a process

### Detection strategy

Add `_check_program_transition()` at the top of every `preop()`:

```python
def _check_program_transition(self, binary) -> None:
    binary_id = binary.id
    if self._current_binary_id is None:
        self._current_binary_id = binary_id
        self._op_index = 0
        return

    if binary_id != self._current_binary_id:
        self._handle_binary_change(binary)
        return

    # Same binary — detect program restart via op counter
    # When op_index has reached end of registry and preop fires again,
    # it's a new program execution
```

**Same-binary program restart detection**: Since `ProgramContext` is opaque (no
`getProgramIndex()` in Python), detect boundaries by tracking the op counter.
When all ops in the registry have been processed and `preop` fires again, a new
program has started.

**Cleaner alternative**: Also expose `get_program_index(CallbackContext)` to
Python bindings. This gives explicit program index detection without heuristics.

## Problem 3: Golden Tensor Sharing Across Programs

### Key insight

Golden tensors are pure CPU/PyTorch — they have **no `TensorRef` dependency**
and remain valid across `ProgramContext` boundaries. Only device tensors have
stale refs. The reset must therefore be **asymmetric**:

```python
def _reset_for_new_program(self) -> None:
    """Clear device state. Golden pool is PRESERVED for cross-program reuse."""
    self.device_tensor_pool.clear()       # Stale TensorRefs — must clear
    # self.golden_tensor_pool NOT cleared  # Reuse across programs
    self._op_index = 0
    self.report.start_program()            # New report section
```

### What resets vs. what's preserved

| State | Reset | Preserved | Reason |
|-------|-------|-----------|--------|
| `device_tensor_pool` | Yes | | `TensorRef`/`DeviceHandle` objects are invalid |
| `golden_tensor_pool` | | Yes | CPU tensors, no device dependency |
| `device_ir_module` | | Yes | Same MLIR module for same binary |
| `registry` | | Yes | Op groups derived from module |
| `executor` | | Yes | References registry + golden pool |
| `_current_binary_id` | | Yes | Tracks binary identity |
| `_op_index` | Yes | | Reset for new program |
| `report` section | Yes | | New program gets new section |

### Golden pool preservation enables three scenarios

1. **Shared weights**: Forward and backward passes use the same model weights.
   Golden weight tensors computed in program 0 are found in the pool by name
   when program 1's preop looks them up — no recomputation needed.

2. **Output→input chaining**: If program 0 produces golden output for tensor
   `%5` and program 1 takes `%5` as input, the golden value is already in the
   pool from program 0's postop.

3. **Re-execution**: If the same program re-runs (training loop), golden tensors
   from the previous iteration serve as a warm cache for inputs that haven't
   changed.

### Name matching across programs

Tensor names are derived from MLIR SSA values in the TTNN module. Since all
programs in the same binary share the same TTNN module, tensor names for shared
weights and inter-program connections are identical.

### Stale golden data

If program 1 produces a tensor with the same name as program 0 but different
values, the pool entry gets overwritten in postop — this is the desired behavior
(latest golden wins). The preop callback reads from the golden pool regardless
of which program produced the entry.

## Different Binary Handling

When `binary.id` changes, the TTNN MLIR module is different — `IRModule` and
`Registry` must be rebuilt.

The Binary flatbuffer does not store a directly parseable MLIR `Module` object,
so Chisel cannot extract it automatically. Two approaches:

### Approach A: `module_provider` callback

Accept an optional `module_provider: Callable[[Binary], Module]` at init time:

```python
def _handle_binary_change(self, binary) -> None:
    if self._module_provider is None:
        raise RuntimeError(
            f"Different binary detected (id={binary.id}, expected "
            f"{self._current_binary_id}). Either re-create ChiselContext "
            f"or provide a module_provider callback."
        )
    mlir_source = self._extract_mlir_from_binary(binary)
    self.device_ir_module = IRModule(mlir_source=mlir_source, ...)
    self.registry = Registry(module=self.device_ir_module)
    self.registry.load_all_ops()
    self.executor = GoldenExecutor(self.registry, self.golden_tensor_pool)
    self.report = ReportWriter(...)
    self._current_binary_id = binary.id
    self.device_tensor_pool.clear()
    self._op_index = 0
```

Golden pool is preserved even across binary changes — if a new binary uses
tensors with the same names (e.g., shared weight names), the golden values
carry over.

### Approach B: Caller re-creates ChiselContext

The builder already creates/destroys `ChiselContext` per `execute_fb()` call
(`builder_runtime.py:737`), so different binaries are handled naturally. For
callers that don't re-create, the `module_provider` provides transparent
handling.

## Cross-Binary Tensor Identity via `Tensor::globalId`

### The problem

When tt-xla (or any caller) creates multiple separate flatbuffers and calls
`submit()` for each one sequentially, tensors flow between completely separate
binaries. The flatbuffer-level `TensorRef.global_id` is useless here — it's
assigned at compile time, scoped per-binary, and has no meaning across binaries.

### Runtime-level `Tensor::globalId`

`tt::runtime::Tensor` already carries a runtime-level `globalId` field
(`runtime/include/tt/runtime/types.h:395-412`) that is distinct from the
flatbuffer `TensorRef.global_id`:

```cpp
// runtime/lib/common/types.cpp:192-194
std::uint64_t Tensor::nextTensorGlobalId() {
  static std::atomic<std::uint64_t> globalId = 0;
  return globalId.fetch_add(1, std::memory_order_relaxed);
}
```

Properties:
- **Process-scoped**: unique across all binaries, programs, and executors
- **Assigned at construction**: every `Tensor` object gets a unique ID
- **Survives across program boundaries**: `ProgramExecutor` stores **pointers**
  to the original input `Tensor` objects (`program_executor.cpp:126-127`), not
  copies — the `globalId` is preserved

This means when tt-xla passes output tensors from `submit(binaryA)` as inputs
to `submit(binaryB)`, the receiving `ProgramExecutor` points to the same
`Tensor` objects with the same `globalId`.

### `TensorRef.global_id` vs `Tensor::globalId`

| | `TensorRef.global_id` | `Tensor::globalId` |
|---|---|---|
| Assigned at | Compile time (flatbuffer emission) | Runtime (`Tensor` construction) |
| Scope | Per-binary | Process-wide |
| Source | `FlatbufferObjectCache::nextGlobalId()` | `Tensor::nextTensorGlobalId()` (static atomic) |
| Purpose | Index into `ProgramTensorPool` | Unique tensor identity across all executions |
| Cross-binary | Not meaningful | Preserved when same `Tensor` object is reused |

### Usage pattern for Chisel

Using `pre_program`/`post_program` callbacks, Chisel can track which tensors
were produced by previous programs and detect cross-program tensor reuse:

```python
self._output_tensor_origins = {}  # Tensor::globalId -> program metadata

def post_program(self, binary, programContext):
    for tensor in get_program_output_tensors(programContext):
        self._output_tensor_origins[tensor.global_id] = {
            "binary_id": binary.id,
            "program_index": get_program_index(programContext),
        }

def pre_program(self, binary, programContext):
    for tensor in get_program_input_tensors(programContext):
        origin = self._output_tensor_origins.get(tensor.global_id)
        if origin is not None:
            # This input was an output of a previous program execution
            # -> reuse its golden value from golden_tensor_pool
```

### Required bindings

This pattern requires new Python bindings not yet exposed:
- `Tensor.global_id` — expose `Tensor::getGlobalId()` to Python
- `get_program_input_tensors(CallbackContext)` — return the `Tensor` objects
  from `ProgramTensorPool` for the program's input IDs
- `get_program_output_tensors(CallbackContext)` — same for output IDs
- `get_program_index(CallbackContext)` — expose
  `ProgramContext::getProgramIndex()` (`types.h:312`)

## Report Per-Program Support

Add `start_program(program_index)` to `ReportWriter`. Two options:

- **Option A**: Single CSV with a `program_index` column — simpler, one file
- **Option B**: Separate CSV per program — cleaner for large models

Builder already uses a similar pattern:
`CallbackRuntimeConfig.start_new_program()` (`builder_runtime.py:446-449`)
resets per-program state.

## Summary of Required Changes

| Component | Change |
|-----------|--------|
| `runtime/python/binary/binary.cpp` | Expose `Binary.id` property (1 line) |
| `runtime/python/runtime/runtime.cpp` | Expose `Tensor.global_id`, `get_program_input_tensors()`, `get_program_output_tensors()`, `get_program_index()` |
| `context.py` (new chisel) | Add `_check_program_transition()`, `_reset_for_new_program()`, `_handle_binary_change()`, `module_provider` parameter, cross-binary tensor tracking via `Tensor::globalId` |
| `tensors.py` (new chisel) | No change — `TensorPool.clear()` from `dict` is sufficient |
| `report.py` (new chisel) | Add `start_program()` method, `program_index` column |
| `callbacks.py` (new chisel) | No change — already passes `binary` through to context |
