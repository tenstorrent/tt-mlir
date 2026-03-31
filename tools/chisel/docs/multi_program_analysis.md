# Multi-Program Execution Design

## Problem Statement

TTRT runs multiple programs back-to-back from the same binary (e.g., forward +
backward passes) or re-runs the same program (training loops). Three concerns
must be addressed:

1. **Stale TensorRefs**: Each `submit()` creates a fresh `ProgramContext` with a
   new tensor pool. Device tensor references from program N are invalid in N+1.
2. **Program identity**: Chisel must know which binary and program is executing
   to look up the correct ops and state.
3. **Golden tensor sharing**: Forward and backward passes share weights. Golden
   tensors from one program should be available to the next.

## Solution: Hierarchical State Model

The hierarchical `ChiselContext → BinaryState → ProgramState` design solves all
three concerns structurally:

```
ChiselContext (singleton)
├── global_tensor_pool: TensorPool       # keyed by Tensor::globalId
├── binaries: Dict[binary_id, BinaryState]
├── current_binary / current_program

BinaryState
├── ir_module, registry                  # parsed once per binary
├── programs: Dict[program_index, ProgramState]

ProgramState
├── golden_tensor_pool: TensorPool       # isolated, preserved across re-executions
├── device_tensor_pool: TensorPool       # cleared each execution
├── executor: GoldenExecutor
├── ops + op_iter                        # iterator advances with callbacks
```

**Stale TensorRefs** — `device_tensor_pool` is per-`ProgramState` and cleared in
`reset_for_new_execution()` at each `preProgram` call.

**Program identity** — `preProgram(binary, program_context)` provides explicit
`binary.id` and `program_index`. No heuristic needed.

**Golden tensor sharing** — Two-level pool: per-program `golden_tensor_pool`
(keyed by SSA name) for execution, `global_tensor_pool` (keyed by
`Tensor::globalId`) for cross-program/cross-binary sharing. `preProgram` copies
global → program, `postProgram` copies program → global.

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

**Program-level callbacks**: `(Binary, CallbackContext)`
**Op-level callbacks**: `(Binary, CallbackContext, OpContext)`

- `Binary` wraps the flatbuffer binary. Has a `binaryId` exposed via `.id`
  property.
- `CallbackContext` wraps a `ProgramContext*`. Exposes `program_index`.
- `OpContext` wraps the flatbuffer operation descriptor.
- Callbacks are registered once globally via `DebugHooks` and fire for ALL
  program executions.

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

| Identifier | Type | Source | Scope |
|---|---|---|---|
| `Binary::binaryId` | `static atomic<uint64_t>` | `runtime/lib/binary.cpp:56-58` | Unique per `Binary` construction, process-wide |
| `Tensor::globalId` | `static atomic<uint64_t>` | `runtime/lib/common/types.cpp:192-194` | Unique per `Tensor` construction, process-wide |
| Program index | `uint32_t` | `program_index` arg to `submit()` | Positional within a binary (0, 1, 2...) |

Programs have **no process-wide counter** of their own. To uniquely identify a
program execution across all binaries, use the composite key
`(binary.id, program_index)`.

## Callback Flow

```
preProgram(binary, program_context):
    binary_id = binary.id
    program_index = get_program_index(program_context)

    # Get or create BinaryState
    if binary_id not in ctx.binaries:
        ctx.binaries[binary_id] = BinaryState(binary)
    state = ctx.binaries[binary_id]

    # Get or create ProgramState
    program = state.get_or_create_program(program_index)
    program.reset_for_new_execution()

    # Copy golden tensors from global pool
    # Matched by Tensor::globalId → SSA name mapping
    for global_id, tensor in ctx.global_tensor_pool.items():
        if matches_program_input(global_id, program):
            program.golden_tensor_pool[ssa_name] = tensor

    ctx.current_binary = state
    ctx.current_program = program

preOp(binary, program_context, op_context):
    op = next(ctx.current_program.op_iter)
    # capture device inputs, copy to golden pool...

postOp(binary, program_context, op_context):
    # capture device output, run golden, compare, report...

postProgram(binary, program_context):
    # Copy program golden tensors → global pool
    for ssa_name, tensor in ctx.current_program.golden_tensor_pool.items():
        global_id = resolve_global_id(ssa_name, program_context)
        ctx.global_tensor_pool[global_id] = tensor

    # Aggregate metrics, finalize report section
```

## State Lifecycle Per Execution

### What resets vs. what's preserved

| State | On `reset_for_new_execution()` | On new binary |
|-------|:------------------------------:|:-------------:|
| `ProgramState.device_tensor_pool` | Cleared | N/A (new ProgramState) |
| `ProgramState.golden_tensor_pool` | Preserved | N/A (new ProgramState) |
| `ProgramState.op_iter` | Reset to start | N/A (new ProgramState) |
| `ProgramState.executor` | Preserved | N/A (new ProgramState) |
| `BinaryState.ir_module` | Preserved | New (parsed from new binary) |
| `BinaryState.registry` | Preserved | New (from new module) |
| `ChiselContext.global_tensor_pool` | Preserved | Preserved |

### Golden Tensor Sharing Across Programs

Preserving the golden tensor pool across program re-executions (and sharing
via the global pool across programs) enables three scenarios:

1. **Shared weights**: Forward and backward passes use the same model weights.
   Golden weight tensors computed in program 0's golden pool are copied to the
   global pool in `postProgram`, then copied into program 1's golden pool in
   `preProgram`.

2. **Output-to-input chaining**: If program 0 produces a golden output for
   tensor `%5` and program 1 takes `%5` as input, the golden value flows
   through the global pool.

3. **Re-execution warm cache**: When the same program re-runs in a training
   loop, its `golden_tensor_pool` is preserved from the previous iteration.
   Unchanged inputs are already present.

### Name Matching

- **Within a binary**: Tensor names are MLIR SSA values from the TTNN module.
  All programs in the same binary share the same module, so names are identical
  for shared data.
- **Across binaries**: The `global_tensor_pool` is keyed by `Tensor::globalId`,
  not SSA names. The `preProgram` callback maps `globalId` to SSA names when
  copying into the per-program pool.

### Stale Golden Data

If a program re-executes and produces a tensor with the same SSA name but
different values, the per-program pool entry is overwritten in `postop` — latest
golden wins. The global pool entry is updated in `postProgram`.

## Cross-Binary Tensor Identity via `Tensor::globalId`

### The problem

When tt-xla (or any caller) creates multiple separate flatbuffers and calls
`submit()` for each one sequentially, tensors flow between completely separate
binaries. The flatbuffer-level `TensorRef.global_id` is useless here — it's
assigned at compile time, scoped per-binary, and has no meaning across binaries.

### Runtime-level `Tensor::globalId`

`tt::runtime::Tensor` carries a runtime-level `globalId` field
(`runtime/include/tt/runtime/types.h:395-412`) distinct from the flatbuffer
`TensorRef.global_id`:

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

### Usage in Chisel

The `global_tensor_pool` on `ChiselContext` is keyed by `Tensor::globalId`.
This enables cross-binary golden tensor sharing:

- `postProgram` stores golden outputs with their `Tensor::globalId` in the
  global pool
- `preProgram` for a different binary matches incoming input tensors by
  `globalId` to find existing golden values

### Required bindings

This pattern requires Python bindings:
- `Tensor.global_id` — expose `Tensor::getGlobalId()` to Python
- `get_program_input_tensors(CallbackContext)` — return the `Tensor` objects
  from `ProgramTensorPool` for the program's input IDs
- `get_program_output_tensors(CallbackContext)` — same for output IDs
- `get_program_index(CallbackContext)` — expose
  `ProgramContext::getProgramIndex()` (`types.h:312`)

## Report Per-Program Support

`ReportWriter` (scoped per-`BinaryState`) supports per-program sections via
`start_program(program_index)`. Two options:

- **Option A**: Single CSV with a `program_index` column — simpler, one file
- **Option B**: Separate CSV per program — cleaner for large models

Builder already uses a similar pattern:
`CallbackRuntimeConfig.start_new_program()` (`builder_runtime.py:446-449`)
resets per-program state.
