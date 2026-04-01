# Chisel Implementation Tasks

## Runtime Changes (PR 0a)

- Add  `preProgram`/`postProgram` hooks to `DebugHooks` in `runtime/include/tt/runtime/debug.h`.
- Add `get_program_index(CallbackContext)` Python binding in `runtime/python/runtime/runtime.cpp`.
- Add `get_program_input_refs(CallbackContext)` → `List[TensorRef]` Python binding in `runtime/python/runtime/runtime.cpp`.
- Add `get_program_output_refs(CallbackContext)` → `List[TensorRef]` Python binding in `runtime/python/runtime/runtime.cpp`.
- Expose `Tensor.global_id` (uint64, runtime-assigned) as a read-only Python property in `runtime/python/runtime/runtime.cpp`.
- Expose `Binary.id` as a read-only Python property in `runtime/python/binary/binary.cpp`.
- Fix `DebugHooks` callback copy semantics to return by const ref (GIL safety).
- Change `getOpOutputRef()` to return `vector<TensorRef>` for multi-output ops (Sort, MaxPool2dWithIndices, etc.).


## Unified Metrics

- Create `tools/golden/metrics.py` with unified PCC/atol/rtol computation (pure torch, no numpy), consolidating duplicates from builder and ttrt.

## Chisel Scaffold & Leaf Modules

- Create `tools/chisel/CMakeLists.txt` with `declare_mlir_python_sources` packaging.
- Create `tools/chisel/chisel/__init__.py` package init.
- Implement `TensorValue` and `TensorPool` (keyed by SSA name or globalId) in `chisel/tensors.py`.
- Implement `IRModule` wrapper with MLIR traversal, `get_op_inputs()`, `get_op_outputs()`, and `hash_location()` in `chisel/ops.py`.

## Chisel Executor & Report (PR 2)

- Implement `GoldenExecutor` that replays TTNN ops via `GOLDEN_MAPPINGS` in `chisel/executor.py`.
- Implement `ReportWriter` with per-op CSV rows (PCC, abs_error, rel_error) and per-program sections in `chisel/report.py`.

## Chisel Logic Layer (PR 3)

- Implement `ChiselContext` singleton, `BinaryState`, and `ProgramState` hierarchy in `chisel/context.py`.
- Implement the 4 callback functions (`pre_program`, `post_program`, `pre_op`, `post_op`) in `chisel/callbacks.py`.
- Implement `chisel/utils.py` with location parsing, dtype maps, and runtime tensor conversion helpers.
