# Chisel Implementation Tasks

## PR 0a-1: GIL-Safety Fix ([detail](../docs/pr0a_hooks_refactor.md))

- Return callbacks by `const std::optional<CallbackFn>&` instead of by value in `runtime/include/tt/runtime/debug.h`.
- Change `runCallback()` to take `const std::optional<...>&` in `runtime/lib/ttnn/program_executor.cpp`.
- Accept callbacks by rvalue ref + `std::move` in `Hooks::get()` in `runtime/lib/common/debug.cpp`.
- Update non-debug path to return static empty optional by const ref in `runtime/include/tt/runtime/debug.h`.

## PR 0a-2a: Named Callback API ([detail](../docs/pr0a2a_named_callback_api.md))

- Replace `Hooks::get(pre, post)` with `Hooks::get()` + `setCallbacks(name, CallbackSet)` in `runtime/include/tt/runtime/debug.h`.
- Add `CallbackSet` struct with `preOp`/`postOp` fields and `unordered_map<string, CallbackSet>` storage.
- Add `unregisterHooks(name)` and `getRegisteredNames()` methods.
- Replace `runCallback` with `runOpCallbacks` that iterates the callback map in `runtime/lib/ttnn/program_executor.cpp`.
- Expose `set_callbacks(name, pre_op=, post_op=)` Python binding in `runtime/python/runtime/runtime.cpp`.
- Update `unregister_hooks` to accept optional name argument.
- Migrate callers: `tools/ttrt/common/run.py`, `tools/builder/base/builder_runtime.py`, `runtime/test/ttnn/python/n150/test_intermidate_tensor_manipulation.py`.

## PR 0a-2b: Program-Level Hooks ([detail](../docs/pr0a2b_program_level_hooks.md))

- Add `ProgramCallbackFn = std::function<void(Binary, CallbackContext)>` type alias in `runtime/include/tt/runtime/debug.h`.
- Add `preProgram`/`postProgram` fields to `CallbackSet`.
- Add `runProgramCallbacks()` in `runtime/include/tt/runtime/detail/ttnn/program_executor.h` and `runtime/lib/ttnn/program_executor.cpp`.
- Call `runProgramCallbacks` before/after the op loop in `ProgramExecutor::execute()`.
- Expose `pre_program`/`post_program` kwargs in Python `set_callbacks()` in `runtime/python/runtime/runtime.cpp`.

## PR 0a-3: Program Introspection Bindings ([detail](../docs/pr0a_program_input_output_refs.md))

- Add `get_program_index(CallbackContext)` Python binding in `runtime/python/runtime/runtime.cpp`.
- Add `getProgramInputRefs(CallbackContext)` → `vector<TensorRef>` in `runtime/include/tt/runtime/runtime.h`, `runtime/lib/ttnn/runtime.cpp`, `runtime/lib/runtime.cpp`.
- Add `getProgramOutputRefs(CallbackContext)` → `vector<TensorRef>` in same files + TTMetal stubs.
- Add `get_program_input_refs` / `get_program_output_refs` Python bindings in `runtime/python/runtime/runtime.cpp`.
- Expose `Tensor.global_id` (uint64) as a read-only Python property in `runtime/python/runtime/runtime.cpp`.
- Expose `Binary.id` as a read-only Python property in `runtime/python/binary/binary.cpp`.
- Add macOS stubs in `runtime/python/runtime/stubs_macos.cpp`.

## PR 0b: Multi-Output Refs ([detail](../docs/pr0b_multi_output_ref.md))

- Change `getOpOutputRef()` to return `vector<TensorRef>` for multi-output ops (Sort, MaxPool2dWithIndices, etc.).


## Unified Metrics

- Create `tools/golden/metrics.py` with unified PCC/atol/rtol computation (pure torch, no numpy), consolidating duplicates from builder and ttrt.

## Chisel Scaffold & Leaf Modules

- Create `tools/chisel/CMakeLists.txt` with `declare_mlir_python_sources` packaging.
- Create `tools/chisel/chisel/__init__.py` package init.
- Implement `TensorValue` and `TensorPool` (keyed by SSA name or globalId) in `chisel/tensors.py`.
- Implement `IRModule` wrapper with MLIR traversal, `get_op_inputs()`, `get_op_outputs()`, and `hash_location()` in `chisel/ops.py`.

## Chisel Executor & Report (PR 2)

- Implement `execute_golden()` standalone function that replays TTNN ops via `GOLDEN_MAPPINGS` in `chisel/executor.py`.
- Implement `ReportWriter` with per-op CSV rows (PCC, abs_error, rel_error) and per-program sections in `chisel/report.py`.

## Chisel Logic Layer (PR 3)

- Implement `ChiselContext` singleton, `BinaryState`, and `ProgramState` hierarchy in `chisel/context.py`.
- Implement the 4 callback functions (`pre_program`, `post_program`, `pre_op`, `post_op`) in `chisel/callbacks.py`.
- Implement `chisel/utils.py` with location parsing, dtype maps, and runtime tensor conversion helpers.
