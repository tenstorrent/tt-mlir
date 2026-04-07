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

## Chisel PR 1: Single Op Isolation ([detail](../docs/pr1_single_op_isolation.md))

Op-level isolation testing — each op tested independently. preOp copies device
inputs, postOp runs golden + compares. No cross-op tensor chaining.

- Create `tools/chisel/CMakeLists.txt` with `declare_mlir_python_sources` packaging.
- Create `tools/chisel/chisel/__init__.py` package init with exports.
- Implement `IRModule` wrapper with MLIR traversal, `get_op_inputs()`, `get_op_outputs()` in `chisel/ops.py`.
- Implement `execute_golden(op, ir_module, inputs: dict)` standalone function that replays a single TTNN op via `GOLDEN_MAPPINGS` in `chisel/executor.py`. Takes plain dict of inputs, returns result without storing.
- Implement slim `ChiselContext` singleton in `chisel/context.py` — holds `ir_module`, `op_iter`, `_stashed_inputs`, `_current_op`. No `BinaryState`/`ProgramState` hierarchy.
- Implement 2 callback functions (`pre_op`, `post_op`) in `chisel/callbacks.py`. preOp copies device inputs to host and stashes them; postOp runs golden with stashed inputs, compares, logs to stdout.
- Implement `chisel/utils.py` with dtype maps and runtime tensor conversion helpers.

## Chisel PR 2: Single Program Flow ([detail](../docs/pr2_single_program.md))

Cross-op golden tensor chaining within a program. Golden outputs persist in
TensorPool and feed subsequent ops. Full state hierarchy and program-level
callbacks.

- Implement `TensorPool(dict)` (keyed by SSA name or globalId, stores `GoldenMapTensor` directly, no disk caching) in `chisel/tensors.py`.
- Expand `ChiselContext` in `chisel/context.py` to full `ChiselContext`/`BinaryState`/`ProgramState` hierarchy.
- Add `execute_golden_from_pool()` pool-aware wrapper in `chisel/executor.py` — pulls inputs from pool, stores outputs.
- Expand `chisel/callbacks.py` from 2 to 4 callbacks — add `pre_program`, `post_program`. Update `pre_op`/`post_op` to use ProgramState's golden pool instead of device-stashed inputs.
- Update `chisel/__init__.py` exports (4 callbacks).
- Add `chisel/tensors.py` to `CMakeLists.txt`.

## Chisel PR 3: Reporting + Cross-Program Sharing ([detail](../docs/pr3_reporting_and_sharing.md))

Add CSV reporting and cross-program/cross-binary tensor sharing via the global
tensor pool. Enables multi-program binary support.

- Implement `ReportWriter` with per-op CSV rows (PCC, abs_error, rel_error) and per-program sections in `chisel/report.py`.
- Wire `ReportWriter` into `postOp` and `postProgram` callbacks.
- Add `global_tensor_pool` ↔ `program.golden_tensor_pool` copying in `preProgram`/`postProgram`.
- Add disk caching support to `TensorPool`.

## Chisel PR 4: Skip Mode ([detail](../docs/pr4_skip_mode.md))

Add skip mode: preOp stashes input tensors before device execution, postOp
replaces device outputs with golden-computed results.

- Add `_skip_stash` to `ProgramState` and skip-mode input stashing in `preOp`.
- Add golden-replace logic in `postOp` — execute golden with stashed inputs, overwrite device tensors.
- Add skip configuration (which ops to skip).

## Chisel PR 5: Builder Integration ([detail](../docs/pr5_builder_integration.md))

Wire Chisel into builder's execution flow via `enable_chisel` parameter.

- Add `enable_chisel`, `chisel_output_dir`, `chisel_report_path` parameters to `execute_fb()` in `builder_runtime.py`.
- Add mutual exclusivity check with `enable_intermediate_verification`.
- Add Chisel callback registration in `execute_fb()`.
- Add `ChiselContext.reset_instance()` cleanup after execution.
- Forward chisel params through `_compile_and_execute()` and `compile_and_execute_ttnn()` in `builder_apis.py`.
