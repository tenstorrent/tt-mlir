# Named Callback API for DebugHooks

## Description

Replace `Hooks::get(pre, post)` with `Hooks::get()` + `setCallbacks(name, CallbackSet)` so multiple clients (chisel, ttrt, builder) can coexist. Store callbacks in `unordered_map<string, CallbackSet>`. Migrate all existing callers.

## Tasks

- Add `CallbackSet` struct with `preOp`/`postOp` fields in `debug.h`
- Replace `Hooks::get(pre, post)` with `Hooks::get()` + `setCallbacks(name, CallbackSet)` in `debug.h`
- Add `unordered_map<string, CallbackSet>` storage
- Add `unregisterHooks(name)` and `getRegisteredNames()` methods
- Replace `runCallback` with `runOpCallbacks` that iterates callback map in `program_executor.cpp`
- Expose `set_callbacks(name, pre_op=, post_op=)` Python binding in `runtime.cpp`
- Update `unregister_hooks` to accept optional name argument
- Migrate caller: `tools/ttrt/common/run.py`
- Migrate caller: `tools/builder/base/builder_runtime.py`
- Migrate caller: `runtime/test/ttnn/python/n150/test_intermidate_tensor_manipulation.py`

