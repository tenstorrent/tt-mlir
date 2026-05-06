# Chisel PR 5: Builder Integration

## Description

Wire Chisel into the builder's execution flow via `enable_chisel` parameter on
`execute_fb()` and `compile_and_execute_ttnn()`. When enabled, `ChiselContext`
is initialized and Chisel's 4 callbacks are registered instead of builder's own
golden callbacks. This is the final PR that makes Chisel usable through the
standard builder API.

No new Chisel code — only modifies `builder_runtime.py` and `builder_apis.py`.

See [pr5_builder_integration.md](../docs/pr5_builder_integration.md) for full
design.

## Acceptance Criteria

- `enable_chisel=True` registers all 4 Chisel callbacks with `DebugHooks`
- `enable_chisel` and `enable_intermediate_verification` are mutually exclusive
  (`ValueError` if both set)
- `ChiselContext` constructed with correct `output_dir` and `report_base_path`
- `ChiselContext.reset_instance()` called after execution completes (even on error)
- `compile_and_execute_ttnn()` forwards chisel params through to `execute_fb()`
- Without `enable_chisel`, builder's own callbacks are used as before

## Dependencies

- Chisel PR 4: Skip Mode
