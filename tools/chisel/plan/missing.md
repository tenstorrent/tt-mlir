# Gaps from dump.md — Open Items

## Still Open

- **Multi-chip handling**: Per-device tensor lists, per-device comparison.
  Not addressed by hierarchical redesign. Needs separate analysis.

## Prerequisites (PR 0a)

- **`get_program_input_ids` / `get_program_output_ids` Python bindings**: New
  nanobind wrappers needed in `runtime/python/runtime/runtime.cpp`. These cast
  `CallbackContext` → `ProgramContext` → `getTensorPool()` →
  `getProgramInputIds()` / `getProgramOutputIds()`. Required so that
  `preProgram` can copy all program input tensors from device into
  `program.golden_tensor_pool` upfront, eliminating per-op "check if exists"
  logic in `preOp`. The C++ API already exists on `ProgramTensorPool`
  (`runtime/detail/ttnn/types/types.h`).
