# Program Introspection Python Bindings

## Description

Add Python bindings for querying program metadata from callbacks: program index, input/output tensor refs, tensor global IDs, and binary IDs. Required so Chisel's `preProgram` callback can copy program input tensors from device into the golden pool.

## Tasks

- [ ] Add `get_program_index(CallbackContext)` Python binding in `runtime.cpp`
- [ ] Add `getProgramInputRefs(CallbackContext)` → `vector<TensorRef>` in `runtime.h`, `ttnn/runtime.cpp`, `runtime.cpp`
- [ ] Add `getProgramOutputRefs(CallbackContext)` → `vector<TensorRef>` in same files + TTMetal stubs
- [ ] Add `get_program_input_refs` / `get_program_output_refs` Python bindings in `runtime.cpp`
- [ ] Expose `Tensor.global_id` (uint64) as read-only Python property in `runtime.cpp`
- [ ] Expose `Binary.id` as read-only Python property in `binary.cpp`
- [ ] Add macOS stubs in `stubs_macos.cpp`
