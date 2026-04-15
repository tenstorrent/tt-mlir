# PR 0a-3: `get_program_input_refs` and `get_program_output_refs`

## Summary

Add two new Python bindings that return `List[TensorRef]` for a program's
inputs and outputs, given only a `CallbackContext`. These wrap existing
flatbuffer `TensorRef*` pointers — no new tensors are allocated.

## Files to modify (8 files, dependency order)

### 1. `runtime/include/tt/runtime/detail/ttnn/ttnn.h` (after `getOpInputRefs` ~line 261)

```cpp
std::vector<tt::runtime::TensorRef>
getProgramInputRefs(CallbackContext programContextHandle);

std::vector<tt::runtime::TensorRef>
getProgramOutputRefs(CallbackContext programContextHandle);
```

### 2. `runtime/include/tt/runtime/detail/ttmetal/ttmetal.h` (after `getOpInputRefs` ~line 164)

```cpp
std::vector<tt::runtime::TensorRef>
getProgramInputRefs(CallbackContext programContextHandle);

std::vector<tt::runtime::TensorRef>
getProgramOutputRefs(CallbackContext programContextHandle);
```

### 3. `runtime/lib/ttnn/runtime.cpp` (after `getOpInputRefs` ~line 1944)

```cpp
std::vector<tt::runtime::TensorRef>
getProgramInputRefs(CallbackContext programContextHandle) {
  auto &programContext =
      programContextHandle.as<ProgramContext>(DeviceRuntime::TTNN);
  const auto *program =
      utils::getProgram(programContext.getExecutableHandle(),
                        programContext.getProgramIndex());
  std::vector<tt::runtime::TensorRef> refs;
  for (const auto *input : *program->inputs()) {
    refs.emplace_back(utils::createRuntimeTensorRefFromTTNN(input));
  }
  return refs;
}

std::vector<tt::runtime::TensorRef>
getProgramOutputRefs(CallbackContext programContextHandle) {
  auto &programContext =
      programContextHandle.as<ProgramContext>(DeviceRuntime::TTNN);
  const auto *program =
      utils::getProgram(programContext.getExecutableHandle(),
                        programContext.getProgramIndex());
  std::vector<tt::runtime::TensorRef> refs;
  for (const auto *output : *program->outputs()) {
    refs.emplace_back(utils::createRuntimeTensorRefFromTTNN(output));
  }
  return refs;
}
```

### 4. `runtime/lib/ttmetal/runtime.cpp` (after `getOpInputRefs` ~line 740)

```cpp
std::vector<tt::runtime::TensorRef>
getProgramInputRefs(CallbackContext programContextHandle) {
  LOG_FATAL("getProgramInputRefs not implemented for metal runtime");
  return {};
}

std::vector<tt::runtime::TensorRef>
getProgramOutputRefs(CallbackContext programContextHandle) {
  LOG_FATAL("getProgramOutputRefs not implemented for metal runtime");
  return {};
}
```

### 5. `runtime/include/tt/runtime/runtime.h` (after `getOpInputRefs` ~line 253)

```cpp
std::vector<TensorRef>
getProgramInputRefs(CallbackContext programContextHandle);

std::vector<TensorRef>
getProgramOutputRefs(CallbackContext programContextHandle);
```

### 6. `runtime/lib/runtime.cpp` (after `getOpInputRefs` dispatch ~line 1112)

```cpp
std::vector<tt::runtime::TensorRef>
getProgramInputRefs(CallbackContext programContextHandle) {
  using RetType = std::vector<tt::runtime::TensorRef>;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getProgramInputRefs(programContextHandle);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getProgramInputRefs(programContextHandle);
      },
      [&]() -> RetType {
        detail::fatalNotImplemented("getProgramInputRefs", HostRuntime::Distributed);
      });
}

std::vector<tt::runtime::TensorRef>
getProgramOutputRefs(CallbackContext programContextHandle) {
  using RetType = std::vector<tt::runtime::TensorRef>;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getProgramOutputRefs(programContextHandle);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getProgramOutputRefs(programContextHandle);
      },
      [&]() -> RetType {
        detail::fatalNotImplemented("getProgramOutputRefs", HostRuntime::Distributed);
      });
}
```

### 7. `runtime/python/runtime/runtime.cpp` (after `get_op_input_refs` ~line 532)

```cpp
  m.def(
      "get_program_input_refs",
      [](tt::runtime::CallbackContext &program_context_handle) {
        return tt::runtime::getProgramInputRefs(program_context_handle);
      },
      nb::arg("program_context_handle"), R"(
    Return references to all input tensors of the program.

    Parameters
    ----------
    program_context_handle : ttrt.runtime.CallbackContext

    Returns
    -------
    List[ttrt.runtime.TensorRef]
    )");

  m.def(
      "get_program_output_refs",
      [](tt::runtime::CallbackContext &program_context_handle) {
        return tt::runtime::getProgramOutputRefs(program_context_handle);
      },
      nb::arg("program_context_handle"), R"(
    Return references to all output tensors of the program.

    Parameters
    ----------
    program_context_handle : ttrt.runtime.CallbackContext

    Returns
    -------
    List[ttrt.runtime.TensorRef]
    )");
```

### 8. `runtime/python/runtime/stubs_macos.cpp` (after `getOpInputRefs` stub ~line 222)

```cpp
std::vector<TensorRef>
getProgramInputRefs(CallbackContext programContextHandle) {
  __builtin_trap();
}
std::vector<TensorRef>
getProgramOutputRefs(CallbackContext programContextHandle) {
  __builtin_trap();
}
```

## Design notes

- **No new tensors** — `createRuntimeTensorRefFromTTNN` wraps existing
  flatbuffer `TensorRef*` via `unsafeBorrowShared`.
- **Only `CallbackContext` needed** — unlike `getOpInputRefs` which also
  requires `OpContext`, program-level refs come from the flatbuffer program
  directly via `ProgramContext.getExecutableHandle()` + `getProgramIndex()`.
- **`utils::getProgram`** is the existing helper that does
  `getBinary` + `programs()->Get(idx)`.
