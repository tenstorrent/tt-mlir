# PR 0b: Support Multi-Output Ops in getOpOutputRef

## Goal

Change `getOpOutputRef` to return `std::vector<TensorRef>` instead of
`std::optional<TensorRef>`, enabling proper handling of multi-output ops
like `SortOp`, `MaxPool2dWithIndicesOp`, and `BatchNormTrainingOp`.

Currently these ops return `std::nullopt` with a log warning, making their
outputs invisible to callbacks (including Chisel's postop comparison).

## Problem Analysis

### Current signature

`runtime/include/tt/runtime/runtime.h:244-248`:

```cpp
std::optional<TensorRef> getOpOutputRef(OpContext opContextHandle,
                                        CallbackContext programContextHandle);
```

### Multi-output ops currently unsupported

`runtime/lib/ttnn/runtime.cpp:1316-1341`:

```cpp
case ::tt::target::ttnn::OpType::BatchNormTrainingOp:
case ::tt::target::ttnn::OpType::MaxPool2dWithIndicesOp:
case ::tt::target::ttnn::OpType::SortOp:
// ... more ops ...
{
  LOG_WARNING("getting output tensor is not supported for ", ...);
  return std::nullopt;
}
```

These ops produce two outputs (e.g., `SortOp` returns sorted values +
indices, `MaxPool2dWithIndicesOp` returns pooled values + indices) but
the single-return API cannot represent them.

### Call chain

```
Python: tt_runtime.runtime.get_op_output_ref(op_ctx, prog_ctx)
  → runtime/lib/runtime.cpp: getOpOutputRef() dispatcher
    → runtime/lib/ttnn/runtime.cpp: ttnn::getOpOutputRef()
      → returns std::optional<TensorRef>

Also used by:
  runtime/lib/ttnn/runtime.cpp: getOpOutputTensor()
    → calls getOpOutputRef() then converts to Tensor
```

## Proposed Change

### New return type: `std::vector<TensorRef>`

```cpp
// BEFORE:
std::optional<TensorRef> getOpOutputRef(OpContext opContextHandle,
                                        CallbackContext programContextHandle);

// AFTER:
std::vector<TensorRef> getOpOutputRef(OpContext opContextHandle,
                                      CallbackContext programContextHandle);
```

- Single-output ops return a vector of size 1
- Multi-output ops return a vector of size N (e.g., 2 for SortOp)
- No-output ops (like `DeallocateOp`) return an empty vector

### Implementation for multi-output ops

For each multi-output op, extract all output refs:

```cpp
case ::tt::target::ttnn::OpType::SortOp: {
  auto *sortOp = opContext.type_as_SortOp();
  std::vector<TensorRef> refs;
  refs.push_back(utils::createRuntimeTensorRefFromTTNN(sortOp->value()));
  refs.push_back(utils::createRuntimeTensorRefFromTTNN(sortOp->indices()));
  return refs;
}

case ::tt::target::ttnn::OpType::MaxPool2dWithIndicesOp: {
  auto *poolOp = opContext.type_as_MaxPool2dWithIndicesOp();
  std::vector<TensorRef> refs;
  refs.push_back(utils::createRuntimeTensorRefFromTTNN(poolOp->out()));
  refs.push_back(utils::createRuntimeTensorRefFromTTNN(poolOp->indices()));
  return refs;
}
```

### Single-output ops

Change from:
```cpp
case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
  tensorRef = opContext.type_as_ToMemoryConfigOp()->out();
  break;
}
// ... at the end:
return utils::createRuntimeTensorRefFromTTNN(tensorRef.value());
```

To:
```cpp
case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
  tensorRef = opContext.type_as_ToMemoryConfigOp()->out();
  break;
}
// ... at the end:
if (!tensorRef.has_value()) {
  return {};
}
return {utils::createRuntimeTensorRefFromTTNN(tensorRef.value())};
```

The internal `tensorRef` variable stays as `std::optional<const TensorRef *>`
for single-output ops; only the final return wraps it in a vector.

## Files to Modify

| File | Change |
|------|--------|
| `runtime/include/tt/runtime/runtime.h` | Change return type declaration |
| `runtime/include/tt/runtime/detail/ttnn/ttnn.h` | Change return type declaration |
| `runtime/include/tt/runtime/detail/ttmetal/ttmetal.h` | Change return type declaration |
| `runtime/lib/ttnn/runtime.cpp` | Implement multi-output extraction, wrap single-output in vector |
| `runtime/lib/ttmetal/runtime.cpp` | Adapt return type (likely empty vector for unsupported) |
| `runtime/lib/runtime.cpp` | Update dispatcher return type |
| `runtime/python/runtime/runtime.cpp` | Update `get_op_output_ref` binding to return list |
| `runtime/python/runtime/stubs_macos.cpp` | Update stub return type |

### Caller updates

**`getOpOutputTensor` in `runtime/lib/ttnn/runtime.cpp:950-1000`:**

This function calls `getOpOutputRef` and uses the result. It needs to be
updated to handle the vector. Two options:

1. **Keep `getOpOutputTensor` single-output** — take `output_index` parameter,
   return the tensor at that index
2. **Make `getOpOutputTensor` return a vector** — parallel to `getOpOutputRef`

Option 1 is less invasive:

```cpp
std::optional<Tensor>
getOpOutputTensor(OpContext opContextHandle,
                  CallbackContext programContextHandle,
                  size_t output_index = 0) {
  auto refs = getOpOutputRef(opContextHandle, programContextHandle);
  if (output_index >= refs.size()) {
    return std::nullopt;
  }
  // ... convert refs[output_index] to Tensor ...
}
```

**Python binding** in `runtime/python/runtime/runtime.cpp:488-508`:

```python
# BEFORE: returns Optional[TensorRef]
# AFTER: returns List[TensorRef]
```

Update docstring accordingly. Existing Python callers that check for `None`
will need to check for empty list instead.

**Builder's `golden()` callback** in `tools/builder/base/builder_runtime.py`:

Currently calls `tt_runtime.runtime.get_op_output_tensor(op_context, program_context)`
which returns a single tensor map. This may need updating depending on how
`getOpOutputTensor` changes. If we keep `getOpOutputTensor` taking an
`output_index` parameter, the builder can call it with `output_index=0` for
backward compatibility.

## Test Plan

### Unit tests (C++)
- Test `getOpOutputRef` with a single-output op — verify vector size 1
- Test `getOpOutputRef` with a multi-output op (SortOp) — verify vector size 2
- Test `getOpOutputRef` with a no-output op (DeallocateOp) — verify empty vector

### Python binding tests
- Verify `get_op_output_ref()` returns a list
- Verify backward compatibility: single-output ops return list of length 1

### Integration tests
- Run existing `verify_intermediates` tests — ensure no regression
- Run a model with SortOp or MaxPool2dWithIndices — verify both outputs are accessible

### Multi-output op coverage

Identify all multi-output ops that currently return `std::nullopt` and
implement their extraction. At minimum:

| Op | Outputs |
|----|---------|
| `SortOp` | sorted_values, indices |
| `MaxPool2dWithIndicesOp` | pooled_output, indices |
| `BatchNormTrainingOp` | output, mean, rstd (3 outputs) |

Check the flatbuffer schema for each op's output fields to determine the
exact accessor methods.

## Dependencies

None — standalone runtime change. **Not required for initial Chisel
integration.** Without this, Chisel skips multi-output ops (they return
`std::nullopt` / empty from `getOpOutputRef`). Can land at any time to extend
Chisel's op coverage to SortOp, MaxPool2dWithIndicesOp, etc.
