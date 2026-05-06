# TensorRef Shape/Dtype Inspection via Nanobind

## Motivation

`get_op_input_refs` and `get_op_output_refs` return `List[TensorRef]` to Python
callbacks. Currently `TensorRef` has **zero methods** in Python — to inspect
shape or dtype you must call `retrieve_tensor_from_pool`, which copies the
tensor to host (allocation + H2D transfer). For Chisel pre-op validation (e.g.
checking that input shapes/dtypes match what MLIR declared), this is
unnecessary overhead.

`TensorRef` wraps a `::tt::target::ttnn::TensorRef*` flatbuffer pointer. That
pointer already carries the shape and dtype that were compiled in from MLIR:

```
TensorRef.desc()
  └── TensorDesc
        ├── shape()                       → logical shape  [int32]
        └── layout() → LayoutDesc
              └── memory_desc() → MemoryDesc
                    └── data_type()       → tt::target::DataType
```

No tensor is allocated — these are static metadata embedded in the compiled
flatbuffer.

---

## API (after change)

```python
# In any preop/postop callback:
for ref in ttrt.runtime.get_op_input_refs(op_ctx, program_ctx):
    shape = ref.get_shape()   # List[int], zero allocation
    dtype = ref.get_dtype()   # ttrt.runtime.DataType enum

for ref in ttrt.runtime.get_op_output_refs(op_ctx, program_ctx):
    shape = ref.get_shape()
    dtype = ref.get_dtype()
```

---

## Implementation

Eight files need touching, all following the existing `retrieveTensorFromPool`
dispatch pattern.

### 1. `runtime/include/tt/runtime/runtime.h`
Add after `getOpInputRefs`:
```cpp
std::vector<uint32_t> getTensorRefShape(TensorRef tensorRef);
::tt::target::DataType getTensorRefDataType(TensorRef tensorRef);
```
`::tt::target::DataType` is already available via `types.h` → `types_generated.h`.

### 2. `runtime/include/tt/runtime/detail/ttnn/ttnn.h`
Add after `getOpInputRefs` in the `tt::runtime::ttnn` namespace:
```cpp
std::vector<uint32_t> getTensorRefShape(tt::runtime::TensorRef tensorRef);
::tt::target::DataType getTensorRefDataType(tt::runtime::TensorRef tensorRef);
```

### 3. `runtime/include/tt/runtime/detail/ttmetal/ttmetal.h`
Add matching declarations in `tt::runtime::ttmetal` (implemented as stubs).

### 4. `runtime/lib/ttnn/runtime.cpp`
Add near `retrieveTensorFromPool` (~line 2097).
`ttmlir/Target/TTNN/types_generated.h` is already included:
```cpp
std::vector<uint32_t> getTensorRefShape(tt::runtime::TensorRef tensorRef) {
  const auto &ref =
      tensorRef.as<::tt::target::ttnn::TensorRef>(DeviceRuntime::TTNN);
  const auto *shape = ref.desc()->shape();
  return std::vector<uint32_t>(shape->begin(), shape->end());
}

::tt::target::DataType getTensorRefDataType(tt::runtime::TensorRef tensorRef) {
  const auto &ref =
      tensorRef.as<::tt::target::ttnn::TensorRef>(DeviceRuntime::TTNN);
  return ref.desc()->layout()->memory_desc()->data_type();
}
```

### 5. `runtime/lib/ttmetal/runtime.cpp`
Add `LOG_FATAL` stubs after `getOpInputRefs` (~line 757):
```cpp
std::vector<uint32_t> getTensorRefShape(tt::runtime::TensorRef) {
  LOG_FATAL("getTensorRefShape for metal runtime is not implemented");
  return {};
}
::tt::target::DataType getTensorRefDataType(tt::runtime::TensorRef) {
  LOG_FATAL("getTensorRefDataType for metal runtime is not implemented");
  return ::tt::target::DataType::Float32;
}
```

### 6. `runtime/lib/runtime.cpp`
Add dispatch after `getOpInputRefs` (~line 1110):
```cpp
std::vector<uint32_t> getTensorRefShape(TensorRef tensorRef) {
  using RetType = std::vector<uint32_t>;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType { return ttnn::getTensorRefShape(tensorRef); },
      [&]() -> RetType { return ttmetal::getTensorRefShape(tensorRef); },
      [&]() -> RetType {
        detail::fatalNotImplemented("getTensorRefShape", HostRuntime::Distributed);
      });
}

::tt::target::DataType getTensorRefDataType(TensorRef tensorRef) {
  using RetType = ::tt::target::DataType;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType { return ttnn::getTensorRefDataType(tensorRef); },
      [&]() -> RetType { return ttmetal::getTensorRefDataType(tensorRef); },
      [&]() -> RetType {
        detail::fatalNotImplemented("getTensorRefDataType", HostRuntime::Distributed);
      });
}
```

### 7. `runtime/python/runtime/runtime.cpp`
Change the empty `TensorRef` binding at line 248:
```cpp
// Before:
nb::class_<tt::runtime::TensorRef>(m, "TensorRef");

// After:
nb::class_<tt::runtime::TensorRef>(m, "TensorRef")
    .def("get_shape",
         [](tt::runtime::TensorRef self) {
           return tt::runtime::getTensorRefShape(self);
         },
         "Logical shape from the flatbuffer — no tensor allocation.")
    .def("get_dtype",
         [](tt::runtime::TensorRef self) {
           return tt::runtime::getTensorRefDataType(self);
         },
         "Data type from the flatbuffer — no tensor allocation.");
```

### 8. `runtime/python/runtime/stubs_macos.cpp`
Add after `retrieveTensorFromPool` (~line 264):
```cpp
std::vector<uint32_t> getTensorRefShape(TensorRef) { __builtin_trap(); }
::tt::target::DataType getTensorRefDataType(TensorRef) { __builtin_trap(); }
```

---

## Verification

```bash
source env/activate
cmake --build build          # confirm it compiles
pytest test/python           # confirm existing tests pass
```

Smoke test: in `test/ttnn/python/n150/test_intermidate_tensor_manipulation.py`,
add assertions in the preop callback:
```python
for ref in ttrt.runtime.get_op_input_refs(op_ctx, program_ctx):
    assert len(ref.get_shape()) > 0
    assert ref.get_dtype() != ttrt.runtime.DataType.MAX
```
