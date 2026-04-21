# `walk_binary`: Fake Runtime for Flatbuffer Op Iteration

## Motivation

Testing shape/dtype inspection via `TensorRef.get_shape()` / `get_dtype()` (see
`tensorref_shape_dtype_nanobind.md`) requires a valid `OpContext` per op, but
the real runtime requires a device and allocates tensors. A minimal
`walk_binary` function can iterate a compiled `.ttnn` flatbuffer, fire a
callback per op with a proper `OpContext`, and skip all op execution — no
hardware, no tensors.

There is no meaningful distinction between pre-op and post-op here: since no op
is executed, both would see identical flatbuffer state. A single callback per op
is sufficient.

---

## Key Insight: What the Callback Needs

`ProgramExecutor::runOpCallback` (lines 162-175 of
`runtime/lib/ttnn/program_executor.cpp`) shows how the real runtime builds
contexts:

```cpp
std::shared_ptr<void> opContextPtr =
    unsafeBorrowShared(const_cast<::tt::target::ttnn::Operation *>(op));

(*callback)(executableHandle,
            CallbackContext(programContextPtr, DeviceRuntime::TTNN),
            OpContext(opContextPtr, DeviceRuntime::TTNN));
```

- `OpContext` = thin wrapper around `Operation*` flatbuffer pointer.
- `CallbackContext` = thin wrapper around `ProgramContext*`.

`getOpInputRefs` and `getOpOutputRefs` (TTNN impl) only call
`opContextHandle.as<::tt::target::ttnn::Operation>()` — they never touch
`programContextHandle`. So `CallbackContext` can wrap a minimal stub struct;
it just must be non-null (`RuntimeCheckedObjectImpl::as<>()` asserts
`handle != nullptr`).

---

## Implementation

### New function: `walkBinary`

```cpp
using OpWalkFn = std::function<void(Binary, CallbackContext, OpContext)>;

void walkBinary(Binary executableHandle, uint32_t programIndex,
                const OpWalkFn &cb);
```

Core logic mirrors `ProgramExecutor::execute()` with `runOperation` removed and
pre/post collapsed into one call:

```cpp
struct WalkContext {};  // Stub — satisfies non-null assertion in as<>()

void walkBinary(Binary executableHandle, uint32_t programIndex,
                const OpWalkFn &cb) {
  const auto *fbb = ::tt::target::ttnn::GetSizePrefixedTTNNBinary(
      executableHandle.handle.get());
  const auto *program = fbb->programs()->Get(programIndex);

  WalkContext stub;
  auto stubHandle = ::tt::runtime::utils::unsafeBorrowShared(&stub);

  for (const ::tt::target::ttnn::Operation *op : *program->operations()) {
    auto opHandle = ::tt::runtime::utils::unsafeBorrowShared(
        const_cast<::tt::target::ttnn::Operation *>(op));

    cb(executableHandle,
       CallbackContext(stubHandle, DeviceRuntime::TTNN),
       OpContext(opHandle, DeviceRuntime::TTNN));
  }
}
```

### Files to touch

| File | Change |
|------|--------|
| `runtime/include/tt/runtime/runtime.h` | Declare `walkBinary` |
| `runtime/include/tt/runtime/detail/ttnn/ttnn.h` | Declare TTNN impl |
| `runtime/lib/ttnn/runtime.cpp` | Implement (`ttmlir/Target/TTNN/program_generated.h` already included) |
| `runtime/lib/runtime.cpp` | Dispatch (TTNN-only; check binary type or bypass `DISPATCH_TO_CURRENT_RUNTIME`) |
| `runtime/python/runtime/runtime.cpp` | Python binding |
| `runtime/python/runtime/stubs_macos.cpp` | `__builtin_trap()` stub |

### Python binding

```cpp
m.def(
    "walk_binary",
    [](tt::runtime::Binary binary, uint32_t program_index, nb::callable cb) {
      tt::runtime::walkBinary(
          binary, program_index,
          [cb](tt::runtime::Binary bin,
               tt::runtime::CallbackContext prog_ctx,
               tt::runtime::OpContext op_ctx) {
            nb::gil_scoped_acquire gil;
            cb(bin, prog_ctx, op_ctx);
          });
    },
    nb::arg("binary"), nb::arg("program_index"), nb::arg("cb"));
```

---

## Result: Python Usage

```python
import ttrt.runtime as rt

binary = rt.Binary.load_from_path("model.ttnn")

def inspect(binary, prog_ctx, op_ctx):
    for ref in rt.get_op_input_refs(op_ctx, prog_ctx):
        print("in: ", ref.get_shape(), ref.get_dtype())
    for ref in rt.get_op_output_refs(op_ctx, prog_ctx):
        print("out:", ref.get_shape(), ref.get_dtype())

rt.walk_binary(binary, program_index=0, cb=inspect)
```

No device. No tensors. No hardware.

---

## Constraints

- `CallbackContext` is a stub — any callback that calls `retrieve_tensor_from_pool`
  will crash (it needs a real `ProgramContext` with a tensor pool). Only
  `get_op_input_refs`, `get_op_output_refs`, and `TensorRef.get_shape()` /
  `get_dtype()` are safe.
- TTMetal binary walking is not implemented (stub / `LOG_FATAL`).

---

## Verification

```bash
source env/activate
cmake --build build

python - <<'EOF'
import ttrt.runtime as rt

binary = rt.Binary.load_from_path("ttrt-artifacts/some_model.ttnn")

def inspect(binary, prog_ctx, op_ctx):
    for ref in rt.get_op_input_refs(op_ctx, prog_ctx):
        print("in: ", ref.get_shape(), ref.get_dtype())
    for ref in rt.get_op_output_refs(op_ctx, prog_ctx):
        print("out:", ref.get_shape(), ref.get_dtype())

rt.walk_binary(binary, 0, inspect)
EOF
```
