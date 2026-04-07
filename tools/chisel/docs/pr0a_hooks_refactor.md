# PR 0a-1: Refactor DebugHooks to Avoid Callback Copies

## Goal

Fix the `debug::Hooks` struct so that callbacks are never copied. Currently,
`getPreOperatorCallback()` / `getPostOperatorCallback()` return
`std::optional<CallbackFn>` **by value**, which copies the `std::function` on
every call. When the `std::function` wraps a `nb::callable` (nanobind Python
callable), the copy manipulates the Python object's reference count. If the
calling thread doesn't hold the GIL (e.g., tt-xla), this causes a segfault.

## Problem Analysis

### Copy chain per op execution

In `ProgramExecutor::execute()` (`runtime/lib/ttnn/program_executor.cpp:168-172`):

```cpp
runCallback(debug::Hooks::get().getPreOperatorCallback(), ...);
runOperation(op);
runCallback(debug::Hooks::get().getPostOperatorCallback(), ...);
```

1. `getPreOperatorCallback()` returns `std::optional<CallbackFn>` **by value**
   → copies the `std::function` (which copies the captured `nb::callable`)
2. `runCallback()` takes `std::optional<CallbackFn>` **by value**
   → copies it again

Each copy of `nb::callable` increments/decrements Python refcounts without GIL.

### Where the `nb::callable` enters

In `runtime/python/runtime/runtime.cpp:648-671`:

```cpp
.def_static("get",
    [](nb::callable pre_op_func, nb::callable post_op_func) {
      return tt::runtime::debug::Hooks::get(
          [pre_op_func](Binary b, CallbackContext pc, OpContext oc) {
            pre_op_func(b, pc, oc);     // <-- nb::callable captured by value
          },
          [post_op_func](Binary b, CallbackContext pc, OpContext oc) {
            post_op_func(b, pc, oc);    // <-- nb::callable captured by value
          });
    })
```

The lambda captures `nb::callable` by value. Every copy of the lambda copies
the `nb::callable`. Every copy of the `std::function` wrapping the lambda
copies the lambda.

### Also in Hooks::get() itself

In `runtime/lib/common/debug.cpp:24-35`:

```cpp
const Hooks &
Hooks::get(std::optional<debug::Hooks::CallbackFn> preOperatorCallback, ...) {
  static Hooks config(preOperatorCallback, postOperatorCallback);
  //                   ^--- copies into Hooks constructor
  if (preOperatorCallback.has_value()) {
    config.preOperatorCallback = preOperatorCallback;  // copies again
  }
  ...
}
```

The parameters are taken by value, then assigned by copy.

## Proposed Fix

### 1. Return callbacks by const reference, not by value

```cpp
// BEFORE:
std::optional<CallbackFn> getPreOperatorCallback() const {
  return preOperatorCallback;  // copies
}

// AFTER:
const std::optional<CallbackFn> &getPreOperatorCallback() const {
  return preOperatorCallback;
}

const std::optional<CallbackFn> &getPostOperatorCallback() const {
  return postOperatorCallback;
}
```

### 2. Change `runCallback` to take by const reference

```cpp
// BEFORE:
void ProgramExecutor::runCallback(
    std::optional<debug::Hooks::CallbackFn> callback, ...);

// AFTER:
void ProgramExecutor::runCallback(
    const std::optional<debug::Hooks::CallbackFn> &callback, ...);
```

### 3. Accept callbacks by rvalue reference in `Hooks::get()`

```cpp
// BEFORE:
static const Hooks &
get(std::optional<CallbackFn> preOperatorCallback = std::nullopt,
    std::optional<CallbackFn> postOperatorCallback = std::nullopt);

// AFTER:
static const Hooks &
get(std::optional<CallbackFn> &&preOperatorCallback = std::nullopt,
    std::optional<CallbackFn> &&postOperatorCallback = std::nullopt);
```

And in the implementation:

```cpp
const Hooks &
Hooks::get(std::optional<CallbackFn> &&preOperatorCallback,
           std::optional<CallbackFn> &&postOperatorCallback) {
  static Hooks config(std::move(preOperatorCallback),
                      std::move(postOperatorCallback));
  if (preOperatorCallback.has_value()) {
    config.preOperatorCallback = std::move(preOperatorCallback);
  }
  if (postOperatorCallback.has_value()) {
    config.postOperatorCallback = std::move(postOperatorCallback);
  }
  return config;
}
```

And the Hooks constructor:

```cpp
Hooks(std::optional<CallbackFn> &&preOperatorCallback,
      std::optional<CallbackFn> &&postOperatorCallback)
    : preOperatorCallback(std::move(preOperatorCallback)),
      postOperatorCallback(std::move(postOperatorCallback)) {}
```

### 4. Non-debug path must match return type

The `constexpr static Hooks get()` return in the non-debug path returns `Hooks`
by value, but the debug path returns `const Hooks &`. The getter methods return
`std::optional<CallbackFn>` by value in non-debug (always `std::nullopt`).
Update to return `const std::optional<CallbackFn> &`:

For the non-debug path, we need a static empty optional:

```cpp
#else
  const std::optional<CallbackFn> &getPreOperatorCallback() const {
    static const std::optional<CallbackFn> empty = std::nullopt;
    return empty;
  }
#endif
```

## Files to Modify

| File | Change |
|------|--------|
| `runtime/include/tt/runtime/debug.h` | Return callbacks by const ref, accept by rvalue ref |
| `runtime/lib/common/debug.cpp` | Move semantics in `Hooks::get()` |
| `runtime/lib/ttnn/program_executor.cpp` | `runCallback` takes const ref |
| `runtime/lib/ttmetal/program_executor.cpp` | Same change if it has `runCallback` |

## Test Plan

- **Existing tests pass** — no behavioral change, only avoids copies
- **GIL-safety test (if possible):** Register Python callbacks, execute from a
  non-GIL thread, verify no segfault. This may need a tt-xla test harness.
- **Verify callback invocation:** Ensure pre/post callbacks still fire correctly
  with the builder's `verify_intermediates=True` flow

## Dependencies

None — standalone runtime fix. **Must land before Chisel PR 1**
(Single Op Isolation), which registers preOp/postOp Python callbacks via
`DebugHooks`. Can be developed in parallel with Chisel documentation work.
