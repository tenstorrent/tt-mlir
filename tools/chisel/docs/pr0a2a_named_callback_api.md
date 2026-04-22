# PR 0a-2a: Named Callback API

## Goal

Replace the current `Hooks::get(pre, post)` API with a named multi-client
callback system. Multiple callers (ttrt, builder, chisel) can register their
own callbacks under distinct names without overwriting each other. This PR
introduces only `preOp`/`postOp` in `CallbackSet` — program-level hooks are
added in PR 0a-2b.

## Problem

`Hooks::get()` overwrites the previous callback on each call. Only one client
can register callbacks at a time. Chisel needs to coexist with ttrt's own
debug callbacks.

## New API Design

### C++ API

```cpp
auto &hooks = debug::Hooks::get();

// Register named callbacks — multiple clients can coexist
hooks.setCallbacks("chisel", {
    .preOp = [](Binary b, CallbackContext pc, OpContext oc) { ... },
    .postOp = [](Binary b, CallbackContext pc, OpContext oc) { ... },
});

// ttrt registers its own callbacks under a different name
hooks.setCallbacks("ttrt", {
    .preOp = [](Binary b, CallbackContext pc, OpContext oc) { ... },
    .postOp = [](Binary b, CallbackContext pc, OpContext oc) { ... },
});

// Unregister by name
hooks.unregisterHooks("chisel");   // removes only chisel's callbacks
```

### Python API

```python
hooks = ttrt.runtime.DebugHooks.get()

hooks.set_callbacks(
    "chisel",
    pre_op=chisel_pre_op_fn,
    post_op=chisel_post_op_fn,
)

# List registered names
hooks.get_registered_names()  # e.g. ["chisel", "ttrt"]

# Unregister
ttrt.runtime.unregister_hooks("chisel")
```

## Implementation

### Step 1: `runtime/include/tt/runtime/debug.h` — Hooks struct

**1a. Add includes and type aliases:**

```cpp
#include <string>
#include <unordered_map>
#include <vector>

struct Hooks {
  using CallbackFn = std::function<void(Binary, CallbackContext, OpContext)>;

  // All callbacks for a single named client (e.g., "chisel", "ttrt")
  struct CallbackSet {
    std::optional<CallbackFn> preOp;
    std::optional<CallbackFn> postOp;
  };
```

**1b. `get()` becomes a no-arg singleton accessor:**

```cpp
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static Hooks &get();
#else
  constexpr static Hooks get() { return Hooks(); }
#endif
```

Note: returns non-const `Hooks &` so setters can be called on it.

**1c. Single setter method for all callbacks:**

```cpp
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  void setCallbacks(const std::string &name, CallbackSet &&callbackSet);
#else
  void setCallbacks(const std::string &, CallbackSet &&) {}
#endif
```

**1d. Getters — return const ref to the map:**

```cpp
  const std::unordered_map<std::string, CallbackSet> &
  getCallbacks() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return callbacks;
#else
    static const std::unordered_map<std::string, CallbackSet> empty;
    return empty;
#endif
  }
```

The executor iterates this single map and picks the relevant field
(`.preOp`, `.postOp`) per hook point.

**1e. Unregister and list methods:**

```cpp
  void unregisterHooks(const std::string &name) const;
  void unregisterHooks() const;  // clear all
  std::vector<std::string> getRegisteredNames() const;
```

**1f. Private storage — single map keyed by client name:**

```cpp
private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  Hooks() = default;

  mutable std::unordered_map<std::string, CallbackSet> callbacks;
#else
  constexpr Hooks() = default;
#endif
```

### Step 2: `runtime/lib/common/debug.cpp` — Implementation

```cpp
Hooks &Hooks::get() {
  static Hooks config;
  return config;
}

void Hooks::setCallbacks(const std::string &name, CallbackSet &&callbackSet) {
  callbacks[name] = std::move(callbackSet);
}

void Hooks::unregisterHooks(const std::string &name) const {
  callbacks.erase(name);
}

void Hooks::unregisterHooks() const {
  callbacks.clear();
}

std::vector<std::string> Hooks::getRegisteredNames() const {
  std::vector<std::string> names;
  names.reserve(callbacks.size());
  for (const auto &[name, _] : callbacks) {
    names.push_back(name);
  }
  return names;
}
```

### Step 3: `runtime/include/tt/runtime/detail/ttnn/program_executor.h`

Replace `runCallback` with `runOpCallbacks`:

```cpp
void runOpCallbacks(
    const ::tt::target::ttnn::Operation *opContext,
    ProgramContext *programContext,
    bool pre);
```

### Step 4: `runtime/lib/ttnn/program_executor.cpp`

Replace `runCallback` with `runOpCallbacks`:

```cpp
void ProgramExecutor::runOpCallbacks(
    const ::tt::target::ttnn::Operation *opContext,
    ProgramContext *programContext,
    bool pre) {
  const auto &allCallbacks = debug::Hooks::get().getCallbacks();
  if (allCallbacks.empty()) {
    return;
  }
  std::shared_ptr<void> programContextPtr =
      ::tt::runtime::utils::unsafeBorrowShared(programContext);
  std::shared_ptr<void> opContextPtr =
      ::tt::runtime::utils::unsafeBorrowShared(
          const_cast<::tt::target::ttnn::Operation *>(opContext));
  CallbackContext cc(programContextPtr, DeviceRuntime::TTNN);
  OpContext oc(opContextPtr, DeviceRuntime::TTNN);
  for (const auto &[name, cbs] : allCallbacks) {
    const auto &fn = pre ? cbs.preOp : cbs.postOp;
    if (fn) {
      (*fn)(executableHandle, cc, oc);
    }
  }
}
```

Update `execute()`:

```cpp
void ProgramExecutor::execute() {
  // ...
  for (const ::tt::target::ttnn::Operation *op : *program->operations()) {
    // ... logging, perf ...
    runOpCallbacks(op, context.get(), /*pre=*/true);
    runOperation(op);
    runOpCallbacks(op, context.get(), /*pre=*/false);
    dumpPerfCountersIfNeeded();
  }
  // ...
}
```

### Step 5: `runtime/python/runtime/runtime.cpp` — Python bindings

**5a. Replace `DebugHooks` binding with `set_callbacks` API:**

```cpp
nb::class_<tt::runtime::debug::Hooks>(m, "DebugHooks")
    .def_static("get", []() -> tt::runtime::debug::Hooks & {
      return tt::runtime::debug::Hooks::get();
    }, nb::rv_policy::reference)
    .def("set_callbacks",
         [](tt::runtime::debug::Hooks &self, const std::string &name,
            nb::object preOp, nb::object postOp) {
           tt::runtime::debug::Hooks::CallbackSet cbs;
           if (!preOp.is_none()) {
             auto fn = nb::cast<nb::callable>(preOp);
             cbs.preOp = [fn](Binary b, CallbackContext pc, OpContext oc) {
               fn(b, pc, oc);
             };
           }
           if (!postOp.is_none()) {
             auto fn = nb::cast<nb::callable>(postOp);
             cbs.postOp = [fn](Binary b, CallbackContext pc, OpContext oc) {
               fn(b, pc, oc);
             };
           }
           self.setCallbacks(name, std::move(cbs));
         },
         nb::arg("name"),
         nb::arg("pre_op") = nb::none(),
         nb::arg("post_op") = nb::none())
    .def("get_registered_names", ...)
    .def("__str__", [...]);
```

**5b. Update `unregister_hooks`:**

```cpp
m.def("unregister_hooks",
      [](nb::object name) {
        if (name.is_none()) {
          ::tt::runtime::debug::Hooks::get().unregisterHooks();
        } else {
          ::tt::runtime::debug::Hooks::get().unregisterHooks(
              nb::cast<std::string>(name));
        }
      },
      nb::arg("name") = nb::none());
```

### Step 6: Migrate existing callers

**ttrt** (`tools/ttrt/common/run.py:621-625`):

```python
# BEFORE:
callback_env = ttrt.runtime.DebugHooks.get(
    pre_op_get_callback_fn(pre_op_callback_runtime_config),
    post_op_get_callback_fn(post_op_callback_runtime_config),
)

# AFTER:
hooks = ttrt.runtime.DebugHooks.get()
hooks.set_callbacks(
    "ttrt",
    pre_op=pre_op_get_callback_fn(pre_op_callback_runtime_config),
    post_op=post_op_get_callback_fn(post_op_callback_runtime_config),
)
```

**builder** (`tools/builder/base/builder_runtime.py:722-726`):

```python
# BEFORE:
tt_runtime.runtime.DebugHooks.get(
    pre_op_get_callback_fn(callback_runtime_config),
    post_op_get_callback_fn(callback_runtime_config),
)

# AFTER:
hooks = tt_runtime.runtime.DebugHooks.get()
hooks.set_callbacks(
    "builder",
    pre_op=pre_op_get_callback_fn(callback_runtime_config),
    post_op=post_op_get_callback_fn(callback_runtime_config),
)
```

**test** (`runtime/test/ttnn/python/n150/test_intermidate_tensor_manipulation.py:104,122`):

```python
# BEFORE:
hooks = ttrt.runtime.DebugHooks.get(identity, postop)
...
ttrt.runtime.unregister_hooks()

# AFTER:
hooks = ttrt.runtime.DebugHooks.get()
hooks.set_callbacks("test", pre_op=identity, post_op=postop)
...
ttrt.runtime.unregister_hooks("test")
```

## Files to Modify

| File | Change |
|------|--------|
| `runtime/include/tt/runtime/debug.h` | Replace `get()` with named setters, store callbacks in map, return by const ref |
| `runtime/lib/common/debug.cpp` | Implement named setter methods, iterate callback maps |
| `runtime/include/tt/runtime/detail/ttnn/program_executor.h` | Replace `runCallback` with `runOpCallbacks` |
| `runtime/lib/ttnn/program_executor.cpp` | Implement `runOpCallbacks`, update `execute()` |
| `runtime/python/runtime/runtime.cpp` | Expose new setter API, update `unregister_hooks` |
| `tools/ttrt/common/run.py` | Migrate to `set_callbacks("ttrt", ...)` |
| `tools/builder/base/builder_runtime.py` | Migrate to `set_callbacks("builder", ...)` |
| `runtime/test/ttnn/python/n150/test_intermidate_tensor_manipulation.py` | Migrate to `set_callbacks("test", ...)` |

## Note on TTMetal

TTMetal runtime (`runtime/lib/ttmetal/`) does **not** use `DebugHooks` at all —
no `runCallback` or `debug::Hooks` references exist there. This PR only affects
the TTNN runtime path.

## Test Plan

1. **Build**: `cmake --build build` — compile succeeds
2. **Existing tests pass**: `cmake --build build --target check-ttmlir`
3. **Existing callback test**:
   `runtime/test/ttnn/python/n150/test_intermidate_tensor_manipulation.py`
   migrated to new API and still passes
4. **New test**: Register callbacks from two different names ("a" and "b"),
   verify both fire. Unregister "a", verify only "b" fires.
5. **Pre-commit**: `pre-commit run --all-files`

## Dependencies

- **PR 0a-1** — GIL-safety fix (const ref returns + move semantics)
