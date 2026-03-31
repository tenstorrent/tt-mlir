# PR 0a: Program-Level Callbacks + DebugHooks Refactor

## Context

ChiselV2 requires four callback hook points (pre-program, pre-op, post-op,
post-program) but the current `DebugHooks` only supports two (pre-op,
post-op). Additionally, there are two design problems with the current API:

1. **GIL-safety bug**: Callbacks are returned/passed by value, copying
   `nb::callable` objects and manipulating Python refcounts without the GIL.
   This causes segfaults when called from non-GIL threads (e.g., tt-xla).

2. **Single-callback limitation**: `Hooks::get()` overwrites the previous
   callback on each call. This means only one client (ttrt, builder, or chisel)
   can register callbacks at a time. Chisel needs to coexist with ttrt's own
   debug callbacks.

This PR:
1. Replaces `Hooks::get(pre, post)` with named setter methods that support
   **multiple callbacks per hook point**
2. Adds pre-program and post-program hook points
3. Fixes the GIL-safety bug via const-ref returns and move semantics

## Files to Modify

| File | Change |
|------|--------|
| `runtime/include/tt/runtime/debug.h` | Replace `get()` with named setters, add program callbacks, store callbacks in maps, return by const ref |
| `runtime/lib/common/debug.cpp` | Implement named setter methods, iterate callback maps in getters |
| `runtime/include/tt/runtime/detail/ttnn/program_executor.h` | Add `runProgramCallbacks` method, update `runCallbacks` signature |
| `runtime/lib/ttnn/program_executor.cpp` | Implement `runProgramCallbacks`, call pre/post-program hooks in `execute()` |
| `runtime/python/runtime/runtime.cpp` | Expose new setter API + `Binary.id` property, update `unregister_hooks` |

Existing callers that need migration:
| Caller | File |
|--------|------|
| ttrt | `tools/ttrt/common/run.py:622` |
| builder | `tools/builder/base/builder_runtime.py:723` |
| test | `runtime/test/ttnn/python/n150/test_intermidate_tensor_manipulation.py:104` |

## Problem Analysis (GIL-Safety)

### Copy chain per op execution

In `ProgramExecutor::execute()` (`runtime/lib/ttnn/program_executor.cpp:169-172`):

```cpp
runCallback(debug::Hooks::get().getPreOperatorCallback(), ...);
runOperation(op);
runCallback(debug::Hooks::get().getPostOperatorCallback(), ...);
```

1. `getPreOperatorCallback()` returns `std::optional<CallbackFn>` **by value**
   — copies the `std::function` (which copies the captured `nb::callable`)
2. `runCallback()` takes `std::optional<CallbackFn>` **by value**
   — copies it again

Each copy of `nb::callable` increments/decrements Python refcounts without GIL.

### Where the `nb::callable` enters

In `runtime/python/runtime/runtime.cpp:658-670`:

```cpp
.def_static("get",
    [](nb::callable pre_op_func, nb::callable post_op_func) {
      return tt::runtime::debug::Hooks::get(
          [pre_op_func](Binary b, CallbackContext pc, OpContext oc) {
            pre_op_func(b, pc, oc);     // nb::callable captured by value
          },
          [post_op_func](Binary b, CallbackContext pc, OpContext oc) {
            post_op_func(b, pc, oc);    // nb::callable captured by value
          });
    })
```

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

## New API Design

### C++ API

```cpp
auto &hooks = debug::Hooks::get();

// Register named callbacks — multiple clients can coexist
hooks.setPreOp("chisel", [](Binary b, CallbackContext pc, OpContext oc) { ... });
hooks.setPostOp("chisel", [](Binary b, CallbackContext pc, OpContext oc) { ... });
hooks.setPreProgram("chisel", [](Binary b, CallbackContext pc) { ... });
hooks.setPostProgram("chisel", [](Binary b, CallbackContext pc) { ... });

// ttrt registers its own callbacks under a different name
hooks.setPreOp("ttrt", [](Binary b, CallbackContext pc, OpContext oc) { ... });
hooks.setPostOp("ttrt", [](Binary b, CallbackContext pc, OpContext oc) { ... });

// Unregister by name
hooks.unregisterHooks("chisel");   // removes only chisel's callbacks
hooks.unregisterHooks();           // removes all callbacks (no argument)
```

### Python API

```python
hooks = ttrt.runtime.DebugHooks.get()

# Named setters — multiple clients coexist
hooks.set_pre_op("chisel", chisel_pre_op_fn)
hooks.set_post_op("chisel", chisel_post_op_fn)
hooks.set_pre_program("chisel", chisel_pre_program_fn)
hooks.set_post_program("chisel", chisel_post_program_fn)

# List registered names
hooks.get_registered_names()  # e.g. ["chisel", "ttrt"]

# Unregister
ttrt.runtime.unregister_hooks("chisel")  # by name
ttrt.runtime.unregister_hooks()          # all
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
  using ProgramCallbackFn = std::function<void(Binary, CallbackContext)>;

  // All callbacks for a single named client (e.g., "chisel", "ttrt")
  struct CallbackSet {
    std::optional<CallbackFn> preOp;
    std::optional<CallbackFn> postOp;
    std::optional<ProgramCallbackFn> preProgram;
    std::optional<ProgramCallbackFn> postProgram;
  };
```

`ProgramCallbackFn` has no `OpContext` — it fires at program boundaries.
`CallbackSet` groups all hooks for one named client — unregister and list
operate on a single map instead of four.

**1b. `get()` becomes a no-arg singleton accessor:**

```cpp
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static Hooks &get();
#else
  constexpr static Hooks get() { return Hooks(); }
#endif
```

Note: returns non-const `Hooks &` so setters can be called on it.

**1c. Named setter methods:**

```cpp
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  void setPreOp(const std::string &name, CallbackFn &&callback);
  void setPostOp(const std::string &name, CallbackFn &&callback);
  void setPreProgram(const std::string &name, ProgramCallbackFn &&callback);
  void setPostProgram(const std::string &name, ProgramCallbackFn &&callback);
#else
  void setPreOp(const std::string &, CallbackFn &&) {}
  void setPostOp(const std::string &, CallbackFn &&) {}
  void setPreProgram(const std::string &, ProgramCallbackFn &&) {}
  void setPostProgram(const std::string &, ProgramCallbackFn &&) {}
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
(`.preOp`, `.postOp`, etc.) per hook point.

**1e. Unregister and list methods:**

```cpp
  void unregisterHooks(const std::string &name) const;  // by name
  void unregisterHooks() const;                          // all
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

void Hooks::setPreOp(const std::string &name, CallbackFn &&callback) {
  callbacks[name].preOp = std::move(callback);
}

void Hooks::setPostOp(const std::string &name, CallbackFn &&callback) {
  callbacks[name].postOp = std::move(callback);
}

void Hooks::setPreProgram(const std::string &name,
                          ProgramCallbackFn &&callback) {
  callbacks[name].preProgram = std::move(callback);
}

void Hooks::setPostProgram(const std::string &name,
                           ProgramCallbackFn &&callback) {
  callbacks[name].postProgram = std::move(callback);
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

**3a. Replace `runCallback` with `runOpCallbacks`:**

```cpp
void runOpCallbacks(
    const ::tt::target::ttnn::Operation *opContext,
    ProgramContext *programContext);
```

**3b. Add `runProgramCallbacks`:**

```cpp
void runProgramCallbacks(ProgramContext *programContext);
```

Both methods get the callback map from `debug::Hooks::get().getCallbacks()`
internally and iterate the relevant `CallbackSet` field.

### Step 4: `runtime/lib/ttnn/program_executor.cpp`

**4a. Replace `runCallback` with `runOpCallbacks`:**

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

**4b. Add `runProgramCallbacks`:**

```cpp
void ProgramExecutor::runProgramCallbacks(
    ProgramContext *programContext, bool pre) {
  const auto &allCallbacks = debug::Hooks::get().getCallbacks();
  if (allCallbacks.empty()) {
    return;
  }
  std::shared_ptr<void> programContextPtr =
      ::tt::runtime::utils::unsafeBorrowShared(programContext);
  CallbackContext cc(programContextPtr, DeviceRuntime::TTNN);
  for (const auto &[name, cbs] : allCallbacks) {
    const auto &fn = pre ? cbs.preProgram : cbs.postProgram;
    if (fn) {
      (*fn)(executableHandle, cc);
    }
  }
}
```

**4c. Update `execute()`:**

```cpp
void ProgramExecutor::execute() {
  LOG_DEBUG(LogType::LogRuntimeTTNN,
            "Starting execution of program: ", program->name()->c_str());

  runProgramCallbacks(context.get(), /*pre=*/true);

  for (const ::tt::target::ttnn::Operation *op : *program->operations()) {
    LOG_DEBUG(LogType::LogRuntimeTTNN,
              "Executing operation: ", op->debug_info()->c_str());
    perf::Env::get().tracyLogOpLocation(std::string(op->loc_info()->c_str()));
    perf::Env::get().tracyLogConstEvalProgram(constEvalProgram);
    perf::Env::get().tracyLogProgramMetadata(
        perf::Env::get().tracyProgramMetadata);
    runOpCallbacks(op, context.get(), /*pre=*/true);
    runOperation(op);
    runOpCallbacks(op, context.get(), /*pre=*/false);
    dumpPerfCountersIfNeeded();
  }

  runProgramCallbacks(context.get(), /*pre=*/false);

  LOG_DEBUG(LogType::LogRuntimeTTNN,
            "Finished execution of program: ", program->name()->c_str());
}
```

### Step 5: `runtime/python/runtime/runtime.cpp` — Python bindings

**5a. Expose `Binary.id` property** (in the Binary class binding):

```cpp
.def_prop_ro("id", &tt::runtime::Binary::id)
```

**5b. Replace `DebugHooks` binding with named setter API:**

```cpp
nb::class_<tt::runtime::debug::Hooks>(m, "DebugHooks")
    .def_static("get", &tt::runtime::debug::Hooks::get,
                nb::rv_policy::reference)
    .def("set_pre_op",
         [](tt::runtime::debug::Hooks &self, const std::string &name,
            nb::callable fn) {
           self.setPreOp(name,
               [fn](tt::runtime::Binary b, tt::runtime::CallbackContext pc,
                    tt::runtime::OpContext oc) { fn(b, pc, oc); });
         },
         nb::arg("name"), nb::arg("callback"))
    .def("set_post_op",
         [](tt::runtime::debug::Hooks &self, const std::string &name,
            nb::callable fn) {
           self.setPostOp(name,
               [fn](tt::runtime::Binary b, tt::runtime::CallbackContext pc,
                    tt::runtime::OpContext oc) { fn(b, pc, oc); });
         },
         nb::arg("name"), nb::arg("callback"))
    .def("set_pre_program",
         [](tt::runtime::debug::Hooks &self, const std::string &name,
            nb::callable fn) {
           self.setPreProgram(name,
               [fn](tt::runtime::Binary b,
                    tt::runtime::CallbackContext pc) { fn(b, pc); });
         },
         nb::arg("name"), nb::arg("callback"))
    .def("set_post_program",
         [](tt::runtime::debug::Hooks &self, const std::string &name,
            nb::callable fn) {
           self.setPostProgram(name,
               [fn](tt::runtime::Binary b,
                    tt::runtime::CallbackContext pc) { fn(b, pc); });
         },
         nb::arg("name"), nb::arg("callback"))
    .def("get_registered_names",
         &tt::runtime::debug::Hooks::getRegisteredNames)
    .def("__str__", [...]);
```

**5c. Update `unregister_hooks`:**

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
hooks.set_pre_op("ttrt", pre_op_get_callback_fn(pre_op_callback_runtime_config))
hooks.set_post_op("ttrt", post_op_get_callback_fn(post_op_callback_runtime_config))
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
hooks.set_pre_op("builder", pre_op_get_callback_fn(callback_runtime_config))
hooks.set_post_op("builder", post_op_get_callback_fn(callback_runtime_config))
```

**test** (`runtime/test/ttnn/python/n150/test_intermidate_tensor_manipulation.py:104,122`):

```python
# BEFORE:
hooks = ttrt.runtime.DebugHooks.get(identity, postop)
...
ttrt.runtime.unregister_hooks()

# AFTER:
hooks = ttrt.runtime.DebugHooks.get()
hooks.set_pre_op("test", identity)
hooks.set_post_op("test", postop)
...
ttrt.runtime.unregister_hooks("test")
```

## Execution Flow

```
for each registered pre-program callback (by name):
  pre-program callback  (Binary, CallbackContext)

for each op:
  for each registered pre-op callback (by name):
    pre-op callback   (Binary, CallbackContext, OpContext)
  HW executes op
  for each registered post-op callback (by name):
    post-op callback  (Binary, CallbackContext, OpContext)

for each registered post-program callback (by name):
  post-program callback (Binary, CallbackContext)
```

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
   verify both fire. Unregister "a", verify only "b" fires. Test all 4 hook
   points fire in correct order: pre-program → (pre-op → op → post-op)* →
   post-program
5. **Pre-commit**: `pre-commit run --all-files`

## Dependencies

None — standalone runtime change. **Must land before Chisel PR 3**
(orchestration), which registers Python callbacks via `DebugHooks`.
PRs 1-2 don't register callbacks and are unaffected.
