# PR 0a-2b: Program-Level Hooks

## Goal

Add `preProgram` and `postProgram` callback hook points to `DebugHooks`.
These fire at program boundaries — before the first op and after the last op
in `ProgramExecutor::execute()`. The `CallbackSet` struct (introduced in
PR 0a-2a) is extended with two new fields.

## New Callback Signature

Program-level callbacks have no `OpContext` — they fire at program boundaries,
not per-operation:

```cpp
using ProgramCallbackFn = std::function<void(Binary, CallbackContext)>;
```

## Implementation

### Step 1: `runtime/include/tt/runtime/debug.h`

**1a. Add type alias:**

```cpp
using ProgramCallbackFn = std::function<void(Binary, CallbackContext)>;
```

**1b. Extend `CallbackSet`:**

```cpp
struct CallbackSet {
  std::optional<CallbackFn> preOp;
  std::optional<CallbackFn> postOp;
  std::optional<ProgramCallbackFn> preProgram;   // NEW
  std::optional<ProgramCallbackFn> postProgram;  // NEW
};
```

### Step 2: `runtime/include/tt/runtime/detail/ttnn/program_executor.h`

Add `runProgramCallbacks` method:

```cpp
void runProgramCallbacks(ProgramContext *programContext, bool pre);
```

### Step 3: `runtime/lib/ttnn/program_executor.cpp`

**3a. Implement `runProgramCallbacks`:**

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

**3b. Update `execute()`:**

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

### Step 4: `runtime/python/runtime/runtime.cpp`

Add `pre_program` and `post_program` kwargs to `set_callbacks`:

```cpp
.def("set_callbacks",
     [](tt::runtime::debug::Hooks &self, const std::string &name,
        nb::object preOp, nb::object postOp,
        nb::object preProgram, nb::object postProgram) {
       tt::runtime::debug::Hooks::CallbackSet cbs;
       // ... wrap preOp/postOp as before ...
       if (!preProgram.is_none()) {
         auto fn = nb::cast<nb::callable>(preProgram);
         cbs.preProgram = [fn](Binary b, CallbackContext pc) {
           fn(b, pc);
         };
       }
       if (!postProgram.is_none()) {
         auto fn = nb::cast<nb::callable>(postProgram);
         cbs.postProgram = [fn](Binary b, CallbackContext pc) {
           fn(b, pc);
         };
       }
       self.setCallbacks(name, std::move(cbs));
     },
     nb::arg("name"),
     nb::arg("pre_op") = nb::none(),
     nb::arg("post_op") = nb::none(),
     nb::arg("pre_program") = nb::none(),
     nb::arg("post_program") = nb::none())
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

## Files to Modify

| File | Change |
|------|--------|
| `runtime/include/tt/runtime/debug.h` | Add `ProgramCallbackFn` type, extend `CallbackSet` |
| `runtime/include/tt/runtime/detail/ttnn/program_executor.h` | Add `runProgramCallbacks` declaration |
| `runtime/lib/ttnn/program_executor.cpp` | Implement `runProgramCallbacks`, update `execute()` |
| `runtime/python/runtime/runtime.cpp` | Add `pre_program`/`post_program` kwargs to `set_callbacks` |

## Test Plan

1. **Build**: `cmake --build build` — compile succeeds
2. **Existing tests pass**: `cmake --build build --target check-ttmlir`
3. **New test**: Register all 4 hook points, run a program, verify callbacks
   fire in correct order: pre-program -> (pre-op -> op -> post-op)* -> post-program
4. **Pre-commit**: `pre-commit run --all-files`

## Dependencies

- **PR 0a-2a** — Named callback API (`setCallbacks`, `CallbackSet`, `runOpCallbacks`)
