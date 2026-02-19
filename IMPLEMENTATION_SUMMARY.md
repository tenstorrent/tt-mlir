# Nanobind Dual-Instance Segfault Fix - Final Implementation

## Summary

Successfully fixed the nanobind dual-instance segfault by:
1. Removing nanobind from static library (TTRuntimeTTNN.a)
2. Using Python C API to call into Python extension
3. All nanobind operations happen in single properly-initialized instance
4. Proper GIL management and reference counting with shared_ptr

---

## Key Changes

### 1. Python Extension (`runtime/python/runtime/runtime.cpp`)

**Added `register_debug_callback()` function:**
- Takes **no arguments** (simplified API)
- Imports `debug_callback` module internally
- Wraps callbacks in `shared_ptr<nb::object>` with custom deleter
- Custom deleter acquires GIL before destroying nb::object
- Lambdas capture shared_ptr and acquire GIL before calling Python

**Why shared_ptr with custom deleter is critical:**
```cpp
auto pre_op_shared = std::shared_ptr<nb::object>(
    new nb::object(pre_op_func),
    [](nb::object* p) {
        nb::gil_scoped_acquire gil;  // ← CRITICAL: GIL before Python cleanup
        delete p;
    }
);
```

Without this, when lambdas are destroyed from C++ context (no GIL), nanobind tries to decrement Python refcounts → segfault.

### 2. Static Library (`runtime/lib/ttnn/auto_debug_hooks.cpp`)

**Pure Python C API implementation:**
```cpp
// Import module
runtime_module = PyImport_ImportModule("ttrt.runtime");

// Get function
register_func = PyObject_GetAttrString(runtime_module, "register_debug_callback");

// Call with no arguments
result = PyObject_CallObject(register_func, nullptr);

// Clean up
Py_XDECREF(result);
Py_XDECREF(register_func);
Py_XDECREF(runtime_module);
```

### 3. Public Header Fix (`runtime/include/tt/runtime/debug.h`)

**Problem:** `unregisterHooks()` needed Python.h but header is included everywhere

**Solution:** Moved implementation to debug.cpp
- Header: Declaration only
- debug.cpp: Implementation with Python.h
- CMake: Added Python::Python to TTRuntimeDebug

---

## Architecture

### Before (Broken)
```
TTRuntimeTTNN.a → nanobind Instance #1 (NULL) → SEGFAULT ✗
_ttmlir_runtime.so → nanobind Instance #2 (initialized) → Types here ✗
```

### After (Fixed)
```
TTRuntimeTTNN.a
  └─ Python C API only (PyImport, PyObject_Call)
       ↓
_ttmlir_runtime.so
  └─ Single nanobind instance
       └─ register_debug_callback()
            └─ shared_ptr with GIL-acquiring deleter
                 └─ Lambdas with GIL acquisition
```

---

## Why Each Piece Matters

### 1. Why no arguments in register_debug_callback()?
- Simplifies C API code (fewer PyObject* to manage)
- All module imports happen in nanobind context
- Less error-prone reference counting

### 2. Why shared_ptr with custom deleter?
- Lambdas stored in C++ `debug::Hooks` singleton
- Destroyed from C++ context without GIL
- Direct capture causes segfault when nb::object destructor runs
- Custom deleter ensures GIL acquired before Python cleanup

**Tested:** Direct capture segfaults. shared_ptr with custom deleter works.

### 3. Why move unregisterHooks() to .cpp?
- Uses `PyGILState_Ensure/Release` → needs Python.h
- Can't include Python.h in public header (used everywhere)
- Move to .cpp where Python.h is OK

### 4. Why GIL in unregisterHooks()?
- Clearing callbacks destroys lambdas
- Lambdas contain shared_ptr to nb::object
- Last shared_ptr destruction triggers custom deleter
- Custom deleter needs GIL available in scope

---

## Files Modified

1. ✅ `runtime/python/runtime/runtime.cpp`
   - Added `register_debug_callback()` with shared_ptr wrapper
   - Added `<memory>` include

2. ✅ `runtime/lib/ttnn/auto_debug_hooks.cpp`
   - Replaced all nanobind with Python C API
   - Calls `register_debug_callback()` with nullptr args

3. ✅ `runtime/lib/ttnn/CMakeLists.txt`
   - Changed from nanobind to Python::Python

4. ✅ `runtime/include/tt/runtime/debug.h`
   - Changed `unregisterHooks()` to declaration only
   - Removed Python.h include

5. ✅ `runtime/lib/common/debug.cpp`
   - Added `unregisterHooks()` implementation
   - Added Python.h include

6. ✅ `runtime/lib/common/CMakeLists.txt`
   - Added Python::Python to TTRuntimeDebug when TT_RUNTIME_DEBUG enabled

---

## Build Verification

```bash
cd /localdev/ndrakulic/tt-xla
source venv/activate
cmake --build build
```

Expected: Clean build, no nanobind in TTRuntimeTTNN.a

---

## Runtime Testing

```bash
export TT_RUNTIME_DEBUG=1
export TT_FORCE_DEBUG_CALLBACKS=1
python debug_callback.py
```

Expected output:
```
[AutoHooks] Attempting to register Python debug callbacks...
[AutoHooks] Importing ttrt.runtime module...
[AutoHooks] Found register_debug_callback function
[AutoHooks] Calling register_debug_callback()...
[register_debug_callback] Importing debug_callback module...
[register_debug_callback] Callbacks loaded, registering...
[register_debug_callback] ✓ Callbacks registered successfully!
[AutoHooks] ✓ Python debug callbacks registered successfully!
```

---

## Lessons Learned

1. **Multiple nanobind instances are deadly** - Each has separate type registry
2. **Direct nb::object capture in lambdas is unsafe** when destroyed from C++ context
3. **shared_ptr with custom GIL-acquiring deleter** is necessary pattern
4. **Python.h belongs in .cpp files only**, never in public headers
5. **Simple capture works for DebugHooks.get()** but not for auto_debug_hooks because different destruction contexts

---

## Date Completed
2026-02-15
