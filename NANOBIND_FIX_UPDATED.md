# Nanobind Fix - Updated Implementation

## Changes from Original Plan

The implementation was **simplified** to reduce complexity in the static library code:

### Key Change: Zero-Argument Registration Function

Instead of passing callbacks as arguments from C++ to Python, the `register_debug_callback()` function now:
1. Takes **no arguments**
2. Imports `debug_callback` module itself
3. Extracts `pre_op_callback` and `post_op_callback` from the module
4. Registers them with proper GIL management

---

## Updated Architecture

### Static Library (C++)
**File:** `runtime/lib/ttnn/auto_debug_hooks.cpp`

```cpp
// Simplified - only 3 Python objects to manage
PyObject* runtime_module = nullptr;
PyObject* register_func = nullptr;
PyObject* result = nullptr;

// Import ttrt.runtime
runtime_module = PyImport_ImportModule("ttrt.runtime");

// Get register_debug_callback function
register_func = PyObject_GetAttrString(runtime_module, "register_debug_callback");

// Call with NO arguments
result = PyObject_CallObject(register_func, nullptr);

// Cleanup
Py_XDECREF(result);
Py_XDECREF(register_func);
Py_XDECREF(runtime_module);
```

**Benefits:**
- ✅ Simpler C API code (fewer PyObject* variables)
- ✅ Less error-prone (fewer reference counting paths)
- ✅ No need to pass Python objects between contexts
- ✅ All nanobind operations stay in Python extension

### Python Extension (C++ with nanobind)
**File:** `runtime/python/runtime/runtime.cpp`

```cpp
m.def("register_debug_callback", []() {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    nb::gil_scoped_acquire gil;

    // Import debug_callback module
    nb::module_ debug_module = nb::module_::import_("debug_callback");

    // Get callbacks from module
    nb::object pre_op_func = debug_module.attr("pre_op_callback");
    nb::object post_op_func = debug_module.attr("post_op_callback");

    // Validate
    if (!pre_op_func.is_none() && !PyCallable_Check(pre_op_func.ptr())) {
        throw nb::type_error("pre_op_callback must be callable or None");
    }
    if (!post_op_func.is_none() && !PyCallable_Check(post_op_func.ptr())) {
        throw nb::type_error("post_op_callback must be callable or None");
    }

    // Create wrappers with GIL acquisition
    auto pre_callback = [pre_op_func](...) {
        nb::gil_scoped_acquire gil;
        try {
            pre_op_func(binary, programContext, opContext);
        } catch (...) { /* handle errors */ }
    };

    // Register
    tt::runtime::debug::Hooks::get(pre_callback, post_callback);
#endif
});
```

**Benefits:**
- ✅ All module imports happen in Python extension context
- ✅ All nanobind type conversions in single instance
- ✅ Proper GIL management throughout
- ✅ Clean error handling with exceptions

---

## Call Flow

```
┌─────────────────────────────────────────────────────┐
│ Static Library: auto_debug_hooks.cpp                │
│ (Pure Python C API)                                 │
├─────────────────────────────────────────────────────┤
│ 1. PyGILState_Ensure()                              │
│ 2. PyImport_ImportModule("ttrt.runtime")            │
│ 3. PyObject_GetAttrString("register_debug_callback")│
│ 4. PyObject_CallObject(func, nullptr) ◄──── NO ARGS│
│ 5. Cleanup & PyGILState_Release()                   │
└─────────────────────────────────────────────────────┘
                     │
                     │ calls
                     ▼
┌─────────────────────────────────────────────────────┐
│ Python Extension: runtime.cpp                       │
│ (Nanobind context)                                  │
├─────────────────────────────────────────────────────┤
│ register_debug_callback():                          │
│   1. nb::gil_scoped_acquire ◄──── GIL already held │
│   2. nb::module_::import_("debug_callback")         │
│   3. Get pre_op_callback from module                │
│   4. Get post_op_callback from module               │
│   5. Create lambda wrappers with GIL                │
│   6. Call debug::Hooks::get(pre, post)              │
└─────────────────────────────────────────────────────┘
                     │
                     │ imports
                     ▼
┌─────────────────────────────────────────────────────┐
│ User Code: debug_callback.py                        │
├─────────────────────────────────────────────────────┤
│ def pre_op_callback(binary, ctx, op):               │
│     print(f"Op: {get_op_debug_str(op)}")            │
│                                                      │
│ def post_op_callback(binary, ctx, op):              │
│     pass                                             │
└─────────────────────────────────────────────────────┘
```

---

## Advantages Over Original Plan

| Aspect | Original Plan | Updated Implementation |
|--------|--------------|----------------------|
| **C API Complexity** | Pass callbacks as PyObject* args | No arguments needed |
| **Reference Counting** | 7 PyObject* to manage | 3 PyObject* to manage |
| **Error Paths** | Many early returns with cleanup | Simpler cleanup logic |
| **Boundary Crossing** | Python objects cross boundary | Only function call crosses |
| **Module Import** | Done in C with PyImport | Done in C++ with nanobind |
| **Type Safety** | Manual PyCallable_Check in C | Nanobind handles type checks |

---

## Files Modified

1. ✅ `runtime/python/runtime/runtime.cpp` - Zero-argument registration function
2. ✅ `runtime/lib/ttnn/auto_debug_hooks.cpp` - Simplified C API calls
3. ✅ `runtime/lib/ttnn/CMakeLists.txt` - Python::Python dependency (unchanged)

---

## Testing

### Build Status
```bash
$ cd /localdev/ndrakulic/tt-xla && source venv/activate && cmake --build build
...
Successfully built pjrt-plugin-tt
Successfully installed pjrt-plugin-tt-0.1.260212+dev.aecd2574
✅ BUILD SUCCESSFUL
```

### Runtime Test
```bash
export TT_RUNTIME_DEBUG=1
export TT_FORCE_DEBUG_CALLBACKS=1
python debug_callback.py
```

**Expected Output:**
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

### API Usage
```python
import ttrt.runtime

# No arguments needed - reads from debug_callback.py
ttrt.runtime.register_debug_callback()
```

---

## Why This Is Better

**Simpler Mental Model:**
- C code: "Just call this Python function, it handles everything"
- Python code: "I know how to import modules and get attributes safely"

**Less Can Go Wrong:**
- Fewer variables = fewer bugs
- Fewer cleanup paths = fewer leaks
- Single responsibility = easier to debug

**Maintains Separation:**
- C API layer: minimal Python interaction
- Nanobind layer: all the smart type handling
- User code: unchanged

---

## Date Updated
2026-02-15
