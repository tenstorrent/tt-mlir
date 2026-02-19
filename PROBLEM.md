# Nanobind Segfault Problem in auto_debug_hooks.cpp

## Symptom

Segfault at `auto_debug_hooks.cpp:72` when calling `nb::type<tt::runtime::Binary>()`:

```
Thread 247 "python" received signal SIGSEGV, Segmentation fault.
nanobind::detail::nb_type_c2p(internals_=0x0, ...)
```

Key observation: **`internals_` is NULL** (0x0)

## Critical Discovery

The segfault occurs **even when testing with `CallbackContext`**, which IS registered in the correct module (`_ttmlir_runtime.runtime`). This proves the problem is NOT about which submodule types are registered in.

## Root Cause: Duplicate Nanobind Instances

### The Problem

**TWO SEPARATE NANOBIND INSTANCES** exist with independent global state:

#### Instance #1 (Uninitialized)
- **Location**: `TTRuntimeTTNN.a` (static library)
- **Created by**: `target_link_libraries(TTRuntimeTTNN PRIVATE nanobind-static)`
  at `runtime/lib/ttnn/CMakeLists.txt:39`
- **State**: Has its own `nanobind::detail::internals` that is NEVER initialized (no `NB_MODULE` call)
- **Problem**: Code in `auto_debug_hooks.cpp` accesses THIS instance

#### Instance #2 (Properly Initialized)
- **Location**: `_ttmlir_runtime.so` (Python extension)
- **Created by**: `nanobind_add_module(_ttmlir_runtime ...)`
  at `runtime/python/CMakeLists.txt:39-42`
- **State**: Has properly initialized `nanobind::detail::internals` (via `NB_MODULE` macro)
- **Success**: All type registrations happen here and work correctly

### What Happens

1. User runs Python script
2. `_ttmlir_runtime.so` loads → Instance #2 initialized via `NB_MODULE`
3. Types registered in Instance #2: `Binary`, `CallbackContext`, `OpContext`, etc.
4. Code in `auto_debug_hooks.cpp` executes (from static library code)
5. `nb::type<tt::runtime::Binary>()` tries to look up type
6. **BUT**: Static library code accesses Instance #1's internals (NULL)
7. Types are in Instance #2's registry → completely separate!
8. Result: Null pointer dereference → SEGFAULT

### Why Even CallbackContext Fails

Even though `CallbackContext` is correctly registered in `_ttmlir_runtime.runtime`, the callback code in the static library accesses the **wrong nanobind instance** (Instance #1, which has no types registered), so it still segfaults with `internals_=0x0`.

## Architecture Diagram

```
Before (Broken):

TTRuntimeTTNN.a (static library)
├── Links: nanobind-static → Instance #1
│   └── internals_ = NULL (never initialized)
├── auto_debug_hooks.cpp
│   └── nb::type<T>() accesses Instance #1 → NULL → SEGFAULT ✗
└── Linked into TTMLIRRuntime.so

_ttmlir_runtime.so (Python extension)
├── Links: nanobind via nanobind_add_module() → Instance #2
│   └── internals_ = properly initialized via NB_MODULE ✓
├── All types registered in Instance #2:
│   ├── Binary (from registerBinaryBindings)
│   ├── CallbackContext (from registerRuntimeBindings)
│   └── OpContext (from registerRuntimeBindings)
└── Instance #2 works perfectly, but static library can't access it!
```

## Key Insights

1. **Not a submodule isolation issue** - Even types in the "correct" submodule fail
2. **Not a linking order issue** - Both instances link correctly, they're just separate
3. **Not fixable by imports** - Importing modules doesn't merge the two instances
4. **Core problem**: Static library has its own uninitialized nanobind instance

## Why Standard Fixes Don't Work

### ❌ Import `ttrt.binary` in auto_debug_hooks.cpp
- **Why it fails**: Only registers types in Instance #2, static library still accesses Instance #1

### ❌ Move Binary to runtime submodule
- **Why it fails**: Doesn't fix the duplicate instances problem

### ❌ Remove validation code only
- **Why it fails**: Automatic type conversion also needs to access the type registry

## Solution Requirements

Any fix MUST ensure:
1. Only ONE nanobind instance exists
2. All nanobind operations happen in the context where that instance is initialized
3. Type registrations and type lookups use the SAME instance

## Viable Solutions

### Option 1: Remove Nanobind from Static Library
- Don't link `nanobind-static` to `TTRuntimeTTNN`
- Only include nanobind headers for type definitions
- Move callback registration to Python extension code
- **Pros**: Clean separation, single nanobind instance
- **Cons**: Requires manual registration from Python

### Option 2: Export Nanobind Symbols from Python Extension
- Make Python extension export nanobind symbols with specific visibility
- Force static library to use those symbols (not its own copy)
- **Pros**: Could keep automatic registration
- **Cons**: Complex linker setup, ABI compatibility risks

## References

- Static library CMake: `/localdev/ndrakulic/tt-xla/third_party/tt-mlir/src/tt-mlir/runtime/lib/ttnn/CMakeLists.txt:39`
- Python extension CMake: `/localdev/ndrakulic/tt-xla/third_party/tt-mlir/src/tt-mlir/runtime/python/CMakeLists.txt:39-42`
- Problematic code: `/localdev/ndrakulic/tt-xla/third_party/tt-mlir/src/tt-mlir/runtime/lib/ttnn/auto_debug_hooks.cpp:72`
