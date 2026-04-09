# EmitC Runtime Missing `.test` Attribute (8 failures)

## Error Signature
```
module '_ttmlir_runtime.runtime' has no attribute 'test'
```

## Representative Test
`test_layer_norm_pre_all_gather[emitc-True-1x1x32x512]`

## Root Cause

The `execute_cpp` function in `tools/builder/base/builder_runtime.py` (lines 1187-1218) unconditionally accesses `tt_runtime.runtime.test.*` methods (`open_so`, `get_so_programs`, `create_inputs`, `run_so_program`), but the `test` submodule is a **conditionally compiled** nanobind extension that only exists when built with `-DTTMLIR_ENABLE_RUNTIME_TESTS=ON`.

The default is `OFF` (`CMakeLists.txt` line 60).

Conditional compilation enforced in:
- `runtime/python/__init__.cpp` lines 36-38: conditionally creates `m_runtime.def_submodule("test")`
- `runtime/python/runtime/test.cpp` line 5: guarded by `#if defined(TTMLIR_ENABLE_RUNTIME_TESTS)`
- `runtime/python/CMakeLists.txt` lines 30-32: conditionally adds `test.cpp`

Note: `tools/ttrt/runtime/__init__.py` (lines 76-81) already handles this with `try/except ImportError`. The builder has no such guard.

## Fix

Add a guard in `execute_cpp` (~line 1186 of `builder_runtime.py`):
```python
if not hasattr(tt_runtime.runtime, "test"):
    raise TTBuilderRuntimeException(
        "EmitC execution requires building with -DTTMLIR_ENABLE_RUNTIME_TESTS=ON"
    )
```

Or skip emitc tests via pytest marker when `tt_runtime.runtime.test` is unavailable.
