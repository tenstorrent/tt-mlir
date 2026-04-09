# Fatal Error (53 failures)

## Error Signature
```
builder.base.builder_runtime.TTBuilderRuntimeException: Fatal error
```

## Representative Test
`test_reduce_scatter_without_workaround[ttnn-f32-1x2-rank3_dim1]`

## Root Cause

The literal string `"Fatal error"` originates from the C++ runtime logger at:

**`runtime/include/tt/runtime/detail/common/logger.h:319`**
```cpp
throw std::runtime_error("Fatal error");
```

This is thrown by `tt_throw_()` (lines 300-320), invoked by macros: `LOG_FATAL`, `LOG_ASSERT`, `DEBUG_ASSERT`.

The key design issue: `tt_throw_` constructs a detailed diagnostic string (file, line, condition, backtrace) and passes it to `log_fatal_()`, but then throws with **only the opaque literal string**. The actual diagnostic info is logged but never included in the exception message.

On the Python side, `tools/builder/base/builder_runtime.py:789` catches this and wraps it as `TTBuilderRuntimeException`.

## All 53 Failures Are Multi-Device (1x2 mesh)

Every failing test has `1x2` in its parameters — a 2-device mesh:

| Test Function | Count | Operation Type |
|---|---|---|
| `test_all_gather` | 42 | CCL: all_gather |
| `test_distributed_rms_norm` | 8 | Distributed normalization |
| `test_reduce_scatter_without_workaround` | 1 | CCL: reduce_scatter |
| `test_moe_dispatch_combine` | 1 | CCL: MoE dispatch/combine |
| `test_presharded_arg` | 1 | Pre-sharded multi-device arg |

## Key File Paths

- **Exception origin (C++)**: `runtime/include/tt/runtime/detail/common/logger.h:319`
- **Python wrapper**: `tools/builder/base/builder_runtime.py:789`
- **Device fixture**: `test/python/golden/conftest.py:207` / `conftest.py:65`

## What Needs Investigation

1. **Multi-device availability**: If the test environment only has a single device (e.g., N150), these ops will crash at runtime.
2. **CCL/fabric configuration**: All failing ops are CCL or distributed ops requiring inter-device communication.
3. **Improve error propagation**: Consider including the diagnostic string in the `std::runtime_error` message so Python consumers get actionable info.
