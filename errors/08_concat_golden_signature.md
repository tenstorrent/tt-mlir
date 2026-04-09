# Concat Golden Signature Mismatch (5 failures)

## Error Signature
```
ttnn_concat_golden() takes 3 positional arguments but 4/31 were given
```

## Representative Test
`test_concat[ttnn-shapes13-0]`

## Root Cause

The Chisel executor's `_build_golden_args()` in `tools/chisel/chisel/executor.py` (lines 40-91) uses `inspect.signature()` to detect if the first parameter expects a `List` (variadic tensor inputs).

The check is in `_is_list_annotation()` (lines 26-37), which calls `typing.get_origin(annotation)`. However, `tools/golden/mapping.py` line 14 has `from __future__ import annotations` (PEP 563), which makes ALL type annotations stored as **strings** at runtime. So `get_origin("List[GoldenMapTensor]")` returns `None`, and `_is_list_annotation()` always returns `False`.

Result: each input tensor is passed as a separate positional argument instead of wrapped in a list:
- **Expected**: `ttnn_concat_golden([A, B, C], IntegerAttr(dim), F32Type)`
- **Actual**: `ttnn_concat_golden(A, B, C, IntegerAttr(dim), F32Type)` -- too many args

## Key Files

| File | Lines | Role |
|------|-------|------|
| `tools/chisel/chisel/executor.py` | 26-37 | `_is_list_annotation()` -- broken by PEP 563 string annotations |
| `tools/chisel/chisel/executor.py` | 68-75 | Input grouping logic falls into wrong branch |
| `tools/golden/mapping.py` | 14 | `from __future__ import annotations` makes annotations strings |
| `tools/golden/mapping.py` | 6484-6492 | `ttnn_concat_golden` expects `(List, IntegerAttr, Type)` |

## Fix

In `_is_list_annotation()`, handle string annotations:
```python
def _is_list_annotation(annotation) -> bool:
    if isinstance(annotation, str):
        return annotation.startswith("List[") or annotation.startswith("list[")
    origin = get_origin(annotation)
    return origin is list
```
