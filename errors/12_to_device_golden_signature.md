# to_device Golden Signature Mismatch (16 failures)

## Error Signature
```
ttnn_to_device_golden() takes 1 positional argument but 2 were given
```

## Representative Test
`test_conv3d[ttnn-f32-temporal_downsampling_192ch_s211]`

## Root Cause

Same root cause as from_device (#01). The Chisel executor's `_build_golden_args` in `tools/chisel/chisel/executor.py` (lines 40-91) does not handle `**kwargs` (VAR_KEYWORD) parameters.

For `ttnn_to_device_golden(input_tensor: GoldenMapTensor, **kwargs)`:
1. `inspect.signature()` returns `params = [input_tensor, kwargs]`
2. After input handling: `args = [golden_input]`, `param_idx = 1`
3. Loop over `params[1:]` = `[kwargs]` -- a VAR_KEYWORD parameter
4. `name = 'kwargs'` doesn't match any condition, falls to else
5. `args.append(output_type)` -- incorrectly adds output_type as 2nd positional arg

## Key Files

| File | Lines | Role |
|------|-------|------|
| `tools/chisel/chisel/executor.py` | 40-91 | `_build_golden_args` -- doesn't skip VAR_KEYWORD |
| `tools/golden/mapping.py` | 6672-6673 | `ttnn_to_device_golden(input_tensor, **kwargs)` |
| `tools/golden/mapping.py` | 6676-6677 | `ttnn_from_device_golden(input_tensor, **kwargs)` |

## Fix

In `_build_golden_args()`, skip VAR_KEYWORD and VAR_POSITIONAL parameters:
```python
for param in params[param_idx:]:
    if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
        continue
    # ... rest of existing logic
```

This fixes both to_device (16) and from_device (632) failures.
