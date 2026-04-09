# rms_norm `normalized_shape` Receives F32Type (12 failures)

## Error Signature
```
rms_norm(): argument 'normalized_shape' (position 2) must be tuple of ints, not ttmlir._mlir_libs._mlir.ir.F32Type
```

## Representative Test
`test_rms_norm[ttnn-True-True-shape2-normalized_shape2]`

## Root Cause

In `tools/builder/ttir/ttir_builder.py`, the `rms_norm` method (line 14202) calls `ttir_rms_norm_golden` with raw MLIR attribute objects instead of Python-native values:

```python
# ttir_builder.py:14226-14232
golden_output = op_golden_function(
    input0,
    weight=weight0,
    bias=bias0,
    normalized_shape=normalized_shape_attr,   # DenseI64ArrayAttr (MLIR attr)
    epsilon=epsilon_attr,                      # FloatAttr (MLIR attr)
    output_type_mlir=mlir_output_type,        # F32Type (MLIR type)
)
```

The golden function `ttir_rms_norm_golden` in `tools/golden/mapping.py` (line 1096) receives these MLIR objects. The `F32Type` from `mlir_output_type = self.get_type(in0)` ends up being passed as `normalized_shape` to `torch.nn.functional.rms_norm`.

The `rms_norm_parser` at line 14289 has the same pattern.

## Key Files

| File | Lines | Role |
|------|-------|------|
| `tools/builder/ttir/ttir_builder.py` | 14202-14258 | `rms_norm` method passes MLIR attrs |
| `tools/builder/ttir/ttir_builder.py` | 14226-14232 | Golden function call with wrong arg types |
| `tools/builder/ttir/ttir_builder.py` | 14261-14300 | `rms_norm_parser` -- same issue |
| `tools/golden/mapping.py` | 1096-1119 | `ttir_rms_norm_golden` |
| `tools/golden/mapping.py` | 1052-1093 | `rms_norm_golden` (TTNN version -- works correctly) |

## Fix

Pass Python-native values instead of MLIR attributes in `ttir_builder.py:14226-14232`:

```python
golden_output = op_golden_function(
    input0,
    weight=weight0,
    bias=bias0,
    normalized_shape=normalized_shape,  # Python list[int], NOT MLIR attr
    epsilon=epsilon,                     # Python float, NOT MLIR attr
    output_type_mlir=mlir_output_type,
)
```

Also fix the parser at line 14289-14296 to unpack MLIR attrs before passing:
```python
normalized_shape=list(normalized_shape_attr),
epsilon=float(epsilon_attr.value),
```

Note: `ttir_layer_norm_golden` and `layer_norm` builder method have the exact same pattern and may need the same fix.
