# Type Confusion: Scalar MLIR Type vs RankedTensorType (36 failures)

## Error Signatures
- `BF16Type/F32Type/IntegerType object has no attribute 'element_type'` (31)
- `BF16Type/F32Type object has no attribute 'shape'` (2)
- `BF16Type object has no attribute 'float'` (2)
- `BF16Type object has no attribute 'shard_map'` (2)

## Representative Test
`test_reduction_ops[ttnn-argmax-dim_arg1-True-f32-32x128x128]`

## Root Cause

Multiple code paths unconditionally assume an MLIR value's `.type` is `RankedTensorType` (which has `.shape`, `.element_type`, `.encoding`). When the type is a scalar MLIR type (`BF16Type`, `F32Type`, `IntegerType`), these attributes don't exist.

## Bug Locations

### 1. `Builder.get_type()` / `Builder.get_shape()`
**File**: `tools/builder/base/builder.py`, lines 250-254
```python
def get_shape(self, input): return self._get_type(input).shape      # FAILS on scalar
def get_type(self, input):  return self._get_type(input).element_type # FAILS on scalar
```
Called from **50+ sites** in `tools/builder/ttnn/ttnn_builder.py`.

### 2. TTNN `@parse` methods
**File**: `tools/builder/ttnn/ttnn_builder.py`, ~50 instances
Pattern: `result = old_op.result.type` then `result.element_type` without type check.

### 3. Chisel executor
**File**: `tools/chisel/chisel/executor.py`, line 124
```python
output_type = op_outputs[0].type.element_type if op_outputs else None
```

### 4. Golden functions
**File**: `tools/golden/mapping.py`, lines 4633, 6660
`ttir_to_layout_golden` and `ttnn_to_layout_golden` call `.element_type` without checking.

Compare with **correct pattern** at line 5675-5678:
```python
if hasattr(output_type_mlir, "element_type"):
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir.element_type)
else:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
```

## Fix

1. **`builder.py` `get_type()`/`get_shape()`**: Add `isinstance(t, RankedTensorType)` guard
2. **`executor.py` line 124**: Use `hasattr(raw_type, "element_type")` guard
3. **`@parse` methods**: Guard `.element_type` with isinstance check
4. **Golden functions**: Adopt `hasattr` defensive pattern
