# Unexpected Attribute Type (38 failures)

## Error Signatures
- `Unexpected attribute type: <class 'golden.mapping.GoldenMapTensor'>` (30 failures)
- `Unexpected attribute type: <class 'ttmlir._mlir_libs._mlir.ir.F32Type'>` (8 failures)

## Representative Tests
- `test_concat[ttnn-shapes9-0]` (GoldenMapTensor)
- `test_layer_norm[ttnn-True-True-shape1-normalized_shape1]` (F32Type)

## Origin
Both originate in Chisel executor: `tools/chisel/chisel/executor.py` -> `_build_golden_args()` -> `golden_fn(*args)` -> `tools/golden/mapping.py:393` `unpack_mlir_attr()` raises `ValueError`.

## Variant 1: GoldenMapTensor (30 failures)

**Root Cause**: Same as concat issue (#08). PEP 563 `from __future__ import annotations` in `mapping.py` makes `_is_list_annotation()` fail, so concat inputs are splatted as separate args. The `dim_attr` parameter receives a `GoldenMapTensor` instead of an `IntegerAttr`, and `unpack_mlir_attr(GoldenMapTensor)` raises the error.

**Fix**: Same as #08 -- fix `_is_list_annotation()` to handle string annotations.

## Variant 2: F32Type (8 failures, all layer_norm)

**Root Cause**: Two problems in `_build_golden_args` (executor.py:77-91):

1. **Optional tensor params not handled**: `ttnn_layer_norm_golden` has Optional tensor params (`weight`, `bias`). When absent, the attribute loop's else clause at line 88-89 passes `output_type` (F32Type) as fallback.

2. **TTIR-vs-TTNN attribute mismatch**: Golden expects `normalized_shape` param but TTNN `LayerNormOp` (`TTNNOps.td:2356-2377`) doesn't have this attribute (only TTIR version does). Missing attribute -> else clause -> F32Type.

**Fix**: In `_build_golden_args` else clause (executor.py:88-89):
1. Detect Optional tensor annotations and pass `None`
2. For missing attributes, use parameter defaults
3. Raise clear error instead of silently passing `output_type`
