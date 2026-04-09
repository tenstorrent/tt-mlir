# Where Op Boolean Condition Cast Failure (78 failures)

## Error Signature
```
where expected condition to be a boolean tensor, but got a tensor with dtype Float/BFloat16
```

## Representative Test
`test_where_nan_inf_implicit_broadcast[neg_inf-ttnn-f32-32x64-32x1-32x64]`

## Root Cause

tt-metal's TTNN `where` op now requires the condition tensor to have **boolean dtype**. The tt-mlir compiler pipeline does not cast the condition to boolean before sending it to the device.

The failure chain:
1. Tests declare condition input with same dtype as value inputs (f32, bf16)
2. TTIR WhereOp accepts `AnyRankedTensor` for all inputs (`TTIROps.td` lines 247-249)
3. The TTNN workaround pass (`TTNNWorkaroundsPass.cpp` lines 351-387) only forces the predicate to match input type when they differ. When predicate and input are already the same float type, **no workaround is applied**
4. At runtime, tt-metal validates the condition tensor is boolean and fails

## Key Files

| File | Lines | Role |
|------|-------|------|
| `lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp` | 340-387 | `createWhereOpOperandsWorkarounds()` -- needs to force bool |
| `tools/golden/mapping.py` | 4573-4580 | `ttir_where_golden` -- no `.to(torch.bool)` cast |
| `tools/golden/mapping.py` | 6505-6512 | `ttnn_where_golden` -- no `.to(torch.bool)` cast |
| `tools/golden/mapping.py` | 5373-5374 | `stablehlo_select_golden` -- CORRECT: does `pred.to(torch.bool)` |

## Fix

**Fix 1 (Compiler):** In `TTNNWorkaroundsPass.cpp:createWhereOpOperandsWorkarounds()`, always force predicate to boolean regardless of current type.

**Fix 2 (Golden):** In both `ttir_where_golden` and `ttnn_where_golden`, add `.to(torch.bool)` on condition:
```python
return torch.where(condition.to(torch.bool), x, y).to(output_dtype)
```
