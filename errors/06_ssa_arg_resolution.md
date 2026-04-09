# SSA Arg Resolution Failure (69 failures)

## Error Signature
```
'%arg0' (64 failures), '%arg1' (5 failures)
```

## Representative Test
`test_hoisted_max_pool2d_with_indices[ttnn-kernel9-stride9-dilation9-padding9-True-i32-4x14x14x64]`

## Root Cause: Type Mismatch Between Device and CPU Modules After ElementTypeNormalization

Only affects **hoisted ops with i32 dtype**. The compilation pipeline creates a signedness mismatch:

1. **Hoisting** (`HoistCPUOps.cpp:257-277`) creates CPU function definition with **signless** `i32` types via `dropSignInformation()`
2. Device module function declaration initially also uses signless types
3. **ElementTypeNormalization** (`ElementTypeNormalization.cpp:37-38`) runs on **device module only** (scoped at `TTNNPipelines.cpp:255-262`), converting signless `i32` to **signed** `si32`
4. Result: device declaration has `si32`, CPU definition has signless `i32` -- mismatch

Float types are unaffected because `f32` maps to `f32` under normalization. Only integer types change signedness.

## Key Files

| File | Lines | Role |
|------|-------|------|
| `lib/Dialect/TTIR/Transforms/HoistCPUOps/HoistCPUOps.cpp` | 257-277 | `dropSignInformation()` strips signedness |
| `lib/Dialect/TTIR/Transforms/HoistCPUOps/HoistCPUOps.cpp` | 280-418 | Creates CPU function def (signless) vs device declaration |
| `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` | 251-262 | Hoisting at 251, ElementTypeNorm at 262 (device-only) |
| `lib/Dialect/TTIR/Transforms/ElementTypeNormalization.cpp` | 37-38 | Converts i32 -> si32 |
| `include/ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h` | 128-245 | Type conversion: signless i32 -> DataType::Int32 -> signed si32 |

## Fix

Either:
1. Run `ElementTypeNormalization` on CPU module function definitions as well
2. Add a reconciliation step after device normalization to sync CPU module signatures
3. In `createCPUHoistedFunctionDeclaration()`, apply `dropSignInformation()` to match the definition
