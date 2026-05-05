# TTIRRankNormalization Scoping Changes

## Problem

`TTIRRankNormalization` operates at module scope. It marks any op with rank < 2 as illegal regardless of dialect. In the `TTIRToTTNN` pipeline with optimizer + D2M fusion enabled, this breaks because:

1. `ConvertTTNNToTTIRPass` only converts ops inside `@d2m_subgraph_*` functions
2. TTNN ops in the main function survive and reach `TTIRRankNormalization`
3. The pass expands their tensor types (1D -> 2D) without updating shape-related attributes
4. This breaks verifiers for `ttnn.concat`, `ttnn.reshape`, `ttnn.arange`, etc.

## Solution: Hybrid of Option A + Option B

The fix combines two approaches:

### Option A: Mark TTNN dialect as legal

```cpp
target.addLegalDialect<ttnn::TTNNDialect>();
```

All TTNN ops are marked legal so the conversion framework never attempts to rewrite them. This alone is insufficient because `populateFunctionOpInterfaceTypeConversionPattern` would still promote function signatures of TTNN-only functions, causing `unrealized_conversion_cast` insertions.

### Option B: Scope to participating functions

Rather than hardcoding D2M subgraph function names, the pass uses a generalized criterion: **a function participates in rank normalization if:**
- It has a body containing at least one TTIR dialect op, **or**
- It is an external declaration (no body) whose signature needs rank expansion (e.g. CPU-hoisted forward declarations)

```
Pre-walk module -> build set of participating funcs
                -> dynamic legality only marks ops illegal if in a participating func
```

Pure-TTNN functions with bodies (const-eval helpers, already-lowered subgraphs) are left entirely untouched: signature, block args, body, and return op all keep their original ranks. External declarations that need promotion still participate so their signatures stay in sync with call sites in TTIR functions.

## How It Works

1. Walk the module and collect participating `func.func` ops:
   - External declarations: participate if their arg/result types need rank expansion.
   - Functions with bodies: participate if they contain at least one TTIR-dialect op.

2. In the dynamic legality callback, if an op needs rank expansion but lives in (or is) a non-participating function, mark it as **legal** (skip it).

3. Combined with `addLegalDialect<ttnn::TTNNDialect>()`, this ensures TTNN-only functions are completely untouched -- no signature promotion, no body rewriting, no stray casts -- while external declarations stay in sync with their call sites.

## External Declaration Handling

External functions (no body) — such as CPU-hoisted forward declarations (`tt.function_type = "ForwardCPUDeclaration"`) — cannot contain TTIR ops. Without special handling they would be excluded from rank normalization, leaving their signatures at rank-1 while call sites in participating TTIR functions are promoted to rank-2. This mismatch would cause verifier failures. The fix treats external declarations as participating when their signatures need rank expansion, keeping them eligible for `FuncOpRankNormalizationPattern`.

## Why Not the Other Options

| Option | Why not |
|--------|---------|
| **C** (Teach RankNorm about TTNN ops) | Large surface area, creates coupling between TTIR pass and TTNN dialect, fragile for new ops |
| **D** (Full TTNN->TTIR reconversion) | Many ops have no TTNN->TTIR pattern, risks round-trip info loss, massive effort |
| **E** (Move RankNorm before D2M) | Doesn't help -- D2M sub-pipeline still calls `createTTIRToTTMetalFrontendPipeline` which includes RankNorm |

## Files Changed

- `lib/Dialect/TTIR/Transforms/RankNormalization.cpp` -- Core logic
- `test/ttmlir/Dialect/TTIR/Transforms/rank_normalization_scoping.mlir` -- Scoping unit test
