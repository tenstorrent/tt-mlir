# Decomposition Workarounds Migration Plan

## Overview
This document outlines the plan for migrating existing decomposition workaround patterns to the new interface-based system.

## Architecture Summary

### New System Components
1. **Base Class**: `DecompositionWorkaround` with virtual `matchAndRewrite()` method
2. **TableGen Interface**: `TTNNDecompositionWorkaroundInterface`
3. **Universal Rewriter**: `DecompositionRewriter` that handles all ops with interface
4. **Op Implementation**: Ops implement `getDecompositionWorkarounds()` to return workaround objects

### Benefits
- Reusable workarounds across different passes
- Type-safe interface via TableGen
- Cleaner separation of concerns
- Easier to test and maintain

## Migration Steps for Each Pattern

For each existing pattern in `lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/`:

1. **Add Interface to Op in TableGen** (`TTNNOps.td`):
   ```tablegen
   def TTNN_OpName : TTNN_Op<"op_name", [
     // ... existing interfaces ...
     DeclareOpInterfaceMethods<TTNN_DecompositionWorkaroundInterface, ["getDecompositionWorkarounds"]>
   ]> {
   ```

2. **Implement Workaround Class in TTNNOps.cpp**:
   ```cpp
   namespace {
   class OpNameWorkaround : public decomposition::DecompositionWorkaround {
   public:
     LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
       // Move logic from existing pattern's matchAndRewrite
     }
   };
   } // namespace
   ```

3. **Implement Interface Method in TTNNOps.cpp**:
   ```cpp
   decomposition::DecompositionWorkarounds
   OpName::getDecompositionWorkarounds() {
     decomposition::DecompositionWorkarounds workarounds;
     workarounds.push_back(std::make_unique<OpNameWorkaround>());
     return workarounds;
   }
   ```

4. **Remove Old Pattern** from `TTNNWorkaroundsPatterns.cpp`

## Patterns to Migrate

### High Priority (Used in multiple places)
- [ ] Conv3dPadOutputChannelsRewritePattern
- [ ] Conv3dDepthPaddingRewritePattern
- [ ] LinearOpRewritePattern
- [ ] ScaledDotProductAttentionPadTileDimsRewritePattern

### Standard Patterns
- [ ] AllGatherOpRewritePattern
- [ ] ArgMaxOpRewritePattern
- [ ] ConcatenateHeadsOpRewritePattern
- [ ] PagedUpdateCacheOpRewritePattern
- [ ] PointToPointOpRewritePattern
- [ ] SplitQueryKeyValueAndSplitHeadsOpRewritePattern
- [ ] CumSumOpDimRewritePattern
- [ ] CumSumOpRankRewritePattern
- [ ] EmbeddingOpSqueezeWeightRewritePattern
- [ ] MultiplyOpDecompositionRewritePattern
- [ ] ReduceScatterOpRewritePattern
- [ ] ScatterOpRewritePattern
- [ ] UpsampleOpRewritePattern
- [ ] RMSNormConfigRewritePattern
- [ ] DistributedRMSNormWidthShardInputRewritePattern
- [ ] ReduceScatterConfigRewritePattern
- [ ] SubtractOpImplicitBroadcastRewritePattern
- [ ] ExplicateOperandBroadcastsRewritePattern
- [ ] PadHighDimRewritePattern

### Already Migrated
- [x] RotaryEmbeddingOpRewritePattern → RotaryEmbeddingPaddingWorkaround

## Testing Strategy

1. **Unit Tests**: Create tests in `test/ttmlir/Dialect/TTNN/Workarounds/` for each migrated pattern
2. **Integration Tests**: Verify workarounds work in both TTNNWorkarounds pass and fusion patterns
3. **Performance Tests**: Ensure no regression in compilation time

## Notes

- Workaround implementations are currently in `TTNNOps.cpp` to avoid linking issues
- Future work could move them to separate files once linking is resolved
- Some ops may have multiple workarounds (e.g., Conv3d has 4 patterns)