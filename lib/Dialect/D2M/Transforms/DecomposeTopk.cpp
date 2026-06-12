// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSETOPK
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Decompose TopkBlockOp into low-level tile operations.
///
/// The topk_block op is replaced by:
///   1. arange_block for index generation
///   2. tile_topk_local_sort for per-tile bitonic sort
///   3. tile_topk_merge for cross-tile merge
///   4. tile_topk_rebuild for final rebuild
struct DecomposeTopkBlockPattern : OpRewritePattern<TopkBlockOp> {
  using OpRewritePattern<TopkBlockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TopkBlockOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value inputValues = op.getInputValues();
    Value scratchIdxTile = op.getScratchIdxTile();
    Value outValues = op.getOutValues();
    Value outIndices = op.getOutIndices();

    auto inputType = dyn_cast<MemRefType>(inputValues.getType());
    TT_assertv(inputType,
               "input must be a memref, run after bufferization");

    ArrayRef<int64_t> inputShape = inputType.getShape();
    TT_assertv(inputShape.size() >= 2ul,
               "input must have at least 2 dimensions");

    int32_t k = op.getK();
    int64_t numElements = op.getNumElements();
    bool stableSort = op.getStableSort();

    auto tileType = cast<ttcore::TileType>(inputType.getElementType());
    int64_t tileWidth = tileType.getWidth();

    // Compute topk parameters.
    int32_t iEndPhase = static_cast<int32_t>(llvm::Log2_64(tileWidth)) - 1;
    int32_t logk = static_cast<int32_t>(llvm::Log2_64_Ceil(k));
    // skip_second=1 when k fits in a single tile width.
    int32_t skipSecond = (k <= tileWidth) ? 1 : 0;

    // Derive index tile type (si32 version of the input tile).
    Type si32Type = IntegerType::get(rewriter.getContext(), 32,
                                     IntegerType::Signed);
    auto idxTileType = ttcore::TileType::get(si32Type, tileType.getShape());
    auto l1MemorySpace = ttcore::MemorySpaceAttr::get(
        rewriter.getContext(), ttcore::MemorySpace::DeviceL1);

    // Helper to allocate a scratch memref with the same shape as input but
    // with a given tile element type.
    auto allocScratch = [&](ArrayRef<int64_t> shape,
                            ttcore::TileType elemType) -> Value {
      auto memrefType = MemRefType::get(shape, elemType,
                                        MemRefLayoutAttrInterface{},
                                        l1MemorySpace);
      return rewriter.create<memref::AllocOp>(loc, memrefType).getResult();
    };

    // Allocate scratch buffers.
    Value bufIdx = allocScratch(inputShape, idxTileType);
    Value sortedVals = allocScratch(inputShape, tileType);
    Value sortedIdx = allocScratch(inputShape, idxTileType);
    Value mergedVals = allocScratch(inputShape, tileType);
    Value mergedIdx = allocScratch(inputShape, idxTileType);

    // 1. Generate indices via arange_block.
    Value bufIdxFilled = rewriter
                             .create<ArangeBlockOp>(loc, scratchIdxTile, bufIdx,
                                                    numElements,
                                                    /*start=*/0,
                                                    /*step=*/1)
                             .getResult();

    // 2. Bitonic sort each tile independently.
    rewriter.create<TileTopkLocalSortOp>(
        loc, inputValues, bufIdxFilled, sortedVals, sortedIdx,
        /*idir=*/rewriter.getI32IntegerAttr(0),
        /*i_end_phase=*/rewriter.getI32IntegerAttr(iEndPhase),
        /*i_start_phase=*/rewriter.getI32IntegerAttr(0),
        /*stable_sort=*/rewriter.getBoolAttr(stableSort));

    // 3. Merge sorted tile pairs.
    rewriter.create<TileTopkMergeOp>(
        loc, sortedVals, sortedIdx, mergedVals, mergedIdx,
        /*m_iter=*/rewriter.getI32IntegerAttr(0),
        /*k=*/rewriter.getI32IntegerAttr(k),
        /*stable_sort=*/rewriter.getBoolAttr(stableSort));

    // 4. Rebuild final sorted order.
    rewriter.create<TileTopkRebuildOp>(
        loc, mergedVals, mergedIdx, outValues, outIndices,
        /*idir=*/rewriter.getI32IntegerAttr(0),
        /*m_iter=*/rewriter.getI32IntegerAttr(0),
        /*k=*/rewriter.getI32IntegerAttr(k),
        /*logk=*/rewriter.getI32IntegerAttr(logk),
        /*skip_second=*/rewriter.getI32IntegerAttr(skipSecond),
        /*stable_sort=*/rewriter.getBoolAttr(stableSort));

    // Replace the block op with its output buffers.
    rewriter.replaceOp(op, {outValues, outIndices});
    return success();
  }
};

struct D2MDecomposeTopk
    : public impl::D2MDecomposeTopkBase<D2MDecomposeTopk> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeTopkBlockPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
