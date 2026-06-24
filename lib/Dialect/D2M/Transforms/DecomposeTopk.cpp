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

template <typename T>
static int32_t floorLog2(T n) {
  int32_t result = 0;
  for (; n > 1; n >>= 1) {
    ++result;
  }
  return result;
}

// Decomposes TopkBlockOp into arange_block and
// tile_topk_{local_sort,merge,rebuild} ops.
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
    TT_assertv(inputType, "input must be a memref, run after bufferization");

    ArrayRef<int64_t> inputShape = inputType.getShape();
    TT_assertv(inputShape.size() >= 2ul,
               "input must have at least 2 dimensions");

    int64_t numElements = op.getNumElements();
    int32_t dim = op.getDim();
    int32_t k = op.getK();

    auto tileType = cast<ttcore::TileType>(inputType.getElementType());
    Type si32Type =
        IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed);
    auto idxTileType = ttcore::TileType::get(si32Type, tileType.getShape());
    auto l1MemorySpace = ttcore::MemorySpaceAttr::get(
        rewriter.getContext(), ttcore::MemorySpace::DeviceL1);

    auto allocScratch = [&](ArrayRef<int64_t> shape,
                            ttcore::TileType elemType) -> Value {
      auto memrefType = MemRefType::get(
          shape, elemType, MemRefLayoutAttrInterface{}, l1MemorySpace);
      return rewriter.create<memref::AllocOp>(loc, memrefType).getResult();
    };

    Value bufIdx = allocScratch(inputShape, idxTileType);
    Value bufIdxFilled =
        rewriter
            .create<ArangeBlockOp>(loc, scratchIdxTile, bufIdx, numElements,
                                   /*start=*/0,
                                   /*step=*/1)
            .getResult();

    int32_t logk = floorLog2(k);

    int64_t numTilesInner = inputShape[dim];
    // logWt is the number of merge-tree iterations. Each iteration doubles the
    // distance between paired tiles, so numTilesInner must be a power of 2.
    int32_t logWt = floorLog2(numTilesInner);

    // When k>32 the result spans 2 tiles, requiring a 3-sub-merge reduction.
    bool useLargeK = (k > 32);

    auto i32Attr = [&](int32_t v) { return rewriter.getI32IntegerAttr(v); };
    auto i64Attr = [&](int64_t v) { return rewriter.getI64IntegerAttr(v); };
    auto boolAttr = [&](bool v) { return rewriter.getBoolAttr(v); };

    // Emit the local_sort, merge, and rebuild stages.
    auto emitSortMergeRebuild = [&](int64_t tA, int64_t tB, int32_t mK,
                                    int32_t mLogk, int32_t mIter,
                                    int32_t skipSecond, bool rfo) {
      rewriter.create<TileTopkLocalSortOp>(
          loc, inputValues, bufIdxFilled, outValues, outIndices, i32Attr(0),
          i32Attr(mLogk - 1), i32Attr(0), i64Attr(tA), i64Attr(tB),
          boolAttr(true), boolAttr(false), boolAttr(rfo));
      rewriter.create<TileTopkMergeOp>(
          loc, inputValues, bufIdxFilled, outValues, outIndices, i32Attr(mIter),
          i32Attr(mK), i64Attr(tA), i64Attr(tB), boolAttr(false),
          boolAttr(false), boolAttr(rfo));
      rewriter.create<TileTopkRebuildOp>(
          loc, inputValues, bufIdxFilled, outValues, outIndices, i32Attr(0),
          i32Attr(mIter), i32Attr(mK), i32Attr(mLogk), i32Attr(skipSecond),
          i64Attr(tA), i64Attr(tB), boolAttr(false), boolAttr(true),
          boolAttr(rfo));
    };

    for (int32_t mIter = 0; mIter < logWt; ++mIter) {
      bool isFirst = (mIter == 0);
      bool isLast = (mIter == logWt - 1);
      int64_t distance = 1LL << mIter;
      bool readFromOutput = !isFirst;

      for (int64_t base = 0; base + distance < numTilesInner;
           base += 2 * distance) {
        int64_t tileA = base;
        int64_t tileB = base + distance;

        if (useLargeK) {
          if (isFirst) {
            emitSortMergeRebuild(tileA, tileB, k, logk, /*mIter=*/mIter,
                                 /*skipSecond=*/0, /*rfo=*/false);
          } else {
            // DST only holds 2 tiles at a time, so we cannot merge all 4 tiles
            // directly. Instead, we use 3 sub-merges to compute the top-k.
            int64_t prevDist = distance / 2;

            // Step 1: Merge the winner tiles (tileA, tileB) from the previous
            // iteration; tileA holds the best-k across both afterward.
            emitSortMergeRebuild(tileA, tileB, k, logk, /*mIter=*/0,
                                 /*skipSecond=*/0, /*rfo=*/true);

            // Step 2: Merge the loser tiles; (tileA+prevDist) holds the best-k
            // across both losers afterward.
            emitSortMergeRebuild(tileA + prevDist, tileB + prevDist, k, logk,
                                 /*mIter=*/0, /*skipSecond=*/0, /*rfo=*/true);

            // Step 3: Merge winners against losers; tileB holds the final top-k
            // across all four tiles afterward.
            emitSortMergeRebuild(tileB, tileA + prevDist, k, logk,
                                 /*mIter=*/0, /*skipSecond=*/0, /*rfo=*/true);
          }
        } else {
          // K=32/logk=5 is used so that sorting spans both tiles; a smaller K
          // would confine it to a single tile, preventing cross-tile merges.

          // Rebuild only on the last iteration. Skip it when k==32 with a
          // single iteration since the merge output is already exactly k
          // elements. The !isFirst check handles the case when logWt = 1.
          bool needsRebuild = isLast && (!isFirst || k < 32);

          rewriter.create<TileTopkLocalSortOp>(
              loc, inputValues, bufIdxFilled, outValues, outIndices,
              i32Attr(mIter), i32Attr(4), i32Attr(0), i64Attr(tileA),
              i64Attr(tileB), boolAttr(true), boolAttr(false),
              boolAttr(readFromOutput));
          rewriter.create<TileTopkMergeOp>(
              loc, inputValues, bufIdxFilled, outValues, outIndices,
              i32Attr(mIter), i32Attr(32), i64Attr(tileA), i64Attr(tileB),
              boolAttr(false), boolAttr(!needsRebuild),
              boolAttr(readFromOutput));

          if (needsRebuild) {
            rewriter.create<TileTopkRebuildOp>(
                loc, inputValues, bufIdxFilled, outValues, outIndices,
                i32Attr(0), i32Attr(mIter), i32Attr(k), i32Attr(5), i32Attr(1),
                i64Attr(tileA), i64Attr(tileB), boolAttr(false), boolAttr(true),
                boolAttr(readFromOutput));
          }
        }
      }
    }

    rewriter.replaceOp(op, {outValues, outIndices});
    return success();
  }
};

struct D2MDecomposeTopk : public impl::D2MDecomposeTopkBase<D2MDecomposeTopk> {
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
