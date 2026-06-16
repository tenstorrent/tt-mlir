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

// Decomposes TopkBlockOp into arange_block +
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

    int32_t logk = 0;
    {
      int32_t tmp = k;
      while (tmp > 1) {
        tmp >>= 1;
        ++logk;
      }
    }

    int64_t numTilesInner = inputShape[dim];
    int32_t logWt = 0;
    {
      int64_t tmp = numTilesInner;
      while (tmp > 1) {
        tmp >>= 1;
        ++logWt;
      }
    }

    bool useLargeK =
        (k > 32); // k>32: result spans 2 tiles, use 3-sub-merge tree reduction.

    auto i32Attr = [&](int32_t v) { return rewriter.getI32IntegerAttr(v); };
    auto i64Attr = [&](int64_t v) { return rewriter.getI64IntegerAttr(v); };
    auto boolAttr = [&](bool v) { return rewriter.getBoolAttr(v); };

    // Emit local_sort + merge + rebuild for an initial pair.
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

    auto emitSortMerge = [&](int64_t sA, int64_t sB, int64_t numPhase,
                             int64_t mK, bool tmp, bool rfo) {
      rewriter.create<TileTopkLocalSortOp>(
          loc, inputValues, bufIdxFilled, outValues, outIndices, i32Attr(0),
          i32Attr(numPhase), i32Attr(0), i64Attr(sA), i64Attr(sB),
          boolAttr(true), boolAttr(false), boolAttr(rfo));
      rewriter.create<TileTopkMergeOp>(
          loc, inputValues, bufIdxFilled, outValues, outIndices, i32Attr(0),
          i32Attr(mK), i64Attr(sA), i64Attr(sB), boolAttr(false), boolAttr(tmp),
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
            emitSortMergeRebuild(tileA, tileB, k, logk, /*mIter=*/0,
                                 /*skipSecond=*/0, /*rfo=*/false);
          } else {
            // 3-sub-merge: top-k of 4 tiles within the 2-tile DST constraint
            int64_t prevDist = distance / 2;

            emitSortMerge(tileA, tileB, logk, k, false, true);
            rewriter.create<TileTopkRebuildOp>(
                loc, inputValues, bufIdxFilled, outValues, outIndices,
                i32Attr(0), i32Attr(0), i32Attr(k), i32Attr(logk), i32Attr(0),
                i64Attr(tileA), i64Attr(tileB), boolAttr(false), boolAttr(true),
                boolAttr(true));

            emitSortMerge(tileA + prevDist, tileB + prevDist, logk, k, false,
                          true);
            rewriter.create<TileTopkRebuildOp>(
                loc, inputValues, bufIdxFilled, outValues, outIndices,
                i32Attr(0), i32Attr(0), i32Attr(k), i32Attr(logk), i32Attr(0),
                i64Attr(tileA + prevDist), i64Attr(tileB + prevDist),
                boolAttr(false), boolAttr(true), boolAttr(true));

            emitSortMerge(tileB, tileA + prevDist, logk, k, false, true);
            rewriter.create<TileTopkRebuildOp>(
                loc, inputValues, bufIdxFilled, outValues, outIndices,
                i32Attr(0), i32Attr(0), i32Attr(k), i32Attr(logk), i32Attr(0),
                i64Attr(tileB), i64Attr(tileA + prevDist), boolAttr(false),
                boolAttr(true), boolAttr(true));
          }
        } else {
          // k<=32: always use K=32/logk=5 to force cross-tile work.
          bool needsRebuild = isLast && (!isFirst || k < 32);

          emitSortMerge(tileA, tileB, 4, 32, !needsRebuild, readFromOutput);

          if (needsRebuild) {
            rewriter.create<TileTopkRebuildOp>(
                loc, inputValues, bufIdxFilled, outValues, outIndices,
                i32Attr(0), i32Attr(mIter), i32Attr(32), i32Attr(5), i32Attr(1),
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
