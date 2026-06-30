// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// Decomposes TopkBlockOp into arange_block +
// tile_topk_{local_sort,merge,rebuild} ops with scf.for loops over tile pairs.
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

    // When k>32 the result spans 2 tiles, requiring a 3-sub-merge reduction.
    // The reduction dim is padded to a power-of-2 tile count in TTIRToD2M
    // (-inf fill), so the large-k path always sees a power-of-2 tile count.
    bool useLargeK = (k > 32);
    int64_t dimIdx = op.getDim();
    int64_t numTilesInner = inputShape[dimIdx];
    // logWt is the merge-tree depth; ceilLog2 ensures the final fold always
    // runs for non-power-of-2 tile counts.
    bool numTilesPow2 =
        (numTilesInner > 0 && (numTilesInner & (numTilesInner - 1)) == 0);
    int32_t fl = floorLog2(numTilesInner);
    int32_t logWt = numTilesPow2 ? fl : fl + 1;
    bool ragged = !numTilesPow2;

    // The shard is row-major [htShard, wtShard]; flat(nt, r) =
    // r*reductionStride
    // + nt*ntStride, where strides depend on which dim is the reduction dim.
    int64_t ntDimIdx = (dimIdx == (int64_t)inputShape.size() - 1)
                           ? (int64_t)inputShape.size() - 2
                           : (int64_t)inputShape.size() - 1;
    int64_t nonTargetCount = inputShape[ntDimIdx];
    int64_t reductionStride =
        (dimIdx == (int64_t)inputShape.size() - 1) ? 1 : nonTargetCount;
    int64_t ntStride =
        (dimIdx == (int64_t)inputShape.size() - 1) ? numTilesInner : 1;

    auto i32Attr = [&](int32_t v) { return rewriter.getI32IntegerAttr(v); };
    auto boolAttr = [&](bool v) { return rewriter.getBoolAttr(v); };

    // Helper to create index-typed SSA constants.
    auto idxVal = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, v);
    };
    // Helper to create i32-typed SSA constants.
    auto i32Val = [&](int32_t v) -> Value {
      return rewriter.create<arith::ConstantOp>(loc,
                                                rewriter.getI32IntegerAttr(v));
    };
    // Helper to create i1-typed SSA constants.
    auto i1Val = [&](bool v) -> Value {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(rewriter.getI1Type(), v ? 1 : 0));
    };

    Value zeroIdx = idxVal(0);
    Value oneIdx = idxVal(1);
    Value zeroI32 = i32Val(0);
    Value trueVal = i1Val(true);
    Value falseVal = i1Val(false);
    Value numTilesIdx = idxVal(numTilesInner);
    Value logWtIdx = idxVal(logWt);

    Value reductionStrideIdx = idxVal(reductionStride);
    Value ntStrideIdx = idxVal(ntStride);

    // Each non-target row runs an independent merge tree with its tile indices
    // offset by ntOffset.
    auto ntLoop = rewriter.create<scf::ForOp>(loc, zeroIdx,
                                              idxVal(nonTargetCount), oneIdx);
    rewriter.setInsertionPointToStart(ntLoop.getBody());
    Value ntIdxVar = ntLoop.getInductionVar();
    Value ntOffset = rewriter.create<arith::MulIOp>(loc, ntIdxVar, ntStrideIdx);

    auto flat = [&](Value r) -> Value {
      Value scaled = rewriter.create<arith::MulIOp>(loc, r, reductionStrideIdx);
      return rewriter.create<arith::AddIOp>(loc, scaled, ntOffset);
    };

    // Emit local_sort + merge + rebuild for the large-k path. The rebuild
    // always runs here (is_group_end=true), so the merge never packs
    // (is_group_end=false).
    auto emitSortMergeRebuild = [&](Value tA, Value tB, int32_t mergeK,
                                    int32_t rebuildK, int32_t mLogk,
                                    Value sortStartPhase, int32_t sortEndPhase,
                                    Value mergeIter, int32_t skipSecond,
                                    Value rfo) {
      rewriter.create<TileTopkLocalSortOp>(
          loc, inputValues, bufIdxFilled, outValues, outIndices, sortStartPhase,
          i32Attr(sortEndPhase), i32Attr(0), tA, tB, boolAttr(true),
          i1Val(false), rfo);
      rewriter.create<TileTopkMergeOp>(loc, inputValues, bufIdxFilled,
                                       outValues, outIndices, mergeIter,
                                       i32Attr(mergeK), tA, tB, boolAttr(false),
                                       /*is_group_end=*/i1Val(false), rfo);
      rewriter.create<TileTopkRebuildOp>(
          loc, inputValues, bufIdxFilled, outValues, outIndices, i32Attr(0),
          mergeIter, i32Attr(rebuildK), i32Attr(mLogk), i32Attr(skipSecond), tA,
          tB, boolAttr(false), /*is_group_end=*/i1Val(true), rfo);
    };

    auto outerLoop =
        rewriter.create<scf::ForOp>(loc, zeroIdx, logWtIdx, oneIdx);
    rewriter.setInsertionPointToStart(outerLoop.getBody());

    Value mIterIdx = outerLoop.getInductionVar();
    Value mIterI32 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), mIterIdx);

    Value distanceIdx = rewriter.create<arith::ShLIOp>(loc, oneIdx, mIterIdx);
    Value innerUB =
        rewriter.create<arith::SubIOp>(loc, numTilesIdx, distanceIdx);
    Value innerStep =
        rewriter.create<arith::MulIOp>(loc, distanceIdx, idxVal(2));
    auto innerLoop =
        rewriter.create<scf::ForOp>(loc, zeroIdx, innerUB, innerStep);
    rewriter.setInsertionPointToStart(innerLoop.getBody());

    // rA/rB are raw reduction indices; tileA/tileB are flat after adding
    // ntOffset.
    Value baseIdx = innerLoop.getInductionVar();
    Value rA = baseIdx;
    Value rB = rewriter.create<arith::AddIOp>(loc, baseIdx, distanceIdx);
    Value tileA = flat(rA);
    Value tileB = flat(rB);

    if (useLargeK) {
      // The first iteration performs a single sort-merge-rebuild. In later
      // iterations, DST only holds 2 tiles at a time, so we cannot merge all 4
      // tiles directly; instead, we use 3 sub-merges to compute the top-k.
      Value isFirst = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, mIterIdx, zeroIdx);
      auto ifOp = rewriter.create<scf::IfOp>(loc, isFirst,
                                             /*withElseRegion=*/true);

      // Iteration 0: emit a single sort-merge-rebuild.
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      emitSortMergeRebuild(tileA, tileB, /*mergeK=*/k, /*rebuildK=*/k, logk,
                           /*sortStartPhase=*/zeroI32,
                           /*sortEndPhase=*/logk - 1, /*mergeIter=*/zeroI32,
                           /*skipSecond=*/0, /*rfo=*/falseVal);

      // Iterations > 0: emit a 3-sub-merge tree.
      rewriter.setInsertionPointToStart(ifOp.elseBlock());
      // prevDist is the stride from the previous level: distance / 2.
      Value prevDistIdx =
          rewriter.create<arith::ShRUIOp>(loc, distanceIdx, oneIdx);

      // Loser indices derive from rA/rB to avoid double-applying the ntOffset.
      Value rALoser = rewriter.create<arith::AddIOp>(loc, rA, prevDistIdx);
      Value rBLoser = rewriter.create<arith::AddIOp>(loc, rB, prevDistIdx);
      Value tileALoser = flat(rALoser);
      Value tileBLoser = flat(rBLoser);

      // Step 1: Merge the winner tiles (tileA, tileB) from the previous
      // iteration; tileA holds the best-k across both afterward.
      emitSortMergeRebuild(tileA, tileB, /*mergeK=*/k, /*rebuildK=*/k, logk,
                           /*sortStartPhase=*/zeroI32,
                           /*sortEndPhase=*/logk - 1, /*mergeIter=*/zeroI32,
                           /*skipSecond=*/0, /*rfo=*/trueVal);

      // Step 2: Merge the loser tiles; (tileALoser) holds the best-k
      // across both losers afterward.
      emitSortMergeRebuild(tileALoser, tileBLoser, /*mergeK=*/k, /*rebuildK=*/k,
                           logk, /*sortStartPhase=*/zeroI32,
                           /*sortEndPhase=*/logk - 1, /*mergeIter=*/zeroI32,
                           /*skipSecond=*/0, /*rfo=*/trueVal);

      // Step 3: Merge winners against losers; tileB holds the final top-k
      // across all four tiles afterward.
      emitSortMergeRebuild(tileB, tileALoser, /*mergeK=*/k, /*rebuildK=*/k,
                           logk, /*sortStartPhase=*/zeroI32,
                           /*sortEndPhase=*/logk - 1, /*mergeIter=*/zeroI32,
                           /*skipSecond=*/0, /*rfo=*/trueVal);
    } else {
      // K=32/logk=5 ensures sorting spans both tiles for cross-tile merges.
      Value readFromOutput = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, mIterIdx, zeroIdx);
      Value lastIterIdx = idxVal(logWt - 1);
      Value isLast = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, mIterIdx, lastIterIdx);

      // Rebuild only on the last iteration. Skip it when k==32 with a
      // single iteration since the merge output is already exactly k
      // elements. The !isFirst check handles the case when logWt = 1.
      Value needsRebuild =
          (k < 32) ? isLast
                   : rewriter.create<arith::AndIOp>(loc, isLast, readFromOutput)
                         .getResult();

      // The merge packs the tiles only when no rebuild follows, so its
      // is_group_end is the complement of needsRebuild.
      Value mergeGroupEnd =
          rewriter.create<arith::XOrIOp>(loc, needsRebuild, trueVal);

      // For ragged N, tileB may be out of bounds at the last level, so guard
      // the sort+merge+rebuild with a bounds check on rB (not the flat tileB).
      // On the ragged path, always use sortStartPhase=0 since carried tiles may
      // have skipped levels and need a full sort.
      Value sortStartPhase = ragged ? i32Val(0) : mIterI32;

      auto emitSortMergeRebuildSmallK = [&]() {
        rewriter.create<TileTopkLocalSortOp>(
            loc, inputValues, bufIdxFilled, outValues, outIndices,
            /*sortStartPhase=*/sortStartPhase, /*sortEndPhase=*/i32Attr(4),
            i32Attr(0), tileA, tileB, boolAttr(true), i1Val(false),
            /*rfo=*/readFromOutput);
        rewriter.create<TileTopkMergeOp>(
            loc, inputValues, bufIdxFilled, outValues, outIndices,
            /*mergeIter=*/mIterI32, i32Attr(32), tileA, tileB, boolAttr(false),
            mergeGroupEnd, /*rfo=*/readFromOutput);

        // Rebuild runs only on the last iteration; k==32 with one iteration
        // skips it since the merge already produces exactly k elements.
        auto rebuildIf = rewriter.create<scf::IfOp>(loc, needsRebuild,
                                                    /*withElseRegion=*/false);
        rewriter.setInsertionPointToStart(rebuildIf.thenBlock());
        rewriter.create<TileTopkRebuildOp>(
            loc, inputValues, bufIdxFilled, outValues, outIndices, i32Attr(0),
            /*mergeIter=*/mIterI32, i32Attr(k), i32Attr(5), i32Attr(1), tileA,
            tileB, boolAttr(false), /*is_group_end=*/trueVal,
            /*rfo=*/readFromOutput);
        rewriter.setInsertionPointAfter(rebuildIf);
      };

      if (ragged) {
        // Guard on rB (the raw reduction index) rather than the flat tile
        // index.
        Value rBInBounds = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, rB, numTilesIdx);
        auto mergeIf = rewriter.create<scf::IfOp>(loc, rBInBounds,
                                                  /*withElseRegion=*/false);
        rewriter.setInsertionPointToStart(mergeIf.thenBlock());
        emitSortMergeRebuildSmallK();
        rewriter.setInsertionPointAfter(mergeIf);
      } else {
        emitSortMergeRebuildSmallK();
      }
    }

    rewriter.setInsertionPointAfter(innerLoop);

    // For ragged N with an odd tile count, tile (N-1) is skipped by the
    // even-indexed level-0 loop. Emit a standalone local_sort for it at level 0
    // using tileB=tileA (sorts the same tile twice, harmlessly) so it is a
    // valid sorted run before level 1 tries to pair it.
    if (ragged && (numTilesInner % 2 == 1) && !useLargeK) {
      Value tailRawIdx = idxVal(numTilesInner - 1);
      Value tailTileIdx = flat(tailRawIdx);
      Value isLevelZero = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, mIterIdx, zeroIdx);
      auto tailSortIf = rewriter.create<scf::IfOp>(loc, isLevelZero,
                                                   /*withElseRegion=*/false);
      rewriter.setInsertionPointToStart(tailSortIf.thenBlock());
      rewriter.create<TileTopkLocalSortOp>(
          loc, inputValues, bufIdxFilled, outValues, outIndices,
          /*sortStartPhase=*/i32Val(0), /*sortEndPhase=*/i32Attr(4),
          /*idir=*/i32Attr(0), /*tileA=*/tailTileIdx, /*tileB=*/tailTileIdx,
          /*is_group_start=*/boolAttr(true), /*is_group_end=*/i1Val(true),
          /*rfo=*/falseVal);
      rewriter.setInsertionPointAfter(tailSortIf);
    }

    rewriter.setInsertionPointAfter(outerLoop);
    rewriter.setInsertionPointAfter(ntLoop);

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
