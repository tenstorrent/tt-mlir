// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/TopKUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

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

struct TopkIndexBuffers {
  Value idxBuf;
  Value laneBuf;
};

static std::optional<TopkIndexBuffers> findIndexBuffers(GenericOp genericOp) {
  TopkIndexBuffers buffers;
  genericOp.getRegion(0).walk([&](memref::AllocOp allocOp) {
    if (allocOp->hasAttr(utils::kTopkIndexBufferAttr)) {
      buffers.idxBuf = allocOp.getResult();
    } else if (allocOp->hasAttr(utils::kTopkLaneBufferAttr)) {
      buffers.laneBuf = allocOp.getResult();
    }
  });
  if (!buffers.idxBuf || !buffers.laneBuf) {
    return std::nullopt;
  }
  return buffers;
}

static void eraseScratchAnchors(PatternRewriter &rewriter, GenericOp genericOp,
                                const TopkIndexBuffers &buffers) {
  SmallVector<ScratchInitOp> anchors;
  genericOp.getRegion(0).walk([&](ScratchInitOp initOp) {
    if (initOp.getScratch() == buffers.idxBuf ||
        initOp.getScratch() == buffers.laneBuf) {
      anchors.push_back(initOp);
    }
  });
  for (ScratchInitOp initOp : anchors) {
    rewriter.eraseOp(initOp);
  }
}

// Builds this core's index buffer: tile t, element (i, j) holds t * 32 + i
// (row index i plus this core's band offset from core_index(dim)), constant
// along non-target axis j.
static Value buildIndexBuffer(PatternRewriter &rewriter, Location loc,
                              TopkBlockOp op, MemRefType inputType,
                              int64_t dimIdx, int64_t numTilesInner,
                              Value idxBuf, Value laneBuf) {
  MLIRContext *ctx = rewriter.getContext();
  ArrayRef<int64_t> shape = inputType.getShape();
  TT_assertv(shape.size() == 2ul,
             "in-kernel index generation expects a 2D shard");

  auto idxTileType = mlir::cast<ttcore::TileType>(
      mlir::cast<MemRefType>(op.getOutIndices().getType()).getElementType());
  // emitLeafTopk always picks an integer index type; arith wants it signless.
  Type scalarType = IntegerType::get(
      ctx, mlir::cast<IntegerType>(idxTileType.getElementType()).getWidth());

  auto idxVal = [&](int64_t v) -> Value {
    return rewriter.create<arith::ConstantIndexOp>(loc, v);
  };
  auto coreIndex = [&](int64_t dim) -> Value {
    return rewriter.create<CoreIndexOp>(loc, dim);
  };
  Value zeroIdx = idxVal(0);
  Value oneIdx = idxVal(1);
  Value const32Idx = idxVal(32);

  // Written once; every tile derives its values from this one lane pattern. A
  // column-major arange puts i in column 0, which a Col bcast then replicates.
  rewriter.create<ArangeBlockOp>(loc, op.getScratchIdxTile(), laneBuf,
                                 /*numElements=*/32, /*start=*/0, /*step=*/1,
                                 /*colMajor=*/true);
  // The loop below reads this back out of L1, so the pack must land first.
  rewriter.create<UnpackStallOnPackOp>(loc);

  // arange_block folds this core's grid position into every element; the
  // per-tile offset below subtracts it back out.
  auto genericOp = op->getParentOfType<GenericOp>();
  TT_assertv(genericOp, "topk_block must be inside a generic");
  ArrayRef<int64_t> gridShape = genericOp.getGrid().getShape();
  int64_t totalTileRows =
      utils::kTopkLaneTileRows * gridShape[gridShape.size() - 2];
  Value arangeBase = rewriter.create<arith::AddIOp>(
      loc,
      rewriter.create<arith::MulIOp>(loc, coreIndex(0),
                                     idxVal(utils::kTopkLaneTileRows * 32)),
      rewriter.create<arith::MulIOp>(loc, coreIndex(1),
                                     idxVal(totalTileRows * 32 * 32)));

  // Tiles this core's band starts at, in whole tiles of the reduction dim.
  Value bandOffset = rewriter.create<arith::MulIOp>(loc, coreIndex(dimIdx),
                                                    idxVal(numTilesInner));

  // The reduction dim is iterated innermost so the compute-root tag lands on a
  // loop that can never fold away for a single trip: topk_block merges
  // reduction tiles pairwise, so a shard always spans at least two.
  bool reductionIsRow = (dimIdx == 0);
  int64_t outerTiles = reductionIsRow ? shape[1] : shape[0];
  int64_t innerTiles = reductionIsRow ? shape[0] : shape[1];

  auto outerLoop =
      rewriter.create<scf::ForOp>(loc, zeroIdx, idxVal(outerTiles), oneIdx);
  rewriter.setInsertionPointToStart(outerLoop.getBody());
  auto innerLoop =
      rewriter.create<scf::ForOp>(loc, zeroIdx, idxVal(innerTiles), oneIdx);
  // Tagged here rather than by linalg-to-affine or d2m-op-scheduler, neither of
  // which processes a directly emitted scf.for; DST syncs go in the inner body.
  innerLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
  innerLoop->setAttr("d2m.scheduled", rewriter.getUnitAttr());
  rewriter.setInsertionPointToStart(innerLoop.getBody());

  Value outerIdx = outerLoop.getInductionVar();
  Value reductionTile = innerLoop.getInductionVar();
  Value rowIdx = reductionIsRow ? reductionTile : outerIdx;
  Value colIdx = reductionIsRow ? outerIdx : reductionTile;

  Value laneTile = rewriter.create<memref::LoadOp>(
      loc, laneBuf, ValueRange{zeroIdx, zeroIdx});
  Value rowTile =
      rewriter
          .create<TileBcastOp>(loc, idxTileType, laneTile, TileBcastType::Col)
          .getResult();

  Value tileOffsetIdx = rewriter.create<arith::SubIOp>(
      loc,
      rewriter.create<arith::MulIOp>(
          loc, rewriter.create<arith::AddIOp>(loc, bandOffset, reductionTile),
          const32Idx),
      arangeBase);
  Value tileOffset =
      rewriter.create<arith::IndexCastOp>(loc, scalarType, tileOffsetIdx);
  Value indexTile =
      rewriter.create<TileAddOp>(loc, idxTileType, rowTile, tileOffset)
          .getResult();

  rewriter.create<memref::StoreOp>(loc, indexTile, idxBuf,
                                   ValueRange{rowIdx, colIdx});

  rewriter.setInsertionPointAfter(outerLoop);
  // The merge tree unpacks these tiles straight out of L1, so the packer's
  // writes must land before the first copy_tile reads them.
  rewriter.create<UnpackStallOnPackOp>(loc);

  return idxBuf;
}

// Decomposes TopkBlockOp into tile_topk_{local_sort,merge,rebuild} ops with
// scf.for loops over tile pairs.
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

    int32_t k = op.getK();
    int32_t logk = floorLog2(k);

    // When k>32 the result spans 2 tiles. The large-k path left-folds over
    // the reduction tiles, so it handles any tile count without padding.
    bool useLargeK = (k > 32);
    int64_t dimIdx = op.getDim();
    int64_t numTilesInner = inputShape[dimIdx];

    // A leaf builds its own index buffer; a merge stage is handed the real
    // indices its children produced.
    Value bufIdxFilled = scratchIdxTile;
    if (op.getGenerateIndices()) {
      auto parentGeneric = op->getParentOfType<GenericOp>();
      TT_assertv(parentGeneric, "topk_block must be inside a generic");
      std::optional<TopkIndexBuffers> buffers = findIndexBuffers(parentGeneric);
      if (!buffers) {
        return failure();
      }
      // d2m-lower-scratch-allocate reads a leftover anchor as the spill pool.
      eraseScratchAnchors(rewriter, parentGeneric, *buffers);
      bufIdxFilled =
          buildIndexBuffer(rewriter, loc, op, inputType, dimIdx, numTilesInner,
                           buffers->idxBuf, buffers->laneBuf);
    }

    // logWt is the merge-tree depth; ceilLog2 ensures the final fold always
    // runs for non-power-of-2 tile counts.
    bool numTilesPow2 =
        (numTilesInner > 0 && (numTilesInner & (numTilesInner - 1)) == 0);
    int32_t fl = floorLog2(numTilesInner);
    int32_t logWt = numTilesPow2 ? fl : fl + 1;
    bool ragged = !numTilesPow2;

    // Shard is row-major [htShard, wtShard] with flat index r*reductionStride +
    // nt*ntStride; strides depend on which dim is the reduction dim.
    int64_t ntDimIdx = (dimIdx == static_cast<int64_t>(inputShape.size()) - 1)
                           ? static_cast<int64_t>(inputShape.size()) - 2
                           : static_cast<int64_t>(inputShape.size()) - 1;
    int64_t nonTargetCount = inputShape[ntDimIdx];
    int64_t reductionStride =
        (dimIdx == static_cast<int64_t>(inputShape.size()) - 1)
            ? 1
            : nonTargetCount;
    int64_t ntStride = (dimIdx == static_cast<int64_t>(inputShape.size()) - 1)
                           ? numTilesInner
                           : 1;

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
          loc, inputValues, bufIdxFilled, outValues, outIndices,
          /*idir=*/i32Attr(0), /*i_end_phase=*/i32Attr(sortEndPhase),
          /*i_start_phase=*/sortStartPhase, tA, tB, boolAttr(true),
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

    if (useLargeK) {
      // Left-fold: keeps a running 2-tile accumulator, with the winner always
      // in tile 0 and the loser always in tile 1, so extraction never needs
      // to track which tile is which. This works for any tile count without
      // power-of-2 padding, at ~2 sort-merge-rebuilds per tile: folding in a
      // complete pair takes a 3-sort-merge-rebuild, folding in a lone tail tile
      // takes a 2-sort-merge-rebuild.
      Value accWin = zeroIdx;
      Value accLos = oneIdx;

      // Build the initial accumulator from reduction tiles 0 and 1.
      emitSortMergeRebuild(flat(accWin), flat(accLos), /*mergeK=*/k,
                           /*rebuildK=*/k, logk, /*sortStartPhase=*/zeroI32,
                           /*sortEndPhase=*/logk - 1, /*mergeIter=*/zeroI32,
                           /*skipSecond=*/0, /*rfo=*/falseVal);

      // Fold in every COMPLETE right pair (t, t+1) for t = 2, 4, ... The upper
      // bound excludes an odd trailing tile, which is handled after the loop.
      int64_t completeUB = numTilesInner - (numTilesInner % 2);
      if (completeUB > 2) {
        auto foldLoop = rewriter.create<scf::ForOp>(
            loc, idxVal(2), idxVal(completeUB), idxVal(2));
        rewriter.setInsertionPointToStart(foldLoop.getBody());
        Value tIdx = foldLoop.getInductionVar();
        Value tPlus1 = rewriter.create<arith::AddIOp>(loc, tIdx, oneIdx);

        // Build the complete right block q = (t, t+1) from raw input.
        emitSortMergeRebuild(flat(tIdx), flat(tPlus1), /*mergeK=*/k,
                             /*rebuildK=*/k, logk, /*sortStartPhase=*/zeroI32,
                             /*sortEndPhase=*/logk - 1, /*mergeIter=*/zeroI32,
                             /*skipSecond=*/0, /*rfo=*/falseVal);

        // 3-sub-merge combine of acc=(0, 1) with q=(t, t+1). All operands were
        // previously packed, so rfo=true.
        // Step 1: winners (0, t) -> tile 0 holds the global top-k; t holds the
        // losers of the winner tiles.
        emitSortMergeRebuild(flat(accWin), flat(tIdx), /*mergeK=*/k,
                             /*rebuildK=*/k, logk, /*sortStartPhase=*/zeroI32,
                             /*sortEndPhase=*/logk - 1, /*mergeIter=*/zeroI32,
                             /*skipSecond=*/0, /*rfo=*/trueVal);
        // Step 2: losers (1, t+1) -> tile 1 holds the best of the loser tiles.
        emitSortMergeRebuild(flat(accLos), flat(tPlus1), /*mergeK=*/k,
                             /*rebuildK=*/k, logk, /*sortStartPhase=*/zeroI32,
                             /*sortEndPhase=*/logk - 1, /*mergeIter=*/zeroI32,
                             /*skipSecond=*/0, /*rfo=*/trueVal);
        // Step 3: (1, t) -> tile 1 holds the final rank-(k/2..k). Writing the
        // winner back to tile 1 (not t) keeps the accumulator canonical at
        // (0, 1).
        emitSortMergeRebuild(flat(accLos), flat(tIdx), /*mergeK=*/k,
                             /*rebuildK=*/k, logk, /*sortStartPhase=*/zeroI32,
                             /*sortEndPhase=*/logk - 1, /*mergeIter=*/zeroI32,
                             /*skipSecond=*/0, /*rfo=*/trueVal);

        rewriter.setInsertionPointAfter(foldLoop);
      }

      // Odd tile count: fold in the lone trailing tile with a winner-only
      // 2-sub-merge. This keeps the accumulator canonical at (0, 1).
      if (numTilesInner % 2 == 1) {
        Value tailIdx = idxVal(numTilesInner - 1);
        // Prime the raw tail tile into a valid sorted run (sorts one tile;
        // tileA==tileB is harmless).
        rewriter.create<TileTopkLocalSortOp>(
            loc, inputValues, bufIdxFilled, outValues, outIndices,
            /*idir=*/i32Attr(0), /*i_end_phase=*/i32Attr(4),
            /*i_start_phase=*/zeroI32, /*tileA=*/flat(tailIdx),
            /*tileB=*/flat(tailIdx), /*is_group_start=*/boolAttr(true),
            /*is_group_end=*/i1Val(true), /*rfo=*/falseVal);
        // Step A: winners (0, tail) -> tile 0 holds the global top-k.
        emitSortMergeRebuild(flat(accWin), flat(tailIdx), /*mergeK=*/k,
                             /*rebuildK=*/k, logk, /*sortStartPhase=*/zeroI32,
                             /*sortEndPhase=*/logk - 1, /*mergeIter=*/zeroI32,
                             /*skipSecond=*/0, /*rfo=*/trueVal);
        // Step B: (1, tail) -> tile 1 holds the final rank-(k/2..k).
        emitSortMergeRebuild(flat(accLos), flat(tailIdx), /*mergeK=*/k,
                             /*rebuildK=*/k, logk, /*sortStartPhase=*/zeroI32,
                             /*sortEndPhase=*/logk - 1, /*mergeIter=*/zeroI32,
                             /*skipSecond=*/0, /*rfo=*/trueVal);
      }
    } else {
      // Level 0 is emitted unrolled, levels [1,logWt-1) run in a middle
      // scf.for loop (if logWt > 1), and level logWt-1 is emitted unrolled
      // (if logWt > 1). Each level's behavior is determined by the isLevelZero
      // and isLastLevel parameters.
      bool tailSortNeeded = ragged && (numTilesInner % 2 == 1);

      auto emitLevelBody = [&](Value mIterIdx, bool isLevelZero,
                               bool isLastLevel) {
        Value mIterI32 = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI32Type(), mIterIdx);

        Value distanceIdx =
            rewriter.create<arith::ShLIOp>(loc, oneIdx, mIterIdx);
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

        // K=32/logk=5 ensures sorting spans both tiles for cross-tile merges.
        Value readFromOutput = isLevelZero ? falseVal : trueVal;

        // Rebuild only on the last level. Skip it when k==32 with a single
        // level since the merge output is already exactly k elements.
        bool needsRebuild = isLastLevel && ((k < 32) || !isLevelZero);

        // On the ragged path, always use sortStartPhase=0 since carried
        // tiles may have skipped levels and need a full sort.
        Value sortStartPhase = ragged ? i32Val(0) : mIterI32;

        rewriter.create<TileTopkLocalSortOp>(
            loc, inputValues, bufIdxFilled, outValues, outIndices,
            /*idir=*/i32Attr(0), /*i_end_phase=*/i32Attr(4),
            /*i_start_phase=*/sortStartPhase, tileA, tileB, boolAttr(true),
            i1Val(false), /*rfo=*/readFromOutput);
        rewriter.create<TileTopkMergeOp>(
            loc, inputValues, bufIdxFilled, outValues, outIndices,
            /*mergeIter=*/mIterI32, i32Attr(32), tileA, tileB, boolAttr(false),
            /*is_group_end=*/i1Val(!needsRebuild), /*rfo=*/readFromOutput);

        if (needsRebuild) {
          rewriter.create<TileTopkRebuildOp>(
              loc, inputValues, bufIdxFilled, outValues, outIndices, i32Attr(0),
              /*mergeIter=*/mIterI32, i32Attr(k), i32Attr(5), i32Attr(1), tileA,
              tileB, boolAttr(false), /*is_group_end=*/trueVal,
              /*rfo=*/readFromOutput);
        }

        // For ragged N with an odd tile count, tile (N-1) is skipped by the
        // even-indexed level-0 loop. Emit a standalone local_sort for it at
        // level 0 using tileB=tileA (sorts the same tile twice, harmlessly)
        // so it is a valid sorted run before level 1 tries to pair it.
        if (isLevelZero && tailSortNeeded) {
          Value tailRawIdx = idxVal(numTilesInner - 1);
          Value tailTileIdx = flat(tailRawIdx);
          rewriter.create<TileTopkLocalSortOp>(
              loc, inputValues, bufIdxFilled, outValues, outIndices,
              /*idir=*/i32Attr(0), /*i_end_phase=*/i32Attr(4),
              /*i_start_phase=*/i32Val(0), /*tileA=*/tailTileIdx,
              /*tileB=*/tailTileIdx, /*is_group_start=*/boolAttr(true),
              /*is_group_end=*/i1Val(true), /*rfo=*/falseVal);
        }

        rewriter.setInsertionPointAfter(innerLoop);
      };

      // Level 0: exactly one iteration, run unrolled.
      emitLevelBody(zeroIdx, /*isLevelZero=*/true,
                    /*isLastLevel=*/(logWt == 1));

      if (logWt > 1) {
        // Middle levels [1, logWt-1): no rebuild, no tail-sort.
        auto midLoop =
            rewriter.create<scf::ForOp>(loc, oneIdx, idxVal(logWt - 1), oneIdx);
        rewriter.setInsertionPointToStart(midLoop.getBody());
        emitLevelBody(midLoop.getInductionVar(), /*isLevelZero=*/false,
                      /*isLastLevel=*/false);
        rewriter.setInsertionPointAfter(midLoop);

        // Last level: exactly one iteration, run unrolled.
        emitLevelBody(idxVal(logWt - 1), /*isLevelZero=*/false,
                      /*isLastLevel=*/true);
      }
    }

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
