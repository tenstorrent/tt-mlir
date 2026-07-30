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
#define GEN_PASS_DEF_D2MDECOMPOSESORT
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Full bitonic merge sort over the reduction tiles, built from the two topk
// LLKs that happen to be exactly the primitives a sorting network needs:
//
//   tile_topk_local_sort  sorts the 64 values spanning a tile pair, in a given
//                         direction (phases 0..5 = a complete 64-element sort)
//   tile_topk_merge       with k=64 degenerates to an elementwise
//                         compare-exchange between the two DST tiles, writing
//                         the loser to DST0 and the winner to DST1
//
// This mirrors tt-metal's own ttnn::sort compute kernel
// (sort_single_row_single_core.cpp); notably neither uses tile_topk_rebuild.
struct DecomposeSortBlockPattern : OpRewritePattern<SortBlockOp> {
  using OpRewritePattern<SortBlockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SortBlockOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value inputValues = op.getInputValues();
    Value bufIdxFilled = op.getScratchIdxTile();
    Value outValues = op.getOutValues();
    Value outIndices = op.getOutIndices();

    auto inputType = dyn_cast<MemRefType>(inputValues.getType());
    TT_assertv(inputType, "input must be a memref, run after bufferization");

    ArrayRef<int64_t> inputShape = inputType.getShape();
    TT_assertv(inputShape.size() >= 2ul,
               "input must have at least 2 dimensions");

    int64_t rank = static_cast<int64_t>(inputShape.size());
    int64_t dimIdx = op.getDim();
    int64_t numTilesInner = inputShape[dimIdx];
    TT_assertv(
        (numTilesInner >= 2 && (numTilesInner & (numTilesInner - 1)) == 0),
        "sort_block reduction dim must be a power-of-two tile count "
        ">= 2; TTIRToD2M pads to guarantee this");

    // Shard is row-major [htShard, wtShard] with flat index r*reductionStride +
    // nt*ntStride; strides depend on which dim is the reduction dim.
    int64_t ntDimIdx = (dimIdx == rank - 1) ? rank - 2 : rank - 1;
    int64_t nonTargetCount = inputShape[ntDimIdx];
    int64_t reductionStride = (dimIdx == rank - 1) ? 1 : nonTargetCount;
    int64_t ntStride = (dimIdx == rank - 1) ? numTilesInner : 1;

    // topk_local_sort's idir is 1 for increasing, 0 for decreasing.
    bool ascending = !op.getDescending();

    // A complete 64-element sort across the tile pair.
    constexpr int32_t kFullSortEndPhase = 5;
    // k=64 makes topk_merge a plain compare-exchange of the two DST tiles.
    constexpr int32_t kMergeK = 64;

    int32_t stages = 0;
    for (int64_t i = numTilesInner; i > 1; i >>= 1) {
      ++stages;
    }

    auto i32Attr = [&](int32_t v) { return rewriter.getI32IntegerAttr(v); };
    auto boolAttr = [&](bool v) { return rewriter.getBoolAttr(v); };
    auto idxVal = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, v);
    };
    auto i32Val = [&](int32_t v) -> Value {
      return rewriter.create<arith::ConstantOp>(loc,
                                                rewriter.getI32IntegerAttr(v));
    };
    auto i1Val = [&](bool v) -> Value {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(rewriter.getI1Type(), v ? 1 : 0));
    };

    Value zeroIdx = idxVal(0);
    Value oneIdx = idxVal(1);
    Value zeroI32 = i32Val(0);
    Value oneI32 = i32Val(1);
    Value trueVal = i1Val(true);
    Value falseVal = i1Val(false);

    Value reductionStrideIdx = idxVal(reductionStride);
    Value ntStrideIdx = idxVal(ntStride);

    // Each non-target row is a fully independent sort.
    auto ntLoop = rewriter.create<scf::ForOp>(loc, zeroIdx,
                                              idxVal(nonTargetCount), oneIdx);
    rewriter.setInsertionPointToStart(ntLoop.getBody());
    Value ntOffset = rewriter.create<arith::MulIOp>(
        loc, ntLoop.getInductionVar(), ntStrideIdx);

    auto flat = [&](Value r) -> Value {
      Value scaled = rewriter.create<arith::MulIOp>(loc, r, reductionStrideIdx);
      return rewriter.create<arith::AddIOp>(loc, scaled, ntOffset);
    };

    // Bit twiddling is done in i32 rather than on the index-typed loop
    // variables: the arith->emitc conversion only lowers bitwise/shift ops
    // whose type is a true IntegerType, so index-typed andi/shrui fail to
    // legalize.
    auto toI32 = [&](Value v) -> Value {
      return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), v);
    };

    // idir is 1 for increasing, 0 for decreasing, and every direction in the
    // network is `ascending XOR bit` for some runtime bit. `ascending` is
    // compile-time, so the XOR folds to either `bit` or `1 - bit`, which keeps
    // this out of i1 arithmetic entirely.
    auto dirFromBit = [&](Value bit) -> Value {
      if (!ascending) {
        return bit;
      }
      return rewriter.create<arith::SubIOp>(loc, oneI32, bit);
    };

    // Every compare-exchange is its own DST group: acquire, copy the four
    // tiles in, run the LLK, pack the four tiles back, release.
    auto emitLocalSort = [&](Value tileA, Value tileB, Value idir, Value rfo) {
      rewriter.create<TileTopkLocalSortOp>(
          loc, inputValues, bufIdxFilled, outValues, outIndices, idir,
          /*i_end_phase=*/i32Attr(kFullSortEndPhase),
          /*i_start_phase=*/zeroI32, tileA, tileB,
          /*is_group_start=*/boolAttr(true), /*is_group_end=*/trueVal, rfo);
    };

    // ---- Phase A: build the initial bitonic sequence --------------------
    // Sort each adjacent pair into a 64-run, flipping direction between pairs
    // so neighbouring runs form bitonic sequences for the network below.
    {
      auto pairLoop = rewriter.create<scf::ForOp>(
          loc, zeroIdx, idxVal(numTilesInner), idxVal(2));
      rewriter.setInsertionPointToStart(pairLoop.getBody());
      Value p = pairLoop.getInductionVar();

      // idir = ascending XOR (pairIndex & 1): even pairs sort one way, odd
      // pairs the other, which is what makes neighbouring runs bitonic.
      Value pairIdx = rewriter.create<arith::ShRUIOp>(loc, toI32(p), oneI32);
      Value dirBit = rewriter.create<arith::AndIOp>(loc, pairIdx, oneI32);
      Value idir = dirFromBit(dirBit);

      emitLocalSort(flat(p),
                    flat(rewriter.create<arith::AddIOp>(loc, p, oneIdx)), idir,
                    /*rfo=*/falseVal);

      rewriter.setInsertionPointAfter(pairLoop);
    }

    // ---- Phase B: the bitonic merge network -----------------------------
    // `stage` and `sub` are compile-time (bounded by log2 of the tile count),
    // so only the tile walks become scf.for loops.
    for (int32_t stage = 2; stage <= stages; ++stage) {
      Value mIter = i32Val(stage - 1);
      Value stageShift = i32Val(stage);

      // Direction of the comparison block tile `i` belongs to, as an idir:
      //   bit = (i >> stage) & 1   (0 selects the ascending half of the block)
      //   dir = ascending XOR bit
      auto blockDir = [&](Value i) -> Value {
        Value shifted =
            rewriter.create<arith::ShRUIOp>(loc, toI32(i), stageShift);
        Value bit = rewriter.create<arith::AndIOp>(loc, shifted, oneI32);
        return dirFromBit(bit);
      };

      for (int32_t sub = stage; sub >= 2; --sub) {
        int64_t subDist = int64_t{1} << (sub - 1);
        Value subDistIdx = idxVal(subDist);

        auto blockLoop = rewriter.create<scf::ForOp>(
            loc, zeroIdx, idxVal(numTilesInner), idxVal(2 * subDist));
        rewriter.setInsertionPointToStart(blockLoop.getBody());
        Value blockBase = blockLoop.getInductionVar();

        auto offLoop =
            rewriter.create<scf::ForOp>(loc, zeroIdx, subDistIdx, oneIdx);
        rewriter.setInsertionPointToStart(offLoop.getBody());

        Value i = rewriter.create<arith::AddIOp>(loc, blockBase,
                                                 offLoop.getInductionVar());
        Value j = rewriter.create<arith::AddIOp>(loc, i, subDistIdx);
        Value tileI = flat(i);
        Value tileJ = flat(j);
        Value dir = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, blockDir(i), zeroI32);

        // The LLK writes the loser to DST0 and the winner to DST1, so an
        // ascending block is obtained by swapping the pack destinations
        // rather than by re-running the merge.
        Value storeA = rewriter.create<arith::SelectOp>(loc, dir, tileJ, tileI);
        Value storeB = rewriter.create<arith::SelectOp>(loc, dir, tileI, tileJ);

        rewriter.create<TileTopkMergeOp>(
            loc, inputValues, bufIdxFilled, outValues, outIndices, mIter,
            i32Attr(kMergeK), tileI, tileJ, storeA, storeB,
            /*is_group_start=*/boolAttr(true), /*is_group_end=*/trueVal,
            /*rfo=*/trueVal);

        rewriter.setInsertionPointAfter(blockLoop);
      }

      // sub == 1 pairs adjacent tiles, so a single full local sort finishes
      // the stage in one pass instead of five more compare-exchange steps.
      {
        auto pairLoop = rewriter.create<scf::ForOp>(
            loc, zeroIdx, idxVal(numTilesInner), idxVal(2));
        rewriter.setInsertionPointToStart(pairLoop.getBody());
        Value i = pairLoop.getInductionVar();
        emitLocalSort(flat(i),
                      flat(rewriter.create<arith::AddIOp>(loc, i, oneIdx)),
                      blockDir(i), /*rfo=*/trueVal);
        rewriter.setInsertionPointAfter(pairLoop);
      }
    }

    rewriter.setInsertionPointAfter(ntLoop);

    rewriter.replaceOp(op, {outValues, outIndices});
    return success();
  }
};

struct D2MDecomposeSort : public impl::D2MDecomposeSortBase<D2MDecomposeSort> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeSortBlockPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
