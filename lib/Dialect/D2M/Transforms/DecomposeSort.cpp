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

// Odd-even transposition sort over the reduction tiles, built from the one
// topk LLK that is a strong enough primitive on its own:
//
//   tile_topk_local_sort  sorts the 64 values spanning an ADJACENT tile pair,
//                         in a given direction (phases 0..5 = a complete
//                         64-element sort)
//
// A `local_sort` of tiles (t, t+1) is a full 2-tile sorting comparator: unlike
// `tile_topk_merge` (which is lane-preserving, so it can never move a value
// between lanes) it can permute all 64 values arbitrarily. That makes it
// exactly the comparator odd-even transposition needs, and transposition in
// turn is correct for ANY tile count -- no power-of-two padding, no masked pad
// tiles, no virtual-tile bookkeeping.
//
// This deliberately does NOT use the bitonic network from tt-metal's
// ttnn::sort compute kernel (sort_single_row_single_core.cpp). Bitonic is
// asymptotically cheaper (O(N log^2 N) comparators vs O(N^2) here) but is only
// defined over a power-of-two extent, so it must pad: a 43-tile sort dimension
// rounds up to 64, costing ~49% extra L1 for the input, the values result, the
// indices result and the index arange all at once. L1 capacity is the binding
// constraint for this op, so the extra comparator work is the better trade.
//
// The network is N sweeps over N tiles, alternating which parity of adjacent
// pair is compared:
//
//   sweep 0:  (0,1) (2,3) (4,5) ...
//   sweep 1:  (1,2) (3,4) (5,6) ...
//   sweep 2:  (0,1) (2,3) (4,5) ...
//
// N sweeps is tight, not conservative -- the worst case genuinely needs all N.
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
    TT_assertv((numTilesInner >= 2 && numTilesInner % 2 == 0),
               "sort_block reduction dim must span an even number of tiles "
               ">= 2; the only comparator is a tile-pair sort, so sweep 0 has "
               "to be a perfect matching. TTIRToD2M rounds up to guarantee it");

    // Shard is row-major [htShard, wtShard] with flat index r*reductionStride +
    // nt*ntStride; strides depend on which dim is the reduction dim.
    int64_t ntDimIdx = (dimIdx == rank - 1) ? rank - 2 : rank - 1;
    int64_t nonTargetCount = inputShape[ntDimIdx];
    int64_t reductionStride = (dimIdx == rank - 1) ? 1 : nonTargetCount;
    int64_t ntStride = (dimIdx == rank - 1) ? numTilesInner : 1;

    // topk_local_sort's idir is 1 for increasing, 0 for decreasing.
    bool ascending = !op.getDescending();

    // Phases 0..5 are a complete sort of the 64 datums spanning a tile pair.
    // This is the only sort granularity the LLK offers: a lower `i_end_phase`
    // stops part-way through the merge rather than yielding a sorted 32-run,
    // so a single tile cannot be sorted on its own. TTIRToD2M rounds a 1-tile
    // reduction up to 2 for exactly that reason.
    constexpr int32_t kFullSortEndPhase = 5;

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
    Value twoIdx = idxVal(2);
    Value zeroI32 = i32Val(0);
    Value trueVal = i1Val(true);

    // Every sweep sorts in the same direction; there is no per-pair direction
    // flipping the way a bitonic network needs, so idir is a loop-invariant
    // constant.
    Value idir = i32Val(ascending ? 1 : 0);

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

    // Every comparator is its own DST group: acquire, copy the four tiles in,
    // run the LLK, pack the four tiles back, release. `rfo` (read-from-output)
    // is false only for the very first touch of a tile, when the data still
    // lives in the input CB rather than the output one.
    auto emitLocalSort = [&](Value tileA, Value tileB, Value rfo) {
      rewriter.create<TileTopkLocalSortOp>(
          loc, inputValues, bufIdxFilled, outValues, outIndices, idir,
          /*i_end_phase=*/i32Attr(kFullSortEndPhase),
          /*i_start_phase=*/zeroI32, tileA, tileB,
          /*is_group_start=*/boolAttr(true), /*is_group_end=*/trueVal, rfo);
    };

    {
      // ---- sweep 0, peeled ------------------------------------------------
      // Peeled because it is the only sweep that reads from the input CB
      // (rfo=false); every later sweep reads back what this one wrote to the
      // output CB. Its pairs (0,1), (2,3), ... are a perfect matching on the
      // tiles, so each tile is copied out of the input exactly once. That is
      // why TTIRToD2M guarantees an even tile count: with an odd count the
      // trailing tile would have no unread partner, and pairing it with an
      // already-sorted tile would re-read that tile's stale input data and
      // clobber the sorted result.
      {
        auto pairLoop = rewriter.create<scf::ForOp>(
            loc, zeroIdx, idxVal(numTilesInner - 1), twoIdx);
        rewriter.setInsertionPointToStart(pairLoop.getBody());
        Value t = pairLoop.getInductionVar();
        emitLocalSort(flat(t),
                      flat(rewriter.create<arith::AddIOp>(loc, t, oneIdx)),
                      /*rfo=*/i1Val(false));
        rewriter.setInsertionPointAfter(pairLoop);
      }

      // ---- sweeps 1..N-1, rolled -----------------------------------------
      // Both loops are scf.for so the emitted kernel stays a fixed size
      // regardless of the reduction extent. Unrolling the sweep loop would
      // make the kernel grow with N and overflow the Tensix kernel config
      // buffer, which is what a fully unrolled network did previously.
      auto sweepLoop = rewriter.create<scf::ForOp>(
          loc, oneIdx, idxVal(numTilesInner), oneIdx);
      rewriter.setInsertionPointToStart(sweepLoop.getBody());
      Value sweep = sweepLoop.getInductionVar();

      // Sweep parity selects which adjacent pairs are compared: even sweeps
      // start at tile 0, odd sweeps start at tile 1.
      //
      // The parity is computed as `(i32)sweep & 1` and cast back to index.
      // Doing it in i32 rather than on the index-typed loop variable is
      // deliberate: the arith->emitc conversion only lowers bitwise/shift ops
      // whose type is a true IntegerType, and index-typed remui does not
      // legalize either. Casting to i32, masking, and casting back is the same
      // shape the bitonic implementation used for its bit twiddling.
      Value sweepI32 = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), sweep);
      Value parityI32 =
          rewriter.create<arith::AndIOp>(loc, sweepI32, i32Val(1));
      Value parity = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), parityI32);

      auto pairLoop = rewriter.create<scf::ForOp>(
          loc, parity, idxVal(numTilesInner - 1), twoIdx);
      rewriter.setInsertionPointToStart(pairLoop.getBody());
      Value t = pairLoop.getInductionVar();
      emitLocalSort(flat(t),
                    flat(rewriter.create<arith::AddIOp>(loc, t, oneIdx)),
                    /*rfo=*/trueVal);

      rewriter.setInsertionPointAfter(sweepLoop);
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
