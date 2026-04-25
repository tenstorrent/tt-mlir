// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSESORT
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// `topk_local_sort`'s `idir` arg is `0 == descending, 1 == ascending`.
constexpr int32_t kTopkAscending = 1;
constexpr int32_t kTopkDescending = 0;

// Bitonic primitive parameters used by the SFPU topk pipeline. `k=64` and
// `logk=6` come from the tt-metal `topk_*` LLK contract for sorting two
// 32x32 tiles together.
constexpr int32_t kEndPhase = 5;
constexpr int32_t kK = 64;

// Build a list of indices that targets the `linearIdx`-th tile of a tile
// memref, padding the leading dims with zeros for rank > 1 tile memrefs (e.g.
// `memref<1x4x!tile, l1>` is addressed as `[0, linearIdx]`).
static SmallVector<Value> buildTileIndices(OpBuilder &builder, Location loc,
                                           Value memref, int64_t linearIdx) {
  auto memrefType = mlir::cast<MemRefType>(memref.getType());
  unsigned rank = memrefType.getRank();
  SmallVector<Value> indices;
  indices.reserve(rank);
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (unsigned i = 0; i + 1 < rank; ++i) {
    indices.push_back(c0);
  }
  indices.push_back(builder.create<arith::ConstantIndexOp>(loc, linearIdx));
  return indices;
}

// Helper: copy one tile from CB `src` slot `srcIdx` into DST slot `dstIdx`.
// The IR pattern `memref.load %cb[i] -> memref.store %t, %dst[k]` is matched
// by D2MToTTKernel's MemrefStoreRewriter and lowered to copy_tile_init +
// copy_tile.
static void emitCopyTileToDst(OpBuilder &builder, Location loc, Value srcCb,
                              int64_t srcIdx, Value dst, int64_t dstIdx) {
  SmallVector<Value> srcIdxs = buildTileIndices(builder, loc, srcCb, srcIdx);
  SmallVector<Value> dstIdxs = buildTileIndices(builder, loc, dst, dstIdx);
  Value tile = builder.create<memref::LoadOp>(loc, srcCb, srcIdxs);
  builder.create<memref::StoreOp>(loc, tile, dst, dstIdxs);
}

// Helper: pack one tile from DST slot `dstIdx` into CB `dst` slot `dstCbIdx`.
// The IR pattern `memref.load %dst[k] -> memref.store %t, %cb[i]` is matched
// by D2MToTTKernel's MemrefStoreRewriter::lowerPackTile.
static void emitPackTileFromDst(OpBuilder &builder, Location loc, Value dst,
                                int64_t dstIdx, Value dstCb, int64_t dstCbIdx) {
  SmallVector<Value> dstIdxs = buildTileIndices(builder, loc, dst, dstIdx);
  SmallVector<Value> dstCbIdxs =
      buildTileIndices(builder, loc, dstCb, dstCbIdx);
  Value tile = builder.create<memref::LoadOp>(loc, dst, dstIdxs);
  builder.create<memref::StoreOp>(loc, tile, dstCb, dstCbIdxs);
}

// Helper: load tile from CB `src` slot `srcIdx`, transpose-WH, store to DST
// slot `dstIdx`. Lowers to transpose_init + transpose_tile in TTKernel.
static void emitTransposeTileToDst(OpBuilder &builder, Location loc,
                                   ttcore::TileType tileTy, Value srcCb,
                                   int64_t srcIdx, Value dst, int64_t dstIdx) {
  SmallVector<Value> srcIdxs = buildTileIndices(builder, loc, srcCb, srcIdx);
  SmallVector<Value> dstIdxs = buildTileIndices(builder, loc, dst, dstIdx);
  Value loaded = builder.create<memref::LoadOp>(loc, srcCb, srcIdxs);
  Value transposed =
      builder.create<TileTransposeOp>(loc, tileTy, loaded).getResult();
  builder.create<memref::StoreOp>(loc, transposed, dst, dstIdxs);
}

// Helper: fill DST slot `dstIdx` with a zero-tile of the value tile type.
static void emitZeroTileToDst(OpBuilder &builder, Location loc,
                              ttcore::TileType tileTy, Value dst,
                              int64_t dstIdx) {
  TypedAttr zeroAttr;
  Type elemTy = tileTy.getElementType();
  if (auto floatTy = mlir::dyn_cast<FloatType>(elemTy)) {
    zeroAttr = builder.getFloatAttr(floatTy, 0.0);
  } else {
    auto intTy = mlir::cast<IntegerType>(elemTy);
    auto signlessTy = IntegerType::get(builder.getContext(), intTy.getWidth());
    zeroAttr = builder.getIntegerAttr(signlessTy, 0);
  }
  Value zero = builder.create<arith::ConstantOp>(loc, zeroAttr);
  Value zeroTile = builder.create<TileFillOp>(loc, tileTy, zero).getResult();
  SmallVector<Value> dstIdxs = buildTileIndices(builder, loc, dst, dstIdx);
  builder.create<memref::StoreOp>(loc, zeroTile, dst, dstIdxs);
}

// Acquire 8 tile slots in DST. Returns the result memref.
static Value emitAcquireDst(OpBuilder &builder, Location loc,
                            ttcore::TileType tileTy) {
  auto dstSpace = ttcore::MemorySpaceAttr::get(
      builder.getContext(), ttcore::MemorySpace::RegisterDst);
  auto dstType =
      MemRefType::get({8}, tileTy, MemRefLayoutAttrInterface{}, dstSpace);
  return builder.create<AcquireDstOp>(loc, dstType).getResult();
}

// Emit `topk_tile_init`.
static void emitTopkInit(OpBuilder &builder, Location loc) {
  builder.create<TileTopkInitOp>(loc);
}

// Emit `topk_local_sort(dst, idir, end_phase=5, 0, 0, 0)`.
static void emitTopkLocalSort(OpBuilder &builder, Location loc, int64_t idst,
                              int32_t idir, bool stable) {
  Value idstV = builder.create<arith::ConstantIndexOp>(loc, idst);
  Value idirV = builder.create<arith::ConstantIntOp>(loc, idir, 32);
  Value endPhaseV = builder.create<arith::ConstantIntOp>(loc, kEndPhase, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  builder.create<TileTopkLocalSortOp>(loc, idstV, idirV, endPhaseV, zero, zero,
                                      zero, builder.getBoolAttr(stable));
}

// Emit `topk_merge(dst, m_iter, k=64)` with idir baked in as a template attr.
static void emitTopkMerge(OpBuilder &builder, Location loc, int64_t idst,
                          int32_t mIter, bool idir, bool stable) {
  Value idstV = builder.create<arith::ConstantIndexOp>(loc, idst);
  Value mIterV = builder.create<arith::ConstantIntOp>(loc, mIter, 32);
  Value kV = builder.create<arith::ConstantIntOp>(loc, kK, 32);
  builder.create<TileTopkMergeOp>(loc, idstV, mIterV, kV,
                                  builder.getBoolAttr(idir),
                                  builder.getBoolAttr(stable));
}

/// Decompose `d2m.sort_block` into the bitonic-sort sequence of
/// transpose / topk_* / pack tile ops, mirroring the tt-metal kernel
/// `data_movement/sort/device/kernels/compute/sort_single_row_single_core.cpp`.
///
/// Today only handles the `Ht=1`, `Wt=4` shape (input `1x4` tiles).
struct DecomposeSortBlockPattern : OpRewritePattern<SortBlockOp> {
  using OpRewritePattern<SortBlockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SortBlockOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto inputType = mlir::dyn_cast<MemRefType>(op.getInput().getType());
    auto valuesType = mlir::dyn_cast<MemRefType>(op.getValuesOut().getType());
    auto indicesType = mlir::dyn_cast<MemRefType>(op.getIndicesOut().getType());
    if (!inputType || !valuesType || !indicesType) {
      return rewriter.notifyMatchFailure(
          op, "expected memref operands (run after bufferization)");
    }
    if (inputType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "only rank-2 sort_block supported");
    }

    auto valueTileTy =
        mlir::dyn_cast<ttcore::TileType>(inputType.getElementType());
    if (!valueTileTy) {
      return rewriter.notifyMatchFailure(op, "expected tile element type");
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t Ht = inputShape[0];
    int64_t Wt = inputShape[1];
    if (Ht != 1 || Wt != 4) {
      return rewriter.notifyMatchFailure(
          op, "only Ht=1, Wt=4 sort_block supported today");
    }

    bool descending = op.getDescending();
    bool stable = op.getStable();
    int32_t initialAscDir = descending ? kTopkDescending : kTopkAscending;
    int32_t initialDescDir = descending ? kTopkAscending : kTopkDescending;
    bool mergeIdir = !descending; // template `idir` for topk_merge.

    // The transposed scratch CBs (Wt tiles each) are passed as additional
    // inputs to the surrounding d2m.generic so the standard CB-binding
    // machinery materializes them as L1 CBs. We use the per-core buffers
    // directly here.
    rewriter.setInsertionPoint(op);
    Value valuesTransposed = op.getValuesScratch();
    Value indicesTransposed = op.getIndicesScratch();

    // === Stage 0: bitonic sequence formation ===
    // Sort pair (0,1) ascending, pair (2,3) descending (or flipped if
    // `descending`). After this stage, valuesTransposed[0..3] holds the
    // transposed bitonic sequence, indicesTransposed[0..3] holds garbage
    // indices that were carried through the sort.
    auto emitInitialPair = [&](int64_t pairBase, int32_t dir) {
      Value dst = emitAcquireDst(rewriter, loc, valueTileTy);
      // Load + transpose value tiles into DST[0,1].
      emitTransposeTileToDst(rewriter, loc, valueTileTy, op.getInput(),
                             pairBase + 0, dst, 0);
      emitTransposeTileToDst(rewriter, loc, valueTileTy, op.getInput(),
                             pairBase + 1, dst, 1);
      // Index slots [2,3] just get zero tiles (we discard the indices output).
      emitZeroTileToDst(rewriter, loc, valueTileTy, dst, 2);
      emitZeroTileToDst(rewriter, loc, valueTileTy, dst, 3);

      emitTopkInit(rewriter, loc);
      emitTopkLocalSort(rewriter, loc, /*idst=*/0, dir, stable);

      // Pack DST[0,1] back into valuesTransposed[pairBase, pairBase+1].
      emitPackTileFromDst(rewriter, loc, dst, 0, valuesTransposed,
                          pairBase + 0);
      emitPackTileFromDst(rewriter, loc, dst, 1, valuesTransposed,
                          pairBase + 1);
      // Pack DST[2,3] back into indicesTransposed[pairBase, pairBase+1].
      emitPackTileFromDst(rewriter, loc, dst, 2, indicesTransposed,
                          pairBase + 0);
      emitPackTileFromDst(rewriter, loc, dst, 3, indicesTransposed,
                          pairBase + 1);
    };

    emitInitialPair(/*pairBase=*/0, initialAscDir);
    emitInitialPair(/*pairBase=*/2, initialDescDir);

    // === Stage 1 (m_iter=1): merge pairs with sub_dist=2, then sort
    //     pairs with sub_dist=1. ===

    // Per-pair body: copy values+indices into DST[0,1,2,3], invoke either
    // `topk_merge` or `topk_local_sort` based on `subEqOne`, then pack back.
    auto emitMergeOrSortPair = [&](int64_t left, int64_t right, bool subEqOne) {
      Value dst = emitAcquireDst(rewriter, loc, valueTileTy);
      emitCopyTileToDst(rewriter, loc, valuesTransposed, left, dst, 0);
      emitCopyTileToDst(rewriter, loc, valuesTransposed, right, dst, 1);
      emitCopyTileToDst(rewriter, loc, indicesTransposed, left, dst, 2);
      emitCopyTileToDst(rewriter, loc, indicesTransposed, right, dst, 3);

      emitTopkInit(rewriter, loc);
      if (subEqOne) {
        emitTopkLocalSort(rewriter, loc, /*idst=*/0,
                          descending ? kTopkDescending : kTopkAscending,
                          stable);
        // Pack values back to (left, right).
        emitPackTileFromDst(rewriter, loc, dst, 0, valuesTransposed, left);
        emitPackTileFromDst(rewriter, loc, dst, 1, valuesTransposed, right);
        emitPackTileFromDst(rewriter, loc, dst, 2, indicesTransposed, left);
        emitPackTileFromDst(rewriter, loc, dst, 3, indicesTransposed, right);
      } else {
        // `topk_merge`'s `idir` template arg controls which DST slot ends up
        // with the larger value: idir=false → DST[0]=larger; idir=true →
        // DST[0]=smaller. We set `mergeIdir = !descending` so that DST[0]
        // always holds the value that should land at the LEFT tile (the
        // "first" tile in spatial order):
        //   - descending: idir=false → DST[0]=larger → larger to left ✓
        //   - ascending : idir=true  → DST[0]=smaller → smaller to left ✓
        // No additional pack-side swap is needed.
        emitTopkMerge(rewriter, loc, /*idst=*/0, /*m_iter=*/1, mergeIdir,
                      stable);
        emitPackTileFromDst(rewriter, loc, dst, /*dstIdx=*/0, valuesTransposed,
                            left);
        emitPackTileFromDst(rewriter, loc, dst, /*dstIdx=*/1, valuesTransposed,
                            right);
        emitPackTileFromDst(rewriter, loc, dst, /*dstIdx=*/2, indicesTransposed,
                            left);
        emitPackTileFromDst(rewriter, loc, dst, /*dstIdx=*/3, indicesTransposed,
                            right);
      }
    };

    // sub=2: distance=2, pairs (0,2) and (1,3).
    emitMergeOrSortPair(/*left=*/0, /*right=*/2, /*subEqOne=*/false);
    emitMergeOrSortPair(/*left=*/1, /*right=*/3, /*subEqOne=*/false);
    // sub=1: distance=1, pairs (0,1) and (2,3).
    emitMergeOrSortPair(/*left=*/0, /*right=*/1, /*subEqOne=*/true);
    emitMergeOrSortPair(/*left=*/2, /*right=*/3, /*subEqOne=*/true);

    // === Final transpose-and-pack: undo the WH transpose for each output
    //     tile, writing into the values_out / indices_out shards. ===
    auto emitTransposeAndPack = [&](Value srcCb, Value dstCb,
                                    ttcore::TileType srcTileTy) {
      for (int64_t w = 0; w < Wt; ++w) {
        Value dst = emitAcquireDst(rewriter, loc, srcTileTy);
        emitTransposeTileToDst(rewriter, loc, srcTileTy, srcCb, w, dst, 0);
        emitPackTileFromDst(rewriter, loc, dst, /*dstIdx=*/0, dstCb,
                            /*dstCbIdx=*/w);
      }
    };

    emitTransposeAndPack(valuesTransposed, op.getValuesOut(), valueTileTy);

    // For indices, just zero-fill since we don't track real argsort indices
    // through the sort yet. The test discards the indices output.
    auto indicesTileTy =
        mlir::cast<ttcore::TileType>(indicesType.getElementType());
    for (int64_t w = 0; w < Wt; ++w) {
      Value dst = emitAcquireDst(rewriter, loc, indicesTileTy);
      emitZeroTileToDst(rewriter, loc, indicesTileTy, dst, 0);
      emitPackTileFromDst(rewriter, loc, dst, /*dstIdx=*/0, op.getIndicesOut(),
                          /*dstCbIdx=*/w);
    }

    // sort_block has memref operands and produces memref results that alias
    // the DPS inits. Replace results with the inits so users (e.g. yields
    // inside the surrounding generic) see the populated shards.
    rewriter.replaceOp(op, ValueRange{op.getValuesOut(), op.getIndicesOut()});
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
