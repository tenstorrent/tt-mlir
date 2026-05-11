// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <functional>
#include <limits>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSEMASKING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kTileHeight = 32;
constexpr int64_t kTileWidth = 32;

struct BoundsInterval {
  Value start;
  Value end;
};

// Build an element-typed OOB fill attribute. For integer types, Inf/NegInf
// are saturated to the representable min/max of the type.
static TypedAttr getFillValueAttr(Builder &builder, Type elemType,
                                  ttcore::OOBVal oobVal) {
  if (auto floatTy = dyn_cast<FloatType>(elemType)) {
    double v = 0.0;
    switch (oobVal) {
    case ttcore::OOBVal::Undef:
    case ttcore::OOBVal::Zero:
      v = 0.0;
      break;
    case ttcore::OOBVal::One:
      v = 1.0;
      break;
    case ttcore::OOBVal::Inf:
      v = std::numeric_limits<double>::infinity();
      break;
    case ttcore::OOBVal::NegInf:
      v = -std::numeric_limits<double>::infinity();
      break;
    }
    return builder.getFloatAttr(floatTy, v);
  }
  if (auto intTy = dyn_cast<IntegerType>(elemType)) {
    unsigned w = intTy.getWidth();
    bool isUnsigned = intTy.isUnsigned();
    APInt v(w, 0);
    switch (oobVal) {
    case ttcore::OOBVal::Undef:
    case ttcore::OOBVal::Zero:
      break;
    case ttcore::OOBVal::One:
      v = APInt(w, 1);
      break;
    case ttcore::OOBVal::Inf:
      v = isUnsigned ? APInt::getMaxValue(w) : APInt::getSignedMaxValue(w);
      break;
    case ttcore::OOBVal::NegInf:
      v = isUnsigned ? APInt(w, 0) : APInt::getSignedMinValue(w);
      break;
    }
    // arith.constant requires signless int types; TileFillOp accepts any
    // integer so this is safe for the downstream use.
    return builder.getIntegerAttr(IntegerType::get(builder.getContext(), w), v);
  }
  llvm_unreachable("unsupported element type for OOB fill");
}

/// Decompose MaskOp with multi-core support.
///
/// The key change from single-core: loop bounds become dynamic based on
/// which portion of the global tile space this core is responsible for.
///
/// For each loop:
///   start = stride * coreIndex.
///   end = min(regionEnd, stride * (coreIndex + 1)).
///   localIdx = globalIdx - (stride * coreIndex).
///
/// If start >= end, the loop doesn't run--we only pad rightmost + downmost
/// regions, so this should be correct w/o modifying start with max().
struct DecomposeMaskPattern : OpRewritePattern<MaskOp> {
  DecomposeMaskPattern(MLIRContext *ctx, unsigned numStreamBuffers)
      : OpRewritePattern<MaskOp>(ctx), numStreamBuffers(numStreamBuffers) {}

  unsigned numStreamBuffers;

  // Compute local loop bounds for a core given a global region
  // [globalRegionStart, globalRegionEnd). Returns (localStart, localEnd) such
  // that iterating [localStart, localEnd) in local coordinates covers exactly
  // the tiles that fall within both the global region and this core's shard.
  static std::pair<Value, Value>
  computeLocalBounds(PatternRewriter &rewriter, Location loc,
                     int64_t globalRegionStart, int64_t globalRegionEnd,
                     Value coreIdx, int64_t shardSize) {

    Value shardSizeVal =
        rewriter.create<arith::ConstantIndexOp>(loc, shardSize);
    Value globalCoreStart =
        rewriter.create<arith::MulIOp>(loc, coreIdx, shardSizeVal);

    Value globalRegionStartVal =
        rewriter.create<arith::ConstantIndexOp>(loc, globalRegionStart);
    Value globalRegionEndVal =
        rewriter.create<arith::ConstantIndexOp>(loc, globalRegionEnd);

    // We define localStart = max(globalRegionStart - globalCoreStart, 0); in
    // turn this can be rewritten as localStart = globalRegionStart -
    // min(globalRegionStart, globalCoreStart).
    Value clampedStart = rewriter.create<arith::MinUIOp>(
        loc, globalRegionStartVal, globalCoreStart);
    Value localStart =
        rewriter.create<arith::SubIOp>(loc, globalRegionStartVal, clampedStart);

    // Similarly, we define localEnd = min(globalRegionEnd - globalCoreStart,
    // shardSize). However, to avoid underflow on unsigned, we re-express it as
    // clampedEnd = max(min(globalRegionEnd, globalCoreEnd), globalCoreStart),
    // and localEnd = clampedEnd - globalCoreStart, which is equivalent.
    Value globalCoreEnd =
        rewriter.create<arith::AddIOp>(loc, globalCoreStart, shardSizeVal);
    Value clampedEnd =
        rewriter.create<arith::MinUIOp>(loc, globalRegionEndVal, globalCoreEnd);
    clampedEnd =
        rewriter.create<arith::MaxUIOp>(loc, clampedEnd, globalCoreStart);
    Value localEnd =
        rewriter.create<arith::SubIOp>(loc, clampedEnd, globalCoreStart);

    return {localStart, localEnd};
  }

  LogicalResult matchAndRewrite(MaskOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value globalInput = op.getInput();
    Value globalOutput = op.getOutput();
    ttcore::OOBVal fillOOBVal = op.getFillValue();

    if (isa<RankedTensorType>(globalInput.getType())) {
      return rewriter.notifyMatchFailure(op, "tensor semantics not supported");
    }

    auto globalInputType = cast<MemRefType>(globalInput.getType());
    auto globalOutputType = cast<MemRefType>(globalOutput.getType());
    if (!ttcore::hasDeviceLayout(globalInput) ||
        !ttcore::hasDeviceLayout(globalOutput)) {
      return rewriter.notifyMatchFailure(op, "mask operands need layouts");
    }

    ArrayRef<int64_t> gridShape = ttcore::getGridShape(globalOutput);
    if (gridShape.size() < 2) {
      return rewriter.notifyMatchFailure(
          op, "grid must have at least 2 dimensions");
    }

    SmallVector<int64_t> shardShape =
        llvm::to_vector(ttcore::getShardShape(globalInput));
    if (shardShape.size() < 2) {
      return rewriter.notifyMatchFailure(op, "input must have at least 2 dims");
    }

    ArrayRef<int64_t> logicalShape = op.getLogicalShape();
    int64_t logicalRows = logicalShape[logicalShape.size() - 2];
    int64_t logicalCols = logicalShape[logicalShape.size() - 1];

    auto gridAttr = ttcore::GridAttr::get(rewriter.getContext(), gridShape);
    ArrayAttr emptyArray = rewriter.getArrayAttr({});
    ArrayAttr threads = rewriter.getArrayAttr(
        rewriter.getAttr<ThreadAttr>(ThreadType::Unified));
    auto genericOp = rewriter.create<GenericOp>(
        loc, TypeRange{}, ValueRange{globalInput}, ValueRange{globalOutput},
        ValueRange{}, gridAttr, emptyArray, emptyArray, emptyArray, threads,
        /*fabricConnectionConfig=*/nullptr, /*regionsCount=*/1);

    Region &region = genericOp.getRegion(0);
    rewriter.createBlock(&region);
    rewriter.setInsertionPointToStart(&region.front());

    Attribute memorySpace = globalOutputType.getMemorySpace();
    Type tileElementType = globalInputType.getElementType();
    auto inputType = MemRefType::get(shardShape, tileElementType,
                                     MemRefLayoutAttrInterface{}, memorySpace);
    auto outputType = MemRefType::get(shardShape, tileElementType,
                                      MemRefLayoutAttrInterface{}, memorySpace);
    auto maskLayout = ttcore::CBLayoutAttr::get(
        rewriter.getContext(), {1, 1},
        ttcore::getElementSizeBytes(tileElementType), numStreamBuffers);
    auto maskType =
        MemRefType::get({1, 1}, tileElementType, maskLayout, memorySpace);

    Value input = rewriter.create<memref::AllocOp>(loc, inputType);
    Value output = rewriter.create<memref::AllocOp>(loc, outputType);
    Value rowMaskCB = rewriter.create<memref::AllocOp>(loc, maskType);
    Value colMaskCB = rewriter.create<memref::AllocOp>(loc, maskType);

    SmallVector<Value> remoteIndices;
    remoteIndices.reserve(gridShape.size());
    for (size_t dim = 0; dim < gridShape.size(); ++dim) {
      remoteIndices.push_back(rewriter.create<CoreIndexOp>(
          loc, rewriter.getIndexType(),
          rewriter.getI64IntegerAttr(static_cast<int64_t>(dim)), nullptr));
    }

    rewriter.create<RemoteLoadOp>(loc, inputType, input, globalInput,
                                  remoteIndices);

    ArrayRef<int64_t> inputShape = inputType.getShape();
    auto tileType = cast<ttcore::TileType>(inputType.getElementType());
    Type elemType = tileType.getElementType();

    int64_t shardTileRows = inputShape[inputShape.size() - 2];
    int64_t shardTileCols = inputShape[inputShape.size() - 1];

    // Compute tile-level boundaries (compile-time constants).
    // lastValidRow: the last tile row that contains any valid data (may be
    // partial).
    // lastValidCol: the last tile col that contains any valid data
    // (may be partial).
    int64_t lastValidRow = (logicalRows - 1) / kTileHeight;
    int64_t lastValidCol = (logicalCols - 1) / kTileWidth;

    // Compute how many elements are valid in the last partial tile to generate
    // bitmask.
    int64_t validRowsInLastTile = logicalRows % kTileHeight;
    if (validRowsInLastTile == 0) {
      validRowsInLastTile = kTileHeight;
    }
    int64_t validColsInLastTile = logicalCols % kTileWidth;
    if (validColsInLastTile == 0) {
      validColsInLastTile = kTileWidth;
    }

    // Total tiles in the padded shape.
    int64_t totalTileRows = shardTileRows * gridShape[gridShape.size() - 2];
    int64_t totalTileCols = shardTileCols * gridShape[gridShape.size() - 1];

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    TypedAttr fillAttr = getFillValueAttr(rewriter, elemType, fillOOBVal);
    Value fillScalar =
        rewriter.create<arith::ConstantOp>(loc, fillAttr.getType(), fillAttr);

    // Get this core's coordinates.
    Value coreY = rewriter.create<CoreIndexOp>(
        loc, rewriter.getIndexType(), rewriter.getI64IntegerAttr(0), nullptr);
    Value coreX = rewriter.create<CoreIndexOp>(
        loc, rewriter.getIndexType(), rewriter.getI64IntegerAttr(1), nullptr);

    // Write the mask tiles.
    Value validRowsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, validRowsInLastTile);
    Value validColsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, validColsInLastTile);

    TT_assert(rowMaskCB);
    rewriter.create<WriteRowMaskTileOp>(loc, validRowsVal, rowMaskCB);
    TT_assert(colMaskCB);
    rewriter.create<WriteColMaskTileOp>(loc, validColsVal, colMaskCB);

    // === Tile operation helpers ===
    auto createTileFill = [&]() {
      return rewriter.create<TileFillOp>(loc, tileType, fillScalar).getResult();
    };

    auto emitPassthrough = [&](Value localRowIdx, Value localColIdx) {
      auto inputTile = rewriter.create<memref::LoadOp>(
          loc, input, ValueRange{localRowIdx, localColIdx});
      rewriter.create<memref::StoreOp>(loc, inputTile.getResult(), output,
                                       ValueRange{localRowIdx, localColIdx});
    };

    auto emitRowMasked = [&](Value localRowIdx, Value localColIdx) {
      auto inputTile = rewriter.create<memref::LoadOp>(
          loc, input, ValueRange{localRowIdx, localColIdx});
      auto tileFill = createTileFill();
      auto rowMaskTile = rewriter.create<memref::LoadOp>(
          loc, rowMaskCB, ValueRange{zeroIdx, zeroIdx});
      auto result =
          rewriter.create<TileWhereOp>(loc, tileType, rowMaskTile.getResult(),
                                       inputTile.getResult(), tileFill);
      rewriter.create<memref::StoreOp>(loc, result.getResult(), output,
                                       ValueRange{localRowIdx, localColIdx});
    };

    auto emitColMasked = [&](Value localRowIdx, Value localColIdx) {
      auto inputTile = rewriter.create<memref::LoadOp>(
          loc, input, ValueRange{localRowIdx, localColIdx});
      auto tileFill = createTileFill();
      auto colMaskTile = rewriter.create<memref::LoadOp>(
          loc, colMaskCB, ValueRange{zeroIdx, zeroIdx});
      auto result =
          rewriter.create<TileWhereOp>(loc, tileType, colMaskTile.getResult(),
                                       inputTile.getResult(), tileFill);
      rewriter.create<memref::StoreOp>(loc, result.getResult(), output,
                                       ValueRange{localRowIdx, localColIdx});
    };

    auto emitCornerMasked = [&](Value localRowIdx, Value localColIdx) {
      auto inputTile = rewriter.create<memref::LoadOp>(
          loc, input, ValueRange{localRowIdx, localColIdx});
      auto tileFill1 = createTileFill();
      auto rowMaskTile = rewriter.create<memref::LoadOp>(
          loc, rowMaskCB, ValueRange{zeroIdx, zeroIdx});
      auto rowMaskedResult =
          rewriter.create<TileWhereOp>(loc, tileType, rowMaskTile.getResult(),
                                       inputTile.getResult(), tileFill1);
      auto tileFill2 = createTileFill();
      auto colMaskTile = rewriter.create<memref::LoadOp>(
          loc, colMaskCB, ValueRange{zeroIdx, zeroIdx});
      auto finalResult =
          rewriter.create<TileWhereOp>(loc, tileType, colMaskTile.getResult(),
                                       rowMaskedResult.getResult(), tileFill2);
      rewriter.create<memref::StoreOp>(loc, finalResult.getResult(), output,
                                       ValueRange{localRowIdx, localColIdx});
    };

    auto emitFill = [&](Value localRowIdx, Value localColIdx) {
      auto tileFill = createTileFill();
      rewriter.create<memref::StoreOp>(loc, tileFill, output,
                                       ValueRange{localRowIdx, localColIdx});
    };

    // Helper to create a nested loop over local coordinates.
    auto createLocalLoop = [&](Value rowStart, Value rowEnd, Value colStart,
                               Value colEnd,
                               std::function<void(Value, Value)> emitBody) {
      auto outerLoop =
          rewriter.create<scf::ForOp>(loc, rowStart, rowEnd, oneIdx);
      rewriter.setInsertionPointToStart(outerLoop.getBody());

      auto innerLoop =
          rewriter.create<scf::ForOp>(loc, colStart, colEnd, oneIdx);
      // Mark the INNER loop as the compute root, since that's where
      // the actual compute operations are emitted. This ensures DST
      // syncs are placed inside the inner loop body, not the outer.
      // Since we emit an scf.for directly, we must tag this here
      // since linalg-to-affine and d2m-op-scheduler won't process this.
      innerLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
      innerLoop->setAttr("d2m.scheduled", rewriter.getUnitAttr());
      rewriter.setInsertionPointToStart(innerLoop.getBody());

      emitBody(outerLoop.getInductionVar(), innerLoop.getInductionVar());

      return outerLoop;
    };

    OpBuilder::InsertionGuard guard(rewriter);
    Operation *insertionPoint = &region.front().back();

    // =========================================================================
    // LOOP 0: Interior tiles - fully valid.
    // Global region: [0, lastValidRow) x [0, lastValidCol).
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto [rowStart, rowEnd] = computeLocalBounds(
          rewriter, loc, 0, lastValidRow, coreY, shardTileRows);
      auto [colStart, colEnd] = computeLocalBounds(
          rewriter, loc, 0, lastValidCol, coreX, shardTileCols);
      auto loop =
          createLocalLoop(rowStart, rowEnd, colStart, colEnd, emitPassthrough);
      insertionPoint = loop;
    }

    // =========================================================================
    // LOOP 1: Last valid row - needs row masking.
    // Global region: [lastValidRow, lastValidRow+1) x [0, lastValidCol).
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto [rowStart, rowEnd] = computeLocalBounds(
          rewriter, loc, lastValidRow, lastValidRow + 1, coreY, shardTileRows);
      auto [colStart, colEnd] = computeLocalBounds(
          rewriter, loc, 0, lastValidCol, coreX, shardTileCols);
      auto loop =
          createLocalLoop(rowStart, rowEnd, colStart, colEnd, emitRowMasked);
      insertionPoint = loop;
    }

    // =========================================================================
    // LOOP 2: Last valid col - needs col masking.
    // Global region: [0, lastValidRow) x [lastValidCol, lastValidCol+1).
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto [rowStart, rowEnd] = computeLocalBounds(
          rewriter, loc, 0, lastValidRow, coreY, shardTileRows);
      auto [colStart, colEnd] = computeLocalBounds(
          rewriter, loc, lastValidCol, lastValidCol + 1, coreX, shardTileCols);
      auto loop =
          createLocalLoop(rowStart, rowEnd, colStart, colEnd, emitColMasked);
      insertionPoint = loop;
    }

    // =========================================================================
    // LOOP 3: Corner tile - needs both row and col masking.
    // Global region: [lastValidRow, lastValidRow+1) x [lastValidCol,
    // lastValidCol+1).
    // =========================================================================
    if (rowMaskCB && colMaskCB) {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto [rowStart, rowEnd] = computeLocalBounds(
          rewriter, loc, lastValidRow, lastValidRow + 1, coreY, shardTileRows);
      auto [colStart, colEnd] = computeLocalBounds(
          rewriter, loc, lastValidCol, lastValidCol + 1, coreX, shardTileCols);
      auto loop =
          createLocalLoop(rowStart, rowEnd, colStart, colEnd, emitCornerMasked);
      insertionPoint = loop;
    }

    // =========================================================================
    // LOOP 4: OOB rows - fill entire rows beyond valid region.
    // Global region: [lastValidRow+1, totalTileRows) x [0, totalTileCols).
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto [rowStart, rowEnd] = computeLocalBounds(
          rewriter, loc, lastValidRow + 1, totalTileRows, coreY, shardTileRows);
      auto [colStart, colEnd] = computeLocalBounds(
          rewriter, loc, 0, totalTileCols, coreX, shardTileCols);
      auto loop = createLocalLoop(rowStart, rowEnd, colStart, colEnd, emitFill);
      insertionPoint = loop;
    }

    // =========================================================================
    // LOOP 5: OOB cols - fill columns beyond valid region (for valid rows
    // only). Global region: [0, lastValidRow+1) x [lastValidCol+1,
    // totalTileCols).
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto [rowStart, rowEnd] = computeLocalBounds(
          rewriter, loc, 0, lastValidRow + 1, coreY, shardTileRows);
      auto [colStart, colEnd] = computeLocalBounds(
          rewriter, loc, lastValidCol + 1, totalTileCols, coreX, shardTileCols);
      auto loop = createLocalLoop(rowStart, rowEnd, colStart, colEnd, emitFill);
      insertionPoint = loop;
    }

    rewriter.setInsertionPointAfter(insertionPoint);
    rewriter.create<RemoteStoreOp>(loc, globalOutput.getType(), globalOutput,
                                   remoteIndices, output);

    rewriter.replaceOp(op, op.getOutput());
    return success();
  }
};

struct D2MDecomposeMasking
    : public impl::D2MDecomposeMaskingBase<D2MDecomposeMasking> {
  using impl::D2MDecomposeMaskingBase<
      D2MDecomposeMasking>::D2MDecomposeMaskingBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeMaskPattern>(ctx, numStreamBuffers);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
