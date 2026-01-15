// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cmath>
#include <limits>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSEMASKING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kTileHeight = 32;
constexpr int64_t kTileWidth = 32;

static double getFillValueAsDouble(ttcore::OOBVal oobVal) {
  switch (oobVal) {
  case ttcore::OOBVal::Undef:
    return 0.0;
  case ttcore::OOBVal::Zero:
    return 0.0;
  case ttcore::OOBVal::One:
    return 1.0;
  case ttcore::OOBVal::Inf:
    return std::numeric_limits<double>::infinity();
  case ttcore::OOBVal::NegInf:
    return -std::numeric_limits<double>::infinity();
  }
  return 0.0;
}

struct DecomposeBlockMaskWritePattern : OpRewritePattern<BlockMaskWriteOp> {
  using OpRewritePattern<BlockMaskWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockMaskWriteOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value rowMaskCB = op.getRowMaskCb();
    Value colMaskCB = op.getColMaskCb();
    Value logicalRowsVal = op.getLogicalRows();
    Value logicalColsVal = op.getLogicalCols();

    Value tileHeightVal =
        rewriter.create<arith::ConstantIndexOp>(loc, kTileHeight);
    Value tileWidthVal =
        rewriter.create<arith::ConstantIndexOp>(loc, kTileWidth);
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    Value rowRemainder =
        rewriter.create<arith::RemUIOp>(loc, logicalRowsVal, tileHeightVal);
    Value rowRemIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rowRemainder, zeroIdx);
    Value validRows = rewriter.create<arith::SelectOp>(
        loc, rowRemIsZero, tileHeightVal, rowRemainder);

    Value colRemainder =
        rewriter.create<arith::RemUIOp>(loc, logicalColsVal, tileWidthVal);
    Value colRemIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, colRemainder, zeroIdx);
    Value validCols = rewriter.create<arith::SelectOp>(
        loc, colRemIsZero, tileWidthVal, colRemainder);

    rewriter.create<ExperimentalWriteRowMaskTileOp>(loc, validRows, rowMaskCB);
    rewriter.create<ExperimentalWriteColMaskTileOp>(loc, validCols, colMaskCB);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Decompose BlockMaskOp with multi-core support.
///
/// The key change from single-core: loop bounds become dynamic based on
/// which portion of the global tile space this core is responsible for.
///
/// For each loop:
///   start = max(regionStart, stride * coreIndex)
///   end = min(regionEnd, stride * (coreIndex + 1))
///   localIdx = globalIdx - (stride * coreIndex)  // for memref access
///
/// If start >= end, the loop doesn't run (scf.for semantics).
struct DecomposeBlockMaskPattern : OpRewritePattern<BlockMaskOp> {
  using OpRewritePattern<BlockMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockMaskOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    Value rowMaskCB = op.getRowMaskCb();
    Value colMaskCB = op.getColMaskCb();
    Value logicalRowsVal = op.getLogicalRows();
    Value logicalColsVal = op.getLogicalCols();
    ttcore::OOBVal fillOOBVal = op.getFillValue();

    if (isa<RankedTensorType>(input.getType())) {
      return rewriter.notifyMatchFailure(op, "tensor semantics not supported");
    }

    auto inputType = cast<MemRefType>(input.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    if (inputShape.size() < 2) {
      return rewriter.notifyMatchFailure(op, "input must have at least 2 dims");
    }

    // Get the enclosing GenericOp for grid info
    auto genericOp = op->getParentOfType<GenericOp>();
    if (!genericOp) {
      return rewriter.notifyMatchFailure(
          op, "BlockMaskOp must be inside a GenericOp");
    }

    ttcore::GridAttr gridAttr = genericOp.getGrid();
    ArrayRef<int64_t> gridShape = gridAttr.getShape();
    if (gridShape.size() < 2) {
      return rewriter.notifyMatchFailure(
          op, "grid must have at least 2 dimensions");
    }

    int64_t gridRowsStatic = gridShape[gridShape.size() - 2];
    int64_t gridColsStatic = gridShape[gridShape.size() - 1];

    auto tileType = cast<ttcore::TileType>(inputType.getElementType());
    Type elemType = tileType.getElementType();

    int64_t shardTileRows = inputShape[inputShape.size() - 2];
    int64_t shardTileCols = inputShape[inputShape.size() - 1];

    double fillValueDouble = getFillValueAsDouble(fillOOBVal);
    Value fillScalar = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, fillValueDouble));

    // Constants
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value shardTileRowsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, shardTileRows);
    Value shardTileColsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, shardTileCols);

    // Get core coordinates
    Value coreY = rewriter.create<CoreIndexOp>(loc, rewriter.getIndexType(),
                                               rewriter.getI64IntegerAttr(0));
    Value coreX = rewriter.create<CoreIndexOp>(loc, rewriter.getIndexType(),
                                               rewriter.getI64IntegerAttr(1));

    // Extract logical shape constants
    auto getConstantIndex = [](Value v) -> std::optional<int64_t> {
      if (auto constOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
        return constOp.value();
      }
      return std::nullopt;
    };

    std::optional<int64_t> logicalRowsOpt = getConstantIndex(logicalRowsVal);
    std::optional<int64_t> logicalColsOpt = getConstantIndex(logicalColsVal);

    if (!logicalRowsOpt || !logicalColsOpt) {
      return rewriter.notifyMatchFailure(op, "logical shape must be constant");
    }

    int64_t logicalRows = *logicalRowsOpt;
    int64_t logicalCols = *logicalColsOpt;

    // Compute mask values at C++ compile time - these are global properties
    int64_t validRowsStatic = logicalRows % kTileHeight;
    if (validRowsStatic == 0) {
      validRowsStatic = kTileHeight;
    }
    int64_t validColsStatic = logicalCols % kTileWidth;
    if (validColsStatic == 0) {
      validColsStatic = kTileWidth;
    }

    Value validRows =
        rewriter.create<arith::ConstantIndexOp>(loc, validRowsStatic);
    Value validCols =
        rewriter.create<arith::ConstantIndexOp>(loc, validColsStatic);

    // Compute region boundaries at C++ compile time
    int64_t logicalTileRows = (logicalRows + kTileHeight - 1) / kTileHeight;
    int64_t logicalTileCols = (logicalCols + kTileWidth - 1) / kTileWidth;
    int64_t lastValidRowStatic = (logicalRows - 1) / kTileHeight;
    int64_t lastValidColStatic = (logicalCols - 1) / kTileWidth;

    // Value logicalTileRowsVal = rewriter.create<arith::ConstantIndexOp>(loc,
    // logicalTileRows); Value logicalTileColsVal =
    // rewriter.create<arith::ConstantIndexOp>(loc, logicalTileCols);
    Value lastValidRow =
        rewriter.create<arith::ConstantIndexOp>(loc, lastValidRowStatic);
    Value lastValidCol =
        rewriter.create<arith::ConstantIndexOp>(loc, lastValidColStatic);
    Value lastValidRowPlusOne =
        rewriter.create<arith::ConstantIndexOp>(loc, lastValidRowStatic + 1);
    Value lastValidColPlusOne =
        rewriter.create<arith::ConstantIndexOp>(loc, lastValidColStatic + 1);

    // Write mask tiles once - all cores write the same mask values,
    // but only the core owning the boundary tile will actually use them
    if (rowMaskCB) {
      rewriter.create<ExperimentalWriteRowMaskTileOp>(loc, validRows,
                                                      rowMaskCB);
    }
    if (colMaskCB) {
      rewriter.create<ExperimentalWriteColMaskTileOp>(loc, validCols,
                                                      colMaskCB);
    }

    // Compute stride per core (can be done at C++ compile time too)
    int64_t strideRowsStatic =
        (logicalTileRows + gridRowsStatic - 1) / gridRowsStatic;
    int64_t strideColsStatic =
        (logicalTileCols + gridColsStatic - 1) / gridColsStatic;
    Value strideRows =
        rewriter.create<arith::ConstantIndexOp>(loc, strideRowsStatic);
    Value strideCols =
        rewriter.create<arith::ConstantIndexOp>(loc, strideColsStatic);

    // Compute this core's global start offset (dynamic, depends on core index)
    Value globalRowStart =
        rewriter.create<arith::MulIOp>(loc, coreY, strideRows);
    Value globalColStart =
        rewriter.create<arith::MulIOp>(loc, coreX, strideCols);

    // Compute this core's global end (used for clamping region ends)
    Value coreYPlusOne = rewriter.create<arith::AddIOp>(loc, coreY, oneIdx);
    Value coreXPlusOne = rewriter.create<arith::AddIOp>(loc, coreX, oneIdx);
    Value globalRowEndRaw =
        rewriter.create<arith::MulIOp>(loc, coreYPlusOne, strideRows);
    Value globalColEndRaw =
        rewriter.create<arith::MulIOp>(loc, coreXPlusOne, strideCols);

    // === Helper to compute core-local loop bounds ===
    // Given global region [regionStart, regionEnd), compute:
    //   start = max(regionStart, globalCoreStart)
    //   end = min(regionEnd, globalCoreEnd)
    // Returns (start, end) in global coordinates - caller subtracts
    // globalCoreStart for local access
    auto clampToCore = [&](Value regionStart, Value regionEnd, Value coreStart,
                           Value coreEndRaw) -> std::pair<Value, Value> {
      Value start =
          rewriter.create<arith::MaxUIOp>(loc, regionStart, coreStart);
      Value end = rewriter.create<arith::MinUIOp>(loc, regionEnd, coreEndRaw);
      return {start, end};
    };

    // === Tile operation helpers ===
    auto createFillTile = [&]() {
      return rewriter.create<ExperimentalTileFillOp>(loc, tileType, fillScalar)
          .getResult();
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
      auto fillTile = createFillTile();
      if (rowMaskCB) {
        auto rowMaskTile = rewriter.create<memref::LoadOp>(
            loc, rowMaskCB, ValueRange{zeroIdx, zeroIdx});
        auto result =
            rewriter.create<TileWhereOp>(loc, tileType, rowMaskTile.getResult(),
                                         inputTile.getResult(), fillTile);
        rewriter.create<memref::StoreOp>(loc, result.getResult(), output,
                                         ValueRange{localRowIdx, localColIdx});
      } else {
        auto mask = rewriter.create<ExperimentalTileRowMaskOp>(loc, tileType,
                                                               validRows);
        auto result = rewriter.create<TileWhereOp>(
            loc, tileType, mask.getResult(), inputTile.getResult(), fillTile);
        rewriter.create<memref::StoreOp>(loc, result.getResult(), output,
                                         ValueRange{localRowIdx, localColIdx});
      }
    };

    auto emitColMasked = [&](Value localRowIdx, Value localColIdx) {
      auto inputTile = rewriter.create<memref::LoadOp>(
          loc, input, ValueRange{localRowIdx, localColIdx});
      auto fillTile = createFillTile();
      if (colMaskCB) {
        auto colMaskTile = rewriter.create<memref::LoadOp>(
            loc, colMaskCB, ValueRange{zeroIdx, zeroIdx});
        auto result =
            rewriter.create<TileWhereOp>(loc, tileType, colMaskTile.getResult(),
                                         inputTile.getResult(), fillTile);
        rewriter.create<memref::StoreOp>(loc, result.getResult(), output,
                                         ValueRange{localRowIdx, localColIdx});
      } else {
        auto mask = rewriter.create<ExperimentalTileColMaskOp>(loc, tileType,
                                                               validCols);
        auto result = rewriter.create<TileWhereOp>(
            loc, tileType, mask.getResult(), inputTile.getResult(), fillTile);
        rewriter.create<memref::StoreOp>(loc, result.getResult(), output,
                                         ValueRange{localRowIdx, localColIdx});
      }
    };

    auto emitCornerMasked = [&](Value localRowIdx, Value localColIdx) {
      auto inputTile = rewriter.create<memref::LoadOp>(
          loc, input, ValueRange{localRowIdx, localColIdx});
      auto fillTile1 = createFillTile();
      auto rowMaskTile = rewriter.create<memref::LoadOp>(
          loc, rowMaskCB, ValueRange{zeroIdx, zeroIdx});
      auto rowMaskedResult =
          rewriter.create<TileWhereOp>(loc, tileType, rowMaskTile.getResult(),
                                       inputTile.getResult(), fillTile1);
      auto fillTile2 = createFillTile();
      auto colMaskTile = rewriter.create<memref::LoadOp>(
          loc, colMaskCB, ValueRange{zeroIdx, zeroIdx});
      auto finalResult =
          rewriter.create<TileWhereOp>(loc, tileType, colMaskTile.getResult(),
                                       rowMaskedResult.getResult(), fillTile2);
      rewriter.create<memref::StoreOp>(loc, finalResult.getResult(), output,
                                       ValueRange{localRowIdx, localColIdx});
    };

    auto emitFill = [&](Value localRowIdx, Value localColIdx) {
      auto fillTile = createFillTile();
      rewriter.create<memref::StoreOp>(loc, fillTile, output,
                                       ValueRange{localRowIdx, localColIdx});
    };

    // Helper to create nested loop with global bounds, local memref access
    auto createNestedLoop = [&](Value rowStart, Value rowEnd, Value colStart,
                                Value colEnd, Value rowOffset, Value colOffset,
                                std::function<void(Value, Value)> emitBody) {
      auto outerLoop =
          rewriter.create<scf::ForOp>(loc, rowStart, rowEnd, oneIdx);
      outerLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
      outerLoop->setAttr("d2m.scheduled", rewriter.getUnitAttr());

      rewriter.setInsertionPointToStart(outerLoop.getBody());
      Value globalRowIdx = outerLoop.getInductionVar();
      Value localRowIdx =
          rewriter.create<arith::SubIOp>(loc, globalRowIdx, rowOffset);

      auto innerLoop =
          rewriter.create<scf::ForOp>(loc, colStart, colEnd, oneIdx);
      rewriter.setInsertionPointToStart(innerLoop.getBody());
      Value globalColIdx = innerLoop.getInductionVar();
      Value localColIdx =
          rewriter.create<arith::SubIOp>(loc, globalColIdx, colOffset);

      emitBody(localRowIdx, localColIdx);

      return outerLoop;
    };

    OpBuilder::InsertionGuard guard(rewriter);
    Operation *insertionPoint = op;

    // =========================================================================
    // LOOP 1: Interior tiles [0, lastValidRow) x [0, lastValidCol)
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto [rowStart, rowEnd] =
          clampToCore(zeroIdx, lastValidRow, globalRowStart, globalRowEndRaw);
      auto [colStart, colEnd] =
          clampToCore(zeroIdx, lastValidCol, globalColStart, globalColEndRaw);
      auto loop =
          createNestedLoop(rowStart, rowEnd, colStart, colEnd, globalRowStart,
                           globalColStart, emitPassthrough);
      insertionPoint = loop;
    }

    // =========================================================================
    // LOOP 2: Last row [lastValidRow, lastValidRow+1) x [0, lastValidCol)
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto [rowStart, rowEnd] = clampToCore(lastValidRow, lastValidRowPlusOne,
                                            globalRowStart, globalRowEndRaw);
      auto [colStart, colEnd] =
          clampToCore(zeroIdx, lastValidCol, globalColStart, globalColEndRaw);
      auto loop =
          createNestedLoop(rowStart, rowEnd, colStart, colEnd, globalRowStart,
                           globalColStart, emitRowMasked);
      insertionPoint = loop;
    }

    // =========================================================================
    // LOOP 3: Last col [0, lastValidRow) x [lastValidCol, lastValidCol+1)
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto [rowStart, rowEnd] =
          clampToCore(zeroIdx, lastValidRow, globalRowStart, globalRowEndRaw);
      auto [colStart, colEnd] = clampToCore(lastValidCol, lastValidColPlusOne,
                                            globalColStart, globalColEndRaw);
      auto loop =
          createNestedLoop(rowStart, rowEnd, colStart, colEnd, globalRowStart,
                           globalColStart, emitColMasked);
      insertionPoint = loop;
    }

    // =========================================================================
    // LOOP 4: Corner [lastValidRow, lastValidRow+1) x [lastValidCol,
    // lastValidCol+1)
    // =========================================================================
    if (rowMaskCB && colMaskCB) {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto [rowStart, rowEnd] = clampToCore(lastValidRow, lastValidRowPlusOne,
                                            globalRowStart, globalRowEndRaw);
      auto [colStart, colEnd] = clampToCore(lastValidCol, lastValidColPlusOne,
                                            globalColStart, globalColEndRaw);
      auto loop =
          createNestedLoop(rowStart, rowEnd, colStart, colEnd, globalRowStart,
                           globalColStart, emitCornerMasked);
      insertionPoint = loop;
    }

    // =========================================================================
    // LOOP 5: OOB rows [lastValidRow+1, shardTileRows) x [0, shardTileCols)
    // These are in local coordinates already (shard-relative)
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      // OOB rows start after the last valid row in this core's local space
      // lastValidRow is global; convert to local: lastValidRow - globalRowStart
      // But we need to handle the case where lastValidRow < globalRowStart
      Value localLastValidRowPlusOne = rewriter.create<arith::SubIOp>(
          loc, lastValidRowPlusOne, globalRowStart);
      // Clamp to [0, shardTileRows]
      localLastValidRowPlusOne = rewriter.create<arith::MaxUIOp>(
          loc, localLastValidRowPlusOne, zeroIdx);
      localLastValidRowPlusOne = rewriter.create<arith::MinUIOp>(
          loc, localLastValidRowPlusOne, shardTileRowsVal);

      Value localLastValidColPlusOne = rewriter.create<arith::SubIOp>(
          loc, lastValidColPlusOne, globalColStart);
      localLastValidColPlusOne = rewriter.create<arith::MaxUIOp>(
          loc, localLastValidColPlusOne, zeroIdx);
      localLastValidColPlusOne = rewriter.create<arith::MinUIOp>(
          loc, localLastValidColPlusOne, shardTileColsVal);

      auto outerLoop = rewriter.create<scf::ForOp>(
          loc, localLastValidRowPlusOne, shardTileRowsVal, oneIdx);
      outerLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
      outerLoop->setAttr("d2m.scheduled", rewriter.getUnitAttr());
      rewriter.setInsertionPointToStart(outerLoop.getBody());

      auto innerLoop =
          rewriter.create<scf::ForOp>(loc, zeroIdx, shardTileColsVal, oneIdx);
      rewriter.setInsertionPointToStart(innerLoop.getBody());
      emitFill(outerLoop.getInductionVar(), innerLoop.getInductionVar());

      insertionPoint = outerLoop;
    }

    // =========================================================================
    // LOOP 6: OOB cols [0, lastValidRow+1) x [lastValidCol+1, shardTileCols)
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      Value localLastValidRowPlusOne = rewriter.create<arith::SubIOp>(
          loc, lastValidRowPlusOne, globalRowStart);
      localLastValidRowPlusOne = rewriter.create<arith::MaxUIOp>(
          loc, localLastValidRowPlusOne, zeroIdx);
      localLastValidRowPlusOne = rewriter.create<arith::MinUIOp>(
          loc, localLastValidRowPlusOne, shardTileRowsVal);

      Value localLastValidColPlusOne = rewriter.create<arith::SubIOp>(
          loc, lastValidColPlusOne, globalColStart);
      localLastValidColPlusOne = rewriter.create<arith::MaxUIOp>(
          loc, localLastValidColPlusOne, zeroIdx);
      localLastValidColPlusOne = rewriter.create<arith::MinUIOp>(
          loc, localLastValidColPlusOne, shardTileColsVal);

      auto outerLoop = rewriter.create<scf::ForOp>(
          loc, zeroIdx, localLastValidRowPlusOne, oneIdx);
      outerLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
      outerLoop->setAttr("d2m.scheduled", rewriter.getUnitAttr());
      rewriter.setInsertionPointToStart(outerLoop.getBody());

      auto innerLoop = rewriter.create<scf::ForOp>(
          loc, localLastValidColPlusOne, shardTileColsVal, oneIdx);
      rewriter.setInsertionPointToStart(innerLoop.getBody());
      emitFill(outerLoop.getInductionVar(), innerLoop.getInductionVar());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct D2MDecomposeMasking
    : public impl::D2MDecomposeMaskingBase<D2MDecomposeMasking> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeBlockMaskWritePattern>(ctx);
    patterns.add<DecomposeBlockMaskPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
