// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cmath>
#include <limits>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSEMASKING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Tile dimensions are fixed at 32x32.
constexpr int64_t kTileHeight = 32;
constexpr int64_t kTileWidth = 32;

// Get the fill value as a float based on OOBVal.
static double getFillValueAsDouble(ttcore::OOBVal oobVal) {
  switch (oobVal) {
  case ttcore::OOBVal::Undef:
    return 0.0; // Shouldn't happen, but default to 0
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

/// Decompose BlockMaskWriteOp (DM region) into ops that write
/// row_mask and col_mask tiles to scratch CBs.
///
/// The DM region writes exactly 2 tiles:
/// - Row mask tile at scratch[0] with validRows
/// - Col mask tile at scratch[1] with validCols
///
/// These are computed once based on the logical shape and reused
/// across all tiles in the compute region.
struct DecomposeBlockMaskWritePattern : OpRewritePattern<BlockMaskWriteOp> {
  using OpRewritePattern<BlockMaskWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockMaskWriteOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value rowMaskCB = op.getRowMaskCb();
    Value colMaskCB = op.getColMaskCb();
    Value logicalRowsVal = op.getLogicalRows();
    Value logicalColsVal = op.getLogicalCols();

    // Compute validRows and validCols for this shard.
    // For single-core: validRows = logicalRows % 32 (or 32 if aligned)
    //                  validCols = logicalCols % 32 (or 32 if aligned)
    // The mask tiles represent the partial tile at the boundary.

    Value tileHeightVal =
        rewriter.create<arith::ConstantIndexOp>(loc, kTileHeight);
    Value tileWidthVal =
        rewriter.create<arith::ConstantIndexOp>(loc, kTileWidth);
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // validRows = logicalRows % tileHeight
    // If validRows == 0, it means the last tile is fully valid, so use
    // tileHeight.
    Value rowRemainder =
        rewriter.create<arith::RemUIOp>(loc, logicalRowsVal, tileHeightVal);
    Value rowRemIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rowRemainder, zeroIdx);
    Value validRows = rewriter.create<arith::SelectOp>(
        loc, rowRemIsZero, tileHeightVal, rowRemainder);

    // validCols = logicalCols % tileWidth
    Value colRemainder =
        rewriter.create<arith::RemUIOp>(loc, logicalColsVal, tileWidthVal);
    Value colRemIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, colRemainder, zeroIdx);
    Value validCols = rewriter.create<arith::SelectOp>(
        loc, colRemIsZero, tileWidthVal, colRemainder);

    // Write row mask tile to rowMaskCB.
    rewriter.create<ExperimentalWriteRowMaskTileOp>(loc, validRows, rowMaskCB);

    // Write col mask tile to colMaskCB.
    rewriter.create<ExperimentalWriteColMaskTileOp>(loc, validCols, colMaskCB);

    // Erase the original op.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Decompose BlockMaskOp (compute region) into multiple affine loop nests
/// that handle different tile regions with appropriate masking.
///
/// Loop structure:
/// 1. Interior tiles: [0, lastValidRow) x [0, lastValidCol) - passthrough
/// 2. Last row tiles: [lastValidRow] x [0, lastValidCol) - apply row_mask
/// 3. Last col tiles: [0, lastValidRow) x [lastValidCol] - apply col_mask
/// 4. Corner tile: [lastValidRow, lastValidCol] - apply both masks
/// 5. OOB row tiles: [lastValidRow+1, shardRows) x [0, shardCols) - fill
/// 6. OOB col tiles: [0, lastValidRow+1) x [lastValidCol+1, shardCols) - fill
///
/// Each loop nest is marked with d2m.linalg_root for independent DST
/// management.
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

    // BlockMaskOp decomposition only supports memref semantics.
    if (isa<RankedTensorType>(input.getType())) {
      return rewriter.notifyMatchFailure(op, "tensor semantics not supported");
    }

    auto inputType = cast<MemRefType>(input.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    if (inputShape.size() < 2) {
      return rewriter.notifyMatchFailure(op, "input must have at least 2 dims");
    }

    // Get the tile type for creating scalar constants.
    auto tileType = cast<ttcore::TileType>(inputType.getElementType());
    Type elemType = tileType.getElementType();

    // Shard dimensions are the last 2 dimensions (tile rows x tile cols).
    int64_t shardTileRows = inputShape[inputShape.size() - 2];
    int64_t shardTileCols = inputShape[inputShape.size() - 1];

    // Get the fill value as a float constant.
    double fillValueDouble = getFillValueAsDouble(fillOOBVal);
    Value fillScalar = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, fillValueDouble));

    // Create index constants for mask CB loading.
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // For static bounds, we know at compile time if masking is needed.
    // For dynamic bounds, we use the computed values.

    // Compute validRows and validCols for this shard.
    Value tileHeightVal =
        rewriter.create<arith::ConstantIndexOp>(loc, kTileHeight);
    Value tileWidthVal =
        rewriter.create<arith::ConstantIndexOp>(loc, kTileWidth);

    // validRows = logicalRows % tileHeight (0 means full tile, use tileHeight)
    Value rowRemainder =
        rewriter.create<arith::RemUIOp>(loc, logicalRowsVal, tileHeightVal);
    Value rowRemIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rowRemainder, zeroIdx);
    Value validRows = rewriter.create<arith::SelectOp>(
        loc, rowRemIsZero, tileHeightVal, rowRemainder);

    // validCols = logicalCols % tileWidth (0 means full tile, use tileWidth)
    Value colRemainder =
        rewriter.create<arith::RemUIOp>(loc, logicalColsVal, tileWidthVal);
    Value colRemIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, colRemainder, zeroIdx);
    Value validCols = rewriter.create<arith::SelectOp>(
        loc, colRemIsZero, tileWidthVal, colRemainder);

    // Write mask tiles to scratch CBs before any loops.
    // The masks will be loaded INSIDE each loop body to ensure they're within
    // the scope of that loop's DST acquire.
    if (rowMaskCB) {
      rewriter.create<ExperimentalWriteRowMaskTileOp>(loc, validRows, rowMaskCB);
    }
    if (colMaskCB) {
      rewriter.create<ExperimentalWriteColMaskTileOp>(loc, validCols, colMaskCB);
    }

    // Helper to create a fill tile using ExperimentalTileFillOp.
    auto createFillTile = [&]() {
      return rewriter.create<ExperimentalTileFillOp>(loc, tileType, fillScalar)
          .getResult();
    };

    // Helper to create a nested affine loop and return the innermost body's
    // insertion point along with the loop IVs.
    auto createNestedLoop = [&](int64_t rowStart, int64_t rowEnd,
                                int64_t colStart, int64_t colEnd,
                                StringRef loopName)
        -> std::tuple<affine::AffineForOp, Value, Value> {
      // Skip if bounds are empty.
      if (rowStart >= rowEnd || colStart >= colEnd) {
        return {nullptr, nullptr, nullptr};
      }

      auto outerLoop =
          rewriter.create<affine::AffineForOp>(loc, rowStart, rowEnd, 1);
      outerLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
      outerLoop->setAttr("d2m.scheduled", rewriter.getUnitAttr());

      rewriter.setInsertionPointToStart(outerLoop.getBody());
      Value tileRowIdx = outerLoop.getInductionVar();

      auto innerLoop =
          rewriter.create<affine::AffineForOp>(loc, colStart, colEnd, 1);
      rewriter.setInsertionPointToStart(innerLoop.getBody());
      Value tileColIdx = innerLoop.getInductionVar();

      return {outerLoop, tileRowIdx, tileColIdx};
    };

    // Helper to create a single-iteration loop (for edge cases).
    auto createSingleRowLoop = [&](int64_t row, int64_t colStart,
                                   int64_t colEnd, StringRef loopName)
        -> std::tuple<affine::AffineForOp, Value, Value> {
      if (colStart >= colEnd) {
        return {nullptr, nullptr, nullptr};
      }

      // Use a 1-iteration outer loop so we still have a loop nest structure.
      auto outerLoop =
          rewriter.create<affine::AffineForOp>(loc, row, row + 1, 1);
      outerLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
      outerLoop->setAttr("d2m.scheduled", rewriter.getUnitAttr());

      rewriter.setInsertionPointToStart(outerLoop.getBody());
      Value tileRowIdx = outerLoop.getInductionVar();

      auto innerLoop =
          rewriter.create<affine::AffineForOp>(loc, colStart, colEnd, 1);
      rewriter.setInsertionPointToStart(innerLoop.getBody());
      Value tileColIdx = innerLoop.getInductionVar();

      return {outerLoop, tileRowIdx, tileColIdx};
    };

    auto createSingleColLoop = [&](int64_t rowStart, int64_t rowEnd,
                                   int64_t col, StringRef loopName)
        -> std::tuple<affine::AffineForOp, Value, Value> {
      if (rowStart >= rowEnd) {
        return {nullptr, nullptr, nullptr};
      }

      auto outerLoop =
          rewriter.create<affine::AffineForOp>(loc, rowStart, rowEnd, 1);
      outerLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
      outerLoop->setAttr("d2m.scheduled", rewriter.getUnitAttr());

      rewriter.setInsertionPointToStart(outerLoop.getBody());
      Value tileRowIdx = outerLoop.getInductionVar();

      // Single-iteration inner loop.
      auto innerLoop =
          rewriter.create<affine::AffineForOp>(loc, col, col + 1, 1);
      rewriter.setInsertionPointToStart(innerLoop.getBody());
      Value tileColIdx = innerLoop.getInductionVar();

      return {outerLoop, tileRowIdx, tileColIdx};
    };

    // For single-core with static shapes, compute bounds at compile time.
    // lastValidTileRow and lastValidTileCol are compile-time computable if
    // logicalRows/logicalCols are constants.

    // For now, extract static values from the ops if possible.
    // This is a simplification - a full implementation would handle dynamic
    // cases.
    auto getConstantIndex = [](Value v) -> std::optional<int64_t> {
      if (auto constOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
        return constOp.value();
      }
      return std::nullopt;
    };

    std::optional<int64_t> logicalRowsOpt = getConstantIndex(logicalRowsVal);
    std::optional<int64_t> logicalColsOpt = getConstantIndex(logicalColsVal);

    if (!logicalRowsOpt || !logicalColsOpt) {
      return rewriter.notifyMatchFailure(
          op, "dynamic logical shape not yet supported");
    }

    int64_t logicalRows = *logicalRowsOpt;
    int64_t logicalCols = *logicalColsOpt;

    // Compute static bounds.
    int64_t lastValidRow = (logicalRows - 1) / kTileHeight;
    int64_t lastValidCol = (logicalCols - 1) / kTileWidth;
    bool needsRowMask = (logicalRows % kTileHeight) != 0;
    bool needsColMask = (logicalCols % kTileWidth) != 0;

    // Clamp to shard bounds.
    lastValidRow = std::min(lastValidRow, shardTileRows - 1);
    lastValidCol = std::min(lastValidCol, shardTileCols - 1);

    // Track the insertion point to restore after each loop nest.
    OpBuilder::InsertionGuard guard(rewriter);
    Operation *insertAfter = op;

    // Helper to emit loop body for passthrough (interior tiles).
    auto emitPassthrough = [&](Value rowIdx, Value colIdx) {
      auto inputTile = rewriter.create<affine::AffineLoadOp>(
          loc, input, ValueRange{rowIdx, colIdx});
      rewriter.create<affine::AffineStoreOp>(loc, inputTile.getResult(), output,
                                             ValueRange{rowIdx, colIdx});
    };

    // Compute validRows/validCols for partial tiles (when no mask CBs
    // provided). These are the remainders: logicalRows % 32 (or 32 if aligned).
    int64_t validRowsStatic = logicalRows % kTileHeight;
    if (validRowsStatic == 0) {
      validRowsStatic = kTileHeight;
    }
    int64_t validColsStatic = logicalCols % kTileWidth;
    if (validColsStatic == 0) {
      validColsStatic = kTileWidth;
    }

    // Helper to emit row-masked tiles (last row, not corner).
    auto emitRowMasked = [&](Value rowIdx, Value colIdx) {
      auto inputTile = rewriter.create<affine::AffineLoadOp>(
          loc, input, ValueRange{rowIdx, colIdx});
      auto fillTile = createFillTile();

      if (rowMaskCB) {
        // Load mask from CB inside the loop body (within DST acquire scope).
        auto rowMaskTile = rewriter.create<affine::AffineLoadOp>(
            loc, rowMaskCB, ValueRange{zeroIdx, zeroIdx});
        auto result = rewriter.create<TileWhereOp>(
            loc, tileType, rowMaskTile.getResult(), inputTile.getResult(),
            fillTile);
        rewriter.create<affine::AffineStoreOp>(loc, result.getResult(), output,
                                               ValueRange{rowIdx, colIdx});
      } else {
        // Generate mask dynamically via SFPU.
        Value validRowsVal =
            rewriter.create<arith::ConstantIndexOp>(loc, validRowsStatic);
        auto mask = rewriter.create<ExperimentalTileRowMaskOp>(loc, tileType,
                                                               validRowsVal);
        auto result = rewriter.create<TileWhereOp>(
            loc, tileType, mask.getResult(), inputTile.getResult(), fillTile);
        rewriter.create<affine::AffineStoreOp>(loc, result.getResult(), output,
                                               ValueRange{rowIdx, colIdx});
      }
    };

    // Helper to emit col-masked tiles (last col, not corner).
    auto emitColMasked = [&](Value rowIdx, Value colIdx) {
      auto inputTile = rewriter.create<affine::AffineLoadOp>(
          loc, input, ValueRange{rowIdx, colIdx});
      auto fillTile = createFillTile();

      if (colMaskCB) {
        // Load mask from CB inside the loop body (within DST acquire scope).
        auto colMaskTile = rewriter.create<affine::AffineLoadOp>(
            loc, colMaskCB, ValueRange{zeroIdx, zeroIdx});
        auto result = rewriter.create<TileWhereOp>(
            loc, tileType, colMaskTile.getResult(), inputTile.getResult(),
            fillTile);
        rewriter.create<affine::AffineStoreOp>(loc, result.getResult(), output,
                                               ValueRange{rowIdx, colIdx});
      } else {
        // Generate mask dynamically via SFPU.
        Value validColsVal =
            rewriter.create<arith::ConstantIndexOp>(loc, validColsStatic);
        auto mask = rewriter.create<ExperimentalTileColMaskOp>(loc, tileType,
                                                               validColsVal);
        auto result = rewriter.create<TileWhereOp>(
            loc, tileType, mask.getResult(), inputTile.getResult(), fillTile);
        rewriter.create<affine::AffineStoreOp>(loc, result.getResult(), output,
                                               ValueRange{rowIdx, colIdx});
      }
    };

    // Helper to emit loop body for OOB fill.
    auto emitFill = [&](Value rowIdx, Value colIdx) {
      auto fillTile = createFillTile();
      rewriter.create<affine::AffineStoreOp>(loc, fillTile, output,
                                             ValueRange{rowIdx, colIdx});
    };

    // Helper to emit corner tile - applies BOTH row and col masks in sequence.
    // This keeps both operations in the same tile_regs block, avoiding the need
    // to read back from the output CB between masks.
    auto emitCornerMasked = [&](Value rowIdx, Value colIdx) {
      // Load input tile
      auto inputTile = rewriter.create<affine::AffineLoadOp>(
          loc, input, ValueRange{rowIdx, colIdx});

      // First fill for row mask
      auto fillTile1 = createFillTile();

      // Load row mask from CB
      auto rowMaskTile = rewriter.create<affine::AffineLoadOp>(
          loc, rowMaskCB, ValueRange{zeroIdx, zeroIdx});

      // Apply row mask: where(rowMask, input, fill) -> rowMaskedResult
      auto rowMaskedResult = rewriter.create<TileWhereOp>(
          loc, tileType, rowMaskTile.getResult(), inputTile.getResult(),
          fillTile1);

      // Second fill for col mask
      auto fillTile2 = createFillTile();

      // Load col mask from CB
      auto colMaskTile = rewriter.create<affine::AffineLoadOp>(
          loc, colMaskCB, ValueRange{zeroIdx, zeroIdx});

      // Apply col mask: where(colMask, rowMaskedResult, fill) -> finalResult
      auto finalResult = rewriter.create<TileWhereOp>(
          loc, tileType, colMaskTile.getResult(), rowMaskedResult.getResult(),
          fillTile2);

      // Store final result
      rewriter.create<affine::AffineStoreOp>(loc, finalResult.getResult(),
                                             output, ValueRange{rowIdx, colIdx});
    };

    // === LOOP 1: Interior tiles (no masking needed) ===
    // Bounds: [0, lastValidRow) x [0, lastValidCol)
    if (lastValidRow > 0 && lastValidCol > 0) {
      rewriter.setInsertionPointAfter(insertAfter);
      auto [loop, rowIdx, colIdx] =
          createNestedLoop(0, lastValidRow, 0, lastValidCol, "interior");
      if (loop) {
        emitPassthrough(rowIdx, colIdx);
        insertAfter = loop;
      }
    }

    // === LOOP 2: Last row tiles (row mask, EXCLUDING corner if col mask
    // needed) === When col mask is needed, corner is handled separately to
    // avoid DST conflicts. Bounds: [lastValidRow] x [0, lastValidCol) when
    // needsColMask
    //         [lastValidRow] x [0, lastValidCol+1) when !needsColMask
    if (needsRowMask) {
      int colEnd = needsColMask ? lastValidCol : lastValidCol + 1;
      if (colEnd > 0) {
        rewriter.setInsertionPointAfter(insertAfter);
        auto [loop, rowIdx, colIdx] =
            createSingleRowLoop(lastValidRow, 0, colEnd, "last_row");
        if (loop) {
          emitRowMasked(rowIdx, colIdx);
          insertAfter = loop;
        }
      }
    } else if (lastValidRow >= 0) {
      // Row is fully valid, just passthrough.
      int colEnd = needsColMask ? lastValidCol : lastValidCol + 1;
      if (colEnd > 0) {
        rewriter.setInsertionPointAfter(insertAfter);
        auto [loop, rowIdx, colIdx] =
            createSingleRowLoop(lastValidRow, 0, colEnd, "last_row_full");
        if (loop) {
          emitPassthrough(rowIdx, colIdx);
          insertAfter = loop;
        }
      }
    }

    // === LOOP 3a: Non-corner col tiles (col mask, reads from INPUT) ===
    // Bounds: [0, lastValidRow) x [lastValidCol] when needsRowMask (corner
    // handled separately) Bounds: [0, lastValidRow+1) x [lastValidCol] when
    // !needsRowMask (no corner special case)
    if (needsColMask) {
      int rowEnd = needsRowMask ? lastValidRow : lastValidRow + 1;
      if (rowEnd > 0) {
        rewriter.setInsertionPointAfter(insertAfter);
        auto [loop, rowIdx, colIdx] =
            createSingleColLoop(0, rowEnd, lastValidCol, "last_col");
        if (loop) {
          emitColMasked(rowIdx, colIdx);
          insertAfter = loop;
        }
      }
    } else if (lastValidCol >= 0) {
      // Col is fully valid, just passthrough.
      int rowEnd = needsRowMask ? lastValidRow : lastValidRow + 1;
      if (rowEnd > 0) {
        rewriter.setInsertionPointAfter(insertAfter);
        auto [loop, rowIdx, colIdx] =
            createSingleColLoop(0, rowEnd, lastValidCol, "last_col_full");
        if (loop) {
          emitPassthrough(rowIdx, colIdx);
          insertAfter = loop;
        }
      }
    }

    // === LOOP 3b: Corner tile - apply BOTH masks in single block ===
    // When both row and col masking apply, corner needs both masks applied
    // sequentially within the SAME tile_regs block to avoid CB sync issues.
    // The combined helper applies: input -> rowMask -> colMask -> output
    if (needsRowMask && needsColMask && lastValidRow >= 0 &&
        lastValidCol >= 0 && rowMaskCB && colMaskCB) {
      rewriter.setInsertionPointAfter(insertAfter);
      auto [loop, rowIdx, colIdx] = createSingleColLoop(
          lastValidRow, lastValidRow + 1, lastValidCol, "corner");
      if (loop) {
        emitCornerMasked(rowIdx, colIdx);
        insertAfter = loop;
      }
    } else if (needsRowMask && !needsColMask && lastValidRow >= 0 &&
               lastValidCol >= 0) {
      // Row mask only - corner already handled by LOOP 2, but we still need
      // to process the corner column for non-masked passthrough.
      rewriter.setInsertionPointAfter(insertAfter);
      auto [loop, rowIdx, colIdx] = createSingleColLoop(
          lastValidRow, lastValidRow + 1, lastValidCol, "corner_passthrough");
      if (loop) {
        emitPassthrough(rowIdx, colIdx);
        insertAfter = loop;
      }
    }

    // === LOOP 5: OOB row tiles (full fill) ===
    // Bounds: [lastValidRow+1, shardTileRows) x [0, shardTileCols)
    if (lastValidRow + 1 < shardTileRows) {
      rewriter.setInsertionPointAfter(insertAfter);
      auto [loop, rowIdx, colIdx] = createNestedLoop(
          lastValidRow + 1, shardTileRows, 0, shardTileCols, "oob_rows");
      if (loop) {
        emitFill(rowIdx, colIdx);
        insertAfter = loop;
      }
    }

    // === LOOP 6: OOB col tiles (full fill, excluding OOB rows already handled)
    // ===
    // Bounds: [0, lastValidRow+1) x [lastValidCol+1, shardTileCols)
    if (lastValidCol + 1 < shardTileCols) {
      rewriter.setInsertionPointAfter(insertAfter);
      auto [loop, rowIdx, colIdx] = createNestedLoop(
          0, lastValidRow + 1, lastValidCol + 1, shardTileCols, "oob_cols");
      if (loop) {
        emitFill(rowIdx, colIdx);
        insertAfter = loop;
      }
    }

    rewriter.replaceOp(op, op.getOutput());
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
