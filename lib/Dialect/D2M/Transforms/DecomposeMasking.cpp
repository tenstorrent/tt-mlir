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

/// Decompose BlockMaskOp into affine loops with tile operations.
///
/// For each tile in the block:
/// - Generate row mask (1.0 for valid rows, 0.0 for OOB rows)
/// - Generate col mask (1.0 for valid cols, 0.0 for OOB cols)
/// - Combine masks via multiplication (AND operation)
/// - Use TileWhereOp to select between input and fill based on mask
///
/// This approach avoids control flow (scf.if) which doesn't work well with
/// DST register management. The mask values naturally handle edge cases:
/// - Fully valid tile: mask is all 1.0, TileWhereOp returns input
/// - Fully OOB tile: mask is all 0.0, TileWhereOp returns fill
/// - Partial tile: mask has 1.0s and 0.0s as expected
struct DecomposeBlockMaskPattern : OpRewritePattern<BlockMaskOp> {
  using OpRewritePattern<BlockMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockMaskOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
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

    // Create tile dimension constants.
    Value tileHeightVal =
        rewriter.create<arith::ConstantIndexOp>(loc, kTileHeight);
    Value tileWidthVal =
        rewriter.create<arith::ConstantIndexOp>(loc, kTileWidth);
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Create nested affine loops over tiles.
    // Mark the outer loop with d2m.linalg_root so InsertDstRegisterAccess
    // processes it for DST management.
    auto outerLoop =
        rewriter.create<affine::AffineForOp>(loc, 0, shardTileRows, 1);
    outerLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());

    rewriter.setInsertionPointToStart(outerLoop.getBody());
    Value tileRowIdx = outerLoop.getInductionVar();

    auto innerLoop =
        rewriter.create<affine::AffineForOp>(loc, 0, shardTileCols, 1);
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    Value tileColIdx = innerLoop.getInductionVar();

    // Calculate global element start positions for this tile.
    // globalStartRow = tileRowIdx * kTileHeight
    // globalStartCol = tileColIdx * kTileWidth
    Value globalStartRow =
        rewriter.create<arith::MulIOp>(loc, tileRowIdx, tileHeightVal);
    Value globalStartCol =
        rewriter.create<arith::MulIOp>(loc, tileColIdx, tileWidthVal);

    // Calculate valid rows/cols within this tile.
    // validRows = max(0, min(kTileHeight, logicalRows - globalStartRow))
    // validCols = max(0, min(kTileWidth, logicalCols - globalStartCol))
    Value rowsRemaining =
        rewriter.create<arith::SubIOp>(loc, logicalRowsVal, globalStartRow);
    Value colsRemaining =
        rewriter.create<arith::SubIOp>(loc, logicalColsVal, globalStartCol);

    // Clamp to [0, tileSize] using cmpi + select (arith.minsi/maxsi don't
    // lower). min(a, b) = select(a < b, a, b) max(a, b) = select(a > b, a, b)
    Value rowsLtTileH = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, rowsRemaining, tileHeightVal);
    Value validRowsRaw = rewriter.create<arith::SelectOp>(
        loc, rowsLtTileH, rowsRemaining, tileHeightVal);
    Value rowsGtZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, validRowsRaw, zeroIdx);
    Value validRows = rewriter.create<arith::SelectOp>(loc, rowsGtZero,
                                                       validRowsRaw, zeroIdx);

    Value colsLtTileW = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, colsRemaining, tileWidthVal);
    Value validColsRaw = rewriter.create<arith::SelectOp>(
        loc, colsLtTileW, colsRemaining, tileWidthVal);
    Value colsGtZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, validColsRaw, zeroIdx);
    Value validCols = rewriter.create<arith::SelectOp>(loc, colsGtZero,
                                                       validColsRaw, zeroIdx);

    // Load tile from input.
    auto inputTile = rewriter.create<affine::AffineLoadOp>(
        loc, input, ValueRange{tileRowIdx, tileColIdx});

    // Generate row mask: 1.0 for rows 0..validRows-1, 0.0 otherwise.
    // Generate col mask: 1.0 for cols 0..validCols-1, 0.0 otherwise.
    auto rowMask =
        rewriter.create<ExperimentalTileRowMaskOp>(loc, tileType, validRows);
    auto colMask =
        rewriter.create<ExperimentalTileColMaskOp>(loc, tileType, validCols);

    // Combined mask = rowMask * colMask (element-wise AND).
    // mask[i,j] = 1.0 if (i < validRows && j < validCols), else 0.0
    auto combinedMask = rewriter.create<TileMulOp>(
        loc, tileType, rowMask.getResult(), colMask.getResult());

    // Create fill tile: input * 0 + fill_value.
    // This gives us a tile with all elements = fill_value.
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, 0.0));
    auto zeros =
        rewriter.create<TileMulOp>(loc, tileType, inputTile.getResult(), zero);
    auto fillTile = rewriter.create<TileAddOp>(loc, tileType, zeros.getResult(),
                                               fillScalar);

    // Apply mask: result = where(mask != 0, input, fill)
    // - Where mask is 1.0 (valid region): select input
    // - Where mask is 0.0 (OOB region): select fill
    auto result = rewriter.create<TileWhereOp>(
        loc, tileType, combinedMask.getResult(), inputTile.getResult(),
        fillTile.getResult());

    // Store result to output.
    rewriter.create<affine::AffineStoreOp>(loc, result.getResult(), output,
                                           ValueRange{tileRowIdx, tileColIdx});

    rewriter.replaceOp(op, op.getOutput());
    return success();
  }
};

struct D2MDecomposeMasking
    : public impl::D2MDecomposeMaskingBase<D2MDecomposeMasking> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeBlockMaskPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
