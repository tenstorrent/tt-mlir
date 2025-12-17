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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSEMASKING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Get float value for OOBVal enum.
static double getOOBValAsFloat(ttcore::OOBVal oobVal) {
  switch (oobVal) {
  case ttcore::OOBVal::Zero:
    return 0.0;
  case ttcore::OOBVal::One:
    return 1.0;
  case ttcore::OOBVal::Inf:
    return std::numeric_limits<double>::infinity();
  case ttcore::OOBVal::NegInf:
    return -std::numeric_limits<double>::infinity();
  case ttcore::OOBVal::Undef:
    llvm_unreachable("Undef OOBVal should not reach masking decomposition");
  }
  llvm_unreachable("Unknown OOBVal");
}

// Create a constant scalar value for use with tile ops.
static Value createScalarConstant(OpBuilder &builder, Location loc, double val,
                                  Type elementType) {
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    return builder.create<arith::ConstantOp>(
        loc, builder.getFloatAttr(floatType, val));
  }
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    return builder.create<arith::ConstantOp>(
        loc, builder.getIntegerAttr(intType, static_cast<int64_t>(val)));
  }
  llvm_unreachable("Unsupported element type for constant creation");
}

// Get the scalar element type from a tile type.
static Type getScalarElementType(Type tileType) {
  if (auto tile = dyn_cast<ttcore::TileType>(tileType)) {
    return tile.getElementType();
  }
  return tileType;
}

/// Decompose tile_mask_boundary into primitive tile arithmetic.
///
/// This pattern handles complete tile out-of-bounds masking. When a tile's
/// starting position is entirely beyond the logical bounds, the entire tile
/// is replaced with the fill value.
///
/// TODO (#6311): Partial tile masking (when a tile straddles the boundary)
/// requires per-element index tiles and will be implemented in a follow-up.
class DecomposeTileMaskBoundaryPattern
    : public OpRewritePattern<TileMaskBoundaryOp> {
public:
  using OpRewritePattern<TileMaskBoundaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TileMaskBoundaryOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Type resultType = op.getResult().getType();
    ArrayRef<int64_t> logicalShape = op.getLogicalShape();
    ttcore::OOBVal fillValue = op.getFillValue();

    Type scalarType = getScalarElementType(resultType);

    // Tile dimensions (assume 32x32).
    constexpr int64_t tileH = 32;
    constexpr int64_t tileW = 32;

    // Get current tile indices using linalg.index.
    // The TileMaskBoundaryOp is inside a linalg.generic that iterates over
    // tiles in the shard. linalg.index gives us the iteration indices.
    Value tileRowIdx = rewriter.create<linalg::IndexOp>(loc, 0);
    Value tileColIdx = rewriter.create<linalg::IndexOp>(loc, 1);

    // Compute global element offsets for this tile.
    Value tileHConst = rewriter.create<arith::ConstantIndexOp>(loc, tileH);
    Value tileWConst = rewriter.create<arith::ConstantIndexOp>(loc, tileW);
    Value globalRowStart =
        rewriter.create<arith::MulIOp>(loc, tileRowIdx, tileHConst);
    Value globalColStart =
        rewriter.create<arith::MulIOp>(loc, tileColIdx, tileWConst);

    // Get logical bounds as index values.
    Value logicalHIdx = rewriter.create<arith::ConstantIndexOp>(
        loc, logicalShape[logicalShape.size() - 2]);
    Value logicalWIdx = rewriter.create<arith::ConstantIndexOp>(
        loc, logicalShape[logicalShape.size() - 1]);

    // Check if tile is completely out of bounds.
    Value rowOOB = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, globalRowStart, logicalHIdx);
    Value colOOB = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, globalColStart, logicalWIdx);
    Value entireTileOOB = rewriter.create<arith::OrIOp>(loc, rowOOB, colOOB);

    // Create constants for blending.
    double fillVal = getOOBValAsFloat(fillValue);
    Value fillConst = createScalarConstant(rewriter, loc, fillVal, scalarType);
    Value zero = createScalarConstant(rewriter, loc, 0.0, scalarType);
    Value one = createScalarConstant(rewriter, loc, 1.0, scalarType);

    // Blend using: result = input * mulFactor + addend
    // Where mulFactor = select(oob, 0, 1) and addend = select(oob, fill, 0).
    //
    // When NOT OOB: result = input * 1 + 0 = input.
    // When OOB:     result = input * 0 + fill = fill.
    Value mulFactor =
        rewriter.create<arith::SelectOp>(loc, entireTileOOB, zero, one);
    Value addend =
        rewriter.create<arith::SelectOp>(loc, entireTileOOB, fillConst, zero);

    // Multiply output by select result, should be zero if OOB.
    Value zeroPadded =
        rewriter.create<TileMulOp>(loc, resultType, input, mulFactor);
    // Add fill value if OOB.
    Value filled =
        rewriter.create<TileAddOp>(loc, resultType, zeroPadded, addend);

    rewriter.replaceOp(op, filled);
    return success();
  }
};

class D2MDecomposeMasking
    : public impl::D2MDecomposeMaskingBase<D2MDecomposeMasking> {
public:
  using D2MDecomposeMaskingBase::D2MDecomposeMaskingBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeTileMaskBoundaryPattern>(ctx);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
