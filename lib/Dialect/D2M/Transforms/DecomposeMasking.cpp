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
static Type getScalarElementType(Type type) {
  if (auto tile = dyn_cast<ttcore::TileType>(type)) {
    return tile.getElementType();
  }
  if (auto memref = dyn_cast<MemRefType>(type)) {
    return getScalarElementType(memref.getElementType());
  }
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    return getScalarElementType(tensor.getElementType());
  }
  return type;
}

/// Decompose BlockMaskOp into linalg.generic with per-tile masking.
///
/// This pattern handles complete tile out-of-bounds masking. When a tile's
/// starting position is entirely beyond the logical bounds, the entire tile
/// is replaced with the fill value.
///
/// For multicore execution, each core processes a shard of tiles. We use
/// CoreIndexOp to get the grid position and compute global tile indices as:
///   globalTileRow = coreRowIdx * shardRows + localTileRow
///   globalTileCol = coreColIdx * shardCols + localTileCol
///
/// TODO (#6311): Partial tile masking (when a tile straddles the boundary)
/// requires per-element index tiles and will be implemented in a follow-up.
class DecomposeBlockMaskPattern : public OpRewritePattern<BlockMaskOp> {
public:
  using OpRewritePattern<BlockMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockMaskOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    Value logicalRows = op.getLogicalRows();
    Value logicalCols = op.getLogicalCols();
    ttcore::OOBVal fillValue = op.getFillValue();

    auto inputType = dyn_cast<ShapedType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "input must be a shaped type");
    }

    // Get the element type (should be a tile type).
    Type elementType = inputType.getElementType();
    auto tileType = dyn_cast<ttcore::TileType>(elementType);
    if (!tileType) {
      return rewriter.notifyMatchFailure(op,
                                         "element type must be a tile type");
    }

    Type scalarType = getScalarElementType(elementType);
    ArrayRef<int64_t> blockShape = inputType.getShape();
    size_t blockRank = blockShape.size();

    // Get shard shape (tiles per core). The block shape is what each core
    // sees, so for a 4x4 grid with 1 tile per core, blockShape is [1, 1].
    int64_t shardRows = blockShape[0];
    int64_t shardCols = blockShape[1];

    // Build identity indexing maps for linalg.generic (input and output).
    SmallVector<AffineMap> linalgIndexingMaps(
        2, rewriter.getMultiDimIdentityMap(blockRank));
    SmallVector<mlir::utils::IteratorType> linalgIteratorTypes(
        blockRank, mlir::utils::IteratorType::parallel);

    // Tile dimensions (assume 32x32).
    constexpr int64_t tileH = 32;
    constexpr int64_t tileW = 32;

    // BlockMaskOp decomposition only supports memref semantics.
    // For tensor semantics, let the op survive through bufferization first.
    if (isa<RankedTensorType>(input.getType())) {
      return rewriter.notifyMatchFailure(
          op, "decomposition requires memref types; run after bufferization");
    }

    // Create linalg.generic that iterates over tiles in the block.
    // For memref semantics, linalg.generic has no results (side-effecting).
    rewriter.create<mlir::linalg::GenericOp>(
        loc,
        /* result types */ TypeRange{},
        /* inputs */ input,
        /* outputs */ output, linalgIndexingMaps, linalgIteratorTypes,
        [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
            mlir::ValueRange bbArgs) {
          // bbArgs[0] is the input tile, bbArgs[1] is the output tile.
          Value inputTile = bbArgs[0];
          Type resultType = bbArgs[1].getType();

          // Get core indices from the surrounding d2m.generic grid.
          // CoreIndexOp(0) = row, CoreIndexOp(1) = col in the grid.
          Value coreRowIdx = bbBuilder.create<CoreIndexOp>(bbLoc, int64_t{0});
          Value coreColIdx = bbBuilder.create<CoreIndexOp>(bbLoc, int64_t{1});

          // Get local tile indices within the shard using linalg.index.
          // For a 1x1 shard this is always (0, 0), but for larger shards
          // we iterate over multiple tiles per core.
          Value localTileRowIdx = bbBuilder.create<linalg::IndexOp>(bbLoc, 0);
          Value localTileColIdx = bbBuilder.create<linalg::IndexOp>(bbLoc, 1);

          // Compute global tile indices:
          // globalTileRow = coreRowIdx * shardRows + localTileRow
          Value shardRowsConst =
              bbBuilder.create<arith::ConstantIndexOp>(bbLoc, shardRows);
          Value shardColsConst =
              bbBuilder.create<arith::ConstantIndexOp>(bbLoc, shardCols);
          Value coreRowOffset = bbBuilder.create<arith::MulIOp>(
              bbLoc, coreRowIdx, shardRowsConst);
          Value coreColOffset = bbBuilder.create<arith::MulIOp>(
              bbLoc, coreColIdx, shardColsConst);
          Value globalTileRowIdx = bbBuilder.create<arith::AddIOp>(
              bbLoc, coreRowOffset, localTileRowIdx);
          Value globalTileColIdx = bbBuilder.create<arith::AddIOp>(
              bbLoc, coreColOffset, localTileColIdx);

          // Compute global element offsets for this tile.
          Value tileHConst =
              bbBuilder.create<arith::ConstantIndexOp>(bbLoc, tileH);
          Value tileWConst =
              bbBuilder.create<arith::ConstantIndexOp>(bbLoc, tileW);
          Value globalRowStart = bbBuilder.create<arith::MulIOp>(
              bbLoc, globalTileRowIdx, tileHConst);
          Value globalColStart = bbBuilder.create<arith::MulIOp>(
              bbLoc, globalTileColIdx, tileWConst);

          // Check if tile is completely out of bounds.
          Value rowOOB = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::sge, globalRowStart, logicalRows);
          Value colOOB = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::sge, globalColStart, logicalCols);
          Value entireTileOOB =
              bbBuilder.create<arith::OrIOp>(bbLoc, rowOOB, colOOB);

          // Create constants for blending.
          double fillVal = getOOBValAsFloat(fillValue);
          Value fillConst =
              createScalarConstant(bbBuilder, bbLoc, fillVal, scalarType);
          Value zero = createScalarConstant(bbBuilder, bbLoc, 0.0, scalarType);
          Value one = createScalarConstant(bbBuilder, bbLoc, 1.0, scalarType);

          // Blend using: result = input * mulFactor + addend
          // Where mulFactor = select(oob, 0, 1) and addend = select(oob, fill,
          // 0).
          //
          // When NOT OOB: result = input * 1 + 0 = input.
          // When OOB:     result = input * 0 + fill = fill.
          Value mulFactor = bbBuilder.create<arith::SelectOp>(
              bbLoc, entireTileOOB, zero, one);
          Value addend = bbBuilder.create<arith::SelectOp>(bbLoc, entireTileOOB,
                                                           fillConst, zero);

          // Multiply output by select result, should be zero if OOB.
          Value zeroPadded = bbBuilder.create<TileMulOp>(bbLoc, resultType,
                                                         inputTile, mulFactor);
          // Add fill value if OOB.
          Value filled = bbBuilder.create<TileAddOp>(bbLoc, resultType,
                                                     zeroPadded, addend);

          bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, filled);
        });

    rewriter.eraseOp(op);
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
    patterns.add<DecomposeBlockMaskPattern>(ctx);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
