// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
#define GEN_PASS_DEF_D2MDECOMPOSEARANGE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Decompose ArangeBlockOp into low-level tile operations.
///
/// The arange_block op generates values: output[i] = start + step * i
struct DecomposeArangeBlockPattern : OpRewritePattern<ArangeBlockOp> {
  using OpRewritePattern<ArangeBlockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ArangeBlockOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value output = op.getOutput();
    Value indexTileMemref = op.getIndexTileTensor();
    int64_t start = op.getStart();
    int64_t step = op.getStep();

    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(
          op, "output must be a memref, run after bufferization");
    }

    ArrayRef<int64_t> outputShape = outputType.getShape();
    if (outputShape.size() < 2) {
      return rewriter.notifyMatchFailure(op,
                                         "output must have at least 2 dims");
    }

    auto tileType = cast<ttcore::TileType>(outputType.getElementType());
    Type elemType = tileType.getElementType();

    int64_t numTileRows = outputShape[outputShape.size() - 2];
    int64_t numTileCols = outputShape[outputShape.size() - 1];

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value numTileRowsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, numTileRows);
    Value numTileColsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, numTileCols);

    // === STEP 1: Write scratch tile ===
    rewriter.create<ExperimentalWriteFullIndexTileOp>(loc, indexTileMemref);

    // === STEP 2: Scalar constants for tile arithmetic ===
    Value startF = rewriter.create<arith::ConstantOp>(
        loc, elemType,
        rewriter.getFloatAttr(elemType, static_cast<double>(start)));
    Value stepF = rewriter.create<arith::ConstantOp>(
        loc, elemType,
        rewriter.getFloatAttr(elemType, static_cast<double>(step)));

    // === STEP 3: Create nested loops over tiles ===
    auto outerLoop =
        rewriter.create<scf::ForOp>(loc, zeroIdx, numTileRowsVal, oneIdx);
    rewriter.setInsertionPointToStart(outerLoop.getBody());

    auto innerLoop =
        rewriter.create<scf::ForOp>(loc, zeroIdx, numTileColsVal, oneIdx);
    // Mark the inner loop as the compute root for DST register management.
    // Necessary?
    innerLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
    innerLoop->setAttr("d2m.scheduled", rewriter.getUnitAttr());
    rewriter.setInsertionPointToStart(innerLoop.getBody());

    Value tileRowIdx = outerLoop.getInductionVar();
    Value tileColIdx = innerLoop.getInductionVar();

    // === STEP 4: Load scratch tile (inside the loop?) ===
    Value localIndexTile =
        rewriter
            .create<memref::LoadOp>(loc, indexTileMemref,
                                    ValueRange{zeroIdx, zeroIdx})
            .getResult();

    // === STEP 5: Compute tile offset as scalar ===
    Value tileRowIdxI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), tileRowIdx);
    Value tileColIdxI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), tileColIdx);
    Value numTileColsI64 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(numTileCols));
    Value const32I64 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(32));

    // Row contribution: tile_row * numTileCols * 32 * 32
    Value rowContrib = rewriter.create<arith::MulIOp>(
        loc,
        rewriter.create<arith::MulIOp>(
            loc,
            rewriter.create<arith::MulIOp>(loc, tileRowIdxI64, numTileColsI64),
            const32I64),
        const32I64);
    // Column contribution: tile_col * 32
    Value colContrib =
        rewriter.create<arith::MulIOp>(loc, tileColIdxI64, const32I64);
    // Total offset
    Value tileOffsetI64 =
        rewriter.create<arith::AddIOp>(loc, rowContrib, colContrib);
    Value tileOffsetF =
        rewriter.create<arith::SIToFPOp>(loc, elemType, tileOffsetI64);

    // === STEP 6: Tile arithmetic with scalar RHS ===
    Value globalIndexTile =
        rewriter.create<TileAddOp>(loc, tileType, localIndexTile, tileOffsetF)
            .getResult();
    Value scaledTile =
        rewriter.create<TileMulOp>(loc, tileType, globalIndexTile, stepF)
            .getResult();
    Value resultTile =
        rewriter.create<TileAddOp>(loc, tileType, scaledTile, startF)
            .getResult();

    // === STEP 7: Store result tile to output ===
    rewriter.create<memref::StoreOp>(loc, resultTile, output,
                                     ValueRange{tileRowIdx, tileColIdx});

    rewriter.setInsertionPointAfter(outerLoop);

    rewriter.replaceOp(op, output);
    return success();
  }
};

struct D2MDecomposeArange
    : public impl::D2MDecomposeArangeBase<D2MDecomposeArange> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeArangeBlockPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
