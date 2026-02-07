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

    auto genericOp = op->getParentOfType<GenericOp>();
    if (!genericOp) {
      return rewriter.notifyMatchFailure(
          op, "ArangeBlockOp must be inside a GenericOp");
    }

    ttcore::GridAttr gridAttr = genericOp.getGrid();
    ArrayRef<int64_t> gridShape = gridAttr.getShape();
    if (gridShape.size() < 2) {
      return rewriter.notifyMatchFailure(
          op, "grid must have at least 2 dimensions");
    }

    auto tileType = cast<ttcore::TileType>(outputType.getElementType());
    Type elemType = tileType.getElementType();

    int64_t numTileRows = outputShape[outputShape.size() - 2];
    int64_t numTileCols = outputShape[outputShape.size() - 1];
    // Total tiles across all cores.
    int64_t totalTileCols = numTileCols * gridShape[gridShape.size() - 1];

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value numTileRowsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, numTileRows);
    Value numTileColsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, numTileCols);

    // === STEP 1: Write the scratch tile ===
    TT_assert(indexTileMemref);
    rewriter.create<ExperimentalWriteFullIndexTileOp>(loc, indexTileMemref);

    // === STEP 2: Scalar constants for arange start and step ===
    Value startF = rewriter.create<arith::ConstantOp>(
        loc, elemType,
        rewriter.getFloatAttr(elemType, static_cast<double>(start)));
    Value stepF = rewriter.create<arith::ConstantOp>(
        loc, elemType,
        rewriter.getFloatAttr(elemType, static_cast<double>(step)));

    // === STEP 3: Create nested loops over tiles ===
    // Get this core's coordinates.
    Value coreY = rewriter.create<CoreIndexOp>(
        loc, rewriter.getIndexType(), rewriter.getI64IntegerAttr(0), nullptr);
    Value coreX = rewriter.create<CoreIndexOp>(
        loc, rewriter.getIndexType(), rewriter.getI64IntegerAttr(1), nullptr);

    auto outerLoop =
        rewriter.create<scf::ForOp>(loc, zeroIdx, numTileRowsVal, oneIdx);
    rewriter.setInsertionPointToStart(outerLoop.getBody());

    auto innerLoop =
        rewriter.create<scf::ForOp>(loc, zeroIdx, numTileColsVal, oneIdx);
    // Mark the INNER loop as the compute root, since that's where
    // the actual compute operations are emitted. This ensures DST
    // syncs are placed inside the inner loop body, not the outer.
    // Since we emit an scf.for directly, we must tag this here
    // since linalg-to-affine and d2m-op-scheduler won't process this.
    innerLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
    innerLoop->setAttr("d2m.scheduled", rewriter.getUnitAttr());
    rewriter.setInsertionPointToStart(innerLoop.getBody());

    Value tileRowIdx = outerLoop.getInductionVar();
    Value tileColIdx = innerLoop.getInductionVar();

    // === STEP 4: Load scratch tile ===
    Value localIndexTile =
        rewriter
            .create<memref::LoadOp>(loc, indexTileMemref,
                                    ValueRange{zeroIdx, zeroIdx})
            .getResult();

    // === STEP 5: Compute tile offset as scalar ===
    Value shardTileRowsIdx =
        rewriter.create<arith::ConstantIndexOp>(loc, numTileRows);
    Value shardTileColsIdx =
        rewriter.create<arith::ConstantIndexOp>(loc, numTileCols);
    Value totalTileColsIdx =
        rewriter.create<arith::ConstantIndexOp>(loc, totalTileCols);
    Value const32Idx = rewriter.create<arith::ConstantIndexOp>(loc, 32);
    // globalTileRow = coreY * shardTileRows + localTileRow
    Value globalTileRow = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, coreY, shardTileRowsIdx),
        tileRowIdx);
    // globalTileCol = coreX * shardTileCols + localTileCol
    Value globalTileCol = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, coreX, shardTileColsIdx),
        tileColIdx);

    // Row contribution: globalTileRow * totalTileCols * 32 * 32
    Value rowContrib = rewriter.create<arith::MulIOp>(
        loc,
        rewriter.create<arith::MulIOp>(
            loc,
            rewriter.create<arith::MulIOp>(loc, globalTileRow,
                                           totalTileColsIdx),
            const32Idx),
        const32Idx);
    // Column contribution: globalTileCol * 32
    Value colContrib =
        rewriter.create<arith::MulIOp>(loc, globalTileCol, const32Idx);
    // Total offset (index type)
    Value tileOffsetIdx =
        rewriter.create<arith::AddIOp>(loc, rowContrib, colContrib);
    Value tileOffsetI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), tileOffsetIdx);
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
