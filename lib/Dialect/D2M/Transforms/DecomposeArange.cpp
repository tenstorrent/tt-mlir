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
    TT_assertv(outputType, "output must be a memref, run after bufferization");

    ArrayRef<int64_t> outputShape = outputType.getShape();
    TT_assertv(outputShape.size() >= 2ul, "output must have at least 2 dims");

    auto genericOp = op->getParentOfType<GenericOp>();
    TT_assertv(genericOp, "ArangeBlockOp must be inside a GenericOp");

    ttcore::GridAttr gridAttr = genericOp.getGrid();
    ArrayRef<int64_t> gridShape = gridAttr.getShape();
    TT_assertv(gridShape.size() >= 2ul, "grid must have at least 2 dimensions");

    auto tileType = cast<ttcore::TileType>(outputType.getElementType());
    Type elemType = tileType.getElementType();

    int64_t numTileRows = outputShape[outputShape.size() - 2];
    int64_t numTileCols = outputShape[outputShape.size() - 1];
    // Total tiles across all cores.
    int64_t totalTileCols = numTileCols * gridShape[gridShape.size() - 1];

    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIdx = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value numTileRowsVal =
        arith::ConstantIndexOp::create(rewriter, loc, numTileRows);
    Value numTileColsVal =
        arith::ConstantIndexOp::create(rewriter, loc, numTileCols);

    // === STEP 1: Write the scratch tile ===
    TT_assert(indexTileMemref);
    FillArangeTileOp::create(rewriter, loc, indexTileMemref);

    // === STEP 2: Scalar constants for arange start and step ===
    Value startF = arith::ConstantOp::create(
        rewriter, loc, elemType,
        rewriter.getFloatAttr(elemType, static_cast<double>(start)));
    Value stepF = arith::ConstantOp::create(
        rewriter, loc, elemType,
        rewriter.getFloatAttr(elemType, static_cast<double>(step)));

    // === STEP 3: Create nested loops over tiles ===
    // Get this core's coordinates.
    Value coreY = CoreIndexOp::create(rewriter, loc, rewriter.getIndexType(),
                                      rewriter.getI64IntegerAttr(0), nullptr);
    Value coreX = CoreIndexOp::create(rewriter, loc, rewriter.getIndexType(),
                                      rewriter.getI64IntegerAttr(1), nullptr);

    auto outerLoop =
        scf::ForOp::create(rewriter, loc, zeroIdx, numTileRowsVal, oneIdx);
    rewriter.setInsertionPointToStart(outerLoop.getBody());

    auto innerLoop =
        scf::ForOp::create(rewriter, loc, zeroIdx, numTileColsVal, oneIdx);
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
        memref::LoadOp::create(rewriter, loc, indexTileMemref,
                               ValueRange{zeroIdx, zeroIdx})
            .getResult();

    // === STEP 5: Compute tile offset as scalar ===
    Value shardTileRowsIdx =
        arith::ConstantIndexOp::create(rewriter, loc, numTileRows);
    Value shardTileColsIdx =
        arith::ConstantIndexOp::create(rewriter, loc, numTileCols);
    Value totalTileColsIdx =
        arith::ConstantIndexOp::create(rewriter, loc, totalTileCols);
    Value const32Idx = arith::ConstantIndexOp::create(rewriter, loc, 32);
    // globalTileRow = coreY * shardTileRows + localTileRow
    Value globalTileRow = arith::AddIOp::create(
        rewriter, loc,
        arith::MulIOp::create(rewriter, loc, coreY, shardTileRowsIdx),
        tileRowIdx);
    // globalTileCol = coreX * shardTileCols + localTileCol
    Value globalTileCol = arith::AddIOp::create(
        rewriter, loc,
        arith::MulIOp::create(rewriter, loc, coreX, shardTileColsIdx),
        tileColIdx);

    // Row contribution: globalTileRow * totalTileCols * 32 * 32
    Value rowContrib = arith::MulIOp::create(
        rewriter, loc,
        arith::MulIOp::create(rewriter, loc,
                              arith::MulIOp::create(rewriter, loc,
                                                    globalTileRow,
                                                    totalTileColsIdx),
                              const32Idx),
        const32Idx);
    // Column contribution: globalTileCol * 32
    Value colContrib =
        arith::MulIOp::create(rewriter, loc, globalTileCol, const32Idx);
    // Total offset (index type)
    Value tileOffsetIdx =
        arith::AddIOp::create(rewriter, loc, rowContrib, colContrib);
    Value tileOffsetI64 = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI64Type(), tileOffsetIdx);
    Value tileOffsetF =
        arith::SIToFPOp::create(rewriter, loc, elemType, tileOffsetI64);

    // === STEP 6: Tile arithmetic with scalar RHS ===
    Value globalIndexTile =
        TileAddOp::create(rewriter, loc, tileType, localIndexTile, tileOffsetF)
            .getResult();
    Value scaledTile =
        TileMulOp::create(rewriter, loc, tileType, globalIndexTile, stepF)
            .getResult();
    Value resultTile =
        TileAddOp::create(rewriter, loc, tileType, scaledTile, startF)
            .getResult();

    // === STEP 7: Store result tile to output ===
    memref::StoreOp::create(rewriter, loc, resultTile, output,
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
