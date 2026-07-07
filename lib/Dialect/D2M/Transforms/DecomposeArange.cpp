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

// Walk up from `op` collecting IVs of enclosing blocking loops (tagged
// `d2m.blocking_loop = <dim>` by D2MGenerateOuterLoops) into `rowBlockIV` and
// `colBlockIV`. These must be folded into the arange offset so each shard
// computes its correct slice rather than identical values. Either IV is null
// if the corresponding blocking loop is absent.
static void collectBlockingLoopIVs(Operation *op, int64_t rank,
                                   Value &rowBlockIV, Value &colBlockIV) {
  rowBlockIV = nullptr;
  colBlockIV = nullptr;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto forOp = dyn_cast<scf::ForOp>(parent);
    if (!forOp) {
      continue;
    }
    auto dimAttr = forOp->getAttrOfType<IntegerAttr>("d2m.blocking_loop");
    if (!dimAttr) {
      continue;
    }
    int64_t dim = dimAttr.getInt();
    if (dim == rank - 1) {
      colBlockIV = forOp.getInductionVar();
    } else if (dim == rank - 2) {
      rowBlockIV = forOp.getInductionVar();
    }
  }
}

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
    bool colMajor = op.getColMajor();

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
    int64_t totalTileRows = numTileRows * gridShape[gridShape.size() - 2];

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value numTileRowsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, numTileRows);
    Value numTileColsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, numTileCols);

    // === STEP 1: Write the scratch tile ===
    TT_assert(indexTileMemref);
    rewriter.create<FillArangeTileOp>(loc, indexTileMemref);

    // === STEP 2: Scalar constants for arange start and step ===
    // arith ops require signless integers; tile element types may be si32/ui32.
    bool isIntElem = isa<IntegerType>(elemType);
    Type scalarType = isIntElem ? cast<Type>(IntegerType::get(
                                      rewriter.getContext(),
                                      cast<IntegerType>(elemType).getWidth()))
                                : elemType;
    auto makeScalarAttr = [&](int64_t val) -> TypedAttr {
      return isIntElem
                 ? cast<TypedAttr>(rewriter.getIntegerAttr(scalarType, val))
                 : cast<TypedAttr>(rewriter.getFloatAttr(
                       scalarType, static_cast<double>(val)));
    };
    Value startVal = rewriter.create<arith::ConstantOp>(loc, scalarType,
                                                        makeScalarAttr(start));
    Value stepVal = rewriter.create<arith::ConstantOp>(loc, scalarType,
                                                       makeScalarAttr(step));

    // === STEP 3: Create nested loops over tiles ===
    // Get this core's coordinates.
    Value coreY = rewriter.create<CoreIndexOp>(
        loc, rewriter.getIndexType(), rewriter.getI64IntegerAttr(0), nullptr);
    Value coreX = rewriter.create<CoreIndexOp>(
        loc, rewriter.getIndexType(), rewriter.getI64IntegerAttr(1), nullptr);

    // For column-major, iterate columns first; for row-major, iterate rows
    // first.
    Value outerLoopBound = colMajor ? numTileColsVal : numTileRowsVal;
    Value innerLoopBound = colMajor ? numTileRowsVal : numTileColsVal;

    auto outerLoop =
        rewriter.create<scf::ForOp>(loc, zeroIdx, outerLoopBound, oneIdx);
    rewriter.setInsertionPointToStart(outerLoop.getBody());

    auto innerLoop =
        rewriter.create<scf::ForOp>(loc, zeroIdx, innerLoopBound, oneIdx);
    // Mark the INNER loop as the compute root, since that's where
    // the actual compute operations are emitted. This ensures DST
    // syncs are placed inside the inner loop body, not the outer.
    // Since we emit an scf.for directly, we must tag this here
    // since linalg-to-affine and d2m-op-scheduler won't process this.
    innerLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
    innerLoop->setAttr("d2m.scheduled", rewriter.getUnitAttr());
    rewriter.setInsertionPointToStart(innerLoop.getBody());

    Value outerIdx = outerLoop.getInductionVar();
    Value innerIdx = innerLoop.getInductionVar();
    Value tileRowIdx = colMajor ? innerIdx : outerIdx;
    Value tileColIdx = colMajor ? outerIdx : innerIdx;

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
    Value totalTileRowsIdx =
        rewriter.create<arith::ConstantIndexOp>(loc, totalTileRows);
    Value const32Idx = rewriter.create<arith::ConstantIndexOp>(loc, 32);
    Value rowBlockIV, colBlockIV;
    collectBlockingLoopIVs(op, static_cast<int64_t>(outputShape.size()),
                           rowBlockIV, colBlockIV);

    // globalTileRow = coreY * shardTileRows + rowBlockIV * shardTileRows
    //               + localTileRow
    Value globalTileRow = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, coreY, shardTileRowsIdx),
        tileRowIdx);
    if (rowBlockIV) {
      globalTileRow = rewriter.create<arith::AddIOp>(
          loc, globalTileRow,
          rewriter.create<arith::MulIOp>(loc, rowBlockIV, shardTileRowsIdx));
    }
    // globalTileCol = coreX * shardTileCols + colBlockIV * shardTileCols
    //               + localTileCol
    Value globalTileCol = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, coreX, shardTileColsIdx),
        tileColIdx);
    if (colBlockIV) {
      globalTileCol = rewriter.create<arith::AddIOp>(
          loc, globalTileCol,
          rewriter.create<arith::MulIOp>(loc, colBlockIV, shardTileColsIdx));
    }

    Value tileOffsetIdx;
    if (colMajor) {
      // Row contribution: globalTileRow * 32.
      Value rowContrib =
          rewriter.create<arith::MulIOp>(loc, globalTileRow, const32Idx);
      // Column contribution: globalTileCol * totalTileRows * 32 * 32.
      Value colContrib = rewriter.create<arith::MulIOp>(
          loc,
          rewriter.create<arith::MulIOp>(
              loc,
              rewriter.create<arith::MulIOp>(loc, globalTileCol,
                                             totalTileRowsIdx),
              const32Idx),
          const32Idx);
      tileOffsetIdx =
          rewriter.create<arith::AddIOp>(loc, rowContrib, colContrib);
    } else {
      // Row contribution: globalTileRow * totalTileCols * 32 * 32.
      Value rowContrib = rewriter.create<arith::MulIOp>(
          loc,
          rewriter.create<arith::MulIOp>(
              loc,
              rewriter.create<arith::MulIOp>(loc, globalTileRow,
                                             totalTileColsIdx),
              const32Idx),
          const32Idx);
      // Column contribution: globalTileCol * 32.
      Value colContrib =
          rewriter.create<arith::MulIOp>(loc, globalTileCol, const32Idx);
      // Total offset (index type)
      tileOffsetIdx =
          rewriter.create<arith::AddIOp>(loc, rowContrib, colContrib);
    }
    Value tileOffsetScalar;
    if (isIntElem) {
      tileOffsetScalar =
          rewriter.create<arith::IndexCastOp>(loc, scalarType, tileOffsetIdx);
    } else {
      Value tileOffsetI64 = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI64Type(), tileOffsetIdx);
      tileOffsetScalar =
          rewriter.create<arith::SIToFPOp>(loc, scalarType, tileOffsetI64);
    }

    // === STEP 6: Tile arithmetic with scalar RHS ===
    // For column-major, transpose the scratch tile so column 0 carries the
    // consecutive within-tile row index [0,1,...,31] instead of [0,32,...,992].
    if (colMajor) {
      localIndexTile =
          rewriter.create<TileTransposeOp>(loc, tileType, localIndexTile)
              .getResult();
    }
    Value globalIndexTile =
        rewriter
            .create<TileAddOp>(loc, tileType, localIndexTile, tileOffsetScalar)
            .getResult();
    Value scaledTile =
        rewriter.create<TileMulOp>(loc, tileType, globalIndexTile, stepVal)
            .getResult();
    Value resultTile =
        rewriter.create<TileAddOp>(loc, tileType, scaledTile, startVal)
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
