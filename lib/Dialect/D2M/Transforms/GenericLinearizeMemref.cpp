// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICLINEARIZEMEMREF
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
template <typename LoadStoreOp>
struct D2MLinearizeMemrefAccessRewriter final
    : public OpRewritePattern<LoadStoreOp> {
public:
  D2MLinearizeMemrefAccessRewriter(
      ::mlir::MLIRContext *context,
      DenseMap<Value, memref::CollapseShapeOp> &collapseOps)
      : OpRewritePattern<LoadStoreOp>(context), collapseOps(&collapseOps) {}

  static mlir::AffineMap linearizeAffineMap(::mlir::MLIRContext *context,
                                            mlir::AffineMap map,
                                            ArrayRef<int64_t> shape) {
    auto evaledShape = ttmlir::utils::evalShape(map, shape);
    mlir::AffineExpr indexing = getAffineConstantExpr(0, context);
    mlir::AffineExpr volumeExpr = getAffineConstantExpr(1, context);

    assert(map.getNumResults() > 0);
    for (int i = map.getNumResults() - 1; i >= 0; i--) {
      mlir::AffineExpr linearIdx = getAffineDimExpr(i, context);
      mlir::AffineExpr dim = getAffineConstantExpr(evaledShape[i], context);
      indexing = linearIdx * volumeExpr + indexing;
      volumeExpr = volumeExpr * dim;
    }

    mlir::AffineMap linearResult =
        mlir::AffineMap::get(map.getNumResults(), 0, indexing, context);
    return linearResult.compose(map);
  }

  LogicalResult matchAndRewrite(LoadStoreOp op,
                                PatternRewriter &rewriter) const final {
    Value val = op.getMemRef();
    auto memref = mlir::cast<MemRefType>(val.getType());
    if (memref.getRank() == 1) {
      // Already linearized.
      return failure();
    }

    auto shape = memref.getShape();
    auto linearMap = linearizeAffineMap(
        rewriter.getContext(), memref.getLayout().getAffineMap(), shape);

    // Create or get collapsed memref
    memref::CollapseShapeOp linearizedArg = collapseOps->lookup(val);
    if (!linearizedArg) {
      rewriter.setInsertionPointAfterValue(val);
      SmallVector<ReassociationIndices, 4> collapsedDims = {
          llvm::to_vector(llvm::seq<int64_t>(0, shape.size()))};
      assert(memref::CollapseShapeOp::isGuaranteedCollapsible(memref,
                                                              collapsedDims) &&
             "linearizeAffineMap assumes that the shape is collapsible aka "
             "has contiguous memory layout");
      linearizedArg = rewriter.create<memref::CollapseShapeOp>(op.getLoc(), val,
                                                               collapsedDims);
      collapseOps->insert({val, linearizedArg});
    }

    // Create new indices using the linear map
    SmallVector<Value> indices(op.getIndices());

    rewriter.setInsertionPoint(op);

    Value linearIndex =
        rewriter.create<affine::AffineApplyOp>(op.getLoc(), linearMap, indices);

    // Create new load/store with linearized access
    if constexpr (std::is_same_v<LoadStoreOp, memref::LoadOp>) {
      rewriter.replaceOpWithNewOp<memref::LoadOp>(op, linearizedArg.getResult(),
                                                  ValueRange{linearIndex});
    } else if constexpr (std::is_same_v<LoadStoreOp, memref::StoreOp>) {
      rewriter.replaceOpWithNewOp<memref::StoreOp>(op, op.getValueToStore(),
                                                   linearizedArg.getResult(),
                                                   ValueRange{linearIndex});
    } else {
      return failure();
    }

    return success();
  }

  DenseMap<Value, memref::CollapseShapeOp> *collapseOps;
};
} // namespace

namespace {
struct D2MLinearizeTileMatmulBlockRewriter final
    : public OpRewritePattern<TileMatmulBlockOp> {
public:
  D2MLinearizeTileMatmulBlockRewriter(
      ::mlir::MLIRContext *context,
      DenseMap<Value, memref::CollapseShapeOp> &collapseOps)
      : OpRewritePattern<TileMatmulBlockOp>(context),
        collapseOps(&collapseOps) {}

  static int64_t getNumColumns(Value view) {
    if (auto castOp = mlir::dyn_cast<memref::CastOp>(view.getDefiningOp())) {
      view = castOp.getSource();
    } else if (auto svOp =
                   mlir::dyn_cast<memref::SubViewOp>(view.getDefiningOp())) {
      view = svOp.getSource();
    }
    auto srcTy = mlir::cast<MemRefType>(view.getType());
    return srcTy.getShape()[1];
  }

  LogicalResult matchAndRewrite(TileMatmulBlockOp op,
                                PatternRewriter &rewriter) const final {
    auto typeA = mlir::cast<MemRefType>(op.getA().getType());
    auto typeB = mlir::cast<MemRefType>(op.getB().getType());

    if (typeA.getRank() == 1 && typeB.getRank() == 1) {
      return failure();
    }

    // If block dimensions are already set, this op has already been processed.
    if (op.hasBlockDims()) {
      return failure();
    }

    int64_t rtDim = typeA.getShape()[0];
    int64_t ktDim = typeA.getShape()[1];
    int64_t ctDim = typeB.getShape()[1];
    int64_t ntDim = getNumColumns(op.getB());

    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<TileMatmulBlockOp>(
        op, op.getA(), op.getB(), op.getOutput(),
        rewriter.getI64IntegerAttr(rtDim), rewriter.getI64IntegerAttr(ktDim),
        rewriter.getI64IntegerAttr(ctDim), rewriter.getI64IntegerAttr(ntDim));

    return success();
  }

  DenseMap<Value, memref::CollapseShapeOp> *collapseOps;
};
} // namespace

namespace {
class D2MGenericLinearizeMemref
    : public impl::D2MGenericLinearizeMemrefBase<D2MGenericLinearizeMemref> {
public:
  using impl::D2MGenericLinearizeMemrefBase<
      D2MGenericLinearizeMemref>::D2MGenericLinearizeMemrefBase;

  void runOnOperation() final {
    DenseMap<Value, memref::CollapseShapeOp> collapseOps;
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MLinearizeMemrefAccessRewriter<memref::LoadOp>,
                 D2MLinearizeMemrefAccessRewriter<memref::StoreOp>>(
        &getContext(), collapseOps);
    patterns.add<D2MLinearizeTileMatmulBlockRewriter>(&getContext(),
                                                      collapseOps);
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
