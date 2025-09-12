// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICLINEARIZEMEMREF
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
template <typename LoadStoreOp>
struct TTIRLinearizeMemrefAccessRewriter final
    : public OpRewritePattern<LoadStoreOp> {
public:
  TTIRLinearizeMemrefAccessRewriter(
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
struct TTIRLinearizeTileMatmulBlockRewriter final
    : public OpRewritePattern<TileMatmulBlockOp> {
public:
  TTIRLinearizeTileMatmulBlockRewriter(
      ::mlir::MLIRContext *context,
      DenseMap<Value, memref::CollapseShapeOp> &collapseOps)
      : OpRewritePattern<TileMatmulBlockOp>(context),
        collapseOps(&collapseOps) {}

  static memref::CollapseShapeOp
  linearize(Value val, PatternRewriter &rewriter,
            DenseMap<Value, memref::CollapseShapeOp> *cache) {
    auto memrefTy = mlir::cast<MemRefType>(val.getType());
    memref::CollapseShapeOp linearized = cache->lookup(val);
    if (!linearized) {
      rewriter.setInsertionPointAfterValue(val);
      SmallVector<ReassociationIndices, 4> collapsedDims = {
          llvm::to_vector(llvm::seq<int64_t>(0, memrefTy.getRank()))};
      assert(memref::CollapseShapeOp::isGuaranteedCollapsible(memrefTy,
                                                              collapsedDims) &&
             "Expected collapsible memref layout");
      linearized = rewriter.create<memref::CollapseShapeOp>(val.getLoc(), val,
                                                            collapsedDims);
      cache->insert({val, linearized});
    }
    return linearized;
  }

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

    assert(!op.hasBlockDims() &&
           "Unexpected block dimensions present for matmul_block. Proceeding "
           "would override the present attributes");

    int64_t rtDim = typeA.getShape()[0];
    int64_t ktDim = typeA.getShape()[1];
    int64_t ctDim = typeB.getShape()[1];
    int64_t ntDim = getNumColumns(op.getB());

    auto linA = linearize(op.getA(), rewriter, collapseOps);
    auto linB = linearize(op.getB(), rewriter, collapseOps);
    auto linO = linearize(op.getOutput(), rewriter, collapseOps);

    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<TileMatmulBlockOp>(
        op, linA.getResult(), linB.getResult(), linO.getResult(),
        rewriter.getI64IntegerAttr(rtDim), rewriter.getI64IntegerAttr(ktDim),
        rewriter.getI64IntegerAttr(ctDim), rewriter.getI64IntegerAttr(ntDim));

    return success();
  }

  DenseMap<Value, memref::CollapseShapeOp> *collapseOps;
};
} // namespace

namespace {
class TTIRGenericLinearizeMemref
    : public impl::TTIRGenericLinearizeMemrefBase<TTIRGenericLinearizeMemref> {
public:
  using impl::TTIRGenericLinearizeMemrefBase<
      TTIRGenericLinearizeMemref>::TTIRGenericLinearizeMemrefBase;

  void runOnOperation() final {
    DenseMap<Value, memref::CollapseShapeOp> collapseOps;
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRLinearizeMemrefAccessRewriter<memref::LoadOp>,
                 TTIRLinearizeMemrefAccessRewriter<memref::StoreOp>>(
        &getContext(), collapseOps);
    patterns.add<TTIRLinearizeTileMatmulBlockRewriter>(&getContext(),
                                                       collapseOps);
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
