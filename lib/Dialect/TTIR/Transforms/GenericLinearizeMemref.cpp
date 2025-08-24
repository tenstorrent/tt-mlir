// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
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
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
