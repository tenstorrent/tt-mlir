// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
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

  static mlir::AffineMap createLinearMap(::mlir::MLIRContext *context,
                                         ArrayRef<int64_t> shape) {
    if (shape.empty()) {
      return AffineMap::get(0, 0, context);
    }

    SmallVector<AffineExpr> exprs;
    AffineExpr linearIndex = getAffineConstantExpr(0, context);
    AffineExpr stride = getAffineConstantExpr(1, context);

    // Create a row-major linearization
    for (int i = shape.size() - 1; i >= 0; --i) {
      linearIndex = linearIndex + getAffineDimExpr(i, context) * stride;
      if (i > 0) {
        stride = stride * shape[i];
      }
    }

    return AffineMap::get(shape.size(), 0, linearIndex, context);
  }

  LogicalResult matchAndRewrite(LoadStoreOp op,
                                PatternRewriter &rewriter) const final {
    Value memref = op.getMemRef();
    auto memrefType = mlir::cast<MemRefType>(memref.getType());
    if (memrefType.getRank() == 1) {
      // Already linearized.
      return failure();
    }

    auto shape = memrefType.getShape();
    auto linearMap = createLinearMap(rewriter.getContext(), shape);

    // Create or get collapsed memref
    memref::CollapseShapeOp linearizedArg = collapseOps->lookup(memref);
    if (!linearizedArg) {
      rewriter.setInsertionPointAfterValue(memref);
      SmallVector<ReassociationIndices, 4> collapsedDims = {
          llvm::to_vector(llvm::seq<int64_t>(0, shape.size()))};
      assert(memref::CollapseShapeOp::isGuaranteedCollapsible(memrefType,
                                                              collapsedDims) &&
             "Cannot collapse memref - likely due to non-contiguous layout");
      linearizedArg = rewriter.create<memref::CollapseShapeOp>(
          op.getLoc(), memref, collapsedDims);
      collapseOps->insert({memref, linearizedArg});
    }

    // Create new indices using the linear map
    SmallVector<Value> indices(op.getIndices());
    Location loc = op.getLoc();

    // Apply the linearization to the indices
    AffineMap map = linearMap;
    if (auto mapAttr = memrefType.getLayout().getAffineMap()) {
      map = linearMap.compose(mapAttr);
    }

    rewriter.setInsertionPoint(op);

    Value linearIndex =
        rewriter.create<affine::AffineApplyOp>(loc, map, indices);

    // Create new load/store with linearized access
    if (auto loadOp = dyn_cast<memref::LoadOp>(op.getOperation())) {
      rewriter.replaceOpWithNewOp<memref::LoadOp>(
          loadOp, linearizedArg.getResult(), ValueRange{linearIndex});
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op.getOperation())) {
      rewriter.replaceOpWithNewOp<memref::StoreOp>(
          storeOp, storeOp.getValueToStore(), linearizedArg.getResult(),
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
