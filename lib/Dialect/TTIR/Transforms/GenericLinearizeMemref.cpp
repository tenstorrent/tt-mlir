// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
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
struct TTIRGenericLinearizeMemrefRewriter final
    : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  static bool isLinearizedMemref(BlockArgument arg) {
    auto memref = mlir::cast<MemRefType>(arg.getType());
    if (memref.getShape().size() == 1) {
      return true;
    }

    return llvm::all_of(arg.getUsers(), [](Operation *user) {
      return mlir::isa<memref::CollapseShapeOp>(user);
    });
  }

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

  static void linearizeMemref(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value val) {
    auto memref = mlir::cast<MemRefType>(val.getType());
    auto shape = memref.getShape();
    SmallVector<ReassociationIndices, 4> collapsedDims = {
        llvm::to_vector(llvm::seq<int64_t>(0, shape.size()))};
    assert(memref::CollapseShapeOp::isGuaranteedCollapsible(memref,
                                                            collapsedDims) &&
           "linearizeAffineMap assumes that the shape is collapsible aka "
           "has contiguous memory layout");
    auto linearMap = linearizeAffineMap(
        rewriter.getContext(), memref.getLayout().getAffineMap(), shape);
    auto linearizedArg =
        rewriter.create<memref::CollapseShapeOp>(loc, val, collapsedDims);
    rewriter.replaceAllUsesExcept(val, linearizedArg->getResult(0),
                                  linearizedArg);
    for (auto *user : linearizedArg->getUsers()) {
      if (auto load = mlir::dyn_cast<affine::AffineLoadOp>(user)) {
        load.setMap(linearMap.compose(load.getMap()));
      } else if (auto store = mlir::dyn_cast<affine::AffineStoreOp>(user)) {
        store.setMap(linearMap.compose(store.getMap()));
      }
    }
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    assert(op.getNumRegions() == 1 &&
           "expected single compute region at this stage");
    Block *entry = &op.getRegion(0).front();
    rewriter.setInsertionPointToStart(entry);
    auto args = entry->getArguments();
    if (llvm::all_of(args, isLinearizedMemref)) {
      return failure();
    }

    rewriter.modifyOpInPlace(op, [&]() {
      for (auto arg : args) {
        if (isLinearizedMemref(arg)) {
          continue;
        }
        linearizeMemref(rewriter, arg.getLoc(), arg);
      }

      op->walk([&](ttir::AcquireDstOp acquireDst) {
        rewriter.setInsertionPointAfter(acquireDst);
        linearizeMemref(rewriter, acquireDst.getLoc(), acquireDst.getResult());
      });
    });

    return success();
  }
};
} // namespace

namespace {
class TTIRGenericLinearizeMemref
    : public impl::TTIRGenericLinearizeMemrefBase<TTIRGenericLinearizeMemref> {
public:
  using impl::TTIRGenericLinearizeMemrefBase<
      TTIRGenericLinearizeMemref>::TTIRGenericLinearizeMemrefBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericLinearizeMemrefRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }
};
} // namespace

} // namespace mlir::tt::ttir
