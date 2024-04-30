// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/TTOpsTypes.h"

#include "ttmlir/Dialect/TTIR/TTIRPasses.h"

template <typename T> T div_up(T n, T d) { return (n + d - 1) / d; }

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRDISPATCH
#define GEN_PASS_DEF_TTIRLAYOUT
#define GEN_PASS_DEF_TTIRSHARD
#define GEN_PASS_DEF_TTIRLOWER
#include "ttmlir/Dialect/TTIR/TTIRPasses.h.inc"

class TTIRLinalgGenericToDispatchRewriter
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  template <typename TensorTyT>
  static TensorTyT unparallelize(TensorTyT tensorTy) {
    SmallVector<int64_t> shape(tensorTy.getShape());
    assert(shape.size() > 2);
    SmallVector<int64_t> newShape(shape.begin() + 2, shape.end());
    return tensorTy.clone(newShape);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    return failure();
  }
};

template <typename TosaEltwiseOp>
class TTIRTosaElementwiseToDispatchRewriter
    : public OpRewritePattern<TosaEltwiseOp> {
public:
  using OpRewritePattern<TosaEltwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TosaEltwiseOp op,
                                PatternRewriter &rewriter) const final {
    assert(op.getShift() == 0);
    op.dump();
    return failure();
  }
};

class TTIRDispatch : public impl::TTIRDispatchBase<TTIRDispatch> {
public:
  using impl::TTIRDispatchBase<TTIRDispatch>::TTIRDispatchBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRLinalgGenericToDispatchRewriter,
                 TTIRTosaElementwiseToDispatchRewriter<tosa::MulOp>>(
        &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
  }
};

class TTIRLayout : public impl::TTIRLayoutBase<TTIRLayout> {
public:
  using impl::TTIRLayoutBase<TTIRLayout>::TTIRLayoutBase;

  void runOnOperation() final { assert(false); }
};

class TTIRShard : public impl::TTIRShardBase<TTIRShard> {
public:
  using impl::TTIRShardBase<TTIRShard>::TTIRShardBase;

  void runOnOperation() final { assert(false); }
};

class TTIRLower : public impl::TTIRLowerBase<TTIRLower> {
public:
  using impl::TTIRLowerBase<TTIRLower>::TTIRLowerBase;

  void runOnOperation() final { assert(false); }
};

} // namespace mlir::tt::ttir
