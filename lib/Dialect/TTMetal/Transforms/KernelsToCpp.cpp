// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/Passes.h"

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_CONVERTTTMETALKERNELSTOEMITC
#include "ttmlir/Dialect/TTMetal/Passes.h.inc"

class TTMetalToEmitCKernelRewriter
    : public OpRewritePattern<ttmetal::KernelOp> {
public:
  using OpRewritePattern<ttmetal::KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttmetal::KernelOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange(), op.getOp(), rewriter.getArrayAttr({}),
        rewriter.getArrayAttr({}), ValueRange());
    return success();
  }
};

class TTMetalToEmitCYieldRewriter : public OpRewritePattern<ttmetal::YieldOp> {
public:
  using OpRewritePattern<ttmetal::YieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttmetal::YieldOp op,
                                PatternRewriter &rewriter) const final {
    if (not isa<func::FuncOp>(op.getOperation()->getParentOp()))
      return rewriter.notifyMatchFailure(op, "Not inside of func op");
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, ValueRange());
    return success();
  }
};

class ConvertTTMetalKernelsToEmitC
    : public impl::ConvertTTMetalKernelsToEmitCBase<
          ConvertTTMetalKernelsToEmitC> {
public:
  using impl::ConvertTTMetalKernelsToEmitCBase<
      ConvertTTMetalKernelsToEmitC>::ConvertTTMetalKernelsToEmitCBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTMetalToEmitCYieldRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttmetal::TTMetalDialect>();
    registry.insert<mlir::emitc::EmitCDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};

} // namespace mlir::tt::ttmetal
