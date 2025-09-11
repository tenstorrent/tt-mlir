// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttcore {
#define GEN_PASS_DEF_TTCOREOPTIMIZATIONBARRIERFOLD
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h.inc"

class OptimizationBarrierFoldPattern
    : public mlir::OpRewritePattern<ttcore::OptimizationBarrierOp> {
public:
  using mlir::OpRewritePattern<ttcore::OptimizationBarrierOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ttcore::OptimizationBarrierOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Replace the optimization barrier with its operands
    rewriter.replaceOp(op, op.getInputs());
    return success();
  }
};

class TTCoreOptimizationBarrierFold
    : public impl::TTCoreOptimizationBarrierFoldBase<
          TTCoreOptimizationBarrierFold> {
public:
  using impl::TTCoreOptimizationBarrierFoldBase<
      TTCoreOptimizationBarrierFold>::TTCoreOptimizationBarrierFoldBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<OptimizationBarrierFoldPattern>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttcore::TTCoreDialect>();
  }
};

} // namespace mlir::tt::ttcore
