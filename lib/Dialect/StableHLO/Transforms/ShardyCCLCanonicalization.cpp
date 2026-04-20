// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shardy/dialect/sdy/ir/dialect.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_SHARDYCCLCANONICALIZATIONPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

struct ShardyCCLCanonicalizationPass
    : public impl::ShardyCCLCanonicalizationPassBase<
          ShardyCCLCanonicalizationPass> {
public:
  using impl::ShardyCCLCanonicalizationPassBase<
      ShardyCCLCanonicalizationPass>::ShardyCCLCanonicalizationPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    RewritePatternSet patterns(ctx);
    mlir::sdy::AllGatherOp::getCanonicalizationPatterns(patterns, ctx);
    mlir::sdy::AllSliceOp::getCanonicalizationPatterns(patterns, ctx);
    mlir::sdy::AllReduceOp::getCanonicalizationPatterns(patterns, ctx);
    mlir::sdy::AllToAllOp::getCanonicalizationPatterns(patterns, ctx);
    mlir::sdy::ReduceScatterOp::getCanonicalizationPatterns(patterns, ctx);

    GreedyRewriteConfig config;
    config.enableConstantCSE(false);

    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      signalPassFailure();
      return;
    }

    // Remove dead CCL ops that may remain after canonicalization
    // (e.g. unused all_reduce on argmax's max value output).
    IRRewriter rewriter(ctx);
    module.walk([&](Operation *op) {
      if (isa<mlir::sdy::AllReduceOp, mlir::sdy::AllGatherOp,
              mlir::sdy::AllSliceOp, mlir::sdy::AllToAllOp,
              mlir::sdy::ReduceScatterOp>(op) &&
          op->use_empty()) {
        rewriter.eraseOp(op);
      }
    });
  }
};
} // namespace mlir::tt::stablehlo
