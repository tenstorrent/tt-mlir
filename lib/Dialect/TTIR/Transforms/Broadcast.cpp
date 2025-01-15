// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRBROADCASTFOLD
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRBroadcastFoldRewriter : public RewritePattern {
public:
  TTIRBroadcastFoldRewriter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    // First check if the op itself has any broadcastable traits
    if (op->hasTrait<partiallyBroadcastable::Trait>()) {

      // This operation can only fold broadcast operation for Operand 0.
      ttir::BroadcastOp broadcastOp =
          op->getOperand(0).getDefiningOp<ttir::BroadcastOp>();
      if (broadcastOp) {
        rewriter.replaceOp(broadcastOp, broadcastOp.getInput());
        return success();
      }
    } else if (op->hasTrait<fullyBroadcastable::Trait>()) {
      bool changed = false;
      // Check all operands for this op
      ttir::BroadcastOp broadcastOp0 =
          op->getOperand(0).getDefiningOp<ttir::BroadcastOp>();
      ttir::BroadcastOp broadcastOp1 =
          op->getOperand(1).getDefiningOp<ttir::BroadcastOp>();
      if (broadcastOp0) {
        rewriter.replaceOp(broadcastOp0, broadcastOp0.getInput());
        changed = true;
      } else if (broadcastOp1) {
        rewriter.replaceOp(broadcastOp1, broadcastOp1.getInput());
        changed = true;
      }
      return changed ? success() : failure();
    }

    return failure();
  }
};

class TTIRBroadcastFold
    : public impl::TTIRBroadcastFoldBase<TTIRBroadcastFold> {
public:
  using impl::TTIRBroadcastFoldBase<TTIRBroadcastFold>::TTIRBroadcastFoldBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRBroadcastFoldRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

} // namespace mlir::tt::ttir
