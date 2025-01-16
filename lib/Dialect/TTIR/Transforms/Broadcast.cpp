// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRIMPLICITBROADCASTFOLD
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRImplicitBroadcastFoldRewriter : public RewritePattern {
public:
  TTIRImplicitBroadcastFoldRewriter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    if (!op->hasTrait<partiallyBroadcastable::Trait>() &&
        !op->hasTrait<fullyBroadcastable::Trait>()) {
      // The op should support implicit broadcast to fold them.
      return failure();
    }

    if (op->getNumOperands() < 2) {
      // This optimization is only applicable to binary ops.
      return failure();
    }

    if (op->getNumResults() == 0) {
      return failure();
    }

    // Only one operand can implicitly broadcasted, so verify if
    // an exisiting operand is already implicitly broadcasting.
    RankedTensorType resultType =
        mlir::cast<RankedTensorType>(op->getResult(0).getType());
    for (Type type : op->getOperands().getTypes()) {
      if (mlir::cast<RankedTensorType>(type).getShape() !=
          resultType.getShape()) {
        // Only a single operand is allowed to perform implicit broadcast.
        return failure();
      }
    }

    if (op->hasTrait<partiallyBroadcastable::Trait>()) {

      // This operation only support implicit broadcast for Operand 0.
      ttir::BroadcastOp broadcastOp =
          op->getOperand(0).getDefiningOp<ttir::BroadcastOp>();
      if (broadcastOp) {
        rewriter.replaceOp(broadcastOp, broadcastOp.getInput());
        return success();
      }
    } else if (op->hasTrait<fullyBroadcastable::Trait>()) {
      bool changed = false;
      // Check all operands of this op
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

class TTIRImplicitBroadcastFold
    : public impl::TTIRImplicitBroadcastFoldBase<TTIRImplicitBroadcastFold> {
public:
  using impl::TTIRImplicitBroadcastFoldBase<
      TTIRImplicitBroadcastFold>::TTIRImplicitBroadcastFoldBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRImplicitBroadcastFoldRewriter>(&getContext());
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
