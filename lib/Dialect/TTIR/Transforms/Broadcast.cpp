// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRIMPLICITBROADCASTFOLD
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRImplicitBroadcastFoldRewriter : public RewritePattern {
public:
  TTIRImplicitBroadcastFoldRewriter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    if (!op->hasTrait<PartiallyBroadcastable::Trait>() &&
        !op->hasTrait<FullyBroadcastable::Trait>()) {
      // The op should support implicit broadcast to fold them.
      return failure();
    }

    if (op->getNumOperands() < 2) {
      // This optimization is only applicable to binary ops.
      assert(op->getNumOperands() < 2 &&
             "Implicit broadcast requires at least a binary operation.");
      return failure();
    }

    if (op->getNumResults() == 0) {
      assert(op->getNumResults() == 0 &&
             "Implicit broadcast requires the operation to produce a result.");
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

    bool changed = false;
    if (op->hasTrait<PartiallyBroadcastable::Trait>()) {
      // This operation only support implicit broadcast for Operand 0.
      ttir::BroadcastOp broadcastOp =
          op->getOperand(0).getDefiningOp<ttir::BroadcastOp>();
      if (broadcastOp) {
        Operation *newOp = rewriter.clone(*op);
        newOp->setOperand(0, broadcastOp.getInput());
        rewriter.replaceOp(op, newOp);
        changed = true;
      }
    } else if (op->hasTrait<FullyBroadcastable::Trait>()) {
      // Check all operands of this op
      ttir::BroadcastOp broadcastOp0 =
          op->getOperand(0).getDefiningOp<ttir::BroadcastOp>();
      ttir::BroadcastOp broadcastOp1 =
          op->getOperand(1).getDefiningOp<ttir::BroadcastOp>();
      if (broadcastOp0) {
        Operation *newOp = rewriter.clone(*op);
        newOp->setOperand(0, broadcastOp0.getInput());
        rewriter.replaceOp(op, newOp);
        changed = true;
      } else if (broadcastOp1) {
        Operation *newOp = rewriter.clone(*op);
        newOp->setOperand(1, broadcastOp1.getInput());
        rewriter.replaceOp(op, newOp);
        changed = true;
      }
    }

    return changed ? success() : failure();
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
