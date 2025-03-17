// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRIMPLICITBROADCASTFOLD
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRImplicitBroadcastFoldRewriter : public RewritePattern {
public:
  TTIRImplicitBroadcastFoldRewriter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    if (!op->hasTrait<FirstOperandBroadcastable::Trait>() &&
        !op->hasTrait<SecondOperandBroadcastable::Trait>()) {
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

    // If the op is FullyBroadcastable and both operands are broadcasted, we
    // will only broadcast one of them to keep the shape consistent. Choosing
    // the first operand to broadcast implicitly.
    bool changed = false;
    ttir::BroadcastOp broadcastOpLhs =
        op->getOperand(0).getDefiningOp<ttir::BroadcastOp>();
    ttir::BroadcastOp broadcastOpRhs =
        op->getOperand(1).getDefiningOp<ttir::BroadcastOp>();
    if (op->hasTrait<FirstOperandBroadcastable::Trait>() && broadcastOpLhs) {
      // This operation only support implicit broadcast for Operand 0.
      Operation *newOp = rewriter.clone(*op);
      newOp->setOperand(0, broadcastOpLhs.getInput());
      rewriter.replaceOp(op, newOp);
      changed = true;

    } else if (op->hasTrait<SecondOperandBroadcastable::Trait>() &&
               broadcastOpRhs) {
      // This operation only support implicit broadcast for Operand 0.
      Operation *newOp = rewriter.clone(*op);
      newOp->setOperand(1, broadcastOpRhs.getInput());
      rewriter.replaceOp(op, newOp);
      changed = true;
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

    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
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
