// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OperationSupport.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRERASEINVERSEOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRCommuteTmsAboveElementwiseUnaryRewriter : public RewritePattern {
public:
  TTIRCommuteTmsAboveElementwiseUnaryRewriter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    if (!op->hasTrait<ElementwiseUnary::Trait>()) {
      // The op should support implicit broadcast to fold them.
      return failure();
    }

    SmallVector<Operation *> users(op->getUsers());
    if (failed(checkAllUsersAreIdenticalTms(users))) {
      return failure();
    }

    if (auto transpose = dyn_cast<ttir::TransposeOp>(users[0])) {
      commuteTmsThroughEltwise(op, users, op->getOperand(0), rewriter);
      return success();
    }

    return failure();
  }

private:
  LogicalResult
  checkAllUsersAreIdenticalTms(SmallVector<Operation *> users) const {
    Operation *firstUser = users[0];
    for (auto *user : users) {
      if (user->getAttrDictionary() != firstUser->getAttrDictionary()) {
        return failure();
      }
    }
    return success(
        isa<ttir::TransposeOp, ttir::PermuteOp, ttir::ReshapeOp>(firstUser));
  }

  void commuteTmsThroughEltwise(Operation *op, SmallVector<Operation *> users,
                                Value operand,
                                PatternRewriter &rewriter) const {
    Operation *user = users[0];
    auto newEltwiseType = cast<RankedTensorType>(user->getResult(0).getType());

    mlir::tensor::EmptyOp newTransposeDPS = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), newEltwiseType.getShape(),
        newEltwiseType.getElementType());
    TransposeOp newTranspose = cast<TransposeOp>(rewriter.clone(*user));
    mlir::tensor::EmptyOp newEltwiseDPS = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), newEltwiseType.getShape(),
        newEltwiseType.getElementType());
    Operation *newEltwise = rewriter.clone(*op);

    newEltwise->setOperand(0, newTranspose->getResult(0));
    newEltwise->setOperand(1, newEltwiseDPS->getResult(0));
    newEltwise->getResult(0).setType(newEltwiseType);
    newTranspose->setOperand(0, operand);
    newTranspose->setOperand(1, newTransposeDPS->getResult(0));

    for (auto *user : users) {
      rewriter.replaceOp(user, newEltwise);
    }
  }
};

class TTIREraseInverseOps
    : public impl::TTIREraseInverseOpsBase<TTIREraseInverseOps> {
public:
  using impl::TTIREraseInverseOpsBase<
      TTIREraseInverseOps>::TTIREraseInverseOpsBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRCommuteTmsAboveElementwiseUnaryRewriter>(&getContext());
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
