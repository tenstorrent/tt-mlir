// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

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

    if (op->hasTrait<PartiallyBroadcastable::Trait>()) {
      // This operation only support implicit broadcast for Operand 0.
      ttir::BroadcastOp broadcastOp =
          op->getOperand(0).getDefiningOp<ttir::BroadcastOp>();
      if (broadcastOp) {
        rewriter.modifyOpInPlace(
            op, [&]() { op->setOperand(0, broadcastOp.getInput()); });

        return success();
      }
    } else if (op->hasTrait<FullyBroadcastable::Trait>()) {
      // Check all operands of this op.
      ttir::BroadcastOp broadcastOp0 =
          op->getOperand(0).getDefiningOp<ttir::BroadcastOp>();
      ttir::BroadcastOp broadcastOp1 =
          op->getOperand(1).getDefiningOp<ttir::BroadcastOp>();

      if (broadcastOp0 && broadcastOp1) {
        Operation *newOp = rewriter.clone(*op);

        // First operand can always be implicitly broadcast.
        newOp->setOperand(0, broadcastOp0.getInput());

        if (broadcastOp0.getBroadcastDimensions() ==
            broadcastOp1.getBroadcastDimensions()) {
          // If the broadcast dimensions for the operands are the same, we can
          // remove them and add a broadcast to the output of the operation.
          newOp->setOperand(1, broadcastOp1.getInput());

          rewriter.setInsertionPoint(newOp);
          auto initTensor = rewriter.create<ttir::EmptyOp>(
              op->getLoc(), broadcastOp0.getInput().getType().getShape(),
              broadcastOp0.getInput().getType().getElementType());
          newOp->setOperand(2, initTensor.getResult());
          newOp->getResult(0).setType(broadcastOp0.getInput().getType());
          rewriter.replaceOp(op, newOp);
          rewriter.setInsertionPointAfter(newOp);

          // Insert a broadcast to the output.
          ::llvm::ArrayRef<int64_t> broadcastDims =
              broadcastOp0.getBroadcastDimensions();
          auto newBroadcastOp = ttir::utils::createDPSOp<ttir::BroadcastOp>(
              rewriter, newOp->getLoc(), resultType, newOp->getResult(0),
              broadcastDims);

          rewriter.replaceAllUsesExcept(
              newOp->getResult(0), newBroadcastOp.getResult(), newBroadcastOp);
        } else {
          // If the broadcast dimensions for the operands are different, we can
          // only implicitly broadcast the second operand if there are no common
          // elements between broadcast dimensions other than 1.
          bool dimensionsCompatible =
              !llvm::any_of(llvm::zip(broadcastOp0.getBroadcastDimensions(),
                                      broadcastOp1.getBroadcastDimensions()),
                            [](std::tuple<int64_t, int64_t> pair) {
                              return std::get<0>(pair) == std::get<1>(pair) &&
                                     std::get<0>(pair) != 1;
                            });

          if (dimensionsCompatible) {
            newOp->setOperand(1, broadcastOp1.getInput());
          }

          rewriter.replaceOp(op, newOp);
        }

        return success();
      }

      if (broadcastOp0 || broadcastOp1) {
        rewriter.modifyOpInPlace(op, [&]() {
          if (broadcastOp0) {
            op->setOperand(0, broadcastOp0.getInput());
          } else if (broadcastOp1) {
            op->setOperand(1, broadcastOp1.getInput());
          }
        });

        return success();
      }
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
