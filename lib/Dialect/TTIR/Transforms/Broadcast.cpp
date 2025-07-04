// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRIMPLICITBROADCASTFOLD
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

// This algorithm works by first removing all explicit broadcasts from the
// operands of an operation. While doing so, it calculates the result shape that
// would result from implicit broadcasting, taking all operands into
// consideration. If this shape differs from the target shape, we add an
// explicit broadcast to the operation’s output to match the target shape.
class TTIRImplicitBroadcastFoldRewriter : public RewritePattern {
public:
  TTIRImplicitBroadcastFoldRewriter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasTrait<ttir::Broadcastable>()) {
      return llvm::failure();
    }

    auto dps = mlir::cast<mlir::DestinationStyleOpInterface>(op);

    bool operandsChanged = false;
    llvm::SmallVector<int64_t> implicitBroadcastedShape;

    // Remove all explicit broadcasts from the operands and compute the shape
    // that would result from implicit broadcasting.
    for (int64_t i = 0; i < dps.getNumDpsInputs(); ++i) {
      mlir::Value operand = dps->getOperand(i);
      ::llvm::ArrayRef<int64_t> originalOperandShape;
      if (auto broadcastOp = mlir::dyn_cast_if_present<ttir::BroadcastOp>(
              operand.getDefiningOp())) {
        originalOperandShape = broadcastOp.getInput().getType().getShape();
        rewriter.modifyOpInPlace(
            dps, [&]() { dps->setOperand(i, broadcastOp.getInput()); });
        operandsChanged = true;
      } else {
        originalOperandShape =
            mlir::cast<RankedTensorType>(operand.getType()).getShape();
      }

      llvm::SmallVector<int64_t> prevShape = implicitBroadcastedShape;
      assert(mlir::OpTrait::util::getBroadcastedShape(
                 prevShape, originalOperandShape, implicitBroadcastedShape) &&
             "Operands must be broadcast-compatible");
    }

    auto resultType = mlir::cast<RankedTensorType>(dps->getResult(0).getType());
    llvm::ArrayRef<int64_t> resultShape = resultType.getShape();

    if (implicitBroadcastedShape == resultShape) {
      return llvm::success(operandsChanged);
    }

    // If the shape from implicit broadcasting differs from the target shape,
    // add an explicit broadcast to the operation’s output.
    auto newResultType = mlir::RankedTensorType::get(
        implicitBroadcastedShape, resultType.getElementType());
    rewriter.modifyOpInPlace(dps, [&]() {
      dps.getDpsInits()[0].setType(newResultType);
      dps->getResult(0).setType(newResultType);
    });

    rewriter.setInsertionPointAfter(dps);
    auto broadcastDimensions = ttmlir::utils::getBroadcastDimensions<int64_t>(
        implicitBroadcastedShape, resultShape);
    auto broadcastOp = ttir::utils::createDPSOp<ttir::BroadcastOp>(
        rewriter, dps->getLoc(), resultShape, newResultType.getElementType(),
        newResultType.getEncoding(), dps->getResult(0), broadcastDimensions);
    rewriter.replaceAllUsesExcept(dps->getResult(0), broadcastOp.getResult(),
                                  broadcastOp);

    return llvm::success();
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
    registry.insert<mlir::tt::ttcore::TTCoreDialect>();
  }
};

} // namespace mlir::tt::ttir
