// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRMOVERESHAPETOCONSTANT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Check if a value traces back to a constant op.
static bool isFromConstant(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }

  // Direct constant.
  if (isa<ttir::ConstantOp>(defOp)) {
    return true;
  }

  // Reshape/broadcast of a constant is still "from constant".
  if (isa<ttir::ReshapeOp, ttir::BroadcastOp>(defOp)) {
    return isFromConstant(defOp->getOperand(0));
  }

  return false;
}

// Pattern to move reshapes from activation paths to constant paths in
// elementwise binary operations.
//
// Matches patterns like:
//   %const = ttir.constant() : tensor<32x1x2560xf32>
//   %act = ... : tensor<32x2560xf32>
//   %reshaped = ttir.reshape(%act) : tensor<32x2560xf32> ->
//   tensor<32x1x2560xf32> %result = ttir.pow(%reshaped, %const) :
//   tensor<32x1x2560xf32>
//
// Transforms to:
//   %const = ttir.constant() : tensor<32x1x2560xf32>
//   %const_reshaped = ttir.reshape(%const) : tensor<32x1x2560xf32> ->
//   tensor<32x2560xf32> %act = ... : tensor<32x2560xf32> %result =
//   ttir.pow(%act, %const_reshaped) : tensor<32x2560xf32>
class MoveReshapeToConstantPattern
    : public OpInterfaceRewritePattern<ElementwiseBinary> {
public:
  using OpInterfaceRewritePattern<ElementwiseBinary>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseBinary op,
                                PatternRewriter &rewriter) const override {
    // Get operands - expect exactly 2 for binary ops.
    auto operands = op->getOperands();
    if (operands.size() != 2) {
      return failure();
    }

    // Find which operand is a reshape (not from constant) and which traces to a
    // constant.
    ReshapeOp reshapeOp = nullptr;
    Value constOperand = nullptr;
    size_t reshapeOperandIdx = 0;

    for (size_t i = 0; i < 2; ++i) {
      auto reshape = operands[i].getDefiningOp<ReshapeOp>();
      if (reshape && !isFromConstant(operands[i]) &&
          isFromConstant(operands[1 - i])) {
        reshapeOp = reshape;
        constOperand = operands[1 - i];
        reshapeOperandIdx = i;
        break;
      }
    }

    if (!reshapeOp) {
      return failure();
    }

    // The reshape must have a single use (the elementwise op).
    if (!reshapeOp->hasOneUse()) {
      return failure();
    }

    // Get the pre-reshape shape (activation's original shape).
    auto preReshapeType =
        cast<RankedTensorType>(reshapeOp.getInput().getType());
    auto postReshapeType = cast<RankedTensorType>(reshapeOp.getType());

    // The constant operand should have the same shape as the post-reshape
    // (that's why the reshape was added).
    auto constType = cast<RankedTensorType>(constOperand.getType());
    if (constType.getShape() != postReshapeType.getShape()) {
      return failure();
    }

    // Create inverse reshape for the constant: from post-reshape shape to
    // pre-reshape shape.
    SmallVector<int32_t> newShape;
    for (int64_t dim : preReshapeType.getShape()) {
      newShape.push_back(static_cast<int32_t>(dim));
    }

    auto newConstType = RankedTensorType::get(preReshapeType.getShape(),
                                              constType.getElementType());

    auto constReshape = rewriter.create<ReshapeOp>(
        constOperand.getLoc(), newConstType, constOperand,
        rewriter.getI32ArrayAttr(newShape));

    // Create the new elementwise op with the original activation and reshaped
    // constant. Preserve operand order from the original op.
    SmallVector<Value, 2> newOperands(2);
    newOperands[reshapeOperandIdx] = reshapeOp.getInput();
    newOperands[1 - reshapeOperandIdx] = constReshape.getResult();

    // The result type should match the pre-reshape type.
    auto newResultType = RankedTensorType::get(
        preReshapeType.getShape(),
        cast<RankedTensorType>(op->getResult(0).getType()).getElementType());

    Operation *newOp = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(op->getName().getStringRef()),
        newOperands, newResultType, op->getAttrs());

    // Replace the original op result with the new op result.
    // We need a reshape to match the original output shape for downstream
    // users.
    auto outputReshape = rewriter.create<ReshapeOp>(
        op->getLoc(), op->getResult(0).getType(), newOp->getResult(0),
        reshapeOp.getShapeAttr());

    rewriter.replaceOp(op, outputReshape.getResult());

    return success();
  }
};

class TTIRMoveReshapeToConstant
    : public impl::TTIRMoveReshapeToConstantBase<TTIRMoveReshapeToConstant> {
public:
  using impl::TTIRMoveReshapeToConstantBase<
      TTIRMoveReshapeToConstant>::TTIRMoveReshapeToConstantBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<MoveReshapeToConstantPattern>(&getContext());
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

} // namespace
} // namespace mlir::tt::ttir
