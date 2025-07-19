// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ExplicateOperandBroadcastsRewritePattern.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult ExplicateOperandBroadcastsRewritePattern::matchAndRewrite(
    Operation *srcOp, PatternRewriter &rewriter) const {

  if (!srcOp->hasTrait<ttnn::ExplicateOperandBroadcastsTrait>()) {
    return failure();
  }

  auto resultShape =
      mlir::cast<mlir::RankedTensorType>(srcOp->getResult(0).getType())
          .getShape();
  bool hasChanged = false;

  for (unsigned i = 0; i < srcOp->getNumOperands(); ++i) {
    mlir::Value operand = srcOp->getOperand(i);
    auto operandShape =
        mlir::cast<mlir::RankedTensorType>(operand.getType()).getShape();
    if (operandShape == resultShape) {
      continue;
    }

    auto broadcastDims = ttmlir::utils::getBroadcastDimensions<int64_t>(
        operandShape, resultShape);
    auto shapeAttr = ttnn::ShapeAttr::get(rewriter.getContext(), broadcastDims);
    auto repeatOp = rewriter.create<ttnn::RepeatOp>(
        srcOp->getLoc(),
        mlir::cast<mlir::RankedTensorType>(srcOp->getResult(0).getType()),
        operand, shapeAttr);

    rewriter.modifyOpInPlace(srcOp, [&]() { srcOp->setOperand(i, repeatOp); });
    hasChanged = true;
  }

  return success(hasChanged);
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
