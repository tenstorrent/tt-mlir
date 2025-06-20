// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EXPLICATEOPERANDBROADCASTSREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EXPLICATEOPERANDBROADCASTSREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Currently, not all binary eltwise ops in tt-metal support implicit
// broadcasting on their operands. For those that don't, we have to explicate
// the broadcast on each operand.
template <typename EltwiseOp>
class ExplicateOperandBroadcastsRewritePattern
    : public OpRewritePattern<EltwiseOp> {
  using OpRewritePattern<EltwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(EltwiseOp srcOp,
                                PatternRewriter &rewriter) const override {
    assert(srcOp.getNumOperands() >= 2);

    bool hasChanged = false;
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(srcOp->getResult(0).getType());
    for (int64_t i = 0; i < srcOp->getNumOperands(); ++i) {
      mlir::Value operand = srcOp->getOperand(i);
      ::llvm::ArrayRef<int64_t> operandShape =
          mlir::cast<RankedTensorType>(operand.getType()).getShape();
      if (operandShape != resultType.getShape()) {
        llvm::SmallVector<int64_t> broadcastDimensions =
            ttmlir::utils::getBroadcastDimensions<int64_t>(
                operandShape, resultType.getShape());
        ttnn::ShapeAttr shapeAttr =
            ttnn::ShapeAttr::get(rewriter.getContext(), broadcastDimensions);
        auto repeatOp = rewriter.create<ttnn::RepeatOp>(
            srcOp->getLoc(), resultType, operand, shapeAttr);
        rewriter.modifyOpInPlace(srcOp,
                                 [&]() { srcOp->setOperand(i, repeatOp); });
        hasChanged = true;
      }
    }

    return success(hasChanged);
  }
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EXPLICATEOPERANDBROADCASTSREWRITEPATTERN_H
