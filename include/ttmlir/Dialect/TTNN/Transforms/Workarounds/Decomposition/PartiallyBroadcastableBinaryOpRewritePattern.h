// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PARTIALLYBROADCASTABLEBINARYOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PARTIALLYBROADCASTABLEBINARYOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
template <typename BinaryOp>
class PartiallyBroadcastableBinaryOpRewritePattern
    : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp srcOp,
                                PatternRewriter &rewriter) const override {
    assert(srcOp.getNumOperands() == 2);

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(srcOp->getResult(0).getType());
    auto rhsType = mlir::cast<mlir::RankedTensorType>(srcOp.getRhs().getType());
    if (rhsType.getShape() != resultType.getShape()) {
      // Explicate the broadcast/repeat on second operand.
      llvm::SmallVector<int64_t> broadcastDimensions =
          ttmlir::utils::getBroadcastDimensions<int64_t>(rhsType.getShape(),
                                                         resultType.getShape());
      ttnn::ShapeAttr shapeAttr =
          ttnn::ShapeAttr::get(rewriter.getContext(), broadcastDimensions);
      auto repeatOp = rewriter.create<ttnn::RepeatOp>(
          srcOp->getLoc(), srcOp->getResult(0), shapeAttr);
      rewriter.modifyOpInPlace(
          srcOp, [&]() { srcOp.getRhsMutable().assign(repeatOp); });

      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PARTIALLYBROADCASTABLEBINARYOPREWRITEPATTERN_H
