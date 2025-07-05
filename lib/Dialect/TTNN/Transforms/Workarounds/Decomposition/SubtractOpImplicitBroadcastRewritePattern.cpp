// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/SubtractOpImplicitBroadcastRewritePattern.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult SubtractOpImplicitBroadcastRewritePattern::matchAndRewrite(
    ttnn::SubtractOp srcOp, PatternRewriter &rewriter) const {
  RankedTensorType lhsType =
      mlir::cast<mlir::RankedTensorType>(srcOp.getLhs().getType());
  RankedTensorType rhsType =
      mlir::cast<mlir::RankedTensorType>(srcOp.getRhs().getType());

  if (lhsType.getShape() == rhsType.getShape()) {
    return failure();
  }

  ttnn::NegOp negOp = rewriter.create<ttnn::NegOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_neg"), rhsType,
      srcOp.getRhs());
  rewriter.replaceOpWithNewOp<ttnn::AddOp>(srcOp, srcOp.getResult().getType(),
                                           srcOp.getLhs(), negOp);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
