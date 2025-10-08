// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv2dSliceConfigRewritePattern.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

mlir::LogicalResult Conv2dSliceConfigRewritePattern::matchAndRewrite(
    ttnn::Conv2dOp srcOp, PatternRewriter &rewriter) const {
  if (srcOp.getConv2dSliceConfig()) {
    // Conv2dSliceConfig is already set, no need to apply the workaround.
    return failure();
  }

  auto conv2dSliceConfigAttr = mlir::tt::ttnn::Conv2dSliceConfigAttr::get(
      rewriter.getContext(), mlir::tt::ttnn::Conv2dSliceType::L1Full, 0);

  rewriter.replaceOp(
      srcOp,
      rewriter.create<ttnn::Conv2dOp>(
          srcOp->getLoc(), srcOp.getResult().getType(), srcOp.getInput(),
          srcOp.getWeight(), srcOp.getBias(), srcOp.getDevice(),
          srcOp.getInChannels(), srcOp.getOutChannels(), srcOp.getBatchSize(),
          srcOp.getInputHeight(), srcOp.getInputWidth(), srcOp.getKernelSize(),
          srcOp.getStride(), srcOp.getPadding(), srcOp.getDilation(),
          srcOp.getGroups(), srcOp.getDtypeAttr(), srcOp.getConv2dConfigAttr(),
          srcOp.getComputeConfigAttr(), conv2dSliceConfigAttr));

  return success();
}
} // namespace mlir::tt::ttnn::workarounds::decomposition
