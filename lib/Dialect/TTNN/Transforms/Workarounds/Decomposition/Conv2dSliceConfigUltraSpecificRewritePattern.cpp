// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv2dSliceConfigUltraSpecificRewritePattern.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

mlir::LogicalResult
Conv2dSliceConfigUltraSpecificRewritePattern::matchAndRewrite(
    ttnn::Conv2dOp srcOp, PatternRewriter &rewriter) const {

  // TL;DR: This is a temporary workaround to avoid L1 OOM issues with particular convolutions in SpeechT5 vocoder model (particularly, conv_post module).
  // Going from L1Full memory config to DramWidth avoids L1 OOM issues.

  auto existingConfig = srcOp.getConv2dSliceConfig();
  if (!existingConfig) {
    return failure();
  }

  if (existingConfig->getSliceType() != mlir::tt::ttnn::Conv2dSliceType::L1Full ||
      existingConfig->getNumSlices() != 0) {
    return failure();
  }

  if (srcOp.getInputHeight() != 1) {
    return failure();
  }

  auto kernelSize = srcOp.getKernelSize();

  if (kernelSize.size() != 2 || kernelSize[0] != 1 || kernelSize[1] != 7) {
    return failure();
  }

  auto conv2dSliceConfigAttr = mlir::tt::ttnn::Conv2dSliceConfigAttr::get(
      rewriter.getContext(), mlir::tt::ttnn::Conv2dSliceType::DramWidth, 0);

  rewriter.modifyOpInPlace(
      srcOp, [&]() { srcOp.setConv2dSliceConfigAttr(conv2dSliceConfigAttr); });

  return success();
}
} // namespace mlir::tt::ttnn::workarounds::decomposition
