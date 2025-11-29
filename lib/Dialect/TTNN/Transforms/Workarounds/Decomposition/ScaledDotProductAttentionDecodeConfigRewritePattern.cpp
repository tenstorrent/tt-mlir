// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionDecodeConfigRewritePattern.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

// Workaround which adds a default SDPAProgramConfig to
// ScaledDotProductAttentionDecodeOp when no config is present because
// it will fail otherwise.
// Metal issue reference: https://github.com/tenstorrent/tt-metal/issues/32641
namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
ScaledDotProductAttentionDecodeConfigRewritePattern::matchAndRewrite(
    ScaledDotProductAttentionDecodeOp srcOp, PatternRewriter &rewriter) const {
  if (srcOp.getProgramConfigAttr()) {
    return failure();
  }

  MLIRContext *context = srcOp.getContext();

  // Create a default SDPA program config
  auto computeGridSize = CoreCoordAttr::get(context, /*x=*/8, /*y=*/8);
  uint64_t qChunkSize = 32; // Default chunk size
  uint64_t kChunkSize = 32; // Default chunk size

  auto programConfig = SDPAProgramConfigAttr::get(
      context, computeGridSize,
      /*sub_core_grids=*/nullptr, qChunkSize, kChunkSize,
      /*exp_approx_mode=*/nullptr,
      /*max_cores_per_head_batch=*/std::nullopt);

  // Create a new operation with the program config set
  auto newOp = rewriter.create<ScaledDotProductAttentionDecodeOp>(
      srcOp.getLoc(), srcOp.getResult().getType(), srcOp.getQuery(),
      srcOp.getKey(), srcOp.getValue(), srcOp.getIsCausal(),
      srcOp.getAttentionMask(), srcOp.getCurPosTensor(),
      srcOp.getAttentionSink(), srcOp.getScaleAttr(),
      srcOp.getMemoryConfigAttr(), programConfig);

  rewriter.replaceOp(srcOp, newOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
