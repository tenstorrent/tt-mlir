// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PagedScaledDotProductAttentionDecodeComputeConfigRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult PagedScaledDotProductAttentionDecodeComputeConfigRewritePattern::
    matchAndRewrite(PagedScaledDotProductAttentionDecodeOp srcOp,
                    PatternRewriter &rewriter) const {
  // Skip if compute config is already set
  if (srcOp.getComputeConfigAttr()) {
    return failure();
  }

  MLIRContext *context = srcOp.getContext();

  // Create a high-precision compute config with FP32 accumulation enabled.
  // This improves numerical accuracy for paged SDPA decode operations.
  // Using HiFi4 math fidelity for better precision in attention computations.
  auto computeConfig = DeviceComputeKernelConfigAttr::get(
      context,
      /*mathFidelity=*/MathFidelity::HiFi4,
      /*mathApproxMode=*/BoolAttr::get(context, false),
      /*fp32DestAccEn=*/BoolAttr::get(context, false),
      /*packerL1Acc=*/BoolAttr::get(context, false),
      /*dstFullSyncEn=*/nullptr);

  // Create a new operation with the compute config set
  rewriter.replaceOpWithNewOp<PagedScaledDotProductAttentionDecodeOp>(
      srcOp, srcOp.getResult().getType(), srcOp.getQuery(), srcOp.getKey(),
      srcOp.getValue(), srcOp.getPageTable(), srcOp.getIsCausalAttr(),
      srcOp.getAttentionMask(), srcOp.getCurPosTensor(),
      srcOp.getAttentionSink(), srcOp.getScaleAttr(),
      srcOp.getMemoryConfigAttr(), srcOp.getCoreGridAttr(), computeConfig);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
