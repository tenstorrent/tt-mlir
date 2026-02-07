// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReduceScatterConfigRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
ReduceScatterConfigRewritePattern::matchAndRewrite(ReduceScatterOp srcOp,
                                                   PatternRewriter &rewriter) const {
  // Skip if compute config is already set
  if (srcOp.getComputeConfigAttr()) {
    return failure();
  }

  MLIRContext *context = srcOp.getContext();

  // Create a high-precision compute config with FP32 accumulation enabled.
  // This improves numerical accuracy for reduce_scatter operations.
  auto computeConfig = DeviceComputeKernelConfigAttr::get(
      context,
      /*mathFidelity=*/MathFidelity::HiFi4,
      /*mathApproxMode=*/BoolAttr::get(context, false),
      /*fp32DestAccEn=*/BoolAttr::get(context, true),
      /*packerL1Acc=*/BoolAttr::get(context, false),
      /*dstFullSyncEn=*/nullptr);

  // Create a new operation with the compute config set
  auto newOp = rewriter.create<ReduceScatterOp>(
      srcOp.getLoc(), srcOp.getResult().getType(), srcOp.getInput(),
      srcOp.getReduceTypeAttr(), srcOp.getScatterDimAttr(),
      srcOp.getClusterAxisAttr(), srcOp.getSubDeviceIdAttr(),
      srcOp.getMemoryConfigAttr(), srcOp.getNumLinksAttr(),
      srcOp.getTopologyAttr(), computeConfig);

  rewriter.replaceOp(srcOp, newOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
