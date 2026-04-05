// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RMSNormConfigRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
RMSNormConfigRewritePattern::matchAndRewrite(RMSNormOp srcOp,
                                             PatternRewriter &rewriter) const {
  // Skip if compute config is already set
  if (srcOp.getComputeConfigAttr()) {
    return failure();
  }

  MLIRContext *context = srcOp.getContext();

  // Create a maximum-precision compute config:
  // - math_fidelity: HiFi4 (highest fidelity)
  // - math_approx_mode: false (disable approximations for better accuracy)
  // - fp32_dest_acc_en: true (enable FP32 accumulation in destination
  // registers)
  // - packer_l1_acc: true (use FP32 precision for L1 accumulation)
  // - dst_full_sync_en: false (default)
  //
  // This provides maximum numerical precision for RMS norm operations.
  auto computeConfig = DeviceComputeKernelConfigAttr::get(
      context,
      /*mathFidelity=*/MathFidelity::HiFi4,
      /*mathApproxMode=*/BoolAttr::get(context, false),
      /*fp32DestAccEn=*/BoolAttr::get(context, true),
      /*packerL1Acc=*/BoolAttr::get(context, true),
      /*dstFullSyncEn=*/nullptr);

  // Create a new operation with the compute config set
  auto newOp = rewriter.create<RMSNormOp>(
      srcOp.getLoc(), srcOp.getResult().getType(), srcOp.getInput(),
      srcOp.getWeight(), srcOp.getBias(), srcOp.getEpsilonAttr(),
      srcOp.getMemoryConfigAttr(), computeConfig);

  rewriter.replaceOp(srcOp, newOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
