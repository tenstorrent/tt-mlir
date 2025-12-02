// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionDecodeConfigRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
ScaledDotProductAttentionDecodeConfigRewritePattern::matchAndRewrite(
    ScaledDotProductAttentionDecodeOp srcOp, PatternRewriter &rewriter) const {
  if (srcOp.getProgramConfigAttr()) {
    return failure();
  }

  MLIRContext *context = srcOp.getContext();

  // Get actual grid size from device
  ttcore::DeviceAttr deviceAttr = mlir::tt::ttcore::lookupDevice(srcOp);
  if (!deviceAttr) {
    return failure();
  }

  auto workerGrid = deviceAttr.getWorkerGrid();
  auto gridShape = workerGrid.getShape();

  auto computeGridSize =
      CoreCoordAttr::get(context, /*x=*/gridShape[1], /*y=*/gridShape[0]);
  uint64_t qChunkSize = 32;
  uint64_t kChunkSize = 32;

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
