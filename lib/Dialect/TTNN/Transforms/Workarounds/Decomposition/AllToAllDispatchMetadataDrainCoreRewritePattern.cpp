// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/AllToAllDispatchMetadataDrainCoreRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult AllToAllDispatchMetadataDrainCoreRewritePattern::matchAndRewrite(
    AllToAllDispatchMetadataOp srcOp, PatternRewriter &rewriter) const {
  if (srcOp.getDrainCoreAttr()) {
    return failure();
  }

  MLIRContext *context = srcOp.getContext();

  // Default drain core: CoreCoord(0, 0).
  auto drainCore = CoreCoordAttr::get(context, /*x=*/0, /*y=*/0);

  rewriter.replaceOpWithNewOp<AllToAllDispatchMetadataOp>(
      srcOp, srcOp.getDispatched().getType(), srcOp.getIndices().getType(),
      srcOp.getScores().getType(), srcOp.getInputTensor(),
      srcOp.getExpertIndices(), srcOp.getExpertScores(),
      srcOp.getExpertMapping(), srcOp.getOptionalDispatchedOutputTensor(),
      srcOp.getOptionalIndicesOutputTensor(),
      srcOp.getOptionalScoresOutputTensor(), srcOp.getNumDevicesAttr(),
      srcOp.getClusterAxisAttr(), srcOp.getMemoryConfigAttr(), drainCore);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
