// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

namespace mlir::tt::ttnn::workarounds::decomposition {

// SDPA decode allocates per-core CBs whose footprint scales with head_dim
// (and num_cores_per_head for the tree-reduction partial-accumulator). For
// head_dim >= 256 the default schedule overflows L1. Pin k_chunk_size=32
// and cap max_cores_per_head_batch at 32.
LogicalResult PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern::
    matchAndRewrite(ttnn::PagedScaledDotProductAttentionDecodeOp srcOp,
                    PatternRewriter &rewriter) const {
  if (optimizationLevel != 0) {
    return failure();
  }

  if (srcOp.getProgramConfigAttr()) {
    return failure();
  }

  RankedTensorType queryType =
      mlir::cast<RankedTensorType>(srcOp.getQuery().getType());
  RankedTensorType pageTableType =
      mlir::cast<RankedTensorType>(srcOp.getPageTable().getType());

  // queryShape: [1, B, Hq, head_dim] -- pick head_dim from last dim.
  const int64_t headDim = queryType.getShape().back();
  constexpr int64_t kOverrideHeadDimThreshold = 256;
  if (headDim < kOverrideHeadDimThreshold) {
    return failure();
  }

  // pageTableShape: [B, num_blocks] -- pick block count from last dim.
  const int64_t pageTableBlocks = pageTableType.getShape().back();

  MLIRContext *context = srcOp.getContext();
  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(srcOp.getOperation());
  auto workerGridShape = deviceAttr.getWorkerGrid().getShape();
  // workerGrid is [Y, X]; CoreCoord is (x, y).
  const auto gridX = static_cast<uint32_t>(workerGridShape[1]);
  const auto gridY = static_cast<uint32_t>(workerGridShape[0]);
  const uint32_t totalCores = gridX * gridY;

  constexpr uint32_t kMaxSdpaDecodeCoresPerHeadBatch = 64u;
  constexpr uint32_t kOverrideCoresCap = 32u;
  constexpr uint32_t kPageTokens = 32u;

  const uint32_t coresCap =
      std::min(totalCores, kMaxSdpaDecodeCoresPerHeadBatch);
  const uint32_t overrideCap = std::min(coresCap, kOverrideCoresCap);
  const uint32_t maxCoresPerHeadBatch =
      std::min(static_cast<uint32_t>(pageTableBlocks), overrideCap);

  auto computeGrid = CoreCoordAttr::get(context, /*x=*/gridX, /*y=*/gridY);
  auto programConfig = SDPAProgramConfigAttr::get(
      context, computeGrid, /*sub_core_grids=*/nullptr,
      /*q_chunk_size=*/0u, /*k_chunk_size=*/kPageTokens,
      /*exp_approx_mode=*/nullptr,
      /*max_cores_per_head_batch=*/
      std::optional<uint32_t>(maxCoresPerHeadBatch));

  rewriter.replaceOpWithNewOp<ttnn::PagedScaledDotProductAttentionDecodeOp>(
      srcOp, srcOp.getResult().getType(), srcOp.getQuery(), srcOp.getKey(),
      srcOp.getValue(), srcOp.getPageTable(), srcOp.getIsCausalAttr(),
      srcOp.getAttentionMask(), srcOp.getCurPosTensor(),
      srcOp.getAttentionSink(), srcOp.getScaleAttr(),
      srcOp.getSlidingWindowSizeAttr(), programConfig);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
