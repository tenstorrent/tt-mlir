// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

namespace mlir::tt::ttnn::workarounds::decomposition {

// tt-metal's SDPA decode program factory allocates per-core CBs whose
// footprint scales with head_dim (and num_cores_per_head for the tree-
// reduction partial-accumulator buffer):
//
//   k_tiles               = Sk_chunk_t_cb_size * DHt  * 2  // double-buffered
//   v_tiles               = Sk_chunk_t_cb_size * vDHt * 2  // double-buffered
//   intermed_output_tiles = (PNHt * vDHt + 2 * PNHt) * (num_cores_per_head - 1)
//
// where DHt = head_dim / TILE_WIDTH. For large head_dim (Gemma-4 31B uses
// 512 for global attention and 256 for sliding-window layers) this overflows
// per-core L1 on Blackhole's 1.5 MB usable L1 under the default schedule.
//
// This workaround sets an explicit SDPAProgramConfig that:
//   * pins k_chunk_size = 32 (one page) to bound Sk_chunk_t_cb_size, and
//   * caps max_cores_per_head_batch at 32 (vs the 64-core tree-reduction
//     limit) to halve the intermed-output footprint.
//
// Other configurations (small head_dim, Blackhole defaults) are left
// untouched so the runtime / TTNN default schedule applies.
LogicalResult
PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern::matchAndRewrite(
    ttnn::PagedScaledDotProductAttentionDecodeOp srcOp,
    PatternRewriter &rewriter) const {
  // Skip if program_config is already set (e.g. user-specified or another
  // pass populated it).
  if (srcOp.getProgramConfigAttr()) {
    return failure();
  }

  RankedTensorType queryType =
      mlir::cast<RankedTensorType>(srcOp.getQuery().getType());
  RankedTensorType pageTableType =
      mlir::cast<RankedTensorType>(srcOp.getPageTable().getType());

  // queryShape: [1, B, Hq, head_dim] — pick head_dim from last dim.
  const int64_t headDim = queryType.getShape().back();
  constexpr int64_t kOverrideHeadDimThreshold = 256;
  if (headDim < kOverrideHeadDimThreshold) {
    return failure();
  }

  // pageTableShape: [B, num_blocks] — pick block count from last dim.
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
      srcOp.getSlidingWindowSizeAttr(), srcOp.getMemoryConfigAttr(),
      srcOp.getCoreGridAttr(), programConfig);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
