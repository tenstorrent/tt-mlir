// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern::
    matchAndRewrite(ttnn::PagedScaledDotProductAttentionDecodeOp srcOp,
                    PatternRewriter &rewriter) const {
  MLIRContext *context = srcOp.getContext();

  ttcore::SystemDescAttr sysDesc = ttcore::getCurrentScopeSystemDesc(srcOp);
  ttcore::ChipDescAttr chip = sysDesc.getChipDesc(0);
  const bool isBlackhole = chip.getArch().getValue() == ttcore::Arch::Blackhole;

  // Only Blackhole needs a compile-time config (see the pattern header).
  if (!isBlackhole) {
    return failure();
  }

  SDPAProgramConfigAttr existing = srcOp.getProgramConfigAttr();

  // Preserve an existing config and layer exp_approx_mode = false on top;
  // otherwise replicate metal's paged-decode defaults.
  CoreCoordAttr grid;
  CoreRangeSetAttr subCoreGrids = nullptr;
  uint64_t qChunkSize = 32; // kDefaultDecodeChunkSize
  uint64_t kChunkSize = 32; // kDefaultDecodeChunkSize
  std::optional<uint32_t> maxCoresPerHeadBatch =
      1; // kDefaultMaxCoresPerHeadBatch

  if (existing) {
    grid = existing.getComputeWithStorageGridSize();
    subCoreGrids = existing.getSubCoreGrids();
    qChunkSize = existing.getQChunkSize();
    kChunkSize = existing.getKChunkSize();
    maxCoresPerHeadBatch = existing.getMaxCoresPerHeadBatch();
  } else {
    // WorkerGrid shape is [Y, X]; CoreCoord is (x, y).
    ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(srcOp.getOperation());
    llvm::ArrayRef<int64_t> workerGridShape =
        deviceAttr.getWorkerGrid().getShape();
    grid =
        CoreCoordAttr::get(context, static_cast<uint32_t>(workerGridShape[1]),
                           static_cast<uint32_t>(workerGridShape[0]));
  }

  BoolAttr expApproxMode = BoolAttr::get(context, /*value=*/false);

  auto desired = SDPAProgramConfigAttr::get(
      context, grid, subCoreGrids, qChunkSize, kChunkSize, expApproxMode,
      maxCoresPerHeadBatch);

  // Idempotency: bail when nothing changes so the greedy driver terminates.
  if (existing == desired) {
    return failure();
  }

  rewriter.replaceOpWithNewOp<ttnn::PagedScaledDotProductAttentionDecodeOp>(
      srcOp, srcOp.getResult().getType(), srcOp.getQuery(), srcOp.getKey(),
      srcOp.getValue(), srcOp.getPageTable(), srcOp.getIsCausalAttr(),
      srcOp.getAttentionMask(), srcOp.getCurPosTensor(),
      srcOp.getAttentionSink(), srcOp.getScaleAttr(),
      srcOp.getSlidingWindowSizeAttr(), desired);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
