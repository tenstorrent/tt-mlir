// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/SDPAProgramConfig.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

namespace mlir::tt::ttnn {

std::optional<SDPAProgramConfigAttr>
generateSDPADecodeProgramConfig(Operation *op) {
  auto sdpaOp = mlir::dyn_cast<ttnn::PagedScaledDotProductAttentionDecodeOp>(op);
  if (!sdpaOp) {
    return std::nullopt;
  }

  // Respect an explicit, user-/workaround-provided program config.
  if (sdpaOp.getProgramConfigAttr()) {
    return std::nullopt;
  }

  RankedTensorType queryType =
      mlir::cast<RankedTensorType>(sdpaOp.getQuery().getType());
  RankedTensorType pageTableType =
      mlir::cast<RankedTensorType>(sdpaOp.getPageTable().getType());

  // queryShape: [1, B, Hq, head_dim] -- pick head_dim from last dim.
  const int64_t headDim = queryType.getShape().back();
  // pageTableShape: [B, num_blocks] -- pick block count from last dim.
  const int64_t pageTableBlocks = pageTableType.getShape().back();

  MLIRContext *context = sdpaOp.getContext();
  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);
  auto workerGridShape = deviceAttr.getWorkerGrid().getShape();
  // workerGrid is [Y, X]; CoreCoord is (x, y).
  const auto gridX = static_cast<uint32_t>(workerGridShape[1]);
  const auto gridY = static_cast<uint32_t>(workerGridShape[0]);
  const uint32_t totalCores = gridX * gridY;

  // TODO(#44311): Derive these from a proper L1-aware cost model rather than
  // the fixed heuristic ported from the opt-level=0 workaround. The current
  // values pin a small k_chunk_size and cap the per-head/batch core count so
  // that the per-core CB footprint (which scales with head_dim and
  // num_cores_per_head) does not overflow L1 for head_dim >= 256.
  constexpr uint32_t kMaxSdpaDecodeCoresPerHeadBatch = 64u;
  constexpr uint32_t kCoresCap = 32u;
  constexpr uint32_t kPageTokens = 32u;

  const uint32_t coresCap =
      std::min(totalCores, kMaxSdpaDecodeCoresPerHeadBatch);
  const uint32_t cappedCores = std::min(coresCap, kCoresCap);
  const uint32_t maxCoresPerHeadBatch =
      std::min(static_cast<uint32_t>(pageTableBlocks), cappedCores);

  // TODO(#44311): head_dim currently only informs whether the config is needed
  // (large head_dim overflows the default schedule). Once the cost model
  // lands, use it to size q/k chunking.
  (void)headDim;

  auto computeGrid = CoreCoordAttr::get(context, /*x=*/gridX, /*y=*/gridY);
  return SDPAProgramConfigAttr::get(
      context, computeGrid, /*sub_core_grids=*/nullptr,
      /*q_chunk_size=*/0u, /*k_chunk_size=*/kPageTokens,
      /*exp_approx_mode=*/nullptr,
      /*max_cores_per_head_batch=*/
      std::optional<uint32_t>(maxCoresPerHeadBatch));
}

} // namespace mlir::tt::ttnn
