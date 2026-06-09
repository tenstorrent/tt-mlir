// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MoeComputeRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <utility>

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
MoeComputeRewritePattern::matchAndRewrite(MoeComputeOp srcOp,
                                          PatternRewriter &rewriter) const {
  // tt-metal's moe_compute kernel globally-allocates the expert-indices/scores
  // circular buffers against a single backing buffer whose data must live on
  // the tilize drain core; validate() never checks this, so we must reshard
  // those two inputs to L1 HEIGHT_SHARDED on the drain core or the kernel reads
  // garbage L1. Output layouts are device-derived and captured separately by
  // TTNNDeduceMoEComputeLayouts, so this pattern only fixes up the 2 inputs.
  auto indicesType = mlir::cast<RankedTensorType>(
      srcOp.getTilizeExpertIndicesTensor().getType());
  auto indicesEncoding = mlir::cast<TTNNLayoutAttr>(indicesType.getEncoding());

  // Idempotency: once the indices input is L1 HEIGHT_SHARDED we are done.
  if (indicesEncoding.getBufferType() == BufferType::L1 &&
      indicesEncoding.getMemLayout() &&
      indicesEncoding.getMemLayout().getValue() ==
          TensorMemoryLayout::HeightSharded) {
    return failure();
  }

  MLIRContext *ctx = srcOp.getContext();

  // The drain core is the first entry of tt-metal moe_compute
  // get_layout().max_tilize_cores that fits the device's logical worker grid:
  // the program factory filters that list by compute_with_storage_grid_size and
  // takes index 0 (moe_compute_program_factory.cpp). Replicate that here —
  // incl. the {x,y} candidate order — so the reshard lands on the kernel's
  // actual drain core even on harvested grids that drop the leading (y=9)
  // entry.
  ttcore::SystemDescAttr sysDesc = ttcore::getCurrentScopeSystemDesc(srcOp);
  ttcore::ChipDescAttr chip = sysDesc.getChipDesc(0);
  // ChipDescAttr grid is [y, x] (flatbuffer Dim2d), from compute_with_storage.
  llvm::ArrayRef<int64_t> chipGrid = chip.getGrid();
  int64_t gridY = chipGrid[0], gridX = chipGrid[1];
  bool isBlackhole = chip.getArch().getValue() == ttcore::Arch::Blackhole;
  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> tilizeCandidates =
      isBlackhole ? llvm::SmallVector<std::pair<int64_t, int64_t>, 4>{{10, 9},
                                                                      {10, 8},
                                                                      {9, 9},
                                                                      {9, 8}}
                  : llvm::SmallVector<std::pair<int64_t, int64_t>, 4>{
                        {6, 9}, {6, 8}, {5, 9}, {5, 8}};
  int64_t drainX = -1, drainY = -1;
  for (const auto &[x, y] : tilizeCandidates) {
    if (x < gridX && y < gridY) {
      drainX = x;
      drainY = y;
      break;
    }
  }
  if (drainX < 0) {
    return rewriter.notifyMatchFailure(
        srcOp, "no moe_compute tilize drain core fits the worker grid");
  }

  // The expert-indices/scores CBs are globally allocated against a single
  // backing buffer that must live on the drain core.
  CoreRangeSetAttr tilizeDrainCoreRangeSet = CoreRangeSetAttr::get(
      ctx, CoreRangeAttr::get(ctx, CoreCoordAttr::get(ctx, drainX, drainY),
                              CoreCoordAttr::get(ctx, drainX, drainY)));

  // Insert a ttnn.to_memory_config converting `oldInput` to
  // L1 HEIGHT_SHARDED ROW_MAJOR on the drain core.
  auto reshardToTilize = [&](Value oldInput) -> Value {
    auto t = mlir::cast<RankedTensorType>(oldInput.getType());
    auto seed = mlir::cast<TTNNLayoutAttr>(t.getEncoding());
    auto newEncoding = TTNNLayoutAttr::Builder(seed, t.getShape())
                           .setBufferType(BufferType::L1)
                           .setMemoryLayout(TensorMemoryLayout::HeightSharded)
                           .setLayout(Layout::RowMajor)
                           .setGridShape({1, 1})
                           .setCoreRangeSet(tilizeDrainCoreRangeSet)
                           .build();
    auto newType =
        RankedTensorType::get(t.getShape(), t.getElementType(), newEncoding);
    return rewriter.create<ttnn::ToLayoutOp>(
        srcOp.getLoc(), newType, oldInput,
        LayoutAttr::get(ctx, Layout::RowMajor));
  };

  Value newIndices = reshardToTilize(srcOp.getTilizeExpertIndicesTensor());
  Value newScores = reshardToTilize(srcOp.getTilizeExpertScoresTensor());

  rewriter.modifyOpInPlace(srcOp, [&]() {
    srcOp.getTilizeExpertIndicesTensorMutable().assign(newIndices);
    srcOp.getTilizeExpertScoresTensorMutable().assign(newScores);
  });
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
