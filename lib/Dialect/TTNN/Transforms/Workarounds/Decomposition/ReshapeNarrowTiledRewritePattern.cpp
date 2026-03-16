// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReshapeNarrowTiledRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

static constexpr int64_t kTileWidth = ttnn::TILE_WIDTH;
// NOC DMA minimum transfer size (bytes). Writes smaller than this hang the
// tiled reshape kernel on Wormhole B0 (the only arch with DMA reads/writes
// today). See tt_cluster.cpp min_dma_size_bytes.
// No public API exposes this value at the MLIR dialect level.
static constexpr unsigned kNocMinWriteBytes = 32;

LogicalResult ReshapeNarrowTiledRewritePattern::matchAndRewrite(
    ttnn::ReshapeOp op, PatternRewriter &rewriter) const {
  // This workaround only targets the Wormhole B0 NOC DMA hang; skip on other
  // architectures to avoid unnecessary overhead.
  auto systemDesc = ttcore::getCurrentScopeSystemDesc(op);
  auto arch = systemDesc.getChipDescs()[0].getArch().getValue();
  if (arch != ttcore::Arch::WormholeB0) {
    return failure();
  }

  auto inputValue =
      mlir::cast<mlir::TypedValue<RankedTensorType>>(op.getInput());
  RankedTensorType inputType = inputValue.getType();
  RankedTensorType outputType = op.getResult().getType();

  auto inputLayoutAttr =
      mlir::dyn_cast<TTNNLayoutAttr>(inputType.getEncoding());
  if (!inputLayoutAttr || !inputLayoutAttr.isTiled()) {
    return failure();
  }

  auto outputLayoutAttr =
      mlir::dyn_cast<TTNNLayoutAttr>(outputType.getEncoding());
  if (!outputLayoutAttr || !outputLayoutAttr.isTiled()) {
    return failure();
  }

  int64_t inputRank = inputType.getRank();
  int64_t outputRank = outputType.getRank();
  if (inputRank == 0 || outputRank == 0) {
    return failure();
  }

  int64_t outputLastDim = outputType.getShape()[outputRank - 1];
  int64_t partialWidth = outputLastDim % kTileWidth;

  if (partialWidth == 0) {
    return failure();
  }

  mlir::Type elemType = inputType.getElementType();
  if (!mlir::isa<ttcore::TileType>(elemType) && !elemType.isIntOrFloat()) {
    return failure();
  }
  uint64_t elementSizeBytes = ttcore::getElementSizeBytes(elemType);
  if (elementSizeBytes == 0) {
    return failure();
  }
  unsigned segmentBytes = static_cast<unsigned>(partialWidth) *
                          static_cast<unsigned>(elementSizeBytes);

  // Only decompose when the partial-tile-column segment is smaller than
  // the NOC minimum DMA size.  Larger segments are safe in the tiled
  // reshape kernel.
  if (segmentBytes >= kNocMinWriteBytes) {
    return failure();
  }

  // Skip reshapes the runtime handles as views: when the last dim and the
  // second-to-last dim are unchanged (or both tile-aligned), the tiled
  // reshape kernel is never invoked.
  int64_t inputLastDim = inputType.getShape()[inputRank - 1];
  if (inputLastDim == outputLastDim) {
    if (inputRank >= 2 && outputRank >= 2) {
      int64_t inputSecondLast = inputType.getShape()[inputRank - 2];
      int64_t outputSecondLast = outputType.getShape()[outputRank - 2];
      if (inputSecondLast == outputSecondLast ||
          (inputSecondLast % kTileWidth == 0 &&
           outputSecondLast % kTileWidth == 0)) {
        return failure();
      }
    }
  }

  // Step 1: TILE -> ROW_MAJOR
  auto toRMOp = utils::createToLayoutOp(
      op.getOperation(), inputValue, rewriter, Layout::RowMajor,
      inputLayoutAttr.getBufferType(), inputLayoutAttr.getMemLayout(),
      inputLayoutAttr.getDataType(), "_reshape_wa_to_rm");

  // Step 2: reshape in ROW_MAJOR
  auto rmInputResult =
      mlir::cast<mlir::TypedValue<RankedTensorType>>(toRMOp.getResult());
  RankedTensorType rmOutputType = utils::RankedTensorTypeFactory::create(
      rmInputResult.getType(), outputType.getShape());

  auto reshapeOp = rewriter.create<ttnn::ReshapeOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape_wa_rm"),
      rmOutputType, rmInputResult, op.getShapeAttr(),
      /*memory_config=*/nullptr);

  // Step 3: ROW_MAJOR -> TILE
  auto toTileOp = utils::createToLayoutOp(
      op.getOperation(),
      mlir::cast<mlir::TypedValue<RankedTensorType>>(reshapeOp.getResult()),
      rewriter, Layout::Tile, outputLayoutAttr.getBufferType(),
      outputLayoutAttr.getMemLayout(), outputLayoutAttr.getDataType(),
      "_reshape_wa_to_tile");

  rewriter.replaceOp(op, toTileOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
