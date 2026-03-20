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
// tiled reshape kernel. See tt_cluster.cpp min_dma_size_bytes.
static constexpr unsigned kNocMinWriteBytes = 32;
// The RM fallback allocates circular buffers proportional to the larger row-
// major stick. Keep a safety margin relative to usable L1 so wide sticks do
// not overflow per-core memory.
static constexpr uint64_t kMaxRMFallbackStickUsableL1Divisor = 4;

LogicalResult ReshapeNarrowTiledRewritePattern::matchAndRewrite(
    ttnn::ReshapeOp op, PatternRewriter &rewriter) const {
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

  int64_t inputLastDim = inputType.getShape()[inputRank - 1];
  uint64_t inputStickBytes =
      static_cast<uint64_t>(inputLastDim) * elementSizeBytes;
  uint64_t outputStickBytes =
      static_cast<uint64_t>(outputLastDim) * elementSizeBytes;
  uint64_t maxRMFallbackStickBytes =
      ttcore::getOpChipDescAttr(op).getUsableL1Size() /
      kMaxRMFallbackStickUsableL1Divisor;

  // Skip the workaround when the row-major fallback would need very large
  // sticks. Those can exceed L1 in reshape_rm even when the tiled reshape is
  // otherwise valid.
  if (inputStickBytes > maxRMFallbackStickBytes ||
      outputStickBytes > maxRMFallbackStickBytes) {
    return failure();
  }

  // Skip reshapes the runtime handles as views: when the last dim and the
  // second-to-last dim are unchanged (or both tile-aligned), the tiled
  // reshape kernel is never invoked.
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

  // Convert the input from tiled layout to row-major layout.
  auto toRMOp = utils::createToLayoutOp(
      op.getOperation(), inputValue, rewriter, Layout::RowMajor,
      inputLayoutAttr.getBufferType(), inputLayoutAttr.getMemLayout(),
      inputLayoutAttr.getDataType(), "_reshape_wa_to_rm");

  // Perform the reshape in row-major layout.
  auto rmInputResult =
      mlir::cast<mlir::TypedValue<RankedTensorType>>(toRMOp.getResult());
  RankedTensorType rmOutputType = utils::RankedTensorTypeFactory::create(
      rmInputResult.getType(), outputType.getShape());

  auto reshapeOp = rewriter.create<ttnn::ReshapeOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape_wa_rm"),
      rmOutputType, rmInputResult, op.getShapeAttr(),
      /*memory_config=*/nullptr);

  // Convert the reshaped result back to tiled layout.
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
