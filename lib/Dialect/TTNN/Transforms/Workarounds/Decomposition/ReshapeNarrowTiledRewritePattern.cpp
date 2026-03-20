// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReshapeNarrowTiledRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

static constexpr int64_t kTileWidth = ttnn::TILE_WIDTH;
// NOC DMA minimum transfer size (bytes). Writes smaller than this hang the
// tiled reshape kernel. See tt_cluster.cpp min_dma_size_bytes.
static constexpr unsigned kNocMinWriteBytes = 32;

static int64_t roundUpToTile(int64_t val) {
  return ((val + kTileWidth - 1) / kTileWidth) * kTileWidth;
}

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

  // Guard against recursion: step 1 of the decomposition produces a 1D
  // flatten. Without this guard the pattern would fire on that intermediate
  // reshape op as well.
  if (outputRank == 1) {
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

  // Compute total element count from the input shape.
  int64_t totalElements = 1;
  for (int64_t d : inputType.getShape()) {
    totalElements *= d;
  }

  // Compute the outer dimensions product (all output dims except last).
  int64_t outerOutput = 1;
  for (int64_t i = 0; i < outputRank - 1; ++i) {
    outerOutput *= outputType.getShape()[i];
  }

  int64_t paddedLastDim = roundUpToTile(outputLastDim);
  int64_t paddedTotal = outerOutput * paddedLastDim;

  // Step 1: flatten input to 1D [totalElements].
  // outputRank == 1 guard above ensures the pattern won't fire on this op.
  SmallVector<int64_t> flatShape = {totalElements};
  auto flattenOp = ttir_to_ttnn::utils::generateReshape(
      inputValue, flatShape, rewriter,
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape_wa_flatten"));

  // Step 2: pad [totalElements] → [outerOutput * paddedLastDim].
  // padding is [front, back] for the single dimension.
  SmallVector<int32_t> padding = {
      0, static_cast<int32_t>(paddedTotal - totalElements)};
  auto flatValue =
      mlir::cast<mlir::TypedValue<RankedTensorType>>(flattenOp.getResult());
  auto padOp = ttir_to_ttnn::utils::generatePad(
      flatValue, padding, rewriter,
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape_wa_pad"));

  // Step 3: reshape [outerOutput * paddedLastDim] → [..., paddedLastDim].
  // paddedLastDim is tile-aligned so partialWidth == 0 → this reshape is safe.
  SmallVector<int64_t> paddedOutputShape(outputType.getShape().begin(),
                                         outputType.getShape().end());
  paddedOutputShape[outputRank - 1] = paddedLastDim;
  auto paddedValue =
      mlir::cast<mlir::TypedValue<RankedTensorType>>(padOp.getResult());
  auto paddedReshapeOp = ttir_to_ttnn::utils::generateReshape(
      paddedValue, paddedOutputShape, rewriter,
      ttmlir::utils::appendLocationSuffix(op.getLoc(),
                                          "_reshape_wa_padded_reshape"));

  // Step 4: slice [..., paddedLastDim] → [..., outputLastDim].
  ArrayRef<int64_t> originalOutputShape = outputType.getShape();
  SmallVector<int32_t> begins(outputRank, 0);
  SmallVector<int32_t> ends(originalOutputShape.begin(),
                            originalOutputShape.end());
  SmallVector<int32_t> steps(outputRank, 1);

  auto sliceOp = rewriter.create<ttnn::SliceStaticOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape_wa_slice"),
      outputType, paddedReshapeOp.getResult(),
      rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
      rewriter.getI32ArrayAttr(steps));

  rewriter.replaceOp(op, sliceOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
