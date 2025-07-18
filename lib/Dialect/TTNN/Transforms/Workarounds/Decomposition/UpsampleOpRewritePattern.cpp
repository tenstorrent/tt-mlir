// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/UpsampleOpRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>

namespace mlir::tt::ttnn::workarounds::decomposition {
namespace {
enum UpsampleAxisLayout {
  BATCH = 0,
  HEIGHT = 1,
  WIDTH = 2,
  CHANNEL = 3,
  DIM_COUNT = 4
};
} // namespace

LogicalResult UpsampleOpBilinearPaddingRewritePattern::matchAndRewrite(
    ttnn::UpsampleOp srcOp, PatternRewriter &rewriter) const {
  if (srcOp.getMode() != "bilinear") {
    return failure();
  }

  constexpr int64_t TILE_WIDTH = ttcore::TileType::getDefaultShape()[1];

  RankedTensorType inputType = srcOp.getInput().getType();

  // UpsampleOp in `bilinear` mode expects CHANNEL dimension to be a multiple of
  // TILE_WIDTH, hence we don't apply this pattern if that's already the case.
  if (inputType.getDimSize(CHANNEL) % TILE_WIDTH == 0) {
    return failure();
  }

  // Calculate padded dimension size (round up to nearest multiple of
  // TILE_WIDTH).
  int64_t originalChannelSize = inputType.getDimSize(CHANNEL);
  int64_t paddedChannelSize =
      llvm::divideCeil(originalChannelSize, TILE_WIDTH) * TILE_WIDTH;
  int64_t paddingAmount = paddedChannelSize - originalChannelSize;

  // Create padding configuration - PadOp uses a single padding array with pairs
  // (low, high) for each axis.
  SmallVector<int32_t> padding(DIM_COUNT * 2, 0);
  padding[CHANNEL * 2 + 1] = paddingAmount;

  SmallVector<int64_t> paddedShape(inputType.getShape());
  paddedShape[CHANNEL] = paddedChannelSize;

  auto paddedType = RankedTensorType::get(
      paddedShape, inputType.getElementType(),
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding())
          .withTensorShape(paddedShape));

  auto padOp = rewriter.create<ttnn::PadOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getInput().getLoc(), "pad"),
      paddedType, srcOp.getInput(), padding, /*pad_value=*/mlir::APFloat(0.0f),
      /*use_multicore=*/false,
      /*memory_config=*/nullptr);

  RankedTensorType outputType = srcOp.getResult().getType();
  llvm::SmallVector<int64_t> upsamplePaddedShape(outputType.getShape());
  upsamplePaddedShape[CHANNEL] = paddedChannelSize;

  auto upsampledPaddedType = RankedTensorType::get(
      upsamplePaddedShape, outputType.getElementType(),
      mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding())
          .withTensorShape(upsamplePaddedShape));

  auto paddedUpsampleOp = rewriter.create<ttnn::UpsampleOp>(
      srcOp.getLoc(), upsampledPaddedType, padOp, srcOp.getScaleFactorAttr(),
      srcOp.getModeAttr(), /*memory_config=*/nullptr);

  // Create SliceOp to remove padding from the upsampled result.
  SmallVector<int32_t> begins(/*size=*/DIM_COUNT, /*value=*/0);
  SmallVector<int32_t> ends(outputType.getShape());
  SmallVector<int32_t> steps(/*size=*/DIM_COUNT, /*value=*/1);

  auto sliceOp = rewriter.create<ttnn::SliceOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "slice"), outputType,
      paddedUpsampleOp, rewriter.getI32ArrayAttr(begins),
      rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));

  rewriter.replaceOp(srcOp, sliceOp);

  return success();
}
} // namespace mlir::tt::ttnn::workarounds::decomposition
