// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/GroupNormChannelPadRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Pad a 1D affine parameter (weight or bias) from originalC to paddedC by
// inserting a PadOp that zero-pads along the single dimension.
static Value padAffine1D(Value affine, int64_t originalC, int64_t paddedC,
                         StringRef suffix, Location loc,
                         PatternRewriter &rewriter) {
  if (!affine) {
    return affine;
  }

  auto affineType = mlir::cast<RankedTensorType>(affine.getType());
  if (affineType.getRank() != 1 || affineType.getShape()[0] != originalC) {
    return affine;
  }

  SmallVector<int32_t> padding = {0, static_cast<int32_t>(paddedC - originalC)};
  SmallVector<int64_t> paddedShape = {paddedC};
  auto paddedType = RankedTensorType::get(
      paddedShape, affineType.getElementType(),
      mlir::cast<ttnn::TTNNLayoutAttr>(affineType.getEncoding())
          .withTensorShape(paddedShape));

  return rewriter.create<ttnn::PadOp>(
      ttmlir::utils::appendLocationSuffix(loc, suffix), paddedType, affine,
      padding, /*pad_value=*/mlir::APFloat(0.0f),
      /*use_multicore=*/false,
      /*memory_config=*/nullptr);
}

LogicalResult GroupNormChannelPadRewritePattern::matchAndRewrite(
    ttnn::GroupNormOp srcOp, PatternRewriter &rewriter) const {
  constexpr int64_t tileWidth = ttcore::TileType::getDefaultShape()[1];

  RankedTensorType inputType = srcOp.getInput().getType();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  if (inputType.getRank() != 4) {
    return failure();
  }

  int64_t originalC = inputShape[3];

  if (originalC % tileWidth == 0) {
    return failure();
  }

  int64_t paddedC = llvm::divideCeil(originalC, tileWidth) * tileWidth;
  int64_t numGroups = srcOp.getNumGroups();
  int64_t channelsPerGroup = originalC / numGroups;

  // Scale num_groups so channels_per_group is preserved.
  int64_t paddedNumGroups = paddedC / channelsPerGroup;

  Location loc = srcOp.getLoc();

  // Pad input: [N, 1, H*W, C] -> [N, 1, H*W, paddedC].
  SmallVector<int32_t> inputPadding(inputType.getRank() * 2, 0);
  inputPadding.back() = paddedC - originalC;

  SmallVector<int64_t> paddedInputShape(inputShape);
  paddedInputShape[3] = paddedC;

  auto paddedInputType = RankedTensorType::get(
      paddedInputShape, inputType.getElementType(),
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding())
          .withTensorShape(paddedInputShape));

  auto paddedInput = rewriter.create<ttnn::PadOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_pad_input"), paddedInputType,
      srcOp.getInput(), inputPadding, /*pad_value=*/mlir::APFloat(0.0f),
      /*use_multicore=*/false,
      /*memory_config=*/nullptr);

  // Pad weight and bias from [originalC] to [paddedC].
  Value paddedWeight = padAffine1D(srcOp.getWeight(), originalC, paddedC,
                                   "_pad_weight", loc, rewriter);
  Value paddedBias = padAffine1D(srcOp.getBias(), originalC, paddedC,
                                 "_pad_bias", loc, rewriter);

  // Create padded GroupNormOp with scaled num_groups.
  RankedTensorType outputType = srcOp.getResult().getType();
  SmallVector<int64_t> paddedOutputShape(outputType.getShape());
  paddedOutputShape[3] = paddedC;

  auto paddedOutputType = RankedTensorType::get(
      paddedOutputShape, outputType.getElementType(),
      mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding())
          .withTensorShape(paddedOutputShape));

  auto paddedGroupNorm = rewriter.create<ttnn::GroupNormOp>(
      loc, paddedOutputType, paddedInput, srcOp.getInputMask(), paddedWeight,
      paddedBias, rewriter.getI64IntegerAttr(paddedNumGroups),
      srcOp.getEpsilonAttr(), srcOp.getMemoryConfigAttr(),
      srcOp.getCoreGridAttr());

  // Slice output back: [N, 1, H*W, paddedC] -> [N, 1, H*W, originalC].
  SmallVector<int32_t> begins(outputType.getRank(), 0);
  SmallVector<int32_t> ends(outputType.getShape());
  SmallVector<int32_t> steps(outputType.getRank(), 1);

  auto sliceOp = rewriter.create<ttnn::SliceStaticOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_slice_output"), outputType,
      paddedGroupNorm, rewriter.getI32ArrayAttr(begins),
      rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));

  rewriter.replaceOp(srcOp, sliceOp);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
