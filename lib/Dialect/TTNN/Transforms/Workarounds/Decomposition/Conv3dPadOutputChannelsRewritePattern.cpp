// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv3dPadOutputChannelsRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult Conv3dPadOutputChannelsRewritePattern::matchAndRewrite(
    Conv3dOp srcOp, PatternRewriter &rewriter) const {

  constexpr int64_t TILE_WIDTH = ttcore::TileType::getDefaultShape()[1];

  int64_t outChannels = srcOp.getOutChannels();
  if (outChannels % TILE_WIDTH == 0) {
    return failure();
  }

  int64_t paddedOutChannels =
      llvm::divideCeil(outChannels, TILE_WIDTH) * TILE_WIDTH;
  int64_t paddingAmount = paddedOutChannels - outChannels;

  // Pad weight tensor: (patch_size, O) -> (patch_size, O_padded)
  RankedTensorType weightType = srcOp.getWeight().getType();
  SmallVector<int32_t> weightPadding(weightType.getRank() * 2, 0);
  weightPadding[weightType.getRank() * 2 - 1] = paddingAmount; // last dim high

  SmallVector<int64_t> paddedWeightShape(weightType.getShape());
  paddedWeightShape.back() = paddedOutChannels;

  auto paddedWeightType =
      utils::RankedTensorTypeFactory::create(weightType, paddedWeightShape);

  auto paddedWeight =
      rewriter.create<PadOp>(ttmlir::utils::appendLocationSuffix(
                                 srcOp.getWeight().getLoc(), "_pad_out_ch"),
                             paddedWeightType, srcOp.getWeight(), weightPadding,
                             /*pad_value=*/mlir::APFloat(0.0f),
                             /*use_multicore=*/false,
                             /*memory_config=*/nullptr);

  // Pad bias tensor if present: (1, O) -> (1, O_padded)
  Value paddedBias = srcOp.getBias();
  if (paddedBias) {
    RankedTensorType biasType =
        mlir::cast<RankedTensorType>(paddedBias.getType());
    SmallVector<int32_t> biasPadding(biasType.getRank() * 2, 0);
    biasPadding[biasType.getRank() * 2 - 1] = paddingAmount;

    SmallVector<int64_t> paddedBiasShape(biasType.getShape());
    paddedBiasShape.back() = paddedOutChannels;

    auto paddedBiasType =
        utils::RankedTensorTypeFactory::create(biasType, paddedBiasShape);

    paddedBias =
        rewriter.create<PadOp>(ttmlir::utils::appendLocationSuffix(
                                   srcOp.getBias().getLoc(), "_pad_out_ch"),
                               paddedBiasType, srcOp.getBias(), biasPadding,
                               /*pad_value=*/mlir::APFloat(0.0f),
                               /*use_multicore=*/false,
                               /*memory_config=*/nullptr);
  }

  // Build padded output type: (N, D, H, W, O) -> (N, D, H, W, O_padded)
  RankedTensorType outputType = srcOp.getResult().getType();
  constexpr int64_t CHANNEL_DIM = 4;

  SmallVector<int64_t> paddedOutputShape(outputType.getShape());
  paddedOutputShape[CHANNEL_DIM] = paddedOutChannels;

  auto paddedOutputType =
      utils::RankedTensorTypeFactory::create(outputType, paddedOutputShape);

  // Preserve existing optional attributes (null if absent).
  Conv3dConfigAttr conv3dConfigAttr;
  if (auto config = srcOp.getConv3dConfig()) {
    conv3dConfigAttr = *config;
  }
  DeviceComputeKernelConfigAttr computeConfigAttr;
  if (auto config = srcOp.getComputeConfig()) {
    computeConfigAttr = *config;
  }

  // Create new conv3d with padded output channels.
  auto paddedConvOp = rewriter.create<Conv3dOp>(
      srcOp.getLoc(), paddedOutputType, srcOp.getInput(), paddedWeight,
      paddedBias, srcOp.getDevice(),
      rewriter.getI32IntegerAttr(srcOp.getInChannels()),
      rewriter.getI32IntegerAttr(paddedOutChannels),
      rewriter.getI32IntegerAttr(srcOp.getBatchSize()),
      rewriter.getI32IntegerAttr(srcOp.getInputDepth()),
      rewriter.getI32IntegerAttr(srcOp.getInputHeight()),
      rewriter.getI32IntegerAttr(srcOp.getInputWidth()),
      srcOp.getKernelSizeAttr(), srcOp.getStrideAttr(), srcOp.getPaddingAttr(),
      srcOp.getPaddingModeAttr(), rewriter.getI32IntegerAttr(srcOp.getGroups()),
      srcOp.getDtypeAttr(), conv3dConfigAttr, computeConfigAttr);

  // Slice the result to remove padding:
  // (N, D, H, W, O_padded) -> (N, D, H, W, O)
  int64_t rank = outputType.getRank();
  SmallVector<int32_t> begins(rank, 0);
  SmallVector<int32_t> ends(outputType.getShape());
  SmallVector<int32_t> steps(rank, 1);

  auto sliceOp = rewriter.create<SliceStaticOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_slice_out_ch"),
      outputType, paddedConvOp, rewriter.getI32ArrayAttr(begins),
      rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));

  rewriter.replaceOp(srcOp, sliceOp);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
