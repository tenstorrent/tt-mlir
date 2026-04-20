// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/LayerNormPostAllGatherDecompositionRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
LayerNormPostAllGatherDecompositionRewritePattern::matchAndRewrite(
    ttnn::LayerNormPostAllGatherOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());
  RankedTensorType resultType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t W = inputShape.back();

  // Only fire when the last dimension is not tile-aligned.
  // When W % ttnn::TILE_WIDTH == 0, padded_shape[-1] == logical_shape[-1] and
  // the tt-metal kernel computes the correct winv — no workaround needed.
  if (W % ttnn::TILE_WIDTH == 0) {
    return failure();
  }

  RankedTensorType statsType =
      mlir::cast<RankedTensorType>(op.getStats().getType());
  int64_t statsW = statsType.getShape().back();
  if (statsW == 0 || statsW % ttnn::LAYER_NORM_STATS_WIDTH != 0) {
    return op->emitOpError("stats tensor last dimension must be a positive "
                           "multiple of LAYER_NORM_STATS_WIDTH (")
           << ttnn::LAYER_NORM_STATS_WIDTH << "), got " << statsW;
  }
  int64_t numDevices = statsW / ttnn::LAYER_NORM_STATS_WIDTH;

  auto inputEncoding =
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());

  Location loc = op.getLoc();

  // Build the shape for single-column intermediate tensors [..., 1].
  SmallVector<int64_t> scalarRowShape(inputShape.begin(), inputShape.end());
  scalarRowShape.back() = 1;
  RankedTensorType scalarRowType =
      RankedTensorType::get(scalarRowShape, inputType.getElementType(),
                            inputEncoding.withTensorShape(scalarRowShape));

  // Reshape stats from [..., numDevices * 64] → [..., numDevices, 64]
  // to expose the device dimension, then slice + sum-reduce over it.
  // This is O(1) ops regardless of numDevices (vs O(N) with per-device
  // slice+add).
  SmallVector<int64_t> reshapedStatsShape(statsType.getShape().begin(),
                                          statsType.getShape().end());
  reshapedStatsShape.back() = numDevices;
  reshapedStatsShape.push_back(ttnn::LAYER_NORM_STATS_WIDTH);
  int64_t reshapedRank = static_cast<int64_t>(reshapedStatsShape.size());

  auto statsEncoding =
      mlir::cast<ttnn::TTNNLayoutAttr>(statsType.getEncoding());
  RankedTensorType reshapedStatsType =
      RankedTensorType::get(reshapedStatsShape, statsType.getElementType(),
                            statsEncoding.withTensorShape(reshapedStatsShape));

  SmallVector<int32_t> reshapedStatsShape32(reshapedStatsShape.begin(),
                                            reshapedStatsShape.end());
  auto reshapedStats = rewriter.create<ttnn::ReshapeOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_reshape_stats"),
      reshapedStatsType, op.getStats(),
      rewriter.getI32ArrayAttr(reshapedStatsShape32),
      /*memory_config=*/nullptr);

  // Slice column 0 (sum_x2) and column ttnn::LAYER_NORM_SUM_X_OFFSET (sum_x)
  // from the reshaped stats: [..., numDevices, 64] → [..., numDevices, 1].
  SmallVector<int64_t> slicedColShape(reshapedStatsShape);
  slicedColShape.back() = 1;
  RankedTensorType slicedColType =
      RankedTensorType::get(slicedColShape, statsType.getElementType(),
                            statsEncoding.withTensorShape(slicedColShape));

  SmallVector<int32_t> sliceBegins(reshapedRank, 0);
  SmallVector<int32_t> sliceEnds;
  for (int64_t dim : slicedColShape) {
    sliceEnds.push_back(static_cast<int32_t>(dim));
  }
  SmallVector<int32_t> sliceStep(reshapedRank, 1);

  // Slice sum_x2 (column 0).
  auto sliceSumX2 = rewriter.create<ttnn::SliceStaticOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_slice_sum_x2"), slicedColType,
      reshapedStats.getResult(), rewriter.getI32ArrayAttr(sliceBegins),
      rewriter.getI32ArrayAttr(sliceEnds), rewriter.getI32ArrayAttr(sliceStep));

  // Slice sum_x (column ttnn::LAYER_NORM_SUM_X_OFFSET).
  SmallVector<int32_t> sliceBeginsX(sliceBegins);
  SmallVector<int32_t> sliceEndsX(sliceEnds);
  sliceBeginsX.back() = static_cast<int32_t>(ttnn::LAYER_NORM_SUM_X_OFFSET);
  sliceEndsX.back() = static_cast<int32_t>(ttnn::LAYER_NORM_SUM_X_OFFSET + 1);
  auto sliceSumX = rewriter.create<ttnn::SliceStaticOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_slice_sum_x"), slicedColType,
      reshapedStats.getResult(), rewriter.getI32ArrayAttr(sliceBeginsX),
      rewriter.getI32ArrayAttr(sliceEndsX),
      rewriter.getI32ArrayAttr(sliceStep));

  // Sum-reduce over the device dimension (dim = reshapedRank - 2):
  // [..., numDevices, 1] → [..., 1, 1].
  int32_t deviceDim = static_cast<int32_t>(reshapedRank - 2);
  SmallVector<int64_t> summedShape(slicedColShape);
  summedShape[summedShape.size() - 2] = 1;
  RankedTensorType summedType =
      RankedTensorType::get(summedShape, statsType.getElementType(),
                            statsEncoding.withTensorShape(summedShape));

  auto totalSumX2Nd = rewriter.create<ttnn::SumOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_sum_x2"), summedType,
      sliceSumX2.getResult(), /*keep_dim=*/rewriter.getBoolAttr(true),
      rewriter.getI32ArrayAttr({deviceDim}),
      /*compute_config=*/nullptr);
  auto totalSumXNd = rewriter.create<ttnn::SumOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_sum_x"), summedType,
      sliceSumX.getResult(), /*keep_dim=*/rewriter.getBoolAttr(true),
      rewriter.getI32ArrayAttr({deviceDim}),
      /*compute_config=*/nullptr);

  // Reshape [..., 1, 1] back to [..., 1] for downstream broadcast.
  SmallVector<int32_t> scalarRowShape32(scalarRowShape.begin(),
                                        scalarRowShape.end());
  mlir::Value totalSumX2 =
      rewriter
          .create<ttnn::ReshapeOp>(
              ttmlir::utils::appendLocationSuffix(loc, "_reshape_sum_x2"),
              scalarRowType, totalSumX2Nd.getResult(),
              rewriter.getI32ArrayAttr(scalarRowShape32),
              /*memory_config=*/nullptr)
          .getResult();
  mlir::Value totalSumX =
      rewriter
          .create<ttnn::ReshapeOp>(
              ttmlir::utils::appendLocationSuffix(loc, "_reshape_sum_x"),
              scalarRowType, totalSumXNd.getResult(),
              rewriter.getI32ArrayAttr(scalarRowShape32),
              /*memory_config=*/nullptr)
          .getResult();

  // winv = 1 / (W * numDevices) — the correct normalization denominator.
  // We create a FullOp constant tensor of shape [..., 1] filled with winv.
  mlir::Value device = ttnn::utils::getOrInsertDevice(rewriter, op);
  float winvVal =
      1.0f / (static_cast<float>(W) * static_cast<float>(numDevices));
  auto winvTensor = rewriter.create<ttnn::FullOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_winv"), scalarRowType,
      rewriter.getF32FloatAttr(winvVal), device);

  // E[x²] = total_sum_x2 * winv
  auto ex2 = rewriter.create<ttnn::MultiplyOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_ex2"), scalarRowType,
      totalSumX2, winvTensor.getResult());

  // E[x] = total_sum_x * winv
  auto ex = rewriter.create<ttnn::MultiplyOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_ex"), scalarRowType, totalSumX,
      winvTensor.getResult());

  // var = E[x²] - E[x] * E[x]
  auto exSquared = rewriter.create<ttnn::MultiplyOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_ex_sq"), scalarRowType,
      ex.getResult(), ex.getResult());
  auto var = rewriter.create<ttnn::SubtractOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_var"), scalarRowType,
      ex2.getResult(), exSquared.getResult());

  // rstd = rsqrt(var + epsilon)
  auto epsTensor = rewriter.create<ttnn::FullOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_eps"), scalarRowType,
      rewriter.getF32FloatAttr(op.getEpsilon().convertToFloat()), device);
  auto varEps = rewriter.create<ttnn::AddOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_var_eps"), scalarRowType,
      var.getResult(), epsTensor.getResult());
  auto rstd = rewriter.create<ttnn::RsqrtOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_rstd"), scalarRowType,
      varEps.getResult());

  // result = (input - E[x]) * rstd
  // The [..., 1] E[x] and rstd broadcast over the [..., W] input automatically.
  auto centered = rewriter.create<ttnn::SubtractOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_centered"), resultType,
      op.getInput(), ex.getResult());
  mlir::Value result =
      rewriter
          .create<ttnn::MultiplyOp>(
              ttmlir::utils::appendLocationSuffix(loc, "_normalize"),
              resultType, centered.getResult(), rstd.getResult())
          .getResult();

  // Optional weight (gamma): result = result * weight
  if (op.getWeight()) {
    result = rewriter
                 .create<ttnn::MultiplyOp>(
                     ttmlir::utils::appendLocationSuffix(loc, "_weight"),
                     resultType, result, op.getWeight())
                 .getResult();
  }

  // Optional bias (beta): result = result + bias
  if (op.getBias()) {
    result = rewriter
                 .create<ttnn::AddOp>(
                     ttmlir::utils::appendLocationSuffix(loc, "_bias"),
                     resultType, result, op.getBias())
                 .getResult();
  }

  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
