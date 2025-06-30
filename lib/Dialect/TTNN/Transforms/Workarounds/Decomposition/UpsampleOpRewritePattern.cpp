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

constexpr int64_t TILE_WIDTH = tt::TileType::getDefaultShape()[1];

int64_t getNumCores(llvm::ArrayRef<int64_t> workerGridShape, int64_t batchSize,
                    int64_t height, int64_t width) {
  assert(workerGridShape.size() == 2 && "worker grid shape must be 2D");

  int64_t numCores = std::min(batchSize * height * width,
                              workerGridShape[0] * workerGridShape[1]);
  // For height sharding strategy, we need to find the largest number of cores
  // such that whole image (H, W) is kept in a single shard.
  while (numCores > 0) {
    if (batchSize * height % numCores == 0) {
      return numCores;
    }
    --numCores;
  }

  llvm_unreachable("proof that numCores >= 1 is trivial");
}

ttnn::CoreRangeSetAttr getShardGrid(MLIRContext *ctx,
                                    llvm::ArrayRef<int64_t> workerGridShape,
                                    int64_t numCores) {
  int64_t workerGridX = workerGridShape[0];

  // If `numCores` is less than `workerGridX`, we can use a single row of cores.
  if (numCores < workerGridX) {
    return ttnn::CoreRangeSetAttr::get(
        ctx, /*core_ranges=*/{ttnn::CoreRangeAttr::get(
            ctx,
            /*start_coord=*/ttnn::CoreCoordAttr::get(ctx, /*x=*/0, /*y=*/0),
            /*end_coord=*/
            ttnn::CoreCoordAttr::get(ctx, /*x=*/numCores - 1, /*y=*/0))});
  }

  // If `numCores` is a multiple of `workerGridX`, we can use a full grid of
  // cores.
  if (numCores % workerGridX == 0) {
    int64_t coreGridHeight = numCores / workerGridX;
    return ttnn::CoreRangeSetAttr::get(
        ctx,
        /*core_ranges=*/{ttnn::CoreRangeAttr::get(
            ctx,
            /*start_coord=*/ttnn::CoreCoordAttr::get(ctx, /*x=*/0, /*y=*/0),
            /*end_coord=*/
            ttnn::CoreCoordAttr::get(ctx, /*x=*/workerGridX - 1,
                                     /*y=*/coreGridHeight - 1))});
  }

  // Otherwise, we need to split the cores into two ranges, where the first
  // range is a grid of maximal height, and the second range is partially filled
  // row.
  int64_t coreGridXLarger = workerGridX;
  int64_t coreGridYLarger = numCores / coreGridXLarger;

  int64_t remainingCores = numCores % coreGridXLarger;
  int64_t coreGridXSmaller = remainingCores;
  int64_t coreGridYSmaller = coreGridYLarger + 1;

  return ttnn::CoreRangeSetAttr::get(
      ctx, /*core_ranges=*/{
          ttnn::CoreRangeAttr::get(
              ctx,
              /*start_coord=*/ttnn::CoreCoordAttr::get(ctx, /*x=*/0, /*y=*/0),
              /*end_coord=*/
              ttnn::CoreCoordAttr::get(ctx, /*x=*/coreGridXLarger - 1,
                                       /*y=*/coreGridYLarger - 1)),
          ttnn::CoreRangeAttr::get(
              ctx, /*start_coord=*/
              ttnn::CoreCoordAttr::get(ctx, /*x=*/0,
                                       /*y=*/coreGridYSmaller - 1),
              /*end_coord=*/
              ttnn::CoreCoordAttr::get(ctx, /*x=*/coreGridXSmaller - 1,
                                       /*y=*/coreGridYSmaller - 1))});
}
} // namespace

LogicalResult UpsampleOpBilinearShardingRewritePattern::matchAndRewrite(
    ttnn::UpsampleOp srcOp, PatternRewriter &rewriter) const {
  if (srcOp.getMode() != "bilinear") {
    return failure();
  }

  RankedTensorType inputType = srcOp.getInput().getType();

  // UpsampleOp in `bilinear` mode expects the input to be in `HeightSharded`
  // memory layout, hence if it's already in that layout, we don't need to do
  // anything.
  if (auto inputMemoryLayout =
          mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding())
              .getMemLayout();
      inputMemoryLayout &&
      inputMemoryLayout.getValue() == ttnn::TensorMemoryLayout::HeightSharded) {
    return failure();
  }

  auto inputBatch = inputType.getDimSize(BATCH);
  auto inputHeight = inputType.getDimSize(HEIGHT);
  auto inputWidth = inputType.getDimSize(WIDTH);
  auto inputChannel = inputType.getDimSize(CHANNEL);

  assert(inputChannel % TILE_WIDTH == 0 &&
         "input channel size must be a multiple of tile width");

  // We need to find strategy for sharding the input tensor.
  tt::GridAttr workerGrid = lookupDevice(srcOp.getOperation()).getWorkerGrid();
  auto numCores =
      getNumCores(workerGrid.getShape(), inputBatch, inputHeight, inputWidth);
  ttnn::CoreRangeSetAttr shardGrid =
      getShardGrid(getContext(), workerGrid.getShape(), numCores);

  // We are using `HeightSharded` memory layout for the input tensor, hence the
  // shards are split by first three dimensions (N, H, W).
  auto inputShardHeight = inputBatch * inputHeight * inputWidth / numCores;
  auto inputShardWidth = inputChannel;
  auto inputShardShape = ttnn::ShapeAttr::get(
      getContext(), /*shape=*/{inputShardHeight, inputShardWidth});
  auto inputShardSpec = ttnn::ShardSpecAttr::get(
      getContext(), shardGrid, inputShardShape,
      ttnn::ShardOrientationAttr::get(getContext(),
                                      ttnn::ShardOrientation::RowMajor),
      ttnn::ShardModeAttr::get(getContext(), ttnn::ShardMode::Physical),
      /*physical_shard_shape=*/nullptr);

  auto inputShardedLayout = ttnn::TTNNLayoutAttr::get(
      getContext(), inputType.getShape(), inputType.getElementType(),
      ttnn::BufferType::L1,
      tt::GridAttr::get(getContext(), /*shape=*/{numCores, 1}),
      ttnn::TensorMemoryLayoutAttr::get(
          getContext(), ttnn::TensorMemoryLayout::HeightSharded));
  assert(inputShardedLayout.getLayout() == ttnn::Layout::RowMajor &&
         "expected RowMajor layout");
  auto inputShardedMemoryConfig = ttnn::MemoryConfigAttr::get(
      getContext(),
      ttnn::TensorMemoryLayoutAttr::get(
          getContext(), ttnn::TensorMemoryLayout::HeightSharded),
      ttnn::BufferTypeAttr::get(getContext(), ttnn::BufferType::L1),
      inputShardSpec);

  // Convert the input tensor to the target memory configuration.
  auto inputToMemoryConfigOp = rewriter.create<ttnn::ToMemoryConfigOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getInput().getLoc(),
                                          "to_memory_config"),
      inputType.cloneWithEncoding(inputShardedLayout), srcOp.getInput(),
      inputShardedMemoryConfig);

  auto scaleFactor =
      ttmlir::utils::getPairOfInteger<int32_t>(srcOp.getScaleFactor());
  if (auto error = scaleFactor.takeError()) {
    llvm_unreachable(llvm::toString(std::move(error)).c_str());
  }
  const auto scaleH = scaleFactor->first;
  const auto scaleW = scaleFactor->second;

  auto outputShardHeight = inputShardHeight * scaleH * scaleW;
  auto outputShardWidth = inputShardWidth;
  auto outputShardShape = ttnn::ShapeAttr::get(
      getContext(), /*shape=*/{outputShardHeight, outputShardWidth});
  auto outputShardSpec = ttnn::ShardSpecAttr::get(
      getContext(), shardGrid, outputShardShape,
      ttnn::ShardOrientationAttr::get(getContext(),
                                      ttnn::ShardOrientation::RowMajor),
      ttnn::ShardModeAttr::get(getContext(), ttnn::ShardMode::Physical),
      /*physical_shard_shape=*/nullptr);
  auto outputShardedLayout = ttnn::TTNNLayoutAttr::get(
      getContext(), srcOp.getType().getShape(),
      srcOp.getType().getElementType(), ttnn::BufferType::L1,
      tt::GridAttr::get(getContext(), /*shape=*/{numCores, 1}),
      ttnn::TensorMemoryLayoutAttr::get(
          getContext(), ttnn::TensorMemoryLayout::HeightSharded));
  auto outputShardedMemoryConfig = ttnn::MemoryConfigAttr::get(
      getContext(),
      ttnn::TensorMemoryLayoutAttr::get(
          getContext(), ttnn::TensorMemoryLayout::HeightSharded),
      ttnn::BufferTypeAttr::get(getContext(), ttnn::BufferType::L1),
      outputShardSpec);

  // UpsampleOp is done on height sharded input, and produces height shraded
  // output.
  auto shardedUpsampleOp = rewriter.create<ttnn::UpsampleOp>(
      srcOp.getLoc(), srcOp.getType().cloneWithEncoding(outputShardedLayout),
      inputToMemoryConfigOp, srcOp.getScaleFactorAttr(), srcOp.getModeAttr(),
      outputShardedMemoryConfig);

  auto outputLayout =
      mlir::cast<ttnn::TTNNLayoutAttr>(srcOp.getType().getEncoding());
  auto outputMemoryConfig = ttnn::MemoryConfigAttr::get(
      getContext(), outputLayout.getMemLayout(),
      ttnn::BufferTypeAttr::get(getContext(), outputLayout.getBufferType()),
      ttnn::utils::createShardSpecIfNeeded(outputLayout, workerGrid));

  // Since the output of the new UpsampleOp is height sharded, we need to
  // convert it back to the original memory configuration.
  auto outputToMemoryConfigOp = rewriter.create<ttnn::ToMemoryConfigOp>(
      ttmlir::utils::appendLocationSuffix(shardedUpsampleOp.getLoc(),
                                          "to_memory_config"),
      shardedUpsampleOp.getType().cloneWithEncoding(outputLayout),
      shardedUpsampleOp, outputMemoryConfig);
  rewriter.replaceOp(srcOp, outputToMemoryConfigOp);

  return success();
}

LogicalResult UpsampleOpBilinearPaddingRewritePattern::matchAndRewrite(
    ttnn::UpsampleOp srcOp, PatternRewriter &rewriter) const {
  if (srcOp.getMode() != "bilinear") {
    return failure();
  }

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

LogicalResult UpsampleOpLayoutRewritePattern::matchAndRewrite(
    ttnn::UpsampleOp srcOp, PatternRewriter &rewriter) const {
  auto inputType = srcOp.getInput().getType();
  auto outputType = srcOp.getType();

  constexpr ttnn::Layout TARGET_LAYOUT = ttnn::Layout::RowMajor;
  constexpr tt::DataType TARGET_DTYPE = tt::DataType::BFloat16;

  auto inputLayoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
  auto outputLayoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding());

  // UpsampleOp expects the input and the output in `RowMajor` layout, and with
  // `BFloat16` data type. Hence, if the condition is already met, we don't need
  // to do anything.
  if (!inputLayoutAttr.isTiled() && !outputLayoutAttr.isTiled() &&
      inputLayoutAttr.getElementType().isBF16() &&
      outputLayoutAttr.getElementType().isBF16()) {
    return failure();
  }

  ttnn::MemoryConfigAttr outputMemoryConfig = srcOp.getMemoryConfigAttr();
  if (!outputMemoryConfig) {
    tt::DeviceAttr deviceAttr = tt::lookupDevice(srcOp);

    outputMemoryConfig = ttnn::MemoryConfigAttr::get(
        rewriter.getContext(), outputLayoutAttr.getMemLayout(),
        ttnn::BufferTypeAttr::get(rewriter.getContext(),
                                  outputLayoutAttr.getBufferType()),
        utils::createShardSpecIfNeeded(outputLayoutAttr,
                                       deviceAttr.getWorkerGrid()));
  }

  // Convert the input to `RowMajor` layout with `BFloat16` data type.
  auto inputToLayoutOp = ttnn::utils::createToLayoutOp(
      srcOp.getOperation(), srcOp.getInput(), rewriter, TARGET_LAYOUT,
      inputLayoutAttr.getBufferType(), inputLayoutAttr.getMemLayoutOpt(),
      TARGET_DTYPE, "to_layout");

  // UpsampleOp is replace with the new one, that takes the input in the target
  // configuration, and produces the output in the target configuration.
  auto targetLayoutUpsampleOp = rewriter.create<ttnn::UpsampleOp>(
      srcOp.getLoc(),
      outputType.cloneWithEncoding(outputLayoutAttr.withElementType(
          ttnn::utils::getElementType(getContext(), TARGET_LAYOUT,
                                      TARGET_DTYPE),
          outputType.getShape())),
      inputToLayoutOp, srcOp.getScaleFactorAttr(), srcOp.getModeAttr(),
      /*memory_config=*/nullptr);

  // Convert the output back to the original memory configuration.
  auto outputToLayoutOp = rewriter.create<ttnn::ToLayoutOp>(
      ttmlir::utils::appendLocationSuffix(targetLayoutUpsampleOp.getLoc(),
                                          "to_layout"),
      outputType, targetLayoutUpsampleOp, outputLayoutAttr.getLayout(),
      tt::DataTypeAttr::get(getContext(), outputLayoutAttr.getDataType()),
      outputMemoryConfig, utils::getOrInsertDevice(rewriter, srcOp));
  rewriter.replaceOp(srcOp, outputToLayoutOp);

  return success();
}
} // namespace mlir::tt::ttnn::workarounds::decomposition
