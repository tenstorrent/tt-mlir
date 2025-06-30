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
int64_t getNumCores(llvm::ArrayRef<int64_t> workerGridShape, int64_t batchSize,
                    int64_t height, int64_t width) {
  int64_t numCores = std::min(batchSize * height * width,
                              workerGridShape[0] * workerGridShape[1]);
  while (numCores > 0) {
    if (batchSize * height % numCores == 0) {
      return numCores;
    }
    --numCores;
  }
  // If no valid numCores found, return 0 to indicate failure.
  return 0;
}

ttnn::CoreRangeSetAttr getShardGrid(MLIRContext *ctx,
                                    llvm::ArrayRef<int64_t> workerGridShape,
                                    int64_t numCores) {
  int64_t workerGridX = workerGridShape[0];

  // If `numCores` is less than `workerGridX`, we can use a single row of cores.
  if (numCores < workerGridX) {
    return ttnn::CoreRangeSetAttr::get(
        ctx, ttnn::CoreRangeAttr::get(
                 ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
                 ttnn::CoreCoordAttr::get(ctx, numCores - 1, 0)));
  }

  // If `numCores` is a multiple of `workerGridX`, we can use a full grid of
  // cores.
  if (numCores % workerGridX == 0) {
    int64_t coreGridHeight = numCores / workerGridX;
    return ttnn::CoreRangeSetAttr::get(
        ctx,
        ttnn::CoreRangeAttr::get(ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
                                 ttnn::CoreCoordAttr::get(ctx, workerGridX - 1,
                                                          coreGridHeight - 1)));
  }

  // Otherwise, we need to split the cores into two ranges, where the second
  // range is partially filled row.
  int64_t coreGridXLarger = workerGridX;
  int64_t coreGridYLarger = numCores / coreGridXLarger;

  int64_t remainingCores = numCores % coreGridXLarger;
  int64_t coreGridXSmaller = remainingCores;
  int64_t coreGridYSmaller = coreGridYLarger + 1;

  return ttnn::CoreRangeSetAttr::get(
      ctx, {ttnn::CoreRangeAttr::get(
                ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
                ttnn::CoreCoordAttr::get(ctx, coreGridXLarger - 1,
                                         coreGridYLarger - 1)),
            ttnn::CoreRangeAttr::get(
                ctx, ttnn::CoreCoordAttr::get(ctx, 0, coreGridYSmaller - 1),
                ttnn::CoreCoordAttr::get(ctx, coreGridXSmaller - 1,
                                         coreGridYSmaller - 1))});
}
} // namespace

LogicalResult UpsampleOpBilinearShardingRewritePattern::matchAndRewrite(
    ttnn::UpsampleOp srcOp, PatternRewriter &rewriter) const {
  if (srcOp.getMode() != "bilinear") {
    return failure();
  }

  RankedTensorType inputType = srcOp.getInput().getType();
  if (auto inputMemoryLayout =
          mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding())
              .getMemLayout();
      inputMemoryLayout &&
      inputMemoryLayout.getValue() == ttnn::TensorMemoryLayout::HeightSharded) {
    return failure();
  }

  enum UpsampleAxisLayout { BATCH = 0, HEIGHT = 1, WIDTH = 2, CHANNEL = 3 };

  auto inputBatch = inputType.getDimSize(BATCH);
  auto inputHeight = inputType.getDimSize(HEIGHT);
  auto inputWidth = inputType.getDimSize(WIDTH);
  auto inputChannel = inputType.getDimSize(CHANNEL);

  tt::GridAttr workerGrid = lookupDevice(srcOp.getOperation()).getWorkerGrid();
  auto numCores =
      getNumCores(workerGrid.getShape(), inputBatch, inputHeight, inputWidth);
  ttnn::CoreRangeSetAttr shardGrid =
      getShardGrid(getContext(), workerGrid.getShape(), numCores);

  auto inputShardHeight = inputBatch * inputHeight * inputWidth / numCores;
  auto inputShardWidth = inputChannel;
  auto inputShardShape =
      ttnn::ShapeAttr::get(getContext(), {inputShardHeight, inputShardWidth});
  auto inputShardSpec = ttnn::ShardSpecAttr::get(
      getContext(), shardGrid, inputShardShape,
      ttnn::ShardOrientationAttr::get(getContext(),
                                      ttnn::ShardOrientation::RowMajor),
      ttnn::ShardModeAttr::get(getContext(), ttnn::ShardMode::Physical),
      /*physical_shard_shape=*/nullptr);

  auto inputShardedLayout = ttnn::TTNNLayoutAttr::get(
      getContext(), inputType.getShape(), inputType.getElementType(),
      ttnn::BufferType::L1, tt::GridAttr::get(getContext(), {numCores, 1}),
      ttnn::TensorMemoryLayoutAttr::get(
          getContext(), ttnn::TensorMemoryLayout::HeightSharded));
  assert(inputShardedLayout.getLayout() == ttnn::Layout::RowMajor);
  auto inputShardedMemoryConfig = ttnn::MemoryConfigAttr::get(
      getContext(),
      ttnn::TensorMemoryLayoutAttr::get(
          getContext(), ttnn::TensorMemoryLayout::HeightSharded),
      ttnn::BufferTypeAttr::get(getContext(), ttnn::BufferType::L1),
      inputShardSpec);
  auto inputToMemoryConfigOp = rewriter.create<ttnn::ToMemoryConfigOp>(
      srcOp.getInput().getLoc(),
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
  auto outputShardShape =
      ttnn::ShapeAttr::get(getContext(), {outputShardHeight, outputShardWidth});
  auto outputShardSpec = ttnn::ShardSpecAttr::get(
      getContext(), shardGrid, outputShardShape,
      ttnn::ShardOrientationAttr::get(getContext(),
                                      ttnn::ShardOrientation::RowMajor),
      ttnn::ShardModeAttr::get(getContext(), ttnn::ShardMode::Physical),
      /*physical_shard_shape=*/nullptr);
  auto outputShardedLayout = ttnn::TTNNLayoutAttr::get(
      getContext(), srcOp.getType().getShape(),
      srcOp.getType().getElementType(), ttnn::BufferType::L1,
      tt::GridAttr::get(getContext(), {numCores, 1}),
      ttnn::TensorMemoryLayoutAttr::get(
          getContext(), ttnn::TensorMemoryLayout::HeightSharded));
  auto outputShardedMemoryConfig = ttnn::MemoryConfigAttr::get(
      getContext(),
      ttnn::TensorMemoryLayoutAttr::get(
          getContext(), ttnn::TensorMemoryLayout::HeightSharded),
      ttnn::BufferTypeAttr::get(getContext(), ttnn::BufferType::L1),
      outputShardSpec);

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
  return failure();
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

  auto inputToLayoutOp = ttnn::utils::createToLayoutOp(
      srcOp.getOperation(), srcOp.getInput(), rewriter, TARGET_LAYOUT,
      inputLayoutAttr.getBufferType(), inputLayoutAttr.getMemLayoutOpt(),
      TARGET_DTYPE, "to_layout");
  auto targetLayoutUpsampleOp = rewriter.create<ttnn::UpsampleOp>(
      srcOp.getLoc(),
      outputType.cloneWithEncoding(outputLayoutAttr.withElementType(
          ttnn::utils::getElementType(getContext(), TARGET_LAYOUT,
                                      TARGET_DTYPE),
          outputType.getShape())),
      inputToLayoutOp, srcOp.getScaleFactorAttr(), srcOp.getModeAttr(),
      /*memory_config=*/nullptr);
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
