// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReduceScatterOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
TTNNReduceScatterWorkarounds::matchAndRewrite(ttnn::ReduceScatterOp op,
                                              PatternRewriter &rewriter) const {
  RankedTensorType inputType = op.getInput().getType();
  int64_t rank = inputType.getRank();

  // Only apply workaround for tensors with rank < 4
  if (rank >= 4) {
    return failure();
  }

  RankedTensorType outputType = op.getResult().getType();
  SmallVector<int64_t> originalShape(inputType.getShape());
  SmallVector<int64_t> outputShape(outputType.getShape());
  int32_t originalScatterDim = op.getScatterDim();

  // Create padded shape by adding leading dimensions of size 1
  SmallVector<int64_t> paddedInputShape;
  int64_t paddingDims = 4 - rank;
  for (int64_t i = 0; i < paddingDims; ++i) {
    paddedInputShape.push_back(1);
  }
  paddedInputShape.append(originalShape.begin(), originalShape.end());

  // Create padded output shape
  SmallVector<int64_t> paddedOutputShape;
  for (int64_t i = 0; i < paddingDims; ++i) {
    paddedOutputShape.push_back(1);
  }
  paddedOutputShape.append(outputShape.begin(), outputShape.end());

  // Adjust scatter dimension to account for padding
  int32_t adjustedScatterDim = originalScatterDim + paddingDims;

  // Create reshape to 4D
  SmallVector<int32_t> paddedShapeI32(paddedInputShape.begin(),
                                      paddedInputShape.end());
  RankedTensorType reshapeInputType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, paddedInputShape);
  auto reshapeInput = rewriter.create<ttnn::ReshapeOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape_to_4d"),
      reshapeInputType, op.getInput(), rewriter.getI32ArrayAttr(paddedShapeI32),
      ttnn::MemoryConfigAttr());

  // Create 4D output tensor type
  RankedTensorType paddedOutputType =
      ttnn::utils::RankedTensorTypeFactory::create(outputType,
                                                   paddedOutputShape);

  // Create the reduce scatter operation on 4D tensors with adjusted
  // scatter_dim
  auto reduceScatter4D = rewriter.create<ttnn::ReduceScatterOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reduce_scatter_4d"),
      paddedOutputType, reshapeInput.getResult(), op.getReduceType(),
      adjustedScatterDim, op.getClusterAxis(),
      /*sub_device_id=*/nullptr, /*memory_config=*/nullptr,
      /*num_links=*/nullptr, /*topology=*/nullptr,
      /*compute_config=*/op.getComputeConfigAttr());

  // Reshape back to original dimensionality
  SmallVector<int32_t> outputShapeI32(outputShape.begin(), outputShape.end());
  rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
      op, outputType, reduceScatter4D.getResult(),
      rewriter.getI32ArrayAttr(outputShapeI32), ttnn::MemoryConfigAttr());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
