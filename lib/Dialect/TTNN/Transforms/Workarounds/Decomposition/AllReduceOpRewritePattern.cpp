// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/AllReduceOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
TTNNAllReduceWorkarounds::matchAndRewrite(ttnn::AllReduceOp op,
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

  auto allReduce4D = rewriter.create<ttnn::AllReduceOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_all_reduce_4d"),
      paddedOutputType, reshapeInput.getResult(), op.getDevice(),
      op.getReduceType(), op.getClusterAxis());

  // Reshape back to original dimensionality
  SmallVector<int32_t> outputShapeI32(outputShape.begin(), outputShape.end());
  rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
      op, outputType, allReduce4D.getResult(),
      rewriter.getI32ArrayAttr(outputShapeI32), ttnn::MemoryConfigAttr());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
