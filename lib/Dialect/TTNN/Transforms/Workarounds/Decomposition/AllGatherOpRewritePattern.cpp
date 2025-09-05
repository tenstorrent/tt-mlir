// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/AllGatherOpRewritePattern.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// The all_gather_async op does not support tensors with rank other than 4.
// https://github.com/tenstorrent/tt-metal/issues/25143
// This workaround reshapes the tensor to 4D and performs the all_gather_async
// op and reshapes the tensor back to the original shape

LogicalResult
TTNNAllGatherWorkarounds::matchAndRewrite(ttnn::AllGatherOp op,
                                          PatternRewriter &rewriter) const {
  int64_t rank = op.getInput().getType().getRank();
  if (rank > 4) {
    return rewriteAllGatherHighRank(op, rewriter);
  }
  if (rank < 4) {
    return rewriteAllGatherLowRank(op, rewriter);
  }
  return failure();
}

LogicalResult TTNNAllGatherWorkarounds::rewriteAllGatherLowRank(
    ttnn::AllGatherOp op, PatternRewriter &rewriter) const {
  RankedTensorType inputType = op.getInput().getType();
  int64_t rank = inputType.getRank();

  SmallVector<int64_t> inputShape(inputType.getShape());
  int64_t gatherDim = op.getAllGatherDim();

  // Create padded shape by adding leading dimensions of size 1
  SmallVector<int64_t> paddedInputShape;
  int64_t paddingDims = 4 - rank;
  for (int64_t i = 0; i < paddingDims; ++i) {
    paddedInputShape.push_back(1);
  }
  paddedInputShape.append(inputShape.begin(), inputShape.end());

  // Adjust gather dimension to account for padding
  int64_t adjustedGatherDim = gatherDim + paddingDims;
  return applyReshapedAllGather(op, rewriter, paddedInputShape,
                                adjustedGatherDim);
}

LogicalResult TTNNAllGatherWorkarounds::rewriteAllGatherHighRank(
    ttnn::AllGatherOp op, PatternRewriter &rewriter) const {
  RankedTensorType inputType = op.getInput().getType();
  int64_t rank = inputType.getRank();
  SmallVector<int64_t> inputShape(inputType.getShape());

  int64_t gatherDim = op.getAllGatherDim();
  // Transform high-rank tensor (rank > 4) into 4D tensor:
  // Original shape: [d0, d1, ..., d(gather_dim), ..., d(rank-1)]
  // Target 4D shape: [1, frontFold, gather_dim_size, backFold]
  // Where:
  //   - frontFold = d0 * d1 * ... * d(gather_dim-1)
  //   - gather_dim_size = d(gather_dim) (unchanged)
  //   - backFold = d(gather_dim+1) * ... * d(rank-1)
  //   - adjustedGatherDim = 2 (gather dimension position in 4D tensor)

  // fold dimensions in front of all_gather_dim
  int64_t frontFold = 1;
  for (int64_t i = 0; i < gatherDim; ++i) {
    frontFold *= inputShape[i];
  }

  // fold dimensions behind all_gather_dim
  int64_t backFold = 1;
  for (int64_t i = gatherDim + 1; i < rank; ++i) {
    backFold *= inputShape[i];
  }

  // insert 1 on leftmost to make the shape 4D
  SmallVector<int64_t> preShape = {1, frontFold, inputShape[gatherDim],
                                   backFold};
  int64_t adjustedGatherDim = 2;
  return applyReshapedAllGather(op, rewriter, preShape, adjustedGatherDim);
}

LogicalResult TTNNAllGatherWorkarounds::applyReshapedAllGather(
    ttnn::AllGatherOp op, PatternRewriter &rewriter, ArrayRef<int64_t> preShape,
    int64_t adjustedGatherDim) const {
  RankedTensorType inputType = op.getInput().getType();

  RankedTensorType outputType = op.getResult().getType();
  SmallVector<int64_t> outputShape(outputType.getShape());

  // Calculate the expected output shape after AllGather operation
  // The gather dimension will be multiplied by the mesh size along the cluster
  // axis
  auto deviceDesc = ttcore::lookupDevice(op);
  ::llvm::ArrayRef<int64_t> meshShape = deviceDesc.getMeshShape();
  int64_t clusterAxis = op.getClusterAxis();
  SmallVector<int64_t, 4> postShape(preShape.begin(), preShape.end());
  postShape[adjustedGatherDim] =
      postShape[adjustedGatherDim] * meshShape[clusterAxis];

  // Create reshape to 4D
  SmallVector<int32_t> preShapeI32(preShape.begin(), preShape.end());
  RankedTensorType reshapeInputType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, preShape);
  auto reshapeInput = rewriter.create<ttnn::ReshapeOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape_to_4d"),
      reshapeInputType, op.getInput(), rewriter.getI32ArrayAttr(preShapeI32),
      ttnn::MemoryConfigAttr());

  // Create 4D output tensor type
  RankedTensorType paddedOutputType =
      ttnn::utils::RankedTensorTypeFactory::create(outputType, postShape);

  // Create the AllGather operation on 4D tensors with adjusted
  // all_gather_dim
  auto allGather4D = rewriter.create<ttnn::AllGatherOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_all_gather_4d"),
      paddedOutputType, reshapeInput.getResult(), op.getDevice(),
      adjustedGatherDim, op.getClusterAxis());

  // Reshape back to original dimensionality
  SmallVector<int32_t> outputShapeI32(outputShape.begin(), outputShape.end());
  rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
      op, outputType, allGather4D.getResult(),
      rewriter.getI32ArrayAttr(outputShapeI32), ttnn::MemoryConfigAttr());

  return success();
}
} // namespace mlir::tt::ttnn::workarounds::decomposition
