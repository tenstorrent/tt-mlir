// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/AllGatherOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
TTNNAllGatherWorkarounds::matchAndRewrite(ttnn::AllGatherOp op,
                                          PatternRewriter &rewriter) const {
  RankedTensorType inputType = op.getInput().getType();
  int64_t rank = inputType.getRank();

  // Only apply workaround for 1D tensors. 0D tensors (scalars) have no valid
  // gather dimension and are rejected by AllGatherOp verification.
  if (rank != 1) {
    return failure();
  }

  RankedTensorType outputType = op.getResult().getType();
  SmallVector<int64_t> originalShape(inputType.getShape());
  SmallVector<int64_t> outputShape(outputType.getShape());
  int32_t originalGatherDim = op.getAllGatherDim();

  // Create padded shape by adding a leading dimension of size 1
  SmallVector<int64_t> paddedInputShape = {1};
  paddedInputShape.append(originalShape.begin(), originalShape.end());

  // Create padded output shape
  SmallVector<int64_t> paddedOutputShape = {1};
  paddedOutputShape.append(outputShape.begin(), outputShape.end());

  // Normalize negative gather dimension before adjusting for padding.
  // E.g. for a 1D input, dim=-1 should map to dim=0, then become 1 after
  // padding with a leading dimension.
  int32_t normalizedGatherDim =
      originalGatherDim < 0 ? originalGatherDim + rank : originalGatherDim;
  int32_t adjustedGatherDim = normalizedGatherDim + 1;

  // Create reshape to 2D
  SmallVector<int32_t> paddedShapeI32(paddedInputShape.begin(),
                                      paddedInputShape.end());
  RankedTensorType reshapeInputType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, paddedInputShape);
  auto reshapeInput = rewriter.create<ttnn::ReshapeOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape_to_2d"),
      reshapeInputType, op.getInput(),
      rewriter.getI32ArrayAttr(paddedShapeI32));

  // Create 2D output tensor type
  RankedTensorType paddedOutputType =
      ttnn::utils::RankedTensorTypeFactory::create(outputType,
                                                   paddedOutputShape);

  // Create the all gather operation on 2D tensors with adjusted gather_dim
  auto allGather2D = rewriter.create<ttnn::AllGatherOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_all_gather_2d"),
      paddedOutputType, reshapeInput.getResult(), adjustedGatherDim,
      op.getClusterAxis(), op.getSubDeviceIdAttr(), op.getNumLinksAttr(),
      op.getTopologyAttr());

  // Reshape back to original dimensionality
  SmallVector<int32_t> outputShapeI32(outputShape.begin(), outputShape.end());
  rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
      op, outputType, allGather2D.getResult(),
      rewriter.getI32ArrayAttr(outputShapeI32));

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
