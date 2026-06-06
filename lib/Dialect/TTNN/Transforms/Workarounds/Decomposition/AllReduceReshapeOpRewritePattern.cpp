// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/AllReduceReshapeOpRewritePattern.h"
#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult TTNNAllReduceReshapeWorkarounds::matchAndRewrite(
    ttnn::AllReduceOp op, PatternRewriter &rewriter) const {
  RankedTensorType inputType = op.getInput().getType();
  int64_t rank = inputType.getRank();

  // Only apply workaround for 1D tensors.
  if (rank != 1) {
    return failure();
  }

  RankedTensorType outputType = op.getResult().getType();
  ArrayRef<int64_t> outputShape = outputType.getShape();

  // Create padded shape by adding a leading dimension of size 1.
  SmallVector<int64_t> paddedInputShape = {1};
  paddedInputShape.append(inputType.getShape().begin(),
                          inputType.getShape().end());

  SmallVector<int64_t> paddedOutputShape = {1};
  paddedOutputShape.append(outputShape.begin(), outputShape.end());

  // Reshape input 1D -> 2D.
  auto reshapeInput = ttir_to_ttnn::utils::generateReshape(
      op.getInput(), paddedInputShape, rewriter,
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape_to_2d"));

  // Create 2D output tensor type.
  RankedTensorType paddedOutputType =
      ttnn::utils::RankedTensorTypeFactory::create(outputType,
                                                   paddedOutputShape);

  // Create the all_reduce operation on 2D tensors.
  auto allReduce2D = rewriter.create<ttnn::AllReduceOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_all_reduce_2d"),
      paddedOutputType, reshapeInput.getResult(), op.getReduceType(),
      op.getClusterAxis(), op.getSubDeviceIdAttr(), op.getNumLinksAttr(),
      op.getTopologyAttr());

  // Reshape back to original 1D shape.
  rewriter.replaceOp(op, ttir_to_ttnn::utils::generateReshape(
                             allReduce2D.getResult(), outputShape, rewriter,
                             ttmlir::utils::appendLocationSuffix(
                                 op.getLoc(), "_reshape_to_1d")));

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
