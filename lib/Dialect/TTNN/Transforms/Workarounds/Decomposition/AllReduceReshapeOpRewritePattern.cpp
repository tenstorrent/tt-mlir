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

  // [EXPERIMENT] Reshape any sub-4D all_reduce input up to 4D (prepend 1s), so
  // the native all_reduce and its internal reduce_scatter operate on a 4D
  // tensor scattering the last dim -- matching main's compile-time
  // reduce_scatter decomposition path. Skip if already >= 4D (avoids re-match).
  if (rank >= 4) {
    return failure();
  }

  RankedTensorType outputType = op.getResult().getType();
  ArrayRef<int64_t> outputShape = outputType.getShape();

  int64_t pad = 4 - rank;

  // Create padded shape by prepending `pad` leading dimensions of size 1.
  SmallVector<int64_t> paddedInputShape(pad, 1);
  paddedInputShape.append(inputType.getShape().begin(),
                          inputType.getShape().end());

  SmallVector<int64_t> paddedOutputShape(pad, 1);
  paddedOutputShape.append(outputShape.begin(), outputShape.end());

  // Reshape input -> 4D.
  auto reshapeInput = ttir_to_ttnn::utils::generateReshape(
      op.getInput(), paddedInputShape, rewriter,
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape_to_4d"));

  // Create 4D output tensor type.
  RankedTensorType paddedOutputType =
      ttnn::utils::RankedTensorTypeFactory::create(outputType,
                                                   paddedOutputShape);

  // Create the all_reduce operation on 4D tensors (cluster_axis unchanged --
  // it is a mesh axis, not a tensor dim).
  auto allReduce4D = rewriter.create<ttnn::AllReduceOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_all_reduce_4d"),
      paddedOutputType, reshapeInput.getResult(), op.getReduceType(),
      op.getClusterAxis(), op.getSubDeviceIdAttr(), op.getNumLinksAttr(),
      op.getTopologyAttr());

  // Reshape back to original shape.
  rewriter.replaceOp(op, ttir_to_ttnn::utils::generateReshape(
                             allReduce4D.getResult(), outputShape, rewriter,
                             ttmlir::utils::appendLocationSuffix(
                                 op.getLoc(), "_reshape_back")));

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
