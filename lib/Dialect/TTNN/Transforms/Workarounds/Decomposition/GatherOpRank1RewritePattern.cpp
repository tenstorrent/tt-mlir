// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/GatherOpRank1RewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround to support rank-1 input/index tensors: tt-metal's gather requires
// rank >= 2, so unsqueeze a leading unit dimension on the input and index,
// gather, then reshape the result back to the original rank.
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/45155
LogicalResult
GatherOpRank1RewritePattern::matchAndRewrite(ttnn::GatherOp srcOp,
                                             PatternRewriter &rewriter) const {
  RankedTensorType inputType = srcOp.getInput().getType();
  RankedTensorType indexType = srcOp.getIndex().getType();
  RankedTensorType outputType = srcOp.getResult().getType();

  int64_t inputRank = inputType.getRank();
  if (inputRank != 1) {
    return failure();
  }

  Location loc = srcOp.getLoc();

  auto prependUnitDims = [](RankedTensorType type) {
    SmallVector<int64_t> shape(1, 1);
    shape.append(type.getShape().begin(), type.getShape().end());
    return shape;
  };

  SmallVector<int64_t> inputShape2D = prependUnitDims(inputType);
  SmallVector<int64_t> indexShape2D = prependUnitDims(indexType);
  SmallVector<int64_t> outputShape2D = prependUnitDims(outputType);

  ttnn::ReshapeOp inputReshaped = ttir_to_ttnn::utils::generateReshape(
      srcOp.getInput(), inputShape2D, rewriter,
      ttmlir::utils::appendLocationSuffix(loc, "_input_reshape"));
  ttnn::ReshapeOp indexReshaped = ttir_to_ttnn::utils::generateReshape(
      srcOp.getIndex(), indexShape2D, rewriter,
      ttmlir::utils::appendLocationSuffix(loc, "_index_reshape"));

  // Map `dim` into the unsqueezed (rank-2) layout. A non-negative dim shifts by
  // the number of prepended leading dims; a negative dim already counts from
  // the back and stays valid.
  int32_t dim = srcOp.getDim();
  int32_t adjustedDim = dim >= 0 ? dim + 1 : dim;

  RankedTensorType gatherOutputType =
      ttnn::utils::RankedTensorTypeFactory::create(outputType, outputShape2D);
  auto gather = rewriter.create<ttnn::GatherOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_gather"), gatherOutputType,
      inputReshaped.getResult(), indexReshaped.getResult(),
      rewriter.getI32IntegerAttr(adjustedDim));

  // Reshape the result back to the original rank, reusing the original output
  // type so downstream consumers see the exact same layout.
  SmallVector<int32_t> outputShapeI32 =
      llvm::to_vector_of<int32_t>(outputType.getShape());
  rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
      srcOp, outputType, gather.getResult(),
      rewriter.getI32ArrayAttr(outputShapeI32));

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
