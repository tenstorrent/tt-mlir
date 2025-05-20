// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/BroadcastingOpRewritePattern.h"
namespace mlir::tt::ttnn::workarounds::decomposition {

llvm::SmallVector<long, 4> squeezeTo4dim(ArrayRef<long> shape) {
  auto mergedDim = 1;
  for (size_t i = 0; i < shape.size() - 3; ++i)
    mergedDim *= shape[i];

  llvm::SmallVector<long, 4> reducedShape = {mergedDim, shape[shape.size() - 3],
                                             shape[shape.size() - 2],
                                             shape[shape.size() - 1]};
  return reducedShape;
}

ttnn::ReshapeOp insert4DimReshape(mlir::Value target,
                                  PatternRewriter &rewriter) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(target.getType());
  auto inputShape = inputType.getShape();
  // Collapse leading dimensions to reduce to 4D.

  auto reducedShape = squeezeTo4dim(inputShape);

  return ttir_to_ttnn::utils::generateReshape(
      mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(target),
      reducedShape, rewriter);
}

LogicalResult
BroadcastingOpRewritePattern::matchAndRewrite(ttnn::RepeatOp srcOp,
                                              PatternRewriter &rewriter) const {
  auto outputType = cast<mlir::RankedTensorType>(srcOp.getResult().getType());
  auto outputShape = outputType.getShape();
  // Skip if rank is 4 or less (no need to reshape).
  if (outputShape.size() <= 4)
    return failure();
  auto inputTypedValue =
      dyn_cast<mlir::TypedValue<mlir::RankedTensorType>>(srcOp->getOperand(0));
  auto inputType = inputTypedValue.getType();
  auto inputShape = inputType.getShape();
  auto reducedShape = squeezeTo4dim(inputShape);

  auto preReshapeOp = ttir_to_ttnn::utils::generateReshape(
      inputTypedValue, reducedShape, rewriter);

  auto oldRepeatDims = srcOp.getRepeatDims().getShape();
  if (oldRepeatDims.size() != inputShape.size())
    return failure();

  // Collapse repeat dims to match reduced shape.
  int64_t mergedRepeatDim = 1;
  for (size_t i = 0; i < oldRepeatDims.size() - 3; ++i)
    mergedRepeatDim *= oldRepeatDims[i];

  llvm::SmallVector<int64_t> newRepeatDims = {
      mergedRepeatDim, oldRepeatDims[oldRepeatDims.size() - 3],
      oldRepeatDims[oldRepeatDims.size() - 2],
      oldRepeatDims[oldRepeatDims.size() - 1]};

  auto newRepeatDimsAttr =
      ttnn::ShapeAttr::get(srcOp->getContext(), newRepeatDims);

  // Update layout with reduced shape.
  auto newLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding())
                           .withTensorShape(reducedShape);

  // Compute output shape by applying repeat dims (preserve original dim if
  // repeat dim is 1).
  llvm::SmallVector<int64_t> computedOutputShape;
  for (size_t i = 0; i < newRepeatDims.size(); ++i) {
    computedOutputShape.push_back(newRepeatDims[i] != 1 ? newRepeatDims[i]
                                                        : reducedShape[i]);
  }

  auto computedOutputLayoutAttr =
      newLayoutAttr.withTensorShape(computedOutputShape);
  auto newOutputType = mlir::RankedTensorType::get(computedOutputShape,
                                                   inputType.getElementType(),
                                                   computedOutputLayoutAttr);

  // Create new RepeatOp with adjusted shape and repeat dims.
  auto newRepeatOp = rewriter.create<ttnn::RepeatOp>(
      srcOp->getLoc(), newOutputType, preReshapeOp.getResult(),
      newRepeatDimsAttr);

  // Insert reshape to restore original output shape.
  auto postReshapeOp = ttir_to_ttnn::utils::generateReshape(
      newRepeatOp,
      mlir::cast<mlir::RankedTensorType>(srcOp->getResult(0).getType())
          .getShape(),
      rewriter);

  rewriter.replaceOp(srcOp, postReshapeOp);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
