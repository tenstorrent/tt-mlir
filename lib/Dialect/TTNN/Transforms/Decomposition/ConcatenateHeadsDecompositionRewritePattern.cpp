// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/ConcatenateHeadsDecompositionRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::decomposition {

LogicalResult ConcatenateHeadsDecompositionRewritePattern::matchAndRewrite(
    ttnn::ConcatenateHeadsOp srcOp, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(srcOp.getInput().getType());
  RankedTensorType outputType = srcOp.getResult().getType();

  // When validation config is provided, validate the fused ConcatenateHeads
  // op using IsolatedIRValidationWrapper. If validation succeeds, keep the
  // op as-is.
  if (validationConfig.has_value()) {
    IsolatedIRValidationWrapper validator(rewriter.getContext(),
                                          *validationConfig);

    auto validationResult = validator.validateOp<ttnn::ConcatenateHeadsOp>(
        srcOp.getOperation(), srcOp.getLoc(), {outputType}, srcOp.getInput());

    if (validationResult.isSuccess()) {
      return failure();
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::IsolatedIRValidationWrapper,
                 "ConcatenateHeads decomposition triggered "
                 "(validation failed): {0}",
                 validationResult.errorMessage);
  }

  // Decompose ConcatenateHeads into Permute + Reshape.
  using namespace ttmlir::utils::transformer;
  auto inputShape = inputType.getShape();

  // Step 1: Permute to swap num_heads and sequence_size dimensions.
  // [batch_size, num_heads, sequence_size, head_size]
  //   -> [batch_size, sequence_size, num_heads, head_size]
  llvm::SmallVector<int64_t> permutation = {0, 2, 1, 3};
  DenseI64ArrayAttr permutationAttr =
      rewriter.getDenseI64ArrayAttr(permutation);
  SmallVector<int64_t> permutedShape =
      ttmlir::utils::applyPermutation(inputShape, permutation);
  RankedTensorType permutedType =
      utils::RankedTensorTypeFactory::create(outputType, permutedShape);

  PermuteOp permuteOp = rewriter.create<ttnn::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_concat_heads"),
      permutedType, srcOp.getInput(), permutationAttr, mlir::FloatAttr());

  // Step 2: Reshape to concatenate heads.
  // [batch_size, sequence_size, num_heads, head_size]
  //   -> [batch_size, sequence_size, num_heads * head_size]
  auto reshapedShape = outputType.getShape();
  SmallVector<int32_t> reshapedShapeI32(reshapedShape.begin(),
                                        reshapedShape.end());
  mlir::ArrayAttr reshapedShapeAttr =
      rewriter.getI32ArrayAttr(reshapedShapeI32);

  rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
      srcOp, outputType, permuteOp.getResult(), reshapedShapeAttr);

  return success();
}

} // namespace mlir::tt::ttnn::decomposition
