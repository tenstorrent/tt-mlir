// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MultiplyTypecastRemovalRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult MultiplyTypecastRemovalRewritePattern::matchAndRewrite(
    ttnn::MultiplyOp originalMultiplyOp, PatternRewriter &rewriter) const {
  Location loc = originalMultiplyOp.getLoc();
  auto lhs = originalMultiplyOp.getLhs();
  auto rhs = originalMultiplyOp.getRhs();
  auto result = originalMultiplyOp.getResult();

  // Check if both operands are typecast ops
  auto lhsTypecastOp = lhs.getDefiningOp<ttnn::TypecastOp>();
  auto rhsTypecastOp = rhs.getDefiningOp<ttnn::TypecastOp>();

  if (!lhsTypecastOp || !rhsTypecastOp) {
    return failure();
  }

  // Get the input types of the typecasts (should be bf16)
  auto lhsInputType = lhsTypecastOp.getInput().getType();
  auto rhsInputType = rhsTypecastOp.getInput().getType();

  // Get the output types of the typecasts (should be f32)
  auto lhsOutputType =
      mlir::cast<RankedTensorType>(lhsTypecastOp.getResult().getType());
  auto rhsOutputType =
      mlir::cast<RankedTensorType>(rhsTypecastOp.getResult().getType());

  // Get the layout attributes to check data types
  auto lhsInputLayout =
      mlir::cast<ttnn::TTNNLayoutAttr>(lhsInputType.getEncoding());
  auto rhsInputLayout =
      mlir::cast<ttnn::TTNNLayoutAttr>(rhsInputType.getEncoding());
  auto lhsOutputLayout =
      mlir::cast<ttnn::TTNNLayoutAttr>(lhsOutputType.getEncoding());
  auto rhsOutputLayout =
      mlir::cast<ttnn::TTNNLayoutAttr>(rhsOutputType.getEncoding());

  // Check if inputs are bf16 and outputs are f32
  if (lhsInputLayout.getDataType() != ttcore::DataType::BFloat16 ||
      rhsInputLayout.getDataType() != ttcore::DataType::BFloat16 ||
      lhsOutputLayout.getDataType() != ttcore::DataType::Float32 ||
      rhsOutputLayout.getDataType() != ttcore::DataType::Float32) {
    return failure();
  }

  // Check if the multiply op has exactly one user and that user is a typecast
  if (!result.hasOneUse()) {
    return failure();
  }

  auto userOp = *result.getUsers().begin();
  auto outputTypecastOp = dyn_cast<ttnn::TypecastOp>(userOp);

  if (!outputTypecastOp) {
    return failure();
  }

  // Check if the output typecast converts from f32 to bf16
  auto outputTypecastResultType =
      mlir::cast<RankedTensorType>(outputTypecastOp.getResult().getType());
  auto outputTypecastResultLayout =
      mlir::cast<ttnn::TTNNLayoutAttr>(outputTypecastResultType.getEncoding());

  if (outputTypecastResultLayout.getDataType() != ttcore::DataType::BFloat16) {
    return failure();
  }

  // Check if the MultiplyOpDecompositionRewritePattern would also apply
  // If so, skip this optimization to avoid conflict
  auto multiplyResultType = mlir::cast<RankedTensorType>(result.getType());
  auto multiplyResultShape = multiplyResultType.getShape();
  if (multiplyResultShape.size() == 4 && multiplyResultShape[0] >= 1024 &&
      multiplyResultShape[1] >= 1024 && multiplyResultShape[2] == 1 &&
      multiplyResultShape[3] == 1) {
    // The decomposition pattern would apply, so skip this optimization
    return failure();
  }

  // All conditions are met, perform the transformation
  // Create the output type for the multiply op with bf16 data type
  auto multiplyResultLayout =
      mlir::cast<ttnn::TTNNLayoutAttr>(multiplyResultType.getEncoding());

  // Create new layout with bf16 data type
  auto newMultiplyResultLayout =
      multiplyResultLayout.withDataType(ttcore::DataType::BFloat16);

  // Get the scalar element type from the new layout (bf16, not tile<32x32,
  // bf16>)
  Type newElementType = newMultiplyResultLayout.getScalarElementType();

  // Create the new tensor type with bf16 element type and layout
  auto newMultiplyResultType = RankedTensorType::get(
      multiplyResultType.getShape(), newElementType, newMultiplyResultLayout);

  // Create the new multiply op with bf16 inputs and output
  auto dtypeAttr = ttcore::DataTypeAttr::get(rewriter.getContext(),
                                             ttcore::DataType::BFloat16);
  MultiplyOp newMultiplyOp = rewriter.create<ttnn::MultiplyOp>(
      loc, newMultiplyResultType, lhsTypecastOp.getInput(),
      rhsTypecastOp.getInput(), dtypeAttr,
      /*memory_config=*/nullptr);

  // Replace the output typecast with the new multiply result
  rewriter.replaceOp(outputTypecastOp, newMultiplyOp.getResult());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
