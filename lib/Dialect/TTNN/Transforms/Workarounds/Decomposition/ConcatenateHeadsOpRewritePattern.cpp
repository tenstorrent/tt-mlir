// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ConcatenateHeadsOpRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Rewrite ConcatenateHeadsOp into PermuteOp + ReshapeOp when the head size
// (input_shape[3]) is not divisible by tile size (32).
LogicalResult ConcatenateHeadsOpRewritePattern::matchAndRewrite(
    ttnn::ConcatenateHeadsOp srcOp, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(srcOp.getInput().getType());
  auto inputShape = inputType.getShape();
  RankedTensorType outputType = srcOp.getResult().getType();

  // input: [batch_size, num_heads, sequence_size, head_size]
  enum InputDimensions {
    INPUT_BATCH = 0,
    INPUT_NUM_HEADS = 1,
    INPUT_SEQ = 2,
    INPUT_HEAD_SIZE = 3
  };

  // Only apply this pattern when head size is not divisible by tile size (32)
  constexpr int64_t TILE_SIZE = ttnn::TILE_WIDTH; // 32
  if (inputShape[INPUT_HEAD_SIZE] % TILE_SIZE == 0) {
    return failure();
  }

  // Step 1: Create permutation to swap num_heads and sequence_size dimensions
  // Permute: [batch_size, num_heads, sequence_size, head_size]
  //       -> [batch_size, sequence_size, num_heads, head_size]
  llvm::SmallVector<int64_t> permutation = {0, 2, 1, 3};
  DenseI64ArrayAttr permutationAttr =
      rewriter.getDenseI64ArrayAttr(permutation);
  SmallVector<int64_t> permutedShape = {
      inputShape[INPUT_BATCH], inputShape[INPUT_SEQ],
      inputShape[INPUT_NUM_HEADS], inputShape[INPUT_HEAD_SIZE]};
  RankedTensorType permutedType =
      RankedTensorType::Builder(outputType).setShape(permutedShape);
  auto input = srcOp.getInput();

  PermuteOp permuteOp = rewriter.create<ttnn::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_concat_heads"),
      permutedType, input, permutationAttr, ttnn::MemoryConfigAttr(),
      mlir::FloatAttr());

  // Step 2: Create reshape to concatenate heads
  // Reshape: [batch_size, sequence_size, num_heads, head_size]
  //       -> [batch_size, sequence_size, num_heads * head_size]
  SmallVector<int64_t> reshapedShapeI64 = {
      inputShape[INPUT_BATCH], inputShape[INPUT_SEQ],
      inputShape[INPUT_NUM_HEADS] * inputShape[INPUT_HEAD_SIZE]};
  SmallVector<int32_t> reshapedShapeI32(reshapedShapeI64.begin(),
                                        reshapedShapeI64.end());
  RankedTensorType reshapedType =
      RankedTensorType::Builder(outputType).setShape(reshapedShapeI64);
  mlir::ArrayAttr reshapedShapeAttr =
      rewriter.getI32ArrayAttr(reshapedShapeI32);

  rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
      srcOp, reshapedType, permuteOp.getResult(),
      reshapedShapeAttr, // Pass the ArrayAttr here
      ttnn::MemoryConfigAttr());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
