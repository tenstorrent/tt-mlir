// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/SplitQueryKeyValueAndSplitHeadsOpRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Rewrite SplitQueryKeyValueAndSplitHeadsOp into individual operations when
// the head size is not divisible by tile size (32).
LogicalResult SplitQueryKeyValueAndSplitHeadsOpRewritePattern::matchAndRewrite(
    ttnn::SplitQueryKeyValueAndSplitHeadsOp srcOp,
    PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(srcOp.getInputTensor().getType());
  auto inputShape = inputType.getShape();

  // Only apply this pattern when head size is not divisible by tile size (32).
  constexpr int64_t TILE_SIZE = ttnn::TILE_WIDTH; // 32

  // Get the num_heads from the operation attribute.
  uint32_t numHeads = srcOp.getNumHeads();

  // Calculate head_size from the output shape.
  // The query output has shape [batch_size, num_heads, sequence_size,
  // head_size].
  RankedTensorType queryType = srcOp.getQuery().getType();
  auto queryShape = queryType.getShape();
  int64_t headSize = queryShape[3]; // head_size is the last dimension.

  // Only apply this pattern when head size is not divisible by tile size.
  if (headSize % TILE_SIZE == 0) {
    return failure();
  }

  // Input shape: [batch_size, sequence_size, 3 * hidden_size].
  int64_t batchSize = inputShape[0];
  int64_t sequenceSize = inputShape[1];
  int64_t hiddenSize = inputShape[2] / 3; // Hidden size for one of Q, K, or V.

  Value input = srcOp.getInputTensor();
  Location loc = srcOp.getLoc();

  // Step 1: Split the input tensor into Q, K, V along the last dimension.
  // Create SliceStaticOp for each of Q, K, V.
  SmallVector<int32_t> begins_q = {0, 0, 0};
  SmallVector<int32_t> ends_q = {static_cast<int32_t>(batchSize),
                                 static_cast<int32_t>(sequenceSize),
                                 static_cast<int32_t>(hiddenSize)};
  SmallVector<int32_t> step = {1, 1, 1};

  SmallVector<int64_t> qkvIntermediateShape = {batchSize, sequenceSize,
                                               hiddenSize};
  RankedTensorType qkvIntermediateType =
      utils::RankedTensorTypeFactory::create(queryType, qkvIntermediateShape);

  // Slice for Q
  auto sliceQ = rewriter.create<ttnn::SliceStaticOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_split_q"), qkvIntermediateType,
      input, rewriter.getI32ArrayAttr(begins_q),
      rewriter.getI32ArrayAttr(ends_q), rewriter.getI32ArrayAttr(step));

  // Slice for K
  SmallVector<int32_t> begins_k = {0, 0, static_cast<int32_t>(hiddenSize)};
  SmallVector<int32_t> ends_k = {static_cast<int32_t>(batchSize),
                                 static_cast<int32_t>(sequenceSize),
                                 static_cast<int32_t>(hiddenSize * 2)};
  auto sliceK = rewriter.create<ttnn::SliceStaticOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_split_k"), qkvIntermediateType,
      input, rewriter.getI32ArrayAttr(begins_k),
      rewriter.getI32ArrayAttr(ends_k), rewriter.getI32ArrayAttr(step));

  // Slice for V
  SmallVector<int32_t> begins_v = {0, 0, static_cast<int32_t>(hiddenSize * 2)};
  SmallVector<int32_t> ends_v = {static_cast<int32_t>(batchSize),
                                 static_cast<int32_t>(sequenceSize),
                                 static_cast<int32_t>(hiddenSize * 3)};
  auto sliceV = rewriter.create<ttnn::SliceStaticOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_split_v"), qkvIntermediateType,
      input, rewriter.getI32ArrayAttr(begins_v),
      rewriter.getI32ArrayAttr(ends_v), rewriter.getI32ArrayAttr(step));

  // Step 2: For each of Q, K, V: Reshape then Permute
  // Reshape: [batch, seq, hidden] -> [batch, seq, num_heads, head_size]
  SmallVector<int64_t> reshapedShape = {
      batchSize, sequenceSize, static_cast<int64_t>(numHeads), headSize};
  SmallVector<int32_t> reshapedShapeI32(reshapedShape.begin(),
                                        reshapedShape.end());
  mlir::ArrayAttr reshapedShapeAttr =
      rewriter.getI32ArrayAttr(reshapedShapeI32);

  RankedTensorType reshapedType =
      utils::RankedTensorTypeFactory::create(queryType, reshapedShape);

  auto reshapeQ = rewriter.create<ttnn::ReshapeOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_reshape_q"), reshapedType,
      sliceQ.getResult(), reshapedShapeAttr, ttnn::MemoryConfigAttr());

  auto reshapeK = rewriter.create<ttnn::ReshapeOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_reshape_k"), reshapedType,
      sliceK.getResult(), reshapedShapeAttr, ttnn::MemoryConfigAttr());

  auto reshapeV = rewriter.create<ttnn::ReshapeOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_reshape_v"), reshapedType,
      sliceV.getResult(), reshapedShapeAttr, ttnn::MemoryConfigAttr());

  // Step 3: Permute from [batch, seq, num_heads, head_size] to
  // [batch, num_heads, seq, head_size].
  llvm::SmallVector<int64_t> permutation = {0, 2, 1, 3};
  DenseI64ArrayAttr permutationAttr =
      rewriter.getDenseI64ArrayAttr(permutation);
  SmallVector<int64_t> permutedShape = ttmlir::utils::applyPermutation(
      llvm::ArrayRef<int64_t>(reshapedShape), permutation);
  RankedTensorType queryOutputType =
      utils::RankedTensorTypeFactory::create(queryType, permutedShape);

  auto permuteQ = rewriter.create<ttnn::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_permute_q"), queryOutputType,
      reshapeQ.getResult(), permutationAttr, ttnn::MemoryConfigAttr(),
      mlir::FloatAttr());

  auto permuteK = rewriter.create<ttnn::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_permute_k"), queryOutputType,
      reshapeK.getResult(), permutationAttr, ttnn::MemoryConfigAttr(),
      mlir::FloatAttr());

  auto permuteV = rewriter.create<ttnn::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_permute_v"), queryOutputType,
      reshapeV.getResult(), permutationAttr, ttnn::MemoryConfigAttr(),
      mlir::FloatAttr());

  // Step 4: If transpose_key is true, additionally permute K
  // from [batch, num_heads, seq, head_size] to [batch, num_heads, head_size,
  // seq]
  Value finalK = permuteK.getResult();
  if (srcOp.getTransposeKey()) {
    llvm::SmallVector<int64_t> transposePermutation = {0, 1, 3, 2};
    DenseI64ArrayAttr transposePermutationAttr =
        rewriter.getDenseI64ArrayAttr(transposePermutation);
    SmallVector<int64_t> transposedShape = ttmlir::utils::applyPermutation(
        llvm::ArrayRef<int64_t>(permutedShape), transposePermutation);
    RankedTensorType keyOutputType = utils::RankedTensorTypeFactory::create(
        srcOp.getKey().getType(), transposedShape);

    auto transposeK = rewriter.create<ttnn::PermuteOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_transpose_k"), keyOutputType,
        permuteK.getResult(), transposePermutationAttr,
        ttnn::MemoryConfigAttr(), mlir::FloatAttr());

    finalK = transposeK.getResult();
  }

  // Replace the original operation outputs
  rewriter.replaceOp(srcOp,
                     {permuteQ.getResult(), finalK, permuteV.getResult()});

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
