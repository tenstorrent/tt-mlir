// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionPadSequenceDimRewriterPattern.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

// Workaround which adds padding to ScaledDotProductAttention query, key, and
// value sequence dimensions to make them multiples of tile size (TILE_HEIGHT).
// The attention mask is also padded accordingly in both sequence dimensions.
// After the operation, the result is sliced back to the original shape.
// Metal issue reference: https://github.com/tenstorrent/tt-metal/issues/32502
namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

Value padDimension(Value tensor, int64_t targetLen, int64_t dim,
                   PatternRewriter &rewriter, Location loc) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType) {
    return tensor;
  }

  ArrayRef<int64_t> shape = tensorType.getShape();
  int64_t rank = tensorType.getRank();
  int64_t padAmount = targetLen - shape[dim];

  SmallVector<int64_t> paddedShape(shape.begin(), shape.end());
  paddedShape[dim] = targetLen;

  SmallVector<int32_t> padding(rank * 2, 0);
  padding[dim * 2 + 1] = padAmount;

  auto paddedType =
      utils::RankedTensorTypeFactory::create(tensorType, paddedShape);

  return rewriter
      .create<PadOp>(loc, paddedType, tensor,
                     rewriter.getDenseI32ArrayAttr(padding),
                     rewriter.getF32FloatAttr(0.0f), rewriter.getBoolAttr(true),
                     /*memory_config=*/nullptr)
      .getResult();
}

Value sliceSequenceDimension(Value tensor, int64_t originalSeqLen,
                             PatternRewriter &rewriter, Location loc) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType) {
    return tensor;
  }

  ArrayRef<int64_t> shape = tensorType.getShape();
  int64_t rank = tensorType.getRank();
  int64_t seqDim = rank - 2;

  SmallVector<int64_t> slicedShape(shape.begin(), shape.end());
  slicedShape[seqDim] = originalSeqLen;

  SmallVector<int32_t> begins(rank, 0);
  SmallVector<int32_t> ends(shape.begin(), shape.end());
  ends[seqDim] = originalSeqLen;
  SmallVector<int32_t> steps(rank, 1);

  auto slicedType =
      utils::RankedTensorTypeFactory::create(tensorType, slicedShape);

  return rewriter
      .create<SliceStaticOp>(
          loc, slicedType, tensor, rewriter.getI32ArrayAttr(begins),
          rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps))
      .getResult();
}

} // namespace

LogicalResult ScaledDotProductAttentionPadQueryRewritePattern::matchAndRewrite(
    ttnn::ScaledDotProductAttentionOp srcOp, PatternRewriter &rewriter) const {

  // Only apply workaround when attention mask is present
  if (!srcOp.getAttentionMask()) {
    return failure();
  }

  auto queryType = mlir::dyn_cast<RankedTensorType>(srcOp.getQuery().getType());
  auto keyType = mlir::dyn_cast<RankedTensorType>(srcOp.getKey().getType());
  auto valueType = mlir::dyn_cast<RankedTensorType>(srcOp.getValue().getType());
  if (!queryType || !keyType || !valueType) {
    return failure();
  }

  int64_t querySeqLen = queryType.getShape()[queryType.getRank() - 2];
  int64_t keySeqLen = keyType.getShape()[keyType.getRank() - 2];
  int64_t valueSeqLen = valueType.getShape()[valueType.getRank() - 2];

  bool queryNeedsPadding = (querySeqLen % TILE_HEIGHT != 0);
  bool keyNeedsPadding = (keySeqLen % TILE_HEIGHT != 0);
  bool valueNeedsPadding = (valueSeqLen % TILE_HEIGHT != 0);

  if (!queryNeedsPadding && !keyNeedsPadding && !valueNeedsPadding) {
    return failure();
  }

  Value paddedQuery = srcOp.getQuery();
  Value paddedMask = srcOp.getAttentionMask();

  if (queryNeedsPadding) {
    int64_t paddedQuerySeqLen =
        llvm::divideCeil(querySeqLen, TILE_HEIGHT) * TILE_HEIGHT;
    paddedQuery =
        padDimension(srcOp.getQuery(), paddedQuerySeqLen,
                     queryType.getRank() - 2, rewriter, srcOp.getLoc());
    auto maskType = mlir::dyn_cast<RankedTensorType>(paddedMask.getType());
    paddedMask = padDimension(paddedMask, paddedQuerySeqLen,
                              maskType.getRank() - 2, rewriter, srcOp.getLoc());
  }

  Value paddedKey = srcOp.getKey();
  if (keyNeedsPadding) {
    int64_t paddedKeySeqLen =
        llvm::divideCeil(keySeqLen, TILE_HEIGHT) * TILE_HEIGHT;
    paddedKey = padDimension(srcOp.getKey(), paddedKeySeqLen,
                             keyType.getRank() - 2, rewriter, srcOp.getLoc());
    auto maskType = mlir::dyn_cast<RankedTensorType>(paddedMask.getType());
    paddedMask = padDimension(paddedMask, paddedKeySeqLen,
                              maskType.getRank() - 1, rewriter, srcOp.getLoc());
  }

  Value paddedValue = srcOp.getValue();
  if (valueNeedsPadding) {
    int64_t paddedValueSeqLen =
        llvm::divideCeil(valueSeqLen, TILE_HEIGHT) * TILE_HEIGHT;
    paddedValue =
        padDimension(srcOp.getValue(), paddedValueSeqLen,
                     valueType.getRank() - 2, rewriter, srcOp.getLoc());
  }

  auto resultType = paddedQuery.getType();
  auto sdpaOp = rewriter.create<ScaledDotProductAttentionOp>(
      srcOp.getLoc(), resultType, paddedQuery, paddedKey, paddedValue,
      paddedMask, srcOp.getIsCausal(), srcOp.getScaleAttr(),
      srcOp.getSlidingWindowSizeAttr(), srcOp.getMemoryConfigAttr());

  // Slice the result back to original query sequence length if it was padded
  Value result = sdpaOp.getResult();
  if (queryNeedsPadding) {
    result =
        sliceSequenceDimension(result, querySeqLen, rewriter, srcOp.getLoc());
  }

  rewriter.replaceOp(srcOp, result);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
