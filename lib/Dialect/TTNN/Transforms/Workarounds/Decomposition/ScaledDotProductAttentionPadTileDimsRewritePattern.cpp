// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionPadTileDimsRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

Value padDimension(Value tensor, int64_t targetLen, int64_t dim,
                   PatternRewriter &rewriter, Location loc,
                   float padValue = 0.0f) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType) {
    return tensor;
  }

  ArrayRef<int64_t> shape = tensorType.getShape();
  int64_t rank = tensorType.getRank();
  int64_t padAmount = targetLen - shape[dim];

  SmallVector<int64_t> paddedShape(shape);
  paddedShape[dim] = targetLen;

  SmallVector<int32_t> padding(rank * 2, 0);
  padding[dim * 2 + 1] = padAmount;

  auto paddedType =
      utils::RankedTensorTypeFactory::create(tensorType, paddedShape);

  return rewriter
      .create<PadOp>(loc, paddedType, tensor,
                     rewriter.getDenseI32ArrayAttr(padding),
                     rewriter.getF32FloatAttr(padValue),
                     /*use_multicore=*/rewriter.getBoolAttr(true),
                     /*memory_config=*/nullptr)
      .getResult();
}

Value sliceDimension(Value tensor, int64_t originalLen, int64_t dim,
                     PatternRewriter &rewriter, Location loc) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType) {
    return tensor;
  }

  ArrayRef<int64_t> shape = tensorType.getShape();
  int64_t rank = tensorType.getRank();

  SmallVector<int64_t> slicedShape(shape.begin(), shape.end());
  slicedShape[dim] = originalLen;

  SmallVector<int32_t> begins(rank, 0);
  SmallVector<int32_t> ends(shape.begin(), shape.end());
  ends[dim] = originalLen;
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

LogicalResult
ScaledDotProductAttentionPadTileDimsRewritePattern::matchAndRewrite(
    ttnn::ScaledDotProductAttentionOp srcOp, PatternRewriter &rewriter) const {

  auto queryType = mlir::dyn_cast<RankedTensorType>(srcOp.getQuery().getType());
  auto keyType = mlir::dyn_cast<RankedTensorType>(srcOp.getKey().getType());
  auto valueType = mlir::dyn_cast<RankedTensorType>(srcOp.getValue().getType());
  if (!queryType || !keyType || !valueType) {
    return failure();
  }

  int64_t querySeqLen = queryType.getShape()[queryType.getRank() - 2];
  int64_t keySeqLen = keyType.getShape()[keyType.getRank() - 2];
  int64_t valueSeqLen = valueType.getShape()[valueType.getRank() - 2];

  // Query, key, and value share the same head_dim
  int64_t headDim = queryType.getShape()[queryType.getRank() - 1];

  // Sequence padding is only needed when attention mask is present
  // (tt-metal requires seq_len % chunk_size == 0 when mask is provided)
  bool hasMask = srcOp.getAttentionMask() != nullptr;
  bool querySeqNeedsPadding = hasMask && (querySeqLen % TILE_HEIGHT != 0);
  bool keySeqNeedsPadding = hasMask && (keySeqLen % TILE_HEIGHT != 0);
  bool valueSeqNeedsPadding = hasMask && (valueSeqLen % TILE_HEIGHT != 0);

  // Head dim padding is always needed when not tile-aligned
  // (tt-metal requires logical_shape[3] == padded_shape[3])
  bool headDimNeedsPadding = (headDim % TILE_WIDTH != 0);

  bool anySeqNeedsPadding =
      querySeqNeedsPadding || keySeqNeedsPadding || valueSeqNeedsPadding;

  if (!anySeqNeedsPadding && !headDimNeedsPadding) {
    return failure();
  }

  Value paddedQuery = srcOp.getQuery();
  Value paddedMask = srcOp.getAttentionMask();

  if (querySeqNeedsPadding) {
    int64_t paddedQuerySeqLen =
        llvm::divideCeil(querySeqLen, TILE_HEIGHT) * TILE_HEIGHT;
    paddedQuery =
        padDimension(paddedQuery, paddedQuerySeqLen, queryType.getRank() - 2,
                     rewriter, srcOp.getLoc());
    auto maskType = mlir::dyn_cast<RankedTensorType>(paddedMask.getType());
    paddedMask = padDimension(paddedMask, paddedQuerySeqLen,
                              maskType.getRank() - 2, rewriter, srcOp.getLoc(),
                              -std::numeric_limits<float>::infinity());
  }

  Value paddedKey = srcOp.getKey();
  if (keySeqNeedsPadding) {
    int64_t paddedKeySeqLen =
        llvm::divideCeil(keySeqLen, TILE_HEIGHT) * TILE_HEIGHT;
    paddedKey = padDimension(paddedKey, paddedKeySeqLen, keyType.getRank() - 2,
                             rewriter, srcOp.getLoc());
    auto maskType = mlir::dyn_cast<RankedTensorType>(paddedMask.getType());
    paddedMask = padDimension(paddedMask, paddedKeySeqLen,
                              maskType.getRank() - 1, rewriter, srcOp.getLoc(),
                              -std::numeric_limits<float>::infinity());
  }

  Value paddedValue = srcOp.getValue();
  if (valueSeqNeedsPadding) {
    int64_t paddedValueSeqLen =
        llvm::divideCeil(valueSeqLen, TILE_HEIGHT) * TILE_HEIGHT;
    paddedValue =
        padDimension(paddedValue, paddedValueSeqLen, valueType.getRank() - 2,
                     rewriter, srcOp.getLoc());
  }

  // Pad head_dim for all tensors (they share the same head_dim)
  if (headDimNeedsPadding) {
    int64_t paddedHeadDim = llvm::divideCeil(headDim, TILE_WIDTH) * TILE_WIDTH;
    paddedQuery =
        padDimension(paddedQuery, paddedHeadDim, queryType.getRank() - 1,
                     rewriter, srcOp.getLoc());
    paddedKey = padDimension(paddedKey, paddedHeadDim, keyType.getRank() - 1,
                             rewriter, srcOp.getLoc());
    paddedValue =
        padDimension(paddedValue, paddedHeadDim, valueType.getRank() - 1,
                     rewriter, srcOp.getLoc());
  }

  auto resultType = paddedQuery.getType();
  auto sdpaOp = rewriter.create<ScaledDotProductAttentionOp>(
      srcOp.getLoc(), resultType, paddedQuery, paddedKey, paddedValue,
      paddedMask, srcOp.getIsCausal(), srcOp.getScaleAttr(),
      srcOp.getSlidingWindowSizeAttr(), srcOp.getMemoryConfigAttr());

  // Slice the result back to original dimensions if they were padded
  Value result = sdpaOp.getResult();
  auto resultTensorType = mlir::dyn_cast<RankedTensorType>(result.getType());
  int64_t resultRank = resultTensorType.getRank();

  if (querySeqNeedsPadding) {
    result = sliceDimension(result, querySeqLen, resultRank - 2, rewriter,
                            srcOp.getLoc());
  }

  if (headDimNeedsPadding) {
    result = sliceDimension(result, headDim, resultRank - 1, rewriter,
                            srcOp.getLoc());
  }

  rewriter.replaceOp(srcOp, result);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
