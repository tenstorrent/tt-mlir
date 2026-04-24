// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionDecodeHeadPaddingRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "llvm/Support/MathExtras.h"

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
ScaledDotProductAttentionDecodeHeadPaddingRewritePattern::matchAndRewrite(
    ttnn::ScaledDotProductAttentionDecodeOp srcOp,
    PatternRewriter &rewriter) const {

  // Decode query shape: [1, batch, num_heads, head_dim]
  auto queryType = mlir::dyn_cast<RankedTensorType>(srcOp.getQuery().getType());
  if (!queryType || queryType.getRank() != 4) {
    return failure();
  }

  int64_t numHeads = queryType.getShape()[2];

  if (numHeads % static_cast<int64_t>(TILE_HEIGHT) != 0) {
    return failure();
  }

  int64_t numHeadTiles = numHeads / static_cast<int64_t>(TILE_HEIGHT);
  if (llvm::isPowerOf2_64(static_cast<uint64_t>(numHeadTiles))) {
    return failure();
  }

  // Defer if the attention_sink is not yet in rank-2 form so that
  // ScaledDotProductAttentionDecodeAttentionSinkRewritePattern can normalize it
  // first (otherwise the sink may still be 4D [B, num_heads, 1, 1]).
  Value sink = srcOp.getAttentionSink();
  if (sink) {
    auto sinkType = mlir::dyn_cast<RankedTensorType>(sink.getType());
    if (!sinkType || sinkType.getRank() != 2) {
      return failure();
    }
  }

  int64_t paddedNumHeads =
      static_cast<int64_t>(llvm::NextPowerOf2(numHeadTiles)) *
      static_cast<int64_t>(TILE_HEIGHT);
  Location loc = srcOp.getLoc();

  // Pad query: [1, batch, num_heads, head_dim] -> [1, batch, padded, head_dim]
  Value paddedQuery =
      padDimension(srcOp.getQuery(), paddedNumHeads, 2, rewriter, loc);

  // Pad attention_mask if it has already been broadcast to num_heads (dim 2).
  // If dim 2 is 1, ScaledDotProductAttentionDecodeBroadcastMaskRewritePattern
  // will broadcast it to the new padded_num_heads automatically.
  Value attnMask = srcOp.getAttentionMask();
  if (attnMask) {
    auto maskType = mlir::dyn_cast<RankedTensorType>(attnMask.getType());
    if (maskType && maskType.getRank() == 4 &&
        maskType.getShape()[2] == numHeads) {
      attnMask = padDimension(attnMask, paddedNumHeads, 2, rewriter, loc,
                              /*padValue=*/0.0f);
    }
  }

  // Pad attention_sink first dim: [num_heads, tile_w] -> [padded, tile_w]
  if (sink) {
    auto sinkType = mlir::dyn_cast<RankedTensorType>(sink.getType());
    if (sinkType.getShape()[0] == numHeads) {
      sink = padDimension(sink, paddedNumHeads, 0, rewriter, loc);
    }
  }

  // Compute padded result type.
  auto resultType =
      mlir::dyn_cast<RankedTensorType>(srcOp.getResult().getType());
  SmallVector<int64_t> paddedResultShape(resultType.getShape());
  paddedResultShape[2] = paddedNumHeads;
  auto paddedResultType =
      utils::RankedTensorTypeFactory::create(resultType, paddedResultShape);

  Value paddedOut = rewriter.create<ScaledDotProductAttentionDecodeOp>(
      loc, paddedResultType, paddedQuery, srcOp.getKey(), srcOp.getValue(),
      srcOp.getIsCausal(), attnMask, srcOp.getCurPosTensor(), sink,
      srcOp.getScaleAttr(), srcOp.getMemoryConfigAttr(),
      srcOp.getProgramConfigAttr());

  // Slice result back: [1, batch, padded, head_dim] -> [1, batch, num_heads,
  // head_dim]
  Value result = sliceDimension(paddedOut, numHeads, 2, rewriter, loc);

  rewriter.replaceOp(srcOp, result);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
