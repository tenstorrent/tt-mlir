// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RotaryEmbeddingOpRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

// Workaround which adds padding to RotaryEmbedding seq_len and head_dim
// dimensions to satisfy tile alignment constraints.
// - seq_len (dim -2) must be a multiple of TILE_HEIGHT (32).
//   Metal issue reference: https://github.com/tenstorrent/tt-metal/issues/31567
// - head_dim (dim -1) must be a multiple of TILE_WIDTH * 2 (64).
//   Metal issue reference: https://github.com/tenstorrent/tt-metal/issues/41313
namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

// The rotary_embedding kernel requires the last dimension to be a multiple of
// TILE_WIDTH * 2.
constexpr uint32_t ROPE_HEAD_DIM_ALIGNMENT = TILE_WIDTH * 2;

Value padDimension(Value tensor, int64_t targetLen, int64_t dim,
                   PatternRewriter &rewriter, Location loc,
                   float padValue = 0.0f) {
  auto tensorType = mlir::cast<RankedTensorType>(tensor.getType());
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

} // namespace

std::optional<std::pair<RotaryEmbeddingOp, SliceStaticOp>>
getWorkaroundedOp(RotaryEmbeddingOp ropeOp, PatternRewriter &rewriter) {
  RankedTensorType resultType = ropeOp.getType();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  if (resultShape.size() < 2) {
    return std::nullopt;
  }

  int64_t originalSeqLen = resultShape[resultShape.size() - 2];
  int64_t originalHeadDim = resultShape[resultShape.size() - 1];

  bool seqLenNeedsPadding = (originalSeqLen % TILE_HEIGHT != 0);
  bool headDimNeedsPadding = (originalHeadDim % ROPE_HEAD_DIM_ALIGNMENT != 0);

  if (!seqLenNeedsPadding && !headDimNeedsPadding) {
    return std::nullopt;
  }

  int64_t paddedSeqLen =
      llvm::divideCeil(originalSeqLen, TILE_HEIGHT) * TILE_HEIGHT;
  int64_t paddedHeadDim =
      llvm::divideCeil(originalHeadDim, ROPE_HEAD_DIM_ALIGNMENT) *
      ROPE_HEAD_DIM_ALIGNMENT;

  Location loc = ropeOp.getLoc();
  int64_t rank = resultType.getRank();
  int64_t seqDim = rank - 2;
  int64_t headDim = rank - 1;

  // Pad inputs if head_dim needs padding.
  Value input = ropeOp.getInput();
  Value cos = ropeOp.getCosCache();
  Value sin = ropeOp.getSinCache();

  if (headDimNeedsPadding) {
    input = padDimension(input, paddedHeadDim, headDim, rewriter, loc);
    auto cosType = mlir::cast<RankedTensorType>(cos.getType());
    auto sinType = mlir::cast<RankedTensorType>(sin.getType());
    cos =
        padDimension(cos, paddedHeadDim, cosType.getRank() - 1, rewriter, loc);
    sin =
        padDimension(sin, paddedHeadDim, sinType.getRank() - 1, rewriter, loc);
  }

  // Build padded result shape.
  SmallVector<int64_t> paddedResultShape(resultShape);
  if (seqLenNeedsPadding) {
    paddedResultShape[seqDim] = paddedSeqLen;
  }
  if (headDimNeedsPadding) {
    paddedResultShape[headDim] = paddedHeadDim;
  }

  auto paddedType =
      utils::RankedTensorTypeFactory::create(resultType, paddedResultShape);

  auto paddedOp = rewriter.create<RotaryEmbeddingOp>(
      loc, paddedType, input, cos, sin, ropeOp.getTokenIndexAttr(),
      ropeOp.getMemoryConfigAttr(), ropeOp.getComputeConfigAttr());

  // Slice back to original shape.
  SmallVector<int32_t> begins(rank, 0);
  SmallVector<int32_t> ends(paddedResultShape.begin(), paddedResultShape.end());
  SmallVector<int32_t> steps(rank, 1);
  if (seqLenNeedsPadding) {
    ends[seqDim] = originalSeqLen;
  }
  if (headDimNeedsPadding) {
    ends[headDim] = originalHeadDim;
  }

  auto sliceOp = rewriter.create<ttnn::SliceStaticOp>(
      loc, resultType, paddedOp.getResult(), rewriter.getI32ArrayAttr(begins),
      rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));

  return std::make_pair(paddedOp, sliceOp);
}

LogicalResult RotaryEmbeddingOpRewritePattern::matchAndRewrite(
    RotaryEmbeddingOp srcOp, PatternRewriter &rewriter) const {
  auto workaround = getWorkaroundedOp(srcOp, rewriter);
  if (!workaround) {
    return failure();
  }

  auto [paddedOp, sliceOp] = *workaround;
  rewriter.replaceOp(srcOp, sliceOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
