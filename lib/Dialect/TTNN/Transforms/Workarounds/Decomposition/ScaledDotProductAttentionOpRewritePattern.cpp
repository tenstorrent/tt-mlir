// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

// Pad the sequence dimension (dim -2) to targetSeqLen
Value padSequenceDimension(Value tensor, int64_t targetSeqLen,
                           mlir::PatternRewriter &rewriter, Location loc) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(tensor.getType());
  if (!tensorType) {
    return tensor;
  }

  llvm::ArrayRef<int64_t> shape = tensorType.getShape();
  int64_t rank = tensorType.getRank();
  int64_t seqDim = rank - 2;
  int64_t padAmount = targetSeqLen - shape[seqDim];

  SmallVector<int64_t> paddedShape(shape.begin(), shape.end());
  paddedShape[seqDim] = targetSeqLen;

  SmallVector<int32_t> padding(rank * 2, 0);
  padding[seqDim * 2 + 1] = padAmount;

  return rewriter
      .create<PadOp>(loc,
                     RankedTensorType::get(paddedShape,
                                           tensorType.getElementType(),
                                           tensorType.getEncoding()),
                     tensor, rewriter.getDenseI32ArrayAttr(padding),
                     rewriter.getF32FloatAttr(0.0f), rewriter.getBoolAttr(true),
                     /*memory_config=*/nullptr)
      .getResult();
}

// Slice the sequence dimension (dim -2) back to the original sequence length.
Value sliceSequenceDimension(Value tensor, int64_t originalSeqLen,
                             mlir::PatternRewriter &rewriter, Location loc) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(tensor.getType());
  if (!tensorType) {
    return tensor;
  }

  llvm::ArrayRef<int64_t> shape = tensorType.getShape();
  int64_t rank = tensorType.getRank();
  int64_t seqDim = rank - 2;

  llvm::SmallVector<int64_t> slicedShape(shape.begin(), shape.end());
  slicedShape[seqDim] = originalSeqLen;

  llvm::SmallVector<int32_t> begins(rank, 0);
  llvm::SmallVector<int32_t> ends(shape.begin(), shape.end());
  ends[seqDim] = originalSeqLen;
  llvm::SmallVector<int32_t> step(rank, 1);

  return rewriter
      .create<SliceStaticOp>(
          loc,
          RankedTensorType::get(slicedShape, tensorType.getElementType(),
                                tensorType.getEncoding()),
          tensor, rewriter.getI32ArrayAttr(begins),
          rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(step))
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
  if (!queryType) {
    return failure();
  }

  int64_t querySeqLen = queryType.getShape()[queryType.getRank() - 2];
  if (querySeqLen % TILE_HEIGHT == 0) {
    return failure();
  }

  // Pad to the next multiple of TILE_HEIGHT
  int64_t paddedQuerySeqLen =
      ((querySeqLen + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;

  Value paddedQuery = padSequenceDimension(srcOp.getQuery(), paddedQuerySeqLen,
                                           rewriter, srcOp.getLoc());
  Value paddedMask = padSequenceDimension(
      srcOp.getAttentionMask(), paddedQuerySeqLen, rewriter, srcOp.getLoc());

  auto resultType = paddedQuery.getType();
  auto sdpaOp = rewriter.create<ScaledDotProductAttentionOp>(
      srcOp.getLoc(), resultType, paddedQuery, srcOp.getKey(), srcOp.getValue(),
      paddedMask, srcOp.getIsCausal(), srcOp.getScaleAttr(),
      srcOp.getSlidingWindowSizeAttr(), srcOp.getMemoryConfigAttr());

  // Slice the result back to original sequence length
  Value result = sliceSequenceDimension(sdpaOp.getResult(), querySeqLen,
                                        rewriter, srcOp.getLoc());

  rewriter.replaceOp(srcOp, result);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
