// // SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RotaryEmbeddingOpRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

// Workaround which adds padding to RotaryEmbedding seq_len
// dimension to make it a multiple of tile size.
// Metal issue reference: https://github.com/tenstorrent/tt-metal/issues/31567
namespace mlir::tt::ttnn::workarounds::decomposition {
LogicalResult RotaryEmbeddingOpRewritePattern::matchAndRewrite(
    RotaryEmbeddingOp srcOp, PatternRewriter &rewriter) const {
  RankedTensorType resultType = srcOp.getType();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  if (resultShape.size() < 2) {
    return failure();
  }

  int64_t originalSeqLen = resultShape[resultShape.size() - 2];
  if (originalSeqLen % TILE_HEIGHT == 0) {
    return failure();
  }

  SmallVector<int64_t> paddedResultShape(resultShape);
  paddedResultShape[paddedResultShape.size() - 2] =
      llvm::divideCeil(originalSeqLen, TILE_HEIGHT) * TILE_HEIGHT;

  auto paddedType =
      utils::RankedTensorTypeFactory::create(resultType, paddedResultShape);

  auto rope = rewriter.create<RotaryEmbeddingOp>(
      srcOp.getLoc(), paddedType, srcOp.getInput(), srcOp.getCosCache(),
      srcOp.getSinCache(), srcOp.getTokenIndexAttr(),
      srcOp.getMemoryConfigAttr(), srcOp.getComputeConfigAttr());

  // Slice to original shape
  SmallVector<int32_t> begins(resultShape.size(), 0);
  SmallVector<int32_t> ends(paddedResultShape.begin(), paddedResultShape.end());
  SmallVector<int32_t> steps(resultShape.size(), 1);
  ends[ends.size() - 2] = originalSeqLen;

  auto sliced = rewriter.create<ttnn::SliceStaticOp>(
      srcOp.getLoc(), resultType, rope.getResult(),
      rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
      rewriter.getI32ArrayAttr(steps));

  rewriter.replaceOp(srcOp, sliced);

  return success();
}
} // namespace mlir::tt::ttnn::workarounds::decomposition
