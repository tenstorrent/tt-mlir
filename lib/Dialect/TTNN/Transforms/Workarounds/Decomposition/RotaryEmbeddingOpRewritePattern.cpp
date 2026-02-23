// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RotaryEmbeddingOpRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

// Workaround which adds padding to RotaryEmbedding seq_len
// dimension to make it a multiple of tile size.
// Metal issue reference: https://github.com/tenstorrent/tt-metal/issues/31567
namespace mlir::tt::ttnn::workarounds::decomposition {

RotaryEmbeddingOp getWorkaroundedOp(RotaryEmbeddingOp ropeOp,
                                    PatternRewriter &rewriter) {
  RankedTensorType resultType = ropeOp.getType();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  if (resultShape.size() < 2) {
    return ropeOp;
  }

  int64_t originalSeqLen = resultShape[resultShape.size() - 2];
  if (originalSeqLen % TILE_HEIGHT == 0) {
    return ropeOp;
  }

  SmallVector<int64_t> paddedResultShape(resultShape);
  paddedResultShape[paddedResultShape.size() - 2] =
      llvm::divideCeil(originalSeqLen, TILE_HEIGHT) * TILE_HEIGHT;

  auto paddedType =
      utils::RankedTensorTypeFactory::create(resultType, paddedResultShape);

  auto paddedOp = rewriter.create<RotaryEmbeddingOp>(
      ropeOp.getLoc(), paddedType, ropeOp.getInput(), ropeOp.getCosCache(),
      ropeOp.getSinCache(), ropeOp.getTokenIndexAttr(),
      ropeOp.getMemoryConfigAttr(), ropeOp.getComputeConfigAttr());

  // Slice to original shape.
  SmallVector<int32_t> begins(resultShape.size(), 0);
  SmallVector<int32_t> ends(paddedResultShape.begin(), paddedResultShape.end());
  SmallVector<int32_t> steps(resultShape.size(), 1);
  ends[ends.size() - 2] = originalSeqLen;

  rewriter.create<ttnn::SliceStaticOp>(
      ropeOp.getLoc(), resultType, paddedOp.getResult(),
      rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
      rewriter.getI32ArrayAttr(steps));

  return paddedOp;
}

LogicalResult RotaryEmbeddingOpRewritePattern::matchAndRewrite(
    RotaryEmbeddingOp srcOp, PatternRewriter &rewriter) const {
  auto paddedOp = getWorkaroundedOp(srcOp, rewriter);
  if (paddedOp == srcOp) {
    return failure();
  }

  // The utility created paddedOp + SliceStaticOp. Replace srcOp with the
  // slice result.
  rewriter.replaceOp(srcOp, (*paddedOp->getUsers().begin())->getResult(0));
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
