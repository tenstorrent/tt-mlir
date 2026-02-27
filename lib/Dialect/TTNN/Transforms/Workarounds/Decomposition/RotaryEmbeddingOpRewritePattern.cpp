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

std::optional<std::pair<RotaryEmbeddingOp, SliceStaticOp>>
getWorkaroundedOp(RotaryEmbeddingOp ropeOp, PatternRewriter &rewriter) {
  RankedTensorType resultType = ropeOp.getType();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  if (resultShape.size() < 2) {
    return std::nullopt;
  }

  int64_t originalSeqLen = resultShape[resultShape.size() - 2];
  if (originalSeqLen % TILE_HEIGHT == 0) {
    return std::nullopt;
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

  auto sliceOp = rewriter.create<ttnn::SliceStaticOp>(
      ropeOp.getLoc(), resultType, paddedOp.getResult(),
      rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
      rewriter.getI32ArrayAttr(steps));

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
