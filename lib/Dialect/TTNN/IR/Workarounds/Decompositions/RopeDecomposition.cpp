// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNDecompositionWorkaround.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::wa {

// Workaround implementation for RotaryEmbedding padding
class RotaryEmbeddingPaddingWorkaround : public wa::DecompositionWorkaround {
public:
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ropeOp = cast<RotaryEmbeddingOp>(op);
    RankedTensorType resultType = ropeOp.getType();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    if (resultShape.size() < 2) {
      return failure();
    }

    int64_t originalSeqLen = resultShape[resultShape.size() - 2];
    constexpr int64_t TILE_HEIGHT = 32;
    if (originalSeqLen % TILE_HEIGHT == 0) {
      return failure(); // No workaround needed
    }

    // Create padded shape
    SmallVector<int64_t> paddedResultShape(resultShape);
    paddedResultShape[paddedResultShape.size() - 2] =
        llvm::divideCeil(originalSeqLen, TILE_HEIGHT) * TILE_HEIGHT;

    auto paddedType =
        utils::RankedTensorTypeFactory::create(resultType, paddedResultShape);

    // Create padded RotaryEmbedding op
    auto paddedOp = rewriter.create<RotaryEmbeddingOp>(
        ropeOp.getLoc(), paddedType, ropeOp.getInput(), ropeOp.getCosCache(),
        ropeOp.getSinCache(), ropeOp.getTokenIndexAttr(),
        ropeOp.getMemoryConfigAttr(), ropeOp.getComputeConfigAttr());

    // Slice back to original shape
    SmallVector<int32_t> begins(resultShape.size(), 0);
    SmallVector<int32_t> ends(paddedResultShape.begin(), paddedResultShape.end());
    SmallVector<int32_t> steps(resultShape.size(), 1);
    ends[ends.size() - 2] = originalSeqLen;

    auto sliceOp = rewriter.create<SliceStaticOp>(
        ropeOp.getLoc(), resultType, paddedOp.getResult(),
        rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
        rewriter.getI32ArrayAttr(steps));

    rewriter.replaceOp(ropeOp, sliceOp.getResult());
    return success();
  }
};

// Implementation of the decomposition workaround interface method
wa::DecompositionWorkarounds
getRopeDecompositionWorkarounds() { 
  wa::DecompositionWorkarounds workarounds;
  workarounds.push_back(
      std::make_unique<RotaryEmbeddingPaddingWorkaround>());
  return workarounds;
}
} // namespace mlir::tt::ttnn::wa