// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScatterOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
TTNNScatterWorkarounds::matchAndRewrite(ttnn::ScatterOp op,
                                        PatternRewriter &rewriter) const {
  RankedTensorType indexType = op.getIndexTensor().getType();
  int32_t scatterDim = op.getDim();

  // Check if scatter axis size exceeds hardware limit
  constexpr int64_t MAX_SCATTER_SIZE = 256;
  int64_t scatterAxisSize = indexType.getShape()[scatterDim];

  if (scatterAxisSize <= MAX_SCATTER_SIZE) {
    return failure(); // No workaround needed
  }

  // Calculate chunking parameters
  int64_t chunkSize = MAX_SCATTER_SIZE;
  int64_t numChunks =
      (scatterAxisSize + chunkSize - 1) / chunkSize; // ceil division

  Value currentResult = op.getInputTensor();

  // Process each chunk sequentially
  for (int64_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
    int64_t startIdx = chunkIdx * chunkSize;
    int64_t endIdx = std::min((chunkIdx + 1) * chunkSize, scatterAxisSize);

    // Create slice parameters for this chunk
    SmallVector<int32_t> begins(indexType.getRank(), 0);
    SmallVector<int32_t> ends(indexType.getShape().begin(),
                              indexType.getShape().end());
    SmallVector<int32_t> steps(indexType.getRank(), 1);

    begins[scatterDim] = static_cast<int32_t>(startIdx);
    ends[scatterDim] = static_cast<int32_t>(endIdx);

    // Calculate chunk shape
    SmallVector<int64_t> chunkShape(indexType.getShape());
    chunkShape[scatterDim] = endIdx - startIdx;

    // Slice index tensor for this chunk
    RankedTensorType chunkIndexType =
        ttnn::utils::RankedTensorTypeFactory::create(indexType, chunkShape);
    auto chunkIndex = rewriter.create<ttnn::SliceStaticOp>(
        ttmlir::utils::appendLocationSuffix(
            op.getLoc(), "_chunk_" + std::to_string(chunkIdx) + "_index"),
        chunkIndexType, op.getIndexTensor(), rewriter.getI32ArrayAttr(begins),
        rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));

    // Slice source tensor for this chunk
    RankedTensorType sourceType = op.getSourceTensor().getType();
    SmallVector<int64_t> chunkSourceShape(sourceType.getShape());
    chunkSourceShape[scatterDim] = endIdx - startIdx;

    RankedTensorType chunkSourceType =
        ttnn::utils::RankedTensorTypeFactory::create(sourceType,
                                                     chunkSourceShape);
    auto chunkSource = rewriter.create<ttnn::SliceStaticOp>(
        ttmlir::utils::appendLocationSuffix(
            op.getLoc(), "_chunk_" + std::to_string(chunkIdx) + "_source"),
        chunkSourceType, op.getSourceTensor(), rewriter.getI32ArrayAttr(begins),
        rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));

    // Perform scatter operation for this chunk
    auto chunkScatter = rewriter.create<ttnn::ScatterOp>(
        ttmlir::utils::appendLocationSuffix(
            op.getLoc(), "_chunk_" + std::to_string(chunkIdx) + "_scatter"),
        currentResult.getType(), currentResult, chunkIndex.getResult(),
        chunkSource.getResult(), rewriter.getI32IntegerAttr(scatterDim),
        op.getMemoryConfigAttr());

    currentResult = chunkScatter.getResult();
  }

  rewriter.replaceOp(op, currentResult);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
