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

  /*
   * ScatterOp in TTNN has a hardware limit on the scatter axis size.
   * i.e. index_shape[dim] > to_layout_int32_scatter_axis_max_length
   * This is a requirement from
   * third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/scatter/scatter.cpp
   *
   * This function decomposes large scatter operations into sequential smaller
   * chunks, each respecting the 256-element constraint.
   *
   * EXAMPLE:
   * --------
   * Original operation that would fail:
   *   input_tensor: shape [2272]
   *   index_tensor: shape [284] (>256)
   *   source_tensor: shape [284]
   *   scatter_dim: 0
   *
   * Chunking transformation:
   *
   * CHUNK 0: Elements [0:256]
   *   slice_index_0 = index_tensor[0:256]     // shape [256]
   *   slice_source_0 = source_tensor[0:256]   // shape [256]
   *   result_0 = scatter(input_tensor, slice_index_0, slice_source_0, dim=0)
   *
   * CHUNK 1: Elements [256:284]
   *   slice_index_1 = index_tensor[256:284]   // shape [28]
   *   slice_source_1 = source_tensor[256:284] // shape [28]
   *   result_1 = scatter(result_0, slice_index_1, slice_source_1, dim=0)
   *
   * Return result_1 as final output.
   */
  RankedTensorType indexType = op.getIndexTensor().getType();
  int32_t scatterDim = op.getDim();

  constexpr int64_t MAX_SCATTER_SIZE = 256;
  int64_t scatterAxisSize = indexType.getShape()[scatterDim];

  if (scatterAxisSize <= MAX_SCATTER_SIZE) {
    return failure();
  }

  // Calculate number of chunks needed using round-up (ceiling) division.
  int64_t numChunks =
      (scatterAxisSize + MAX_SCATTER_SIZE - 1) / MAX_SCATTER_SIZE;

  Value currentResult = op.getInputTensor();

  // Process each chunk sequentially
  for (int64_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
    int64_t startIdx = chunkIdx * MAX_SCATTER_SIZE;
    int64_t endIdx =
        std::min((chunkIdx + 1) * MAX_SCATTER_SIZE, scatterAxisSize);

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
