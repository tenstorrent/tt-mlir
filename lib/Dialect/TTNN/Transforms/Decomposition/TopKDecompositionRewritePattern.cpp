// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/TopKDecompositionRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::decomposition {

LogicalResult TopKDecompositionRewritePattern::matchAndRewrite(
    ttnn::TopKOp topkOp, PatternRewriter &rewriter) const {

  auto inputType =
      mlir::cast<RankedTensorType>(topkOp.getInputTensor().getType());
  auto valuesResultType =
      mlir::cast<RankedTensorType>(topkOp.getValues().getType());
  auto indicesResultType =
      mlir::cast<RankedTensorType>(topkOp.getIndices().getType());

  // When validation config is provided, validate the TopK operation using
  // IsolatedIRValidationWrapper. If validation succeeds, keep the op as-is.
  if (validationConfig.has_value()) {
    IsolatedIRValidationWrapper validator(rewriter.getContext(),
                                          *validationConfig);

    auto validationResult = validator.validateOp<ttnn::TopKOp>(
        topkOp.getOperation(), topkOp.getLoc(),
        {valuesResultType, indicesResultType}, topkOp.getInputTensor(),
        topkOp.getKAttr(), topkOp.getDimAttr(), topkOp.getLargestAttr(),
        topkOp.getSortedAttr());

    if (validationResult.isSuccess()) {
      // Op is valid — keep it as-is.
      return failure();
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::IsolatedIRValidationWrapper,
                 "TopK decomposition triggered (validation failed): {0}",
                 validationResult.errorMessage);
  } else {
    // No validator available (e.g. optimizer disabled / opt_level=0).
    // Preserve the TopK rather than regenerating sort+slice with default
    // attributes — the regenerated ops can differ from the original sort+slice
    // (e.g. stable=false, null memory_config) and break downstream paths such
    // as trace capture. TopK has native TTNN runtime support, so keeping it
    // is safe.
    return failure();
  }

  // Decompose TopK back into Sort + SliceStatic.
  int32_t k = topkOp.getK();
  int32_t dim = topkOp.getDim();
  bool largest = topkOp.getLargest();
  int64_t rank = inputType.getRank();

  // Normalize negative dimension.
  int64_t normalizedDim = dim < 0 ? dim + rank : dim;

  // Create Sort result types: same as input shape (full, un-sliced).
  RankedTensorType sortValuesType = utils::RankedTensorTypeFactory::create(
      valuesResultType, inputType.getShape());
  // For indices, create with si32 element type and input shape.
  RankedTensorType sortIndicesType = utils::RankedTensorTypeFactory::create(
      indicesResultType, inputType.getShape());

  // Map TopK's largest to Sort's descending: largest=true -> descending=true.
  auto si8Type =
      IntegerType::get(rewriter.getContext(), 8, IntegerType::Signed);
  auto sortOp = rewriter.create<ttnn::SortOp>(
      topkOp.getLoc(), sortValuesType, sortIndicesType, topkOp.getInputTensor(),
      IntegerAttr::get(si8Type, static_cast<int8_t>(dim)),
      rewriter.getBoolAttr(largest), // descending = largest
      rewriter.getBoolAttr(false));  // stable = false

  // Build slice attributes: begins=0, ends=inputShape (except ends[dim]=k),
  // step=1.
  auto inputShape = inputType.getShape();
  SmallVector<Attribute> begins, ends, steps;
  for (int64_t i = 0; i < rank; ++i) {
    begins.push_back(rewriter.getI32IntegerAttr(0));
    ends.push_back(rewriter.getI32IntegerAttr(
        i == normalizedDim ? k : static_cast<int32_t>(inputShape[i])));
    steps.push_back(rewriter.getI32IntegerAttr(1));
  }

  auto beginsAttr = rewriter.getArrayAttr(begins);
  auto endsAttr = rewriter.getArrayAttr(ends);
  auto stepsAttr = rewriter.getArrayAttr(steps);

  // Create SliceStatic for values and indices.
  auto valuesSlice = rewriter.create<ttnn::SliceStaticOp>(
      topkOp.getLoc(), valuesResultType, sortOp.getValues(), beginsAttr,
      endsAttr, stepsAttr);

  auto indicesSlice = rewriter.create<ttnn::SliceStaticOp>(
      topkOp.getLoc(), indicesResultType, sortOp.getIndices(), beginsAttr,
      endsAttr, stepsAttr);

  rewriter.replaceOp(topkOp,
                     {valuesSlice.getResult(), indicesSlice.getResult()});
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
