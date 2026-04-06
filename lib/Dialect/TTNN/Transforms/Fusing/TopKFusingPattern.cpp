// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/TopKFusingPattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir::tt::ttnn::fusing {

namespace {

// Result of extracting k from a slice operation.
// `fromEnd` indicates the slice takes elements from the end of the sorted
// dimension, which requires inverting the `largest` flag when creating TopK.
struct SliceKResult {
  int32_t k;
  bool fromEnd;
};

// Helper to extract k value from slice attributes.
// Handles two cases:
//   1. begin=0, end=k (slice from start) -> k elements, largest matches sort
//   2. begin=n-k, end=n (slice from end) -> k elements, largest is inverted
// Returns nullopt if the slice doesn't match either pattern or has non-unit
// step.
std::optional<SliceKResult> extractKFromSlice(SliceStaticOp sliceOp,
                                              int64_t dim, int64_t dimSize) {
  auto beginsAttr = sliceOp.getBeginsAttr();
  auto endsAttr = sliceOp.getEndsAttr();
  auto stepsAttr = sliceOp.getStepAttr();

  auto begins = llvm::cast<ArrayAttr>(beginsAttr);
  auto ends = llvm::cast<ArrayAttr>(endsAttr);
  auto steps = llvm::cast<ArrayAttr>(stepsAttr);

  auto beginValue = llvm::cast<IntegerAttr>(begins[dim]).getInt();
  auto endValue = llvm::cast<IntegerAttr>(ends[dim]).getInt();
  auto stepValue = llvm::cast<IntegerAttr>(steps[dim]).getInt();

  // Must have unit step
  if (stepValue != 1) {
    return std::nullopt;
  }

  // Case 1: slice from the beginning [0, k)
  if (beginValue == 0) {
    return SliceKResult{static_cast<int32_t>(endValue), /*fromEnd=*/false};
  }

  // Case 2: slice from the end [n-k, n)
  if (endValue == dimSize) {
    int32_t k = static_cast<int32_t>(dimSize - beginValue);
    return SliceKResult{k, /*fromEnd=*/true};
  }

  return std::nullopt;
}

// Check if two slice operations have identical parameters
bool haveSameSliceParams(SliceStaticOp slice1, SliceStaticOp slice2) {
  return slice1.getBeginsAttr() == slice2.getBeginsAttr() &&
         slice1.getEndsAttr() == slice2.getEndsAttr() &&
         slice1.getStepAttr() == slice2.getStepAttr();
}
} // namespace

// Helper to compute the TopK result type for an unused sort result.
// Derives from the input type with the sorted dimension replaced by k,
// preserving the TTNNLayoutAttr encoding (adjusted for the new shape).
RankedTensorType computeTopKResultType(RankedTensorType inputType,
                                       int64_t sortDim, int32_t k,
                                       Type elementType) {
  SmallVector<int64_t> shape(inputType.getShape());
  shape[sortDim] = k;

  // If the input has a TTNNLayoutAttr encoding, create a new layout with the
  // updated shape so the validation function has properly typed results.
  Attribute encoding = nullptr;
  if (auto inputLayout =
          mlir::dyn_cast_or_null<TTNNLayoutAttr>(inputType.getEncoding())) {
    encoding = inputLayout.withElementType(elementType, shape);
  }

  return RankedTensorType::get(shape, elementType, encoding);
}

// Helper to validate slice parameters: verifies the slice only operates on the
// sorted dimension (other dimensions are passed through unchanged) and extracts
// the k value.
std::optional<SliceKResult>
validateAndExtractK(SliceStaticOp sliceOp, int64_t sortDim,
                    llvm::ArrayRef<int64_t> inputShape, int64_t rank) {
  auto sliceResult = extractKFromSlice(sliceOp, sortDim, inputShape[sortDim]);
  if (!sliceResult.has_value()) {
    return std::nullopt;
  }

  auto beginsAttr = llvm::cast<ArrayAttr>(sliceOp.getBeginsAttr());
  auto endsAttr = llvm::cast<ArrayAttr>(sliceOp.getEndsAttr());
  auto stepsAttr = llvm::cast<ArrayAttr>(sliceOp.getStepAttr());

  for (int64_t i = 0; i < rank; ++i) {
    if (i != sortDim) {
      auto beginValue = llvm::cast<IntegerAttr>(beginsAttr[i]).getInt();
      auto endValue = llvm::cast<IntegerAttr>(endsAttr[i]).getInt();
      auto stepValue = llvm::cast<IntegerAttr>(stepsAttr[i]).getInt();

      if (beginValue != 0 || endValue != inputShape[i] || stepValue != 1) {
        return std::nullopt;
      }
    }
  }

  return sliceResult;
}

mlir::LogicalResult
TopKFusing::matchAndRewrite(SortOp srcOp,
                            mlir::PatternRewriter &rewriter) const {
  // Sort operation must have exactly two results: values and indices
  if (srcOp->getNumResults() != 2) {
    return failure();
  }

  Value sortValues = srcOp.getValues();
  Value sortIndices = srcOp.getIndices();

  // Each result must either be unused or have exactly one user
  if (!sortValues.hasOneUse() && !sortValues.use_empty()) {
    return failure();
  }
  if (!sortIndices.hasOneUse() && !sortIndices.use_empty()) {
    return failure();
  }

  // For used results, the single user must be a SliceStaticOp.
  // At least one result must have a SliceStaticOp user for TopK fusion.
  SliceStaticOp valuesSlice = nullptr;
  SliceStaticOp indicesSlice = nullptr;

  if (sortValues.hasOneUse()) {
    valuesSlice = dyn_cast<SliceStaticOp>(*sortValues.getUsers().begin());
    if (!valuesSlice) {
      return failure();
    }
  }

  if (sortIndices.hasOneUse()) {
    indicesSlice = dyn_cast<SliceStaticOp>(*sortIndices.getUsers().begin());
    if (!indicesSlice) {
      return failure();
    }
  }

  // At least one slice must exist for this to be a TopK pattern
  if (!valuesSlice && !indicesSlice) {
    return failure();
  }

  // If both slices exist, they must have identical parameters
  if (valuesSlice && indicesSlice &&
      !haveSameSliceParams(valuesSlice, indicesSlice)) {
    return failure();
  }

  // Get the dimension being sorted
  int64_t sortDim = srcOp.getDim();

  // Handle negative dimension
  auto inputType = mlir::cast<RankedTensorType>(srcOp.getInput().getType());
  int64_t rank = inputType.getRank();
  if (sortDim < 0) {
    sortDim += rank;
  }

  auto inputShape = inputType.getShape();

  // Use whichever slice exists to extract k and validate
  SliceStaticOp activeSlice = valuesSlice ? valuesSlice : indicesSlice;
  auto sliceResult =
      validateAndExtractK(activeSlice, sortDim, inputShape, rank);
  if (!sliceResult.has_value()) {
    return failure();
  }

  // Map sort's descending attribute to topk's largest attribute.
  // When slicing from the end, the relationship is inverted:
  //   descending + slice_from_start -> largest=true
  //   ascending  + slice_from_start -> largest=false
  //   descending + slice_from_end   -> largest=false (tail of descending =
  //   smallest) ascending  + slice_from_end   -> largest=true  (tail of
  //   ascending = largest)
  bool largest =
      sliceResult->fromEnd ? !srcOp.getDescending() : srcOp.getDescending();

  // TopK always produces sorted output when replacing a sort operation
  bool sorted = true;

  // Determine result types for TopK.
  // For used results, use the slice output type.
  // For unused results, derive from input shape with sorted dim replaced by k.
  RankedTensorType valuesResultType =
      valuesSlice ? mlir::cast<RankedTensorType>(valuesSlice.getType())
                  : computeTopKResultType(inputType, sortDim, sliceResult->k,
                                          inputType.getElementType());
  RankedTensorType indicesResultType =
      indicesSlice
          ? mlir::cast<RankedTensorType>(indicesSlice.getType())
          : computeTopKResultType(inputType, sortDim, sliceResult->k,
                                  IntegerType::get(rewriter.getContext(), 32,
                                                   IntegerType::Signed));

  // Validate the fusion before creating it
  FusionValidator validator(rewriter.getContext(), validationConfig);

  // Validate the TopK operation that would be created
  auto validationResult = validator.validateFusion<TopKOp>(
      srcOp.getOperation(), // Pass source op for context
      srcOp.getLoc(), {valuesResultType, indicesResultType}, srcOp.getInput(),
      rewriter.getI32IntegerAttr(sliceResult->k),
      rewriter.getI32IntegerAttr(sortDim), rewriter.getBoolAttr(largest),
      rewriter.getBoolAttr(sorted),
      nullptr // memory_config
  );

  // If validation fails, don't fuse
  if (!validationResult.isSuccess()) {
    // Log the validation failure for debugging
    TTMLIR_DEBUG(ttmlir::LogComponent::FusionValidator,
                 "TopK fusion validation failed: {0}",
                 validationResult.errorMessage);
    return failure();
  }

  // Create the fused TopK operation (now that we know it's valid)
  auto topkOp = TopKOp::create(
      rewriter, srcOp.getLoc(), valuesResultType, indicesResultType,
      srcOp.getInput(),                           // input tensor
      rewriter.getI32IntegerAttr(sliceResult->k), // k value
      rewriter.getI32IntegerAttr(sortDim),        // dimension
      rewriter.getBoolAttr(largest),              // largest
      rewriter.getBoolAttr(sorted),               // sorted
      nullptr                                     // memory_config (optional)
  );

  // Replace the slice operations with TopK results (only for used results)
  if (valuesSlice) {
    rewriter.replaceOp(valuesSlice, topkOp.getValues());
  }
  if (indicesSlice) {
    rewriter.replaceOp(indicesSlice, topkOp.getIndices());
  }

  // Erase the sort operation (it has no more users after slices are replaced)
  rewriter.eraseOp(srcOp);

  return success();
}

} // namespace mlir::tt::ttnn::fusing
