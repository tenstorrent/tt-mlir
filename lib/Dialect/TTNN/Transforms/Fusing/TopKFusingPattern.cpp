// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/TopKFusingPattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#endif

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::tt::ttnn::fusing {

namespace {

// Helper to extract k value from slice attributes
// Returns nullopt if the slice doesn't start at 0 or has non-unit step
std::optional<int32_t> extractKFromSlice(SliceStaticOp sliceOp, int64_t dim) {
  auto beginsAttr = sliceOp.getBeginsAttr();
  auto endsAttr = sliceOp.getEndsAttr();
  auto stepsAttr = sliceOp.getStepAttr();

  auto begins = llvm::cast<ArrayAttr>(beginsAttr);
  auto ends = llvm::cast<ArrayAttr>(endsAttr);
  auto steps = llvm::cast<ArrayAttr>(stepsAttr);

  // Check that we're slicing from the beginning with unit step
  auto beginValue = llvm::cast<IntegerAttr>(begins[dim]).getInt();
  auto stepValue = llvm::cast<IntegerAttr>(steps[dim]).getInt();

  if (beginValue != 0 || stepValue != 1) {
    return std::nullopt;
  }

  // The end value is our k
  return llvm::cast<IntegerAttr>(ends[dim]).getInt();
}

// Check if two slice operations have identical parameters
bool haveSameSliceParams(SliceStaticOp slice1, SliceStaticOp slice2) {
  return slice1.getBeginsAttr() == slice2.getBeginsAttr() &&
         slice1.getEndsAttr() == slice2.getEndsAttr() &&
         slice1.getStepAttr() == slice2.getStepAttr();
}
} // namespace

mlir::LogicalResult
TopKFusing::matchAndRewrite(SortOp srcOp,
                            mlir::PatternRewriter &rewriter) const {
  // Sort operation must have exactly two results: values and indices
  if (srcOp->getNumResults() != 2) {
    return failure();
  }

  Value sortValues = srcOp.getValues();
  Value sortIndices = srcOp.getIndices();

  // Each result must have exactly one user
  if (!sortValues.hasOneUse() || !sortIndices.hasOneUse()) {
    return failure();
  }

  // Both users must be SliceStaticOp
  auto valuesSlice = dyn_cast<SliceStaticOp>(*sortValues.getUsers().begin());
  auto indicesSlice = dyn_cast<SliceStaticOp>(*sortIndices.getUsers().begin());

  if (!valuesSlice || !indicesSlice) {
    return failure();
  }

  // Both slices must have identical parameters
  if (!haveSameSliceParams(valuesSlice, indicesSlice)) {
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

  // Extract k from the slice operation
  auto k = extractKFromSlice(valuesSlice, sortDim);
  if (!k.has_value()) {
    return failure();
  }

  // Verify that the slice is actually on the sorted dimension
  // by checking that other dimensions are not sliced
  auto beginsAttr = llvm::cast<ArrayAttr>(valuesSlice.getBeginsAttr());
  auto endsAttr = llvm::cast<ArrayAttr>(valuesSlice.getEndsAttr());
  auto stepsAttr = llvm::cast<ArrayAttr>(valuesSlice.getStepAttr());
  auto inputShape = inputType.getShape();

  for (int64_t i = 0; i < rank; ++i) {
    if (i != sortDim) {
      // Other dimensions should not be sliced (begin=0, end=dim_size, step=1)
      auto beginValue = llvm::cast<IntegerAttr>(beginsAttr[i]).getInt();
      auto endValue = llvm::cast<IntegerAttr>(endsAttr[i]).getInt();
      auto stepValue = llvm::cast<IntegerAttr>(stepsAttr[i]).getInt();

      if (beginValue != 0 || endValue != inputShape[i] || stepValue != 1) {
        return failure();
      }
    }
  }

  // Map sort's descending attribute to topk's largest attribute
  bool largest = srcOp.getDescending();

  // TopK always produces sorted output when replacing a sort operation
  bool sorted = true;

  // Create the fused TopK operation
  auto topkOp =
      rewriter.create<TopKOp>(srcOp.getLoc(),
                              valuesSlice.getType(),  // values result type
                              indicesSlice.getType(), // indices result type
                              srcOp.getInput(),       // input tensor
                              rewriter.getI32IntegerAttr(*k),      // k value
                              rewriter.getI32IntegerAttr(sortDim), // dimension
                              rewriter.getBoolAttr(largest),       // largest
                              rewriter.getBoolAttr(sorted),        // sorted
                              nullptr // memory_config (optional)
      );

  // Replace the slice operations with TopK results
  rewriter.replaceOp(valuesSlice, topkOp.getValues());
  rewriter.replaceOp(indicesSlice, topkOp.getIndices());

  // Erase the sort operation (it has no more users after slices are replaced)
  rewriter.eraseOp(srcOp);

  return success();
}

} // namespace mlir::tt::ttnn::fusing
