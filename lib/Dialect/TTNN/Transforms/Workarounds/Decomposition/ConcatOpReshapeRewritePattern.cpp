// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ConcatOpReshapeRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
static bool isWorkaroundRequired(ttnn::ConcatOp srcOp) {
  auto areAllDimsExceptLastOne = [](auto shape) {
    return std::all_of(shape.begin(), shape.end() - 1,
                       [](int64_t dim) { return dim == 1; });
  };
  // Check if we need to apply the workaround.
  bool anyInputLastDimOne = false;
  for (auto input : srcOp.getInputs()) {
    auto shape =
        mlir::dyn_cast<mlir::RankedTensorType>(input.getType()).getShape();
    assert(!shape.empty());

    // All dimensions except the last must be 1.
    if (!areAllDimsExceptLastOne(shape)) {
      return false;
    }

    // At least one input must have last dimension of size 1.
    if (shape.back() == 1) {
      anyInputLastDimOne = true;
    }
  }

  return anyInputLastDimOne;
}

// Issue to track in tt-metal repo:
// https://github.com/tenstorrent/tt-metal/issues/21581.
LogicalResult ConcatOpReshapeRewritePattern::matchAndRewrite(
    ttnn::ConcatOp srcOp, PatternRewriter &rewriter) const {
  int64_t dim = srcOp.getDim();
  mlir::RankedTensorType outputType = srcOp.getResult().getType();
  int64_t rank = outputType.getRank();

  // Check if this is a concat on the last dimension with rank 2 or 3.
  if (((dim + 1) != rank) || (rank != 2 && rank != 3)) {
    return failure();
  }

  if (!isWorkaroundRequired(srcOp)) {
    return failure();
  }

  // Apply workaround: reshape inputs by adding an extra dimension.
  llvm::SmallVector<mlir::Value> reshapedInputs;
  for (auto input : srcOp.getInputs()) {
    auto typedInput = dyn_cast<mlir::TypedValue<mlir::RankedTensorType>>(input);
    llvm::SmallVector<int64_t> newShape(typedInput.getType().getShape());
    newShape.push_back(1);

    reshapedInputs.push_back(
        ttir_to_ttnn::utils::generateReshape(typedInput, newShape, rewriter));
  }

  // Create output type with extra dimension.
  llvm::SmallVector<int64_t> newOutputShape(outputType.getShape());
  newOutputShape.push_back(1);

  auto newOutputType = mlir::RankedTensorType::get(
      newOutputShape, outputType.getElementType(),
      mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding())
          .withTensorShape(newOutputShape));

  // Create new concat op with reshaped inputs and new output type.
  auto newConcatOp = rewriter.create<ttnn::ConcatOp>(
      srcOp->getLoc(), newOutputType, reshapedInputs, dim,
      /*memory_config=*/nullptr);

  // Reshape back to the original output shape.
  auto result = ttir_to_ttnn::utils::generateReshape(
      newConcatOp, outputType.getShape(), rewriter);

  rewriter.replaceOp(srcOp, result);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
