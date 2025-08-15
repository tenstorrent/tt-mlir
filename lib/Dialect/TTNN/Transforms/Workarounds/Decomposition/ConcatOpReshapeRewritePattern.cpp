// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ConcatOpReshapeRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
LogicalResult ConcatOpReshapeRewritePattern::matchAndRewrite(
    ttnn::ConcatOp srcOp, PatternRewriter &rewriter) const {
  int64_t dim = srcOp.getDim();
  mlir::RankedTensorType outputType = srcOp.getResult().getType();
  int64_t rank = outputType.getRank();

  // Workaround required for 1 dimensional tensors only.
  if (rank != 1) {
    return failure();
  }

  // Apply workaround: reshape inputs by adding an extra dimension.
  llvm::SmallVector<mlir::Value> reshapedInputs;
  for (auto input : srcOp.getInputs()) {
    auto typedInput = dyn_cast<mlir::TypedValue<mlir::RankedTensorType>>(input);
    llvm::SmallVector<int64_t> newShape(typedInput.getType().getShape());
    newShape.push_back(1);

    reshapedInputs.push_back(ttir_to_ttnn::utils::generateReshape(
        typedInput, newShape, rewriter, srcOp.getLoc()));
  }

  // Create output type with extra dimension.
  llvm::SmallVector<int64_t> newOutputShape(outputType.getShape());
  newOutputShape.push_back(1);

  RankedTensorType newOutputType =
      utils::RankedTensorTypeFactory::create(outputType, newOutputShape);

  // Create new concat op with reshaped inputs and new output type.
  auto newConcatOp = rewriter.create<ttnn::ConcatOp>(
      srcOp->getLoc(), newOutputType, reshapedInputs, dim,
      /*memory_config=*/nullptr);

  // Reshape back to the original output shape.
  auto result = ttir_to_ttnn::utils::generateReshape(
      newConcatOp, outputType.getShape(), rewriter, srcOp.getLoc());

  rewriter.replaceOp(srcOp, result);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
