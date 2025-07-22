// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ConcatOpDecompositionRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Decompose ConcatOp into multiple ConcatOps if number of inputs exceed 50.
LogicalResult ConcatOpDecompositionRewritePattern::matchAndRewrite(
    ttnn::ConcatOp srcOp, PatternRewriter &rewriter) const {
  constexpr int kMaxAllowedInputs = 50;
  auto inputs = srcOp.getInputs();
  int64_t numInputs = inputs.size();
  if (numInputs <= kMaxAllowedInputs) {
    return failure();
  }

  // Lambda to compute the output shape for a partial concatenation
  // It adds dimensions along the concatenation axis (dim).
  auto computeSubOutputShape = [](mlir::Value firstInput,
                                  mlir::OperandRange inputs,
                                  int64_t dim) -> llvm::SmallVector<int64_t> {
    auto outputShape =
        mlir::cast<RankedTensorType>(firstInput.getType()).getShape();
    llvm::SmallVector<int64_t> outputShapeVec{outputShape};
    for (auto input : inputs) {
      auto inputShape =
          mlir::cast<RankedTensorType>(input.getType()).getShape();
      outputShapeVec[dim] += inputShape[dim];
    }
    return outputShapeVec;
  };

  int64_t start = 1;
  int64_t end = kMaxAllowedInputs;
  int64_t dim = srcOp.getDim();
  Value runningConcatOp = inputs.front();
  auto inputType = mlir::cast<RankedTensorType>(runningConcatOp.getType());

  // Iteratively build ConcatOps: the first ConcatOp uses inputs[0–49], and each
  // following op combines the previous result with up to 49 subsequent inputs.
  while (start < numInputs) {
    auto subRange =
        llvm::make_range(inputs.begin() + start, inputs.begin() + end);
    auto shape = computeSubOutputShape(runningConcatOp, subRange, dim);
    RankedTensorType outputType =
        utils::RankedTensorTypeFactory::create(inputType, shape);

    llvm::SmallVector<mlir::Value> subInputs({runningConcatOp});
    subInputs.append(subRange.begin(), subRange.end());

    runningConcatOp =
        rewriter.create<ttnn::ConcatOp>(srcOp->getLoc(), outputType, subInputs,
                                        dim, srcOp.getMemoryConfigAttr());

    start = end;
    end = std::min(end + kMaxAllowedInputs - 1, numInputs);
  }

  rewriter.replaceOp(srcOp, runningConcatOp);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
