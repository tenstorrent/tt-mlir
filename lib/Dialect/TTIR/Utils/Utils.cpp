// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttir::utils {
llvm::SmallVector<int64_t> unsqueezeValue(mlir::PatternRewriter &rewriter,
                                          mlir::Location loc,
                                          mlir::Value &input,
                                          mlir::RankedTensorType desiredType,
                                          bool frontUnsqueeze) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> unsqueezeShape(desiredType.getRank(), 1);
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    int64_t idx =
        frontUnsqueeze ? (desiredType.getRank() - inputType.getRank()) + i : i;
    unsqueezeShape[idx] = inputType.getDimSize(i);
  }

  llvm::SmallVector<int32_t> reshapeDim(unsqueezeShape.begin(),
                                        unsqueezeShape.end());

  auto reshapeDimAttr = rewriter.getI32ArrayAttr(reshapeDim);
  input = rewriter.create<ttir::ReshapeOp>(
      loc,
      RankedTensorType::get(unsqueezeShape, desiredType.getElementType(),
                            desiredType.getEncoding()),
      input, reshapeDimAttr);
  return unsqueezeShape;
}

mlir::LogicalResult broadcastValue(mlir::PatternRewriter &rewriter,
                                   mlir::Value input,
                                   mlir::RankedTensorType desiredType,
                                   mlir::Value &output, mlir::Location loc,
                                   bool frontUnsqueeze) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> inputShape(inputType.getShape());
  llvm::SmallVector<int64_t, 4> broadcastedShape;
  if (!mlir::OpTrait::util::getBroadcastedShape(
          inputShape, desiredType.getShape(), broadcastedShape)) {
    return mlir::failure();
  }

  if (inputShape == desiredType.getShape()) {
    output = input;
    return mlir::success();
  }

  if (inputType.getRank() != desiredType.getRank()) {
    inputShape =
        unsqueezeValue(rewriter, loc, input, desiredType, frontUnsqueeze);
  }

  llvm::SmallVector<int64_t> broadcastDims =
      ttmlir::utils::getBroadcastDimensions<int64_t>(inputShape,
                                                     desiredType.getShape());

  output = rewriter.create<ttir::BroadcastOp>(loc, desiredType, input,
                                              broadcastDims);
  return mlir::success();
}

int64_t findMatchingDimRTL(ReshapeOp reshapeOp, int64_t dimRTL) {
  auto inputShape = reshapeOp.getInput().getType().getShape();
  auto outputShape = reshapeOp.getResult().getType().getShape();
  int64_t inputRank = inputShape.size();
  int64_t outputRank = outputShape.size();

  // Validate dimRTL is within bounds.
  if (dimRTL < 0 || dimRTL >= inputRank) {
    return -1;
  }

  // RTL position 0 is the rightmost dimension.
  // The number of trailing dimensions equals the RTL position.
  int64_t inputDimSize = inputShape[inputRank - 1 - dimRTL];

  // Calculate stride of the input dimension (product of trailing dimensions).
  auto inputTrailing = inputShape.take_back(dimRTL);
  int64_t inputStride =
      std::accumulate(inputTrailing.begin(), inputTrailing.end(), int64_t{1},
                      std::multiplies<>());

  // Search for a dimension in output with the same size and stride.
  // Check dimensions from right to left, accumulating stride incrementally.
  int64_t outputStride = 1;
  for (int64_t i = outputRank - 1; i >= 0; --i) {
    if (outputStride > inputStride) {
      break;
    }
    if (outputStride == inputStride && outputShape[i] == inputDimSize) {
      return outputRank - 1 - i;
    }
    outputStride *= outputShape[i];
  }

  // No dimension with the same size and stride found.
  return -1;
}

bool preservesDim(mlir::Operation *op, int64_t dim) {
  auto inputType = mlir::cast<RankedTensorType>(op->getOperand(0).getType());
  auto outputType = mlir::cast<RankedTensorType>(op->getResult(0).getType());
  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();
  int64_t inputRank = inputType.getRank();
  int64_t outputRank = outputType.getRank();

  // Normalize negative dimension.
  if (dim < 0) {
    dim += inputRank;
  }

  return llvm::TypeSwitch<mlir::Operation *, bool>(op)
      .Case<PermuteOp>([&](PermuteOp permute) {
        auto perm = permute.getPermutation();
        // Dimension must stay in the same position.
        if (perm[dim] != dim) {
          return false;
        }
        // All dimensions after must still map to positions after.
        auto permAfter = perm.drop_front(dim + 1);
        return llvm::all_of(permAfter, [&](int64_t p) { return p > dim; });
      })
      .Case<RepeatInterleaveOp>([&](RepeatInterleaveOp repeat) {
        int64_t repeatDim = repeat.getDim();
        if (repeatDim < 0) {
          repeatDim += inputRank;
        }
        return repeatDim != dim;
      })
      .Case<ReshapeOp>([&](ReshapeOp reshapeOp) {
        // Convert LTR dim to RTL position.
        int64_t inputDimRTL = inputRank - 1 - dim;
        int64_t foundRTL = findMatchingDimRTL(reshapeOp, inputDimRTL);

        // No matching dimension found.
        if (foundRTL == -1) {
          return false;
        }

        // Same RTL position means dimension is preserved.
        if (foundRTL == inputDimRTL) {
          return true;
        }

        // Check if all dimensions in between have size 1.
        int64_t minRTL = std::min(foundRTL, inputDimRTL);
        int64_t maxRTL = std::max(foundRTL, inputDimRTL);
        for (int64_t i = minRTL; i < maxRTL; ++i) {
          if (outputShape[outputRank - 1 - i] != 1) {
            return false;
          }
        }
        return true;
      })
      .Case<TypecastOp>([](TypecastOp) { return true; })
      .Case<BroadcastOp>([&](BroadcastOp) {
        if (inputRank != outputRank) {
          return false;
        }
        return inputShape[dim] == outputShape[dim];
      })
      .Default([](mlir::Operation *) { return false; });
}

} // namespace mlir::tt::ttir::utils
