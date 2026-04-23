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

int64_t findMatchingDim(llvm::ArrayRef<int64_t> fromShape,
                        llvm::ArrayRef<int64_t> toShape, int64_t dim) {
  int64_t fromRank = fromShape.size();
  int64_t toRank = toShape.size();

  if (dim < 0 || dim >= fromRank) {
    return -1;
  }

  int64_t dimSize = fromShape[dim];

  // Stride of `dim` is the product of all dimensions trailing it.
  int64_t fromStride = 1;
  for (int64_t i = dim + 1; i < fromRank; ++i) {
    fromStride *= fromShape[i];
  }

  // Walk toShape right-to-left, accumulating stride. A matching dim must have
  // the same trailing-stride and the same size. Once accumulated stride
  // exceeds fromStride it can never match again, so we can stop early.
  int64_t toStride = 1;
  for (int64_t i = toRank - 1; i >= 0; --i) {
    if (toStride > fromStride) {
      break;
    }
    if (toStride == fromStride && toShape[i] == dimSize) {
      return i;
    }
    toStride *= toShape[i];
  }

  return -1;
}

int64_t findMatchingDimRTL(ReshapeOp reshapeOp, int64_t dimRTL) {
  auto inputShape = reshapeOp.getInput().getType().getShape();
  auto outputShape = reshapeOp.getResult().getType().getShape();
  int64_t inputRank = inputShape.size();
  int64_t outputRank = outputShape.size();

  if (dimRTL < 0 || dimRTL >= inputRank) {
    return -1;
  }

  int64_t ltrDim = inputRank - 1 - dimRTL;
  int64_t ltrResult = findMatchingDim(inputShape, outputShape, ltrDim);
  if (ltrResult == -1) {
    return -1;
  }
  return outputRank - 1 - ltrResult;
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

namespace {

mlir::Type getTypecastInputElementType(mlir::Value value) {
  if (auto typecast =
          ttmlir::utils::findOpThrough<TypecastOp, ReshapeOp, PermuteOp>(
              value)) {
    return mlir::cast<mlir::RankedTensorType>(typecast.getInput().getType())
        .getElementType();
  }
  return nullptr;
}

} // namespace

mlir::Value revertTypecastFolding(mlir::PatternRewriter &rewriter,
                                  mlir::Operation *originalOp,
                                  mlir::Value newValue) {
  llvm::SmallVector<mlir::Type> originalInputElementTypes;
  for (mlir::Value operand : originalOp->getOperands()) {
    if (!mlir::isa<mlir::RankedTensorType>(operand.getType())) {
      continue;
    }
    if (mlir::Type elementType = getTypecastInputElementType(operand)) {
      originalInputElementTypes.push_back(elementType);
    }
  }

  if (originalInputElementTypes.empty() ||
      !llvm::all_equal(originalInputElementTypes)) {
    return newValue;
  }

  auto newValueType =
      mlir::dyn_cast<mlir::RankedTensorType>(newValue.getType());
  if (!newValueType) {
    return newValue;
  }

  mlir::Type originalInputElementType = originalInputElementTypes.front();
  if (newValueType.getElementType() == originalInputElementType) {
    return newValue;
  }

  auto convertedType = mlir::RankedTensorType::get(newValueType.getShape(),
                                                   originalInputElementType,
                                                   newValueType.getEncoding());

  rewriter.setInsertionPointAfterValue(newValue);
  auto convertedValue =
      rewriter.create<TypecastOp>(originalOp->getLoc(), convertedType, newValue,
                                  /*conservative_folding=*/false);

  rewriter.setInsertionPointAfter(convertedValue);
  auto restoredValue = rewriter.create<TypecastOp>(
      originalOp->getLoc(), newValueType, convertedValue,
      /*conservative_folding=*/false);

  mlir::Value valueToReplace =
      originalOp->getNumResults() == 1 ? originalOp->getResult(0) : newValue;
  rewriter.replaceUsesWithIf(valueToReplace, restoredValue,
                             [&](mlir::OpOperand &operand) {
                               mlir::Operation *owner = operand.getOwner();
                               return owner != convertedValue.getOperation() &&
                                      owner != restoredValue.getOperation();
                             });
  return convertedValue;
}

} // namespace mlir::tt::ttir::utils
