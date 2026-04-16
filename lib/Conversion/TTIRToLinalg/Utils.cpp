// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir::tt::ttir_to_linalg {

SmallVector<int64_t, 2> getBroadcastDims(ArrayRef<int64_t> inputShape,
                                         ArrayRef<int64_t> targetShape) {
  const int64_t sizeDiff = targetShape.size() - inputShape.size();
  assert(sizeDiff >= 0 && "targetShape cannot be smaller than inputShape!");

  // Create padded input shape by prepending 1s.
  SmallVector<int64_t> paddedInput;
  paddedInput.append(sizeDiff, 1); // Prepend with 1s.
  paddedInput.append(inputShape.begin(), inputShape.end());

  // Find broadcast dimensions we want to broadcast along (including padding
  // dimensions).
  SmallVector<int64_t, 2> broadcastDims;
  for (const auto &it : llvm::enumerate(llvm::zip(paddedInput, targetShape))) {
    const size_t i = it.index();
    const auto &[inputDim, targetDim] = it.value();
    // Prepended dimensions are always broadcasted.
    if (i < static_cast<size_t>(sizeDiff) || inputDim != targetDim) {
      broadcastDims.push_back(i);
    }
  }

  return broadcastDims;
}

SmallVector<SmallVector<int64_t, 2>, 2>
getCollapseDims(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> targetShape) {
  // Calculate the size difference.
  const int64_t sizeDiff = targetShape.size() - inputShape.size();

  // Create the padded input shape by prepending 1s.
  SmallVector<int64_t> paddedInput(sizeDiff, 1);
  paddedInput.append(inputShape.begin(), inputShape.end());

  SmallVector<int64_t, 2> collapseDims;
  SmallVector<SmallVector<int64_t, 2>, 2> reassocIndexes;
  for (int64_t i = sizeDiff; i < static_cast<int64_t>(targetShape.size());
       ++i) {
    const int64_t inputDim = paddedInput[i];
    const int64_t targetDim = targetShape[i];
    // Adjust the index to account for the prepended dimensions
    // that are not part of the input shape.
    collapseDims.push_back(i - sizeDiff);
    if (inputDim == targetDim) {
      reassocIndexes.push_back(collapseDims);
      collapseDims.clear();
    }
  }

  if (!collapseDims.empty()) {
    if (reassocIndexes.empty()) {
      reassocIndexes.push_back(collapseDims);
    } else {
      reassocIndexes.back().append(collapseDims.begin(), collapseDims.end());
    }
  }

  return reassocIndexes;
}

Value broadcastToShape(Value input, ArrayRef<int64_t> targetShape, Location loc,
                       ConversionPatternRewriter &rewriter) {
  auto inputType = cast<RankedTensorType>(input.getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  SmallVector<int64_t, 2> broadcastDims =
      getBroadcastDims(inputShape, targetShape);

  // No broadcasting needed.
  if (broadcastDims.empty()) {
    return input;
  }

  auto initTensor = rewriter.create<ttir::EmptyOp>(loc, targetShape,
                                                   inputType.getElementType());

  // When all dims need broadcasting (e.g. [1,1] -> [64,128]), extract the
  // scalar element and use linalg.fill instead of linalg.broadcast, since
  // linalg.broadcast expects a lower-rank input (not a full-rank all-ones
  // tensor) and tensor.collapse_shape with empty reassociation is invalid.
  if (broadcastDims.size() == targetShape.size()) {
    SmallVector<Value> zeroIndices(
        inputShape.size(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
    Value scalar = rewriter.create<tensor::ExtractOp>(loc, input, zeroIndices);
    auto fillOp =
        rewriter.create<linalg::FillOp>(loc, scalar, initTensor.getResult());
    return fillOp.getResult(0);
  }

  Value broadcastInput = input;
  // The broadcast op requires we actually collapse any dimensions with
  // size 1 we want to broadcast along.
  SmallVector<SmallVector<int64_t, 2>, 2> collapseDimGroups =
      getCollapseDims(inputShape, targetShape);
  if (collapseDimGroups.size() != inputShape.size()) {
    broadcastInput =
        rewriter.create<tensor::CollapseShapeOp>(loc, input, collapseDimGroups);
  }

  auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
      loc, broadcastInput, initTensor.getResult(), broadcastDims);

  return broadcastOp.getResults().front();
}

Value convertToBooleanTensor(Value input, Location loc,
                             ConversionPatternRewriter &rewriter) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return input;
  }

  // If it's already a boolean tensor, return it as is.
  if (inputType.getElementType().isInteger(1)) {
    return input;
  }

  auto elementType = inputType.getElementType();
  assert((isa<FloatType>(elementType) || isa<IntegerType>(elementType)) &&
         "Only float and integer element types are supported");

  // Create zero constant.
  SmallVector<int64_t> zeroShape(inputType.getRank(), 1);
  auto zeroType = RankedTensorType::get(zeroShape, elementType);
  auto zeroAttr = createDenseElementsAttr(zeroType, 0.0);
  auto zeroConst = rewriter.create<tosa::ConstOp>(loc, zeroType, zeroAttr);

  // For logical operations, non-zero means true.
  // So we need: (input != 0) which we get by computing !(input == 0).
  auto boolType =
      RankedTensorType::get(inputType.getShape(), rewriter.getIntegerType(1));
  auto equalZero =
      rewriter.create<tosa::EqualOp>(loc, boolType, input, zeroConst);
  // Then use LogicalNotOp to invert it, giving us (input != 0).
  auto notEqualZero =
      rewriter.create<tosa::LogicalNotOp>(loc, boolType, equalZero);

  return notEqualZero;
}

DenseElementsAttr createDenseElementsAttr(RankedTensorType resultType,
                                          double value) {
  auto elementType = resultType.getElementType();
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    return SplatElementsAttr::get(resultType, FloatAttr::get(floatType, value));
  }
  if (isa<IntegerType>(elementType)) {
    return SplatElementsAttr::get(
        resultType, IntegerAttr::get(elementType, static_cast<int64_t>(value)));
  }
  return {};
}

Value createTosaConst(ConversionPatternRewriter &rewriter, Location loc,
                      Type elementType, int64_t rank, double value) {
  SmallVector<int64_t> shape(rank, 1);
  auto type = RankedTensorType::get(shape, elementType);
  auto attr = createDenseElementsAttr(type, value);
  return rewriter.create<tosa::ConstOp>(loc, type, attr);
}

Value createTosaMulShift(ConversionPatternRewriter &rewriter, Location loc) {
  auto type = RankedTensorType::get({1}, rewriter.getI8Type());
  auto attr = DenseElementsAttr::get(type, rewriter.getI8IntegerAttr(0));
  return rewriter.create<tosa::ConstOp>(loc, type, attr);
}

int64_t calculateExtraPadding(int64_t dim, int64_t kernel, int64_t stride,
                              int64_t padding1, int64_t padding2,
                              int64_t dilation) {
  if ((dim - 1 + padding1 + padding2 - (kernel - 1) * dilation) % stride != 0) {
    return (stride -
            (dim - 1 + padding1 + padding2 - (kernel - 1) * dilation) % stride);
  }
  return 0;
}

Value createTosaReshape(Value input, RankedTensorType targetType,
                        ConversionPatternRewriter &rewriter, Location loc) {
  ArrayRef<int64_t> newShape = targetType.getShape();
  auto shapeType = tosa::shapeType::get(rewriter.getContext(), newShape.size());
  auto shapeAttr = rewriter.getIndexTensorAttr(newShape);
  auto shapeOp = rewriter.create<tosa::ConstShapeOp>(loc, shapeType, shapeAttr);
  return rewriter
      .create<tosa::ReshapeOp>(loc, targetType, input, shapeOp.getResult())
      .getResult();
}

Value unflattenInput(Value input, ttir::FlattenedCompatInfoAttr flatInfo,
                     ConversionPatternRewriter &rewriter, Location loc) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto unflattenedType =
      RankedTensorType::get({flatInfo.getBatchSize(), flatInfo.getInputHeight(),
                             flatInfo.getInputWidth(), inputType.getShape()[3]},
                            inputType.getElementType());
  return createTosaReshape(input, unflattenedType, rewriter, loc);
}

Value sliceResultToShape(Value result, RankedTensorType targetType,
                         ConversionPatternRewriter &rewriter, Location loc) {
  auto resultType = cast<RankedTensorType>(result.getType());
  if (resultType.getShape() == targetType.getShape()) {
    return result;
  }
  SmallVector<OpFoldResult> offsets, sizes, strides;
  for (int64_t i = 0; i < targetType.getRank(); ++i) {
    offsets.push_back(rewriter.getI64IntegerAttr(0));
    sizes.push_back(rewriter.getI64IntegerAttr(targetType.getShape()[i]));
    strides.push_back(rewriter.getI64IntegerAttr(1));
  }
  return rewriter.create<tensor::ExtractSliceOp>(loc, targetType, result,
                                                 offsets, sizes, strides);
}

int64_t normalizeDim(int64_t dim, int64_t rank) {
  return dim < 0 ? dim + rank : dim;
}

} // namespace mlir::tt::ttir_to_linalg
