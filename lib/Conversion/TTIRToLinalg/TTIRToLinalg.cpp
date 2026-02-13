// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/TTIRToLinalg.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt {
//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//
namespace {
// Convert a tensor of floating-point values to a tensor of boolean values
// by comparing with zero; zero is false and nonzero is true; logical_not op
// uses this pattern.
static Value convertToBooleanTensor(Value input, Location loc,
                                    ConversionPatternRewriter &rewriter) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return input;
  }

  // If it's already a boolean tensor, return it as is
  if (inputType.getElementType().isInteger(1)) {
    return input;
  }

  // Create a constant tensor with 0.0 for comparison
  auto elementType = inputType.getElementType();
  assert(elementType.isF32());
  TypedAttr zeroAttr = rewriter.getF32FloatAttr(0.0f);

  // Create a constant scalar with the zero value
  auto zeroValue = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

  // Create a splat tensor with the zero value
  auto zeroSplat =
      rewriter.create<tensor::SplatOp>(loc, inputType, zeroValue.getResult());

  // For logical operations, non-zero means true
  // So we need: (input != 0) which we get by computing !(input == 0)
  auto boolType =
      RankedTensorType::get(inputType.getShape(), rewriter.getIntegerType(1));

  // Check if input == 0
  auto equalZero =
      rewriter.create<tosa::EqualOp>(loc, boolType, input, zeroSplat);

  // Then use LogicalNotOp to invert it, giving us (input != 0)
  auto notEqualZero =
      rewriter.create<tosa::LogicalNotOp>(loc, boolType, equalZero);

  return notEqualZero;
}

// Convert a tensor of floating-point values to a tensor of boolean values
// using comparison semantics (positive values are true, non-positive are
// false)--whereOp uses this pattern unfortunately.
static Value
convertToBooleanTensorComparison(Value input, Location loc,
                                 ConversionPatternRewriter &rewriter) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return input;
  }

  // If it's already a boolean tensor, return it as is
  if (inputType.getElementType().isInteger(1)) {
    return input;
  }

  // Create a constant tensor with 0.0 for comparison
  auto elementType = inputType.getElementType();
  assert(elementType.isF32());
  TypedAttr zeroAttr = rewriter.getF32FloatAttr(0.0f);

  // Create a constant scalar with the zero value
  auto zeroValue = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

  // Create a splat tensor with the zero value
  auto zeroSplat =
      rewriter.create<tensor::SplatOp>(loc, inputType, zeroValue.getResult());

  // For comparison semantics: positive values are true
  // So we need: (input > 0)
  auto boolType =
      RankedTensorType::get(inputType.getShape(), rewriter.getIntegerType(1));

  // Check if input > 0
  auto greaterThanZero =
      rewriter.create<tosa::GreaterOp>(loc, boolType, input, zeroSplat);

  return greaterThanZero;
}

// Normalize negative dimension to positive. Negative dimensions are interpreted
// as indexing from the end (e.g., -1 is the last dimension).
static int64_t normalizeDim(int64_t dim, int64_t rank) {
  return dim < 0 ? dim + rank : dim;
}

// Get the dimensions to broadcast.
//
// This function calculates the dimensions to broadcast. We assume that input
// and target shapes are broadcastable. For example if input shape is [4, 1, 3]
// and we want to broadcast to [1, 4, 5, 3], the function will return [0, 2]
// since we want to broadcast 0th and 2nd dimension of result shape.
static SmallVector<int64_t, 2> getBroadcastDims(ArrayRef<int64_t> inputShape,
                                                ArrayRef<int64_t> targetShape) {
  const int64_t sizeDiff = targetShape.size() - inputShape.size();
  assert(sizeDiff >= 0 && "targetShape cannot be smaller than inputShape!");

  // Create padded input shape by prepending 1s.
  SmallVector<int64_t> paddedInput;
  paddedInput.append(sizeDiff, 1); // Prepend with 1s
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

// Get the dimensions to collapse.
//
// This function calculates the dimensions to collapse. We assume that input
// and target shapes are broadcastable. linalg.broadcast requires that input
// tensor only contains dimensions that won't be broadcasted in input tensor.
// For example if input shape is [4, 1, 3] and we want to broadcast to [4, 5,
// 3], then we need to collapse the first dimension of input tensor to [4, 3].
// This function calculates the dimensions to collapse. In case above, we will
// return [[0], [1, 2]].
static SmallVector<SmallVector<int64_t, 2>, 2>
getCollapseDims(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> targetShape) {
  // Calculate the size difference.
  const size_t sizeDiff = targetShape.size() - inputShape.size();

  // Create the padded input shape by prepending 1s.
  SmallVector<int64_t> paddedInput(sizeDiff, 1);
  paddedInput.append(inputShape.begin(), inputShape.end());

  SmallVector<int64_t, 2> collapseDims;
  SmallVector<SmallVector<int64_t, 2>, 2> reassocIndexes;
  for (size_t i = sizeDiff; i < targetShape.size(); ++i) {
    const size_t inputDim = paddedInput[i];
    const size_t targetDim = targetShape[i];
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

// Reshape a tensor by prepending 1s to match the target rank.
// This is useful for broadcasting parameters like weight and bias in LayerNorm.
// For example, if input has shape [64] and targetRank is 3 with numTrailingDims
// = 1, the result will have shape [1, 1, 64].
static Value reshapeByPrependingOnes(Value input, int64_t targetRank,
                                     int64_t numTrailingDims, Type elementType,
                                     Location loc,
                                     ConversionPatternRewriter &rewriter) {
  auto inputType = cast<RankedTensorType>(input.getType());

  SmallVector<int64_t> broadcastShape;
  for (int64_t i = 0; i < targetRank - numTrailingDims; ++i) {
    broadcastShape.push_back(1);
  }
  for (auto dim : inputType.getShape()) {
    broadcastShape.push_back(dim);
  }

  auto reshapedType = RankedTensorType::get(broadcastShape, elementType);
  auto shapeType =
      tosa::shapeType::get(rewriter.getContext(), broadcastShape.size());
  auto shapeAttr = rewriter.getIndexTensorAttr(broadcastShape);
  auto shapeOp = rewriter.create<tosa::ConstShapeOp>(loc, shapeType, shapeAttr);
  return rewriter.create<tosa::ReshapeOp>(loc, reshapedType, input,
                                          shapeOp.getResult());
}

// Get dimensions from the dim_arg attribute; if the attribute is not present or
// empty, return all dimensions.
static SmallVector<int64_t> getDimsFromAttribute(Operation *op, int64_t rank) {
  if (auto dimAttr = op->getAttrOfType<ArrayAttr>("dim_arg")) {
    if (dimAttr.size() == 0) {
      // If dim_arg is present but empty, reduce along all dimensions.
      SmallVector<int64_t> allDims(rank);
      std::iota(allDims.begin(), allDims.end(), 0);
      return allDims;
    }

    // Otherwise, use the provided dimensions, normalizing negative indices.
    SmallVector<int64_t> dims;
    for (auto dim : dimAttr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(dim)) {
        int64_t d = intAttr.getInt();
        // Normalize negative dimensions
        if (d < 0) {
          d += rank;
        }
        dims.push_back(d);
      }
    }
    return dims;
  }

  // If no dim_arg attribute, reduce along all dimensions.
  SmallVector<int64_t> allDims(rank);
  std::iota(allDims.begin(), allDims.end(), 0);
  return allDims;
}

// Get keep_dim attribute; return false is not present.
static bool getKeepDimFromAttribute(Operation *op) {
  if (auto keepDimAttr = op->getAttrOfType<BoolAttr>("keep_dim")) {
    return keepDimAttr.getValue();
  }
  return false;
}

// Helper function to create a chain of reduction operations for multiple
// dimensions
template <typename ReductionOp>
static Value createReductionOpChain(Value input, RankedTensorType resultType,
                                    ArrayRef<int64_t> dims, bool keepDim,
                                    Location loc,
                                    ConversionPatternRewriter &rewriter) {
  // Sort dimensions in descending order to avoid changing indices during
  // reduction
  SmallVector<int64_t> sortedDims(dims.begin(), dims.end());
  std::sort(sortedDims.begin(), sortedDims.end(), std::greater<int64_t>());

  Value result = input;
  auto inputType = cast<RankedTensorType>(input.getType());

  SmallVector<int64_t> shape(inputType.getShape().begin(),
                             inputType.getShape().end());
  // For each dimension, create a reduction operation
  for (size_t i = 0; i < sortedDims.size(); ++i) {
    int64_t dim = sortedDims[i];

    // Create the axis attribute for this dimension
    auto axisAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(dim));
    RankedTensorType opResultType;
    shape[dim] = 1;
    opResultType = RankedTensorType::get(shape, inputType.getElementType());

    // Create the reduction operation
    result = rewriter.create<ReductionOp>(loc, opResultType, result, axisAttr);
  }
  if (!keepDim) {
    ArrayRef<int64_t> newShape = resultType.getShape();
    auto shapeType =
        tosa::shapeType::get(rewriter.getContext(), newShape.size());
    auto attr = rewriter.getIndexTensorAttr(newShape);
    auto shapeOp = rewriter.create<tosa::ConstShapeOp>(loc, shapeType, attr);
    result = rewriter.create<tosa::ReshapeOp>(loc, resultType, result, shapeOp);
  }
  return result;
}

// Helper function to create DenseElementsAttr with a specific value based on
// element type.
static DenseElementsAttr createDenseElementsAttr(RankedTensorType resultType,
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

// Helper function to calculate extra padding for MaxPool2dOp.
// This function calculates the extra padding needed to make the output size
// divisible by the stride.
static int64_t calculateExtraPadding(int64_t dim, int64_t kernel,
                                     int64_t stride, int64_t padding1,
                                     int64_t padding2, int64_t dilation) {
  if ((dim - 1 + padding1 + padding2 - (kernel - 1) * dilation) % stride != 0) {
    return (stride -
            (dim - 1 + padding1 + padding2 - (kernel - 1) * dilation) % stride);
  }
  return 0;
}

} // namespace

//===----------------------------------------------------------------------===//
// TOSA Conversions Patterns
//===----------------------------------------------------------------------===//

namespace {
template <typename TTIROpTy, typename TosaOpTy>
class ElementwiseUnaryOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto result = rewriter.create<TosaOpTy>(op.getLoc(), resultType, input);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
template <typename TTIROpTy, typename TosaOpTy>
class TosaElementwiseBinaryOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto result = rewriter.create<TosaOpTy>(op.getLoc(), resultType,
                                            ValueRange{lhs, rhs});

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class WhereOpConversionPattern : public OpConversionPattern<ttir::WhereOp> {
public:
  using OpConversionPattern<ttir::WhereOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::WhereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value condition = adaptor.getFirst();
    Value trueValue = adaptor.getSecond();
    Value falseValue = adaptor.getThird();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    condition =
        convertToBooleanTensorComparison(condition, op.getLoc(), rewriter);

    auto result = rewriter.create<tosa::SelectOp>(
        op.getLoc(), resultType, condition, trueValue, falseValue);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class ReshapeOpConversionPattern : public OpConversionPattern<ttir::ReshapeOp> {
public:
  using OpConversionPattern<ttir::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto newShape = resultType.getShape();
    SmallVector<int64_t> newShapeValues(newShape.begin(), newShape.end());
    auto shapeType =
        tosa::shapeType::get(rewriter.getContext(), newShape.size());
    auto attr = rewriter.getIndexTensorAttr(newShapeValues);
    auto shapeOp =
        rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

    auto reshapeOp = rewriter.create<tosa::ReshapeOp>(
        op.getLoc(), resultType, adaptor.getInput(), shapeOp);

    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    auto copyOp = rewriter.create<linalg::CopyOp>(
        op.getLoc(), ValueRange{reshapeOp}, output.getResult());
    rewriter.replaceOp(op, copyOp.getResult(0));

    return success();
  }
};
} // namespace

namespace {
class TransposeOpConversionPattern
    : public OpConversionPattern<ttir::TransposeOp> {
public:
  using OpConversionPattern<ttir::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    const size_t permSize = inputType.getShape().size();
    SmallVector<int32_t> permutation(permSize);
    for (size_t i = 0; i < permSize; i++) {
      permutation[i] = static_cast<int32_t>(i);
    }

    const int64_t dim0 =
        (op.getDim0() < 0) ? op.getDim0() + permSize : op.getDim0();
    const int64_t dim1 =
        (op.getDim1() < 0) ? op.getDim1() + permSize : op.getDim1();

    permutation[dim1] = static_cast<int32_t>(dim0);
    permutation[dim0] = static_cast<int32_t>(dim1);

    // Create TransposeOp directly with the permutation array
    auto result = rewriter.create<tosa::TransposeOp>(op.getLoc(), resultType,
                                                     input, permutation);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class ConcatOpConversionPattern : public OpConversionPattern<ttir::ConcatOp> {
public:
  using OpConversionPattern<ttir::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputs = adaptor.getOperands();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // TOSA concat requires non-negative axis, so normalize negative dimensions.
    int64_t dim = normalizeDim(op.getDim(), resultType.getRank());

    // TOSA concat requires at least two inputs.
    if (inputs.size() < 2) {
      return failure();
    }

    // Concatenate all inputs at once using the final result type.
    Value result =
        rewriter.create<tosa::ConcatOp>(op.getLoc(), resultType, inputs, dim);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
// Direct comparison operations (where TTIR and TOSA ops match directly)
template <typename TTIROpTy, typename TosaOpTy>
class DirectComparisonOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Create the TOSA comparison operation
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));
    auto boolResult =
        rewriter.create<TosaOpTy>(op.getLoc(), boolType, lhs, rhs);

    // Convert boolean result to original type using cast.
    auto result =
        rewriter.create<tosa::CastOp>(op.getLoc(), resultType, boolResult);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Swapped comparison operations (where TTIR and TOSA ops have swapped operands
// e.g. ttir.lt must use inverted tosa.greater).
template <typename TTIROpTy, typename TosaOpTy>
class SwappedComparisonOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Create the TOSA comparison operation with swapped operands
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));
    auto boolResult =
        rewriter.create<TosaOpTy>(op.getLoc(), boolType, rhs, lhs);

    // Convert boolean result to original type using cast.
    auto result =
        rewriter.create<tosa::CastOp>(op.getLoc(), resultType, boolResult);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Negated comparison operations (where TTIR op is the negation of a TOSA op,
// e.g. ttir.not_equal)
template <typename TTIROpTy, typename TosaOpTy>
class NegatedComparisonOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Create the TOSA comparison operation
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));
    auto boolResult =
        rewriter.create<TosaOpTy>(op.getLoc(), boolType, lhs, rhs);

    // Negate the boolean result
    auto notResult =
        rewriter.create<tosa::LogicalNotOp>(op.getLoc(), boolType, boolResult);

    // Convert boolean result to original type using cast.
    auto result =
        rewriter.create<tosa::CastOp>(op.getLoc(), resultType, notResult);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class BroadcastOpConversionPattern
    : public OpConversionPattern<ttir::BroadcastOp> {
public:
  using OpConversionPattern<ttir::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> targetShape = resultType.getShape();

    // Calculate broadcast dimensions - these are the dimensions we need to
    // broadcast along.
    SmallVector<int64_t, 2> broadcastDims =
        getBroadcastDims(inputShape, targetShape);

    // If no broadcasting needed, just use the input directly.
    if (broadcastDims.empty()) {
      rewriter.replaceOp(op, input);
      return success();
    }

    // linalg.broadcast requires that input tensor only contains dimensions
    // that won't be broadcasted. We need to collapse any size-1 dimensions
    // that will be broadcasted.
    SmallVector<SmallVector<int64_t, 2>, 2> collapseDimGroups =
        (broadcastDims.size() != targetShape.size())
            ? getCollapseDims(inputShape, targetShape)
            : SmallVector<SmallVector<int64_t, 2>, 2>();

    Value broadcastInput = input;

    // The broadcast op requires we actually collapse any dimensions with
    // size 1 we want to broadcast along.
    if (collapseDimGroups.size() != inputShape.size()) {
      broadcastInput = rewriter.create<tensor::CollapseShapeOp>(
          loc, input, collapseDimGroups);
    }

    auto initTensor = rewriter.create<ttir::EmptyOp>(
        loc, targetShape, inputType.getElementType());
    auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
        loc, broadcastInput, initTensor.getResult(), broadcastDims);

    rewriter.replaceOp(op, broadcastOp.getResults().front());
    return success();
  }
};
} // namespace

namespace {
class SinOpConversionPattern : public OpConversionPattern<ttir::SinOp> {
public:
  using OpConversionPattern<ttir::SinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto result = rewriter.create<tosa::SinOp>(op.getLoc(), resultType, input);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
// Cos operation - TOSA doesn't have a direct cos operation
class CosOpConversionPattern : public OpConversionPattern<ttir::CosOp> {
public:
  using OpConversionPattern<ttir::CosOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::CosOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Create a scalar constant for π/2 using arith.constant
    auto elementType = resultType.getElementType();
    auto piOver2 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), elementType, rewriter.getFloatAttr(elementType, M_PI_2));

    // Create a tensor with the same shape as input filled with π/2
    auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
    auto piOver2Tensor = rewriter.create<tensor::SplatOp>(
        op.getLoc(), RankedTensorType::get(inputShape, elementType),
        piOver2.getResult());

    // Add π/2 to the input
    auto shifted = rewriter.create<tosa::AddOp>(op.getLoc(), resultType, input,
                                                piOver2Tensor);

    // Take the sin of the shifted input
    auto result =
        rewriter.create<tosa::SinOp>(op.getLoc(), resultType, shifted);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class MatmulOpConversionPattern : public OpConversionPattern<ttir::MatmulOp> {
public:
  using OpConversionPattern<ttir::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getA();
    Value rhs = adaptor.getB();

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));

    if (!lhsType || !rhsType || !resultType) {
      return rewriter.notifyMatchFailure(
          op, "Operands or result is not a ranked tensor");
    }

    bool transposeA = op.getTransposeA();
    bool transposeB = op.getTransposeB();

    // Handle transposition if needed
    if (transposeA) {
      auto lhsShape = lhsType.getShape();
      SmallVector<int64_t> transposedShape;

      if (lhsShape.size() >= 2) {
        // For 2D+ tensors, transpose the last two dimensions
        for (size_t i = 0; i < lhsShape.size() - 2; ++i) {
          transposedShape.push_back(lhsShape[i]);
        }
        transposedShape.push_back(lhsShape[lhsShape.size() - 1]);
        transposedShape.push_back(lhsShape[lhsShape.size() - 2]);

        auto transposedType =
            RankedTensorType::get(transposedShape, lhsType.getElementType());

        // Create permutation attribute
        SmallVector<int32_t> permutation;
        for (size_t i = 0; i < lhsShape.size() - 2; ++i) {
          permutation.push_back(static_cast<int32_t>(i));
        }
        permutation.push_back(static_cast<int32_t>(lhsShape.size() - 1));
        permutation.push_back(static_cast<int32_t>(lhsShape.size() - 2));

        // Create transpose op
        lhs = rewriter.create<tosa::TransposeOp>(op.getLoc(), transposedType,
                                                 lhs, permutation);
        lhsType = transposedType;
      }
    }

    if (transposeB) {
      auto rhsShape = rhsType.getShape();
      SmallVector<int64_t> transposedShape;

      if (rhsShape.size() >= 2) {
        // For 2D+ tensors, transpose the last two dimensions
        for (size_t i = 0; i < rhsShape.size() - 2; ++i) {
          transposedShape.push_back(rhsShape[i]);
        }
        transposedShape.push_back(rhsShape[rhsShape.size() - 1]);
        transposedShape.push_back(rhsShape[rhsShape.size() - 2]);

        auto transposedType =
            RankedTensorType::get(transposedShape, rhsType.getElementType());

        // Create permutation attribute
        SmallVector<int32_t> permutation;
        for (size_t i = 0; i < rhsShape.size() - 2; ++i) {
          permutation.push_back(static_cast<int32_t>(i));
        }
        permutation.push_back(static_cast<int32_t>(rhsShape.size() - 1));
        permutation.push_back(static_cast<int32_t>(rhsShape.size() - 2));

        // Create transpose op
        rhs = rewriter.create<tosa::TransposeOp>(op.getLoc(), transposedType,
                                                 rhs, permutation);
        rhsType = transposedType;
      }
    }

    // Ensure both tensors are 3D for tosa.matmul
    unsigned lhsRank = lhsType.getRank();
    unsigned rhsRank = rhsType.getRank();

    // Convert to 3D tensors if needed
    Value lhs3D = lhs;
    Value rhs3D = rhs;
    RankedTensorType lhs3DType = lhsType;
    RankedTensorType rhs3DType = rhsType;

    // If LHS is 2D, reshape to 3D with batch size 1
    if (lhsRank == 2) {
      SmallVector<int64_t> newShape = {1, lhsType.getDimSize(0),
                                       lhsType.getDimSize(1)};
      auto newType = RankedTensorType::get(newShape, lhsType.getElementType());

      // Create shape tensor for reshape - matching your original approach
      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      SmallVector<int64_t> shapeValues = {1, lhsType.getDimSize(0),
                                          lhsType.getDimSize(1)};
      auto attr = rewriter.getIndexTensorAttr(shapeValues);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      // Reshape LHS to 3D
      lhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, lhs,
                                               shapeOp.getResult());
      lhs3DType = newType;
    } else if (lhsRank > 3) {
      // For tensors with rank > 3, collapse all but the last two dimensions
      int64_t collapsedBatchSize = 1;
      for (uint32_t i = 0; i < lhsRank - 2; ++i) {
        collapsedBatchSize *= lhsType.getShape()[i];
      }

      SmallVector<int64_t> newShape = {collapsedBatchSize,
                                       lhsType.getShape()[lhsRank - 2],
                                       lhsType.getShape()[lhsRank - 1]};
      auto newType = RankedTensorType::get(newShape, lhsType.getElementType());

      // Create shape tensor for reshape
      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      SmallVector<int64_t> shapeValues = {collapsedBatchSize,
                                          lhsType.getShape()[lhsRank - 2],
                                          lhsType.getShape()[lhsRank - 1]};
      auto attr = rewriter.getIndexTensorAttr(shapeValues);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      // Reshape LHS to 3D
      lhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, lhs,
                                               shapeOp.getResult());
      lhs3DType = newType;
    }

    // If RHS is 2D, reshape to 3D with batch size 1
    if (rhsRank == 2) {
      SmallVector<int64_t> newShape = {1, rhsType.getDimSize(0),
                                       rhsType.getDimSize(1)};
      auto newType = RankedTensorType::get(newShape, rhsType.getElementType());

      // Create shape tensor for reshape
      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      SmallVector<int64_t> shapeValues = {1, rhsType.getDimSize(0),
                                          rhsType.getDimSize(1)};
      auto attr = rewriter.getIndexTensorAttr(shapeValues);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      // Reshape RHS to 3D
      rhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, rhs,
                                               shapeOp.getResult());
      rhs3DType = newType;
    } else if (rhsRank > 3) {
      // For tensors with rank > 3, collapse all but the last two dimensions
      int64_t collapsedBatchSize = 1;
      for (uint32_t i = 0; i < rhsRank - 2; ++i) {
        collapsedBatchSize *= rhsType.getShape()[i];
      }

      SmallVector<int64_t> newShape = {collapsedBatchSize,
                                       rhsType.getShape()[rhsRank - 2],
                                       rhsType.getShape()[rhsRank - 1]};
      auto newType = RankedTensorType::get(newShape, rhsType.getElementType());

      // Create shape tensor for reshape
      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      SmallVector<int64_t> shapeValues = {collapsedBatchSize,
                                          rhsType.getShape()[rhsRank - 2],
                                          rhsType.getShape()[rhsRank - 1]};
      auto attr = rewriter.getIndexTensorAttr(shapeValues);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      // Reshape RHS to 3D
      rhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, rhs,
                                               shapeOp.getResult());
      rhs3DType = newType;
    }

    // Check if we need to broadcast batch dimensions
    if (lhs3DType.getShape()[0] != rhs3DType.getShape()[0]) {
      // We need to broadcast one of the inputs to match the other's batch
      // dimension
      if (lhs3DType.getShape()[0] == 1 && rhs3DType.getShape()[0] > 1) {
        // Use TOSA tile operation for broadcasting
        SmallVector<int64_t> multiples = {rhs3DType.getShape()[0], 1, 1};
        auto newType = RankedTensorType::get({rhs3DType.getShape()[0],
                                              lhs3DType.getShape()[1],
                                              lhs3DType.getShape()[2]},
                                             lhs3DType.getElementType());

        auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
        auto multiplesAttr = rewriter.getIndexTensorAttr(multiples);
        auto multiplesOp = rewriter.create<tosa::ConstShapeOp>(
            op.getLoc(), shapeType, multiplesAttr);

        lhs3D = rewriter.create<tosa::TileOp>(op.getLoc(), newType, lhs3D,
                                              multiplesOp);
        lhs3DType = cast<RankedTensorType>(lhs3D.getType());
      } else if (rhs3DType.getShape()[0] == 1 && lhs3DType.getShape()[0] > 1) {
        // Use TOSA tile operation for broadcasting
        SmallVector<int64_t> multiples = {lhs3DType.getShape()[0], 1, 1};
        auto newType = RankedTensorType::get({lhs3DType.getShape()[0],
                                              rhs3DType.getShape()[1],
                                              rhs3DType.getShape()[2]},
                                             rhs3DType.getElementType());

        auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
        auto multiplesAttr = rewriter.getIndexTensorAttr(multiples);
        auto multiplesOp = rewriter.create<tosa::ConstShapeOp>(
            op.getLoc(), shapeType, multiplesAttr);

        rhs3D = rewriter.create<tosa::TileOp>(op.getLoc(), newType, rhs3D,
                                              multiplesOp);
        rhs3DType = cast<RankedTensorType>(rhs3D.getType());
      }
    }

    // Now both tensors should have the same batch dimension
    auto matmulResultType =
        RankedTensorType::get({lhs3DType.getShape()[0], lhs3DType.getShape()[1],
                               rhs3DType.getShape()[2]},
                              resultType.getElementType());

    // Perform matrix multiplication using tosa.matmul
    Value matmulResult = rewriter.create<tosa::MatMulOp>(
        op.getLoc(), matmulResultType, lhs3D, rhs3D);

    // Reshape result back to original rank if needed
    if (resultType.getRank() != matmulResultType.getRank()) {
      // Create shape tensor for reshape
      auto shapeType =
          tosa::shapeType::get(rewriter.getContext(), resultType.getRank());
      SmallVector<int64_t> shapeValues;
      for (auto dim : resultType.getShape()) {
        shapeValues.push_back(dim);
      }
      auto attr = rewriter.getIndexTensorAttr(shapeValues);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      // Reshape result
      matmulResult = rewriter.create<tosa::ReshapeOp>(
          op.getLoc(), resultType, matmulResult, shapeOp.getResult());
    }

    Value dest = rewriter.create<ttir::EmptyOp>(
        op.getLoc(), op.getType().getShape(), op.getType().getElementType());
    auto copyOp =
        rewriter.create<linalg::CopyOp>(op.getLoc(), matmulResult, dest);
    rewriter.replaceOp(op, copyOp.getResult(0));
    return success();
  }
};
} // namespace

namespace {
class Conv2dOpConversionPattern : public OpConversionPattern<ttir::Conv2dOp> {
public:
  using OpConversionPattern<ttir::Conv2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::Conv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    Value bias = adaptor.getBias();
    Attribute strides = adaptor.getStride();
    Attribute dilations = adaptor.getDilation();
    Attribute padding = adaptor.getPadding();
    uint32_t groups = adaptor.getGroups();

    if (groups > 1) {
      return rewriter.notifyMatchFailure(
          op, "Grouped convolution is not supported yet.");
    }
    auto weightType = cast<RankedTensorType>(weight.getType());

    // TTIR uses (O,C,H,W) but TOSA uses (O,H,W,C) for weight.
    SmallVector<int32_t> permutation = {0, 2, 3, 1};
    auto weightShape = weightType.getShape();
    SmallVector<int64_t> transposedShape = ttmlir::utils::applyPermutation(
        weightShape, llvm::to_vector_of<int64_t>(permutation));

    auto transposedWeightType =
        RankedTensorType::get(transposedShape, weightType.getElementType());

    auto transposedWeight = rewriter.create<tosa::TransposeOp>(
        op.getLoc(), transposedWeightType, weight, permutation);

    // Reshape bias from 4D (1,1,1,B) to 1D (B) for TOSA.
    // If bias is not provided, create a zero bias tensor.
    Value reshapedBias = nullptr;
    if (bias) {
      auto biasType = cast<RankedTensorType>(bias.getType());
      auto biasShape = biasType.getShape();
      assert(biasShape.size() == 4 && "Bias must be 4D");
      assert(biasShape[0] == 1 && biasShape[1] == 1 && biasShape[2] == 1 &&
             "Bias must be 4D with shape (1,1,1,B)");
      SmallVector<int64_t> reshapedBiasShape = {biasShape[3]};
      auto reshapedBiasType =
          RankedTensorType::get(reshapedBiasShape, biasType.getElementType());
      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 1);
      auto shapeAttr = rewriter.getIndexTensorAttr(reshapedBiasShape);
      auto shapeOp = rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType,
                                                         shapeAttr);

      reshapedBias = rewriter
                         .create<tosa::ReshapeOp>(op.getLoc(), reshapedBiasType,
                                                  bias, shapeOp.getResult())
                         .getResult();
    } else {
      int64_t outputChannels = weightShape[0];
      auto biasElementType =
          cast<RankedTensorType>(input.getType()).getElementType();
      auto biasType = RankedTensorType::get({outputChannels}, biasElementType);

      Attribute zeroAttr;
      if (isa<FloatType>(biasElementType)) {
        zeroAttr = DenseElementsAttr::get(
            biasType,
            APFloat::getZero(
                cast<FloatType>(biasElementType).getFloatSemantics()));
      } else if (isa<IntegerType>(biasElementType)) {
        zeroAttr = DenseElementsAttr::get(
            biasType,
            APInt::getZero(cast<IntegerType>(biasElementType).getWidth()));
      } else {
        return rewriter.notifyMatchFailure(op, "Unsupported bias element type");
      }
      reshapedBias = rewriter.create<tosa::ConstOp>(
          op.getLoc(), biasType, cast<DenseElementsAttr>(zeroAttr));
    }
    // Expand stride if it contains only one element.
    auto stridesResult = ttmlir::utils::getPairOfInteger<int32_t>(strides);
    if (!stridesResult) {
      return rewriter.notifyMatchFailure(
          op, "stride must be an integer or array attribute");
    }

    // Expand padding if it contains only one or two elements.
    auto paddingResult = ttmlir::utils::getQuadrupleOfInteger<int32_t>(padding);
    if (!paddingResult) {
      return rewriter.notifyMatchFailure(
          op, "padding must be an integer, 2-element, or 4-element array "
              "attribute");
    }
    // If padding is a single integer, expand it to 4 elements.
    // If padding is a 2-element array, it is (width, height) and expand it to 4
    // elements (width, height, width, height). If padding is a 4-element array,
    // TTIR uses (top, left, bottom, right) but TOSA uses (top, bottom, left,
    // right). Permutation of indices 1 and 2 is needed to match the padding
    // order. (width, height, width, height) -> (width, width, height, height)
    // (top, left, bottom, right) -> (top, bottom, left, right)

    auto [paddingTop, paddingLeft, paddingBottom, paddingRight] =
        *paddingResult;

    // Expand dilation if it contains only one element.
    auto dilationsResult = ttmlir::utils::getPairOfInteger<int32_t>(dilations);
    if (!dilationsResult) {
      return rewriter.notifyMatchFailure(
          op, "dilation must be an integer or array attribute");
    }

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Choose accumulator type based on result element type.
    Type accType;
    if (isa<FloatType>(resultType.getElementType())) {
      accType = rewriter.getF32Type();
    } else if (isa<IntegerType>(resultType.getElementType())) {
      accType = rewriter.getI32Type();
    } else {
      return rewriter.notifyMatchFailure(
          op, "Unsupported result element type for conv2d");
    }

    // Update padding and return shape to be used in the TOSA Conv2DOp.
    // input_height - 1 + pad_top + pad_bottom - (kernel_height - 1) *
    // dilation_y must be divisible by stride_y. input_width - 1 + pad_left +
    // pad_right - (kernel_width - 1) * dilation_x must be divisible by
    // stride_x. The padding values are updated to ensure this condition is met.
    int64_t inputHeight = cast<RankedTensorType>(input.getType()).getShape()[1];
    int64_t inputWidth = cast<RankedTensorType>(input.getType()).getShape()[2];
    int64_t kernelHeight = weightShape[2];
    int64_t kernelWidth = weightShape[3];

    paddingBottom += calculateExtraPadding(
        inputHeight, kernelHeight, stridesResult->first, paddingTop,
        paddingBottom, dilationsResult->first);
    paddingRight += calculateExtraPadding(
        inputWidth, kernelWidth, stridesResult->second, paddingLeft,
        paddingRight, dilationsResult->second);

    auto expandedStridesAttr = rewriter.getDenseI64ArrayAttr(
        {stridesResult->first, stridesResult->second});
    auto expandedDilationsAttr = rewriter.getDenseI64ArrayAttr(
        {dilationsResult->first, dilationsResult->second});
    auto expandedPaddingAttr = rewriter.getDenseI64ArrayAttr(
        {paddingTop, paddingBottom, paddingLeft, paddingRight});

    // Update return shape to be used in the TOSA Conv2DOp.
    // Height and width must be adjusted for extra padding.
    // Batch size and output channels are not changed.
    SmallVector<int64_t> resultShape = {
        resultType.getShape()[0],
        (inputHeight - 1 + paddingTop + paddingBottom -
         (kernelHeight - 1) * dilationsResult->first) /
                stridesResult->first +
            1,
        (inputWidth - 1 + paddingLeft + paddingRight -
         (kernelWidth - 1) * dilationsResult->second) /
                stridesResult->second +
            1,
        resultType.getShape()[3]};

    auto actualResultType =
        RankedTensorType::get(resultShape, resultType.getElementType());

    auto conv2dOp = rewriter.create<tosa::Conv2DOp>(
        op.getLoc(), actualResultType, input, transposedWeight.getResult(),
        reshapedBias, expandedPaddingAttr, expandedStridesAttr,
        expandedDilationsAttr, TypeAttr::get(accType));

    Value result = conv2dOp.getResult();

    // Slice the result back to the original expected shape if needed.
    ArrayRef<int64_t> originalShape = resultType.getShape();
    if (!std::equal(resultShape.begin(), resultShape.end(),
                    originalShape.begin(), originalShape.end())) {
      SmallVector<OpFoldResult> offsets, sizes, strides;
      for (int64_t i = 0; i < resultType.getRank(); ++i) {
        offsets.push_back(rewriter.getI64IntegerAttr(0));
        sizes.push_back(rewriter.getI64IntegerAttr(resultType.getShape()[i]));
        strides.push_back(rewriter.getI64IntegerAttr(1));
      }
      result = rewriter.create<tensor::ExtractSliceOp>(
          op.getLoc(), resultType, result, offsets, sizes, strides);

      // Since tensor::ExtractSliceOp doesn't support DPS, we need to copy
      // the result into the output buffer
      Value dest = rewriter.create<ttir::EmptyOp>(
          op.getLoc(), op.getType().getShape(), op.getType().getElementType());
      auto copyResult =
          rewriter.create<linalg::CopyOp>(op.getLoc(), result, dest);
      rewriter.replaceOp(op, copyResult);

      return success();
    }

    rewriter.replaceOp(op, result);

    return success();
  }
};
} // namespace

namespace {
class MaxPool2dOpConversionPattern
    : public OpConversionPattern<ttir::MaxPool2dOp> {
public:
  using OpConversionPattern<ttir::MaxPool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto input = adaptor.getInput();
    auto strides = adaptor.getStride();
    auto kernel = adaptor.getKernel();
    auto padding = adaptor.getPadding();
    auto dilation = adaptor.getDilation();

    // Expand stride if it contains only one element.
    auto stridesResult = ttmlir::utils::getPairOfInteger<int32_t>(strides);
    if (!stridesResult) {
      return rewriter.notifyMatchFailure(
          op, "stride must be an integer or array attribute");
    }

    auto paddingResult = ttmlir::utils::getQuadrupleOfInteger<int32_t>(padding);
    if (!paddingResult) {
      return rewriter.notifyMatchFailure(
          op, "padding must be an integer, 2-element, or 4-element array "
              "attribute");
    }

    // If padding is a single integer, expand it to 4 elements.
    // If padding is a 2-element array, it is (width, height) and expand it to 4
    // elements (width, height, width, height). If padding is a 4-element array,
    // TTIR uses (top, left, bottom, right) but TOSA uses (top, bottom, left,
    // right). Permutation of indices 1 and 2 is needed to match the padding
    // order. (width, height, width, height) -> (width, width, height, height)
    // (top, left, bottom, right) -> (top, bottom, left, right)

    auto [paddingTop, paddingLeft, paddingBottom, paddingRight] =
        *paddingResult;

    // Expand kernel if it contains only one element.
    auto kernelResult = ttmlir::utils::getPairOfInteger<int32_t>(kernel);
    if (!kernelResult) {
      return rewriter.notifyMatchFailure(
          op, "kernel must be an integer or array attribute");
    }

    auto dilationResult = ttmlir::utils::getPairOfInteger<int32_t>(dilation);
    if (!dilationResult) {
      return rewriter.notifyMatchFailure(
          op, "dilation must be an integer or array attribute");
    }
    assert(dilationResult->first == 1 && dilationResult->second == 1 &&
           "dilation must be 1x1");

    // Update padding and return shape to be used in the TOSA MaxPool2dOp.
    // input_height + pad_top + pad_bottom - kernel_height must be divisible by
    // stride_y. input_width + pad_left + pad_right - kernel_width must be
    // divisible by stride_x. The padding values are updated to ensure this
    // condition is met.
    int64_t inputHeight = cast<RankedTensorType>(input.getType()).getShape()[1];
    int64_t inputWidth = cast<RankedTensorType>(input.getType()).getShape()[2];
    auto [kernelHeight, kernelWidth] = *kernelResult;

    paddingBottom +=
        calculateExtraPadding(inputHeight, kernelHeight, stridesResult->first,
                              paddingTop, paddingBottom, 1);
    paddingRight +=
        calculateExtraPadding(inputWidth, kernelWidth, stridesResult->second,
                              paddingLeft, paddingRight, 1);

    auto expandedStridesAttr = rewriter.getDenseI64ArrayAttr(
        {stridesResult->first, stridesResult->second});
    auto expandedPaddingAttr = rewriter.getDenseI64ArrayAttr(
        {paddingTop, paddingBottom, paddingLeft, paddingRight});
    auto expandedKernelAttr = rewriter.getDenseI64ArrayAttr(
        {kernelResult->first, kernelResult->second});

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");
    // Update return shape to be used in the TOSA MaxPool2dOp.
    // The output size is calculated as (inputSize + padding1 + padding2 -
    // kernelSize) / stride + 1. This is because the output size is the number
    // of elements separated by stride in a kernel window between the first and
    // last element of the input after padding.
    SmallVector<int64_t> resultShape(resultType.getShape());
    resultShape[1] = (inputHeight + paddingTop + paddingBottom - kernelHeight) /
                         stridesResult->first +
                     1;
    resultShape[2] = (inputWidth + paddingLeft + paddingRight - kernelWidth) /
                         stridesResult->second +
                     1;

    auto actualResultType =
        RankedTensorType::get(resultShape, resultType.getElementType());

    // Create the max pool op.
    auto maxPoolOp = rewriter.create<tosa::MaxPool2dOp>(
        op.getLoc(), actualResultType, input, expandedKernelAttr,
        expandedStridesAttr, expandedPaddingAttr);

    Value result = maxPoolOp.getResult();

    // Slice the result back to the original expected shape if needed.
    if (!llvm::equal(resultShape, resultType.getShape())) {
      SmallVector<OpFoldResult> offsets, sizes, strides;
      for (int64_t i = 0; i < resultType.getRank(); ++i) {
        offsets.push_back(rewriter.getI64IntegerAttr(0));
        sizes.push_back(rewriter.getI64IntegerAttr(resultType.getShape()[i]));
        strides.push_back(rewriter.getI64IntegerAttr(1));
      }
      result = rewriter.create<tensor::ExtractSliceOp>(
          op.getLoc(), resultType, result, offsets, sizes, strides);

      // Since tensor::ExtractSliceOp doesn't support DPS, we need to copy
      // the result into the output buffer
      Value output = rewriter.create<ttir::EmptyOp>(
          op.getLoc(), op.getType().getShape(), op.getType().getElementType());
      auto copyResult =
          rewriter.create<linalg::CopyOp>(op.getLoc(), result, output);
      rewriter.replaceOp(op, copyResult);

      return success();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class AvgPool2dOpConversionPattern
    : public OpConversionPattern<ttir::AvgPool2dOp> {
public:
  using OpConversionPattern<ttir::AvgPool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::AvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto strides = adaptor.getStride();
    auto kernel = adaptor.getKernel();
    auto padding = adaptor.getPadding();
    auto dilation = adaptor.getDilation();

    // Parse stride attribute.
    auto stridesResult = ttmlir::utils::getPairOfInteger<int32_t>(strides);
    if (!stridesResult) {
      return rewriter.notifyMatchFailure(
          op, "stride must be an integer or array attribute");
    }
    auto [strideH, strideW] = *stridesResult;

    // Parse padding attribute.
    auto paddingResult = ttmlir::utils::getQuadrupleOfInteger<int32_t>(padding);
    if (!paddingResult) {
      return rewriter.notifyMatchFailure(
          op, "padding must be an integer, 2-element, or 4-element array "
              "attribute");
    }
    auto [paddingTop, paddingLeft, paddingBottom, paddingRight] =
        *paddingResult;

    // Parse kernel attribute.
    auto kernelResult = ttmlir::utils::getPairOfInteger<int32_t>(kernel);
    if (!kernelResult) {
      return rewriter.notifyMatchFailure(
          op, "kernel must be an integer or array attribute");
    }
    auto [kernelH, kernelW] = *kernelResult;

    // Parse dilation attribute.
    auto dilationResult = ttmlir::utils::getPairOfInteger<int32_t>(dilation);
    if (!dilationResult) {
      return rewriter.notifyMatchFailure(
          op, "dilation must be an integer or array attribute");
    }
    auto [dilationH, dilationW] = *dilationResult;
    assert(dilationH == 1 && dilationW == 1 && "dilation must be 1x1");

    bool countIncludePad = adaptor.getCountIncludePad();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    Type elementType = resultType.getElementType();
    int64_t inputH = inputType.getShape()[1];
    int64_t inputW = inputType.getShape()[2];

    // Calculate extra padding needed for output size alignment.
    paddingBottom += calculateExtraPadding(inputH, kernelH, strideH, paddingTop,
                                           paddingBottom, dilationH);
    paddingRight += calculateExtraPadding(inputW, kernelW, strideW, paddingLeft,
                                          paddingRight, dilationW);

    // Calculate output spatial dimensions.
    int64_t outputH =
        (inputH + paddingTop + paddingBottom - kernelH) / strideH + 1;
    int64_t outputW =
        (inputW + paddingLeft + paddingRight - kernelW) / strideW + 1;

    // Compute actual result shape (may differ from expected due to extra
    // padding).
    SmallVector<int64_t> actualResultShape(resultType.getShape());
    actualResultShape[1] = outputH;
    actualResultShape[2] = outputW;
    auto actualResultType =
        RankedTensorType::get(actualResultShape, elementType);

    // Use sum pooling + division for both count_include_pad cases.
    // The difference is only in how the divisor is computed:
    // - count_include_pad=true: constant divisor (kernel_h * kernel_w)
    // - count_include_pad=false: sum-pool a ones tensor to get per-position
    // counts
    int64_t batch = inputType.getShape()[0];
    int64_t channels = inputType.getShape()[3];
    int64_t paddedH = inputH + paddingTop + paddingBottom;
    int64_t paddedW = inputW + paddingLeft + paddingRight;

    bool hasPadding = paddingTop > 0 || paddingBottom > 0 || paddingLeft > 0 ||
                      paddingRight > 0;

    // Create zero constant for padding and fill operations.
    Value zeroVal;
    if (isa<FloatType>(elementType)) {
      zeroVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elementType, 0.0));
    } else {
      zeroVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(elementType, 0));
    }

    // Create padding attributes.
    SmallVector<OpFoldResult> lowPad = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(paddingTop),
        rewriter.getIndexAttr(paddingLeft), rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> highPad = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(paddingBottom),
        rewriter.getIndexAttr(paddingRight), rewriter.getIndexAttr(0)};
    auto paddedType =
        RankedTensorType::get({batch, paddedH, paddedW, channels}, elementType);

    // Pad the input tensor if needed.
    Value paddedInput = input;
    if (hasPadding) {
      paddedInput = rewriter.create<tensor::PadOp>(loc, paddedType, input,
                                                   lowPad, highPad, zeroVal);
    }

    // Create the kernel tensor (shape only, values don't matter for pooling).
    auto linalgKernelType =
        RankedTensorType::get({kernelH, kernelW}, rewriter.getF32Type());
    Value kernelTensor = rewriter.create<tensor::EmptyOp>(
        loc, linalgKernelType.getShape(), linalgKernelType.getElementType());

    // Create strides and dilations attributes for linalg.pooling_nhwc_sum.
    auto linalgStridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        ArrayRef<int64_t>{strideH, strideW});
    auto dilationsAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        ArrayRef<int64_t>{1, 1});

    // Create output tensor initialized to zero for sum accumulation.
    Value sumOutputInit = rewriter.create<tensor::EmptyOp>(
        loc, actualResultType.getShape(), elementType);
    Value sumOutput =
        rewriter.create<linalg::FillOp>(loc, zeroVal, sumOutputInit)
            .getResult(0);

    // Perform sum pooling on input.
    auto sumPoolOp = rewriter.create<linalg::PoolingNhwcSumOp>(
        loc, TypeRange{actualResultType}, ValueRange{paddedInput, kernelTensor},
        ValueRange{sumOutput}, linalgStridesAttr, dilationsAttr);
    Value sumResult = sumPoolOp.getResult(0);

    // Compute divisor tensor.
    Value divisorTensor;
    if (countIncludePad) {
      // Constant divisor: kernel_h * kernel_w.
      double divisorVal = static_cast<double>(kernelH * kernelW);
      Value divisorScalar;
      if (isa<FloatType>(elementType)) {
        divisorScalar = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(elementType, divisorVal));
      } else {
        divisorScalar = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(elementType,
                                         static_cast<int64_t>(divisorVal)));
      }
      divisorTensor = rewriter.create<tensor::SplatOp>(loc, actualResultType,
                                                       divisorScalar);
    } else {
      // Dynamic divisor: sum-pool a ones tensor to count non-padded elements.
      Value oneVal;
      if (isa<FloatType>(elementType)) {
        oneVal = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(elementType, 1.0));
      } else {
        oneVal = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(elementType, 1));
      }

      // Create ones tensor with same shape as input.
      Value onesInit = rewriter.create<tensor::EmptyOp>(
          loc, inputType.getShape(), elementType);
      Value onesTensor =
          rewriter.create<linalg::FillOp>(loc, oneVal, onesInit).getResult(0);

      // Pad ones tensor with zeros.
      Value paddedOnes = rewriter.create<tensor::PadOp>(
          loc, paddedType, onesTensor, lowPad, highPad, zeroVal);

      // Create output tensor for count accumulation.
      Value countOutputInit = rewriter.create<tensor::EmptyOp>(
          loc, actualResultType.getShape(), elementType);
      Value countOutput =
          rewriter.create<linalg::FillOp>(loc, zeroVal, countOutputInit)
              .getResult(0);

      // Perform sum pooling on ones tensor to get counts.
      auto countPoolOp = rewriter.create<linalg::PoolingNhwcSumOp>(
          loc, TypeRange{actualResultType},
          ValueRange{paddedOnes, kernelTensor}, ValueRange{countOutput},
          linalgStridesAttr, dilationsAttr);
      divisorTensor = countPoolOp.getResult(0);
    }

    // Divide sum by divisor to get average.
    Value avgOutputInit = rewriter.create<tensor::EmptyOp>(
        loc, actualResultType.getShape(), elementType);
    auto divOp = rewriter.create<linalg::DivOp>(
        loc, actualResultType, ValueRange{sumResult, divisorTensor},
        avgOutputInit);
    Value result = divOp.getResult(0);

    // Slice the result back to the original expected shape if needed.
    if (outputH != resultType.getShape()[1] ||
        outputW != resultType.getShape()[2]) {
      SmallVector<OpFoldResult> offsets, sizes, sliceStrides;
      for (int64_t i = 0; i < resultType.getRank(); ++i) {
        offsets.push_back(rewriter.getI64IntegerAttr(0));
        sizes.push_back(rewriter.getI64IntegerAttr(resultType.getShape()[i]));
        sliceStrides.push_back(rewriter.getI64IntegerAttr(1));
      }
      result = rewriter.create<tensor::ExtractSliceOp>(
          loc, resultType, result, offsets, sizes, sliceStrides);

      // Copy the result into the output buffer.
      Value output = rewriter.create<ttir::EmptyOp>(
          loc, op.getType().getShape(), op.getType().getElementType());
      auto copyResult = rewriter.create<linalg::CopyOp>(loc, result, output);
      rewriter.replaceOp(op, copyResult);
      return success();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class GlobalAvgPool2dOpConversionPattern
    : public OpConversionPattern<ttir::GlobalAvgPool2dOp> {
public:
  using OpConversionPattern<ttir::GlobalAvgPool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GlobalAvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    int64_t inputHeight = inputType.getShape()[1];
    int64_t inputWidth = inputType.getShape()[2];

    // Global average pooling is equivalent to sum reduction over H,W followed
    // by division by (H * W). We use MeanOp-style implementation but reduce
    // only over dimensions 1 and 2 (H and W).

    // First, reduce along height dimension (dim 1).
    auto afterHeightReduceShape = resultType.getShape().vec();
    afterHeightReduceShape[1] = 1;
    afterHeightReduceShape[2] = inputWidth;
    auto heightReduceType = RankedTensorType::get(afterHeightReduceShape,
                                                  resultType.getElementType());

    auto heightAxisAttr = rewriter.getI32IntegerAttr(1);
    auto heightReduceResult = rewriter.create<tosa::ReduceSumOp>(
        loc, heightReduceType, input, heightAxisAttr);

    // Then, reduce along width dimension (dim 2).
    auto widthAxisAttr = rewriter.getI32IntegerAttr(2);
    auto widthReduceResult = rewriter.create<tosa::ReduceSumOp>(
        loc, resultType, heightReduceResult.getResult(), widthAxisAttr);

    // Divide by the total number of spatial elements (H * W).
    double spatialCount = static_cast<double>(inputHeight * inputWidth);
    auto elementType = resultType.getElementType();

    // Create a constant tensor with 1/spatialCount for division.
    auto divisorAttr = rewriter.getFloatAttr(elementType, 1.0 / spatialCount);
    auto divisorValue = rewriter.create<arith::ConstantOp>(loc, divisorAttr);
    auto divisorTensor = rewriter.create<tensor::SplatOp>(
        loc, resultType, divisorValue.getResult());

    // Create shift tensor for tosa::MulOp (requires i8 tensor).
    auto shiftType = RankedTensorType::get({1}, rewriter.getI8Type());
    auto shiftAttr =
        DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
    Value shift = rewriter.create<tosa::ConstOp>(loc, shiftType, shiftAttr);

    // Multiply by reciprocal to get average.
    auto result = rewriter.create<tosa::MulOp>(
        loc, resultType, widthReduceResult.getResult(), divisorTensor, shift);

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};
} // namespace

namespace {
class GatherOpConversionPattern : public OpConversionPattern<ttir::GatherOp> {
public:
  using OpConversionPattern<ttir::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return decomposeToLinalg(op, adaptor, rewriter);
  }

private:
  LogicalResult decomposeToLinalg(ttir::GatherOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Value startIndices = adaptor.getStartIndices();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto indicesType = cast<RankedTensorType>(startIndices.getType());
    auto resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    // Extract attributes
    auto offsetDims = op.getOffsetDims();
    auto collapsedSliceDims = op.getCollapsedSliceDims();
    auto startIndexMap = op.getStartIndexMap();
    auto indexVectorDim = op.getIndexVectorDim();

    // Create initial tensor for result
    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Build indexing maps for the generic op
    auto resultRank = resultType.getRank();

    // Create identity map for output
    SmallVector<AffineExpr> outputExprs;
    for (int i = 0; i < resultRank; ++i) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    AffineMap outputMap =
        AffineMap::get(resultRank, 0, outputExprs, rewriter.getContext());
    SmallVector<AffineMap> indexingMaps = {outputMap};

    // All dimensions are parallel for gather
    SmallVector<utils::IteratorType> iteratorTypes(
        resultRank, utils::IteratorType::parallel);

    // Create the indexing logic using linalg.generic
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{}, ValueRange{initTensor}, indexingMaps,
        iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] is the current value in the output tensor

          // Get the current output indices
          SmallVector<Value> outputIndices;
          for (int i = 0; i < resultRank; ++i) {
            outputIndices.push_back(b.create<linalg::IndexOp>(loc, i));
          }

          // Build the input indices for the gather
          SmallVector<Value> inputIndices(inputType.getRank());

          // Initialize all indices to zero first to avoid null values
          for (int64_t i = 0; i < inputType.getRank(); ++i) {
            inputIndices[i] = b.create<arith::ConstantIndexOp>(loc, 0);
          }

          // Determine which output dimensions are batch dimensions
          SmallVector<int64_t> batchDims;
          for (int64_t i = 0; i < resultRank; ++i) {
            if (!llvm::is_contained(offsetDims, i)) {
              batchDims.push_back(i);
            }
          }

          // Extract indices from startIndices tensor for each dimension in
          // startIndexMap
          for (size_t i = 0; i < startIndexMap.size(); ++i) {
            SmallVector<Value> fullIndices;

            // Build indices based on the structure of the indices tensor
            if (indexVectorDim == 0 && indicesType.getRank() == 1) {
              // Special case: 1D indices tensor with index_vector_dim=0
              // For gathering, we use the first batch dimension (output dim 0)
              fullIndices.push_back(outputIndices[0]);
            } else if (indexVectorDim ==
                       static_cast<int64_t>(indicesType.getRank())) {
              // Index vector is implicit (size 1)
              // Use batch dimensions from output
              for (auto batchDim : batchDims) {
                fullIndices.push_back(outputIndices[batchDim]);
              }
            } else {
              // Normal case: index vector is at a specific dimension
              // Build indices from batch dimensions
              int batchIdx = 0;
              for (int64_t d = 0; d < indicesType.getRank(); ++d) {
                if (d == indexVectorDim) {
                  // This is the index vector dimension
                  fullIndices.push_back(
                      b.create<arith::ConstantIndexOp>(loc, i));
                } else {
                  // This is a batch dimension
                  if (static_cast<size_t>(batchIdx) < batchDims.size()) {
                    fullIndices.push_back(outputIndices[batchDims[batchIdx]]);
                    batchIdx++;
                  }
                }
              }
            }

            // Extract the index value
            Value idxValue =
                b.create<tensor::ExtractOp>(loc, startIndices, fullIndices);

            // Convert to index type if needed
            Value idx;
            if (idxValue.getType().isF32()) {
              // First convert f32 to i32
              Value i32Val =
                  b.create<arith::FPToSIOp>(loc, b.getI32Type(), idxValue);
              // Then convert i32 to index
              idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), i32Val);
            } else if (idxValue.getType().isInteger(32)) {
              // Direct cast from i32 to index
              idx =
                  b.create<arith::IndexCastOp>(loc, b.getIndexType(), idxValue);
            } else if (idxValue.getType().isInteger(64)) {
              // Direct cast from i64 to index
              idx =
                  b.create<arith::IndexCastOp>(loc, b.getIndexType(), idxValue);
            } else {
              // Already index type
              idx = idxValue;
            }

            inputIndices[startIndexMap[i]] = idx;
          }

          // Map offset dimensions from output to input
          // For each offset dimension in the output, find the corresponding
          // input dimension
          for (size_t i = 0; i < offsetDims.size(); ++i) {
            int64_t outputDim = offsetDims[i];

            // Find the corresponding input dimension
            // We need to skip over gathered dimensions and collapsed dimensions
            int64_t inputDim = 0;
            int64_t nonCollapsedCount = 0;

            // Count non-collapsed dimensions until we reach the i-th one
            while (inputDim < inputType.getRank() &&
                   static_cast<size_t>(nonCollapsedCount) <= i) {
              if (!llvm::is_contained(collapsedSliceDims, inputDim) &&
                  !llvm::is_contained(startIndexMap, inputDim)) {
                if (static_cast<size_t>(nonCollapsedCount) == i) {
                  inputIndices[inputDim] = outputIndices[outputDim];
                  break;
                }
                nonCollapsedCount++;
              }
              inputDim++;
            }
          }

          // Extract the value from input tensor
          Value extracted =
              b.create<tensor::ExtractOp>(loc, input, inputIndices);

          b.create<linalg::YieldOp>(loc, extracted);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};
} // namespace

namespace {
class LogicalNotOpConversionPattern
    : public OpConversionPattern<ttir::LogicalNotOp> {
public:
  using OpConversionPattern<ttir::LogicalNotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LogicalNotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // First convert the input to a boolean tensor
    Value boolInput = convertToBooleanTensor(input, op.getLoc(), rewriter);

    // Get the boolean type for the intermediate result
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));

    // Apply logical not to the boolean tensor
    auto notResult =
        rewriter.create<tosa::LogicalNotOp>(op.getLoc(), boolType, boolInput);

    // Convert boolean result back to original type using cast.
    auto result =
        rewriter.create<tosa::CastOp>(op.getLoc(), resultType, notResult);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
// Logical binary operations pattern (LogicalAnd, LogicalOr, LogicalXor)
// These operations:
// 1. Convert float inputs to boolean (non-zero = true)
// 2. Apply the TOSA logical operation
// 3. Convert boolean result back to float (true = 1.0, false = 0.0)
template <typename TTIROpTy, typename TosaOpTy>
class LogicalBinaryOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Convert both inputs to boolean tensors.
    Value boolLhs = convertToBooleanTensor(lhs, op.getLoc(), rewriter);
    Value boolRhs = convertToBooleanTensor(rhs, op.getLoc(), rewriter);

    // Get the boolean type for the intermediate result.
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));

    // Apply the logical operation to the boolean tensors.
    auto logicalResult =
        rewriter.create<TosaOpTy>(op.getLoc(), boolType, boolLhs, boolRhs);

    // Convert boolean result back to original type using cast.
    auto result =
        rewriter.create<tosa::CastOp>(op.getLoc(), resultType, logicalResult);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class MinOpConversionPattern : public OpConversionPattern<ttir::MinOp> {
public:
  using OpConversionPattern<ttir::MinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Get dimensions to reduce and keep_dim attribute
    SmallVector<int64_t> dims = getDimsFromAttribute(op, rank);
    bool keepDim = getKeepDimFromAttribute(op);

    // Create a chain of reduction operations
    Value result = createReductionOpChain<tosa::ReduceMinOp>(
        input, resultType, dims, keepDim, op.getLoc(), rewriter);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class MaxOpConversionPattern : public OpConversionPattern<ttir::MaxOp> {
public:
  using OpConversionPattern<ttir::MaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Get dimensions to reduce and keep_dim attribute
    SmallVector<int64_t> dims = getDimsFromAttribute(op, rank);
    bool keepDim = getKeepDimFromAttribute(op);

    // Create a chain of reduction operations
    Value result = createReductionOpChain<tosa::ReduceMaxOp>(
        input, resultType, dims, keepDim, op.getLoc(), rewriter);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class SumOpConversionPattern : public OpConversionPattern<ttir::SumOp> {
public:
  using OpConversionPattern<ttir::SumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Get dimensions to reduce and keep_dim attribute
    SmallVector<int64_t> dims = getDimsFromAttribute(op, rank);
    bool keepDim = getKeepDimFromAttribute(op);

    // Create a chain of reduction operations
    Value result = createReductionOpChain<tosa::ReduceSumOp>(
        input, resultType, dims, keepDim, op.getLoc(), rewriter);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class ProdOpConversionPattern : public OpConversionPattern<ttir::ProdOp> {
public:
  using OpConversionPattern<ttir::ProdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ProdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Get dimensions to reduce and keep_dim attribute
    SmallVector<int64_t> dims = getDimsFromAttribute(op, rank);
    bool keepDim = getKeepDimFromAttribute(op);

    // Create a chain of reduction operations
    Value result = createReductionOpChain<tosa::ReduceProductOp>(
        input, resultType, dims, keepDim, op.getLoc(), rewriter);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
// ArgMax conversion pattern.
// TOSA has no ArgMax reduction op, so we implement this using
// linalg::GenericOp with two output tensors (max values + max indices).
// After TTIRToTTIRDecomposition, ArgMaxOp arrives here with either no dim_arg
// (reduce all dimensions) or exactly 1 reduce dim.
// The tracked index is a linearized (flattened) index across reduce dimensions.
class ArgMaxOpConversionPattern : public OpConversionPattern<ttir::ArgMaxOp> {
public:
  using OpConversionPattern<ttir::ArgMaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArgMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    Type elementType = inputType.getElementType();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // After TTIRToTTIRDecomposition, dim_arg is either absent (reduce all
    // dims) or has exactly 1 entry.
    auto dimArg = op.getDimArg();
    assert((!dimArg || dimArg->size() <= 1) &&
           "Multi-dim argmax should have been decomposed");

    SmallVector<int64_t> reduceDims = getDimsFromAttribute(op, rank);
    bool keepDim = getKeepDimFromAttribute(op);

    // Compute the output shape (without keep_dim) and classify each dim.
    SmallVector<int64_t> reducedShape;
    SmallVector<utils::IteratorType> iteratorTypes;
    SmallVector<AffineExpr> outputExprs;
    for (int64_t i = 0; i < rank; ++i) {
      if (llvm::is_contained(reduceDims, i)) {
        iteratorTypes.push_back(utils::IteratorType::reduction);
      } else {
        iteratorTypes.push_back(utils::IteratorType::parallel);
        reducedShape.push_back(inputType.getShape()[i]);
        outputExprs.push_back(rewriter.getAffineDimExpr(i));
      }
    }

    auto maxValuesType = RankedTensorType::get(reducedShape, elementType);
    auto maxIndicesType =
        RankedTensorType::get(reducedShape, rewriter.getI32Type());

    // Initialize max values to -inf and max indices to 0.
    auto negInfAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getInf(cast<FloatType>(elementType).getFloatSemantics(),
                        /*Negative=*/true));
    Value negInf =
        rewriter.create<arith::ConstantOp>(loc, elementType, negInfAttr);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    Value maxValuesFilled =
        rewriter
            .create<linalg::FillOp>(
                loc, negInf,
                rewriter.create<tensor::EmptyOp>(loc, reducedShape, elementType)
                    .getResult())
            .getResult(0);
    Value maxIndicesFilled =
        rewriter
            .create<linalg::FillOp>(
                loc, zero,
                rewriter
                    .create<tensor::EmptyOp>(loc, reducedShape,
                                             rewriter.getI32Type())
                    .getResult())
            .getResult(0);

    // Indexing maps: identity for input, projection for outputs.
    AffineMap inputMap =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    AffineMap outputMap =
        AffineMap::get(rank, 0, outputExprs, rewriter.getContext());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{maxValuesType, maxIndicesType}, ValueRange{input},
        ValueRange{maxValuesFilled, maxIndicesFilled},
        SmallVector<AffineMap>{inputMap, outputMap, outputMap}, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value currentVal = args[0];
          Value currentMax = args[1];
          Value currentIdx = args[2];

          // Compute linearized index across reduce dimensions (row-major).
          Value linearIdx = nullptr;
          for (int64_t d : reduceDims) {
            Value idx = b.create<arith::IndexCastOp>(
                loc, b.getI32Type(), b.create<linalg::IndexOp>(loc, d));
            if (!linearIdx) {
              linearIdx = idx;
            } else {
              Value dimSize = b.create<arith::ConstantOp>(
                  loc, b.getI32Type(),
                  b.getI32IntegerAttr(inputType.getShape()[d]));
              linearIdx = b.create<arith::AddIOp>(
                  loc, b.create<arith::MulIOp>(loc, linearIdx, dimSize), idx);
            }
          }

          Value isGreater = b.create<arith::CmpFOp>(
              loc, arith::CmpFPredicate::OGT, currentVal, currentMax);
          Value newMax =
              b.create<arith::SelectOp>(loc, isGreater, currentVal, currentMax);
          Value newIdx =
              b.create<arith::SelectOp>(loc, isGreater, linearIdx, currentIdx);
          b.create<linalg::YieldOp>(loc, ValueRange{newMax, newIdx});
        });

    Value result = genericOp.getResult(1);

    // If keep_dim, reshape to reinsert reduced dimensions as size-1.
    if (keepDim) {
      SmallVector<int64_t> keepDimShape;
      for (int64_t i = 0; i < rank; ++i) {
        keepDimShape.push_back(
            llvm::is_contained(reduceDims, i) ? 1 : inputType.getShape()[i]);
      }
      auto shapeType =
          tosa::shapeType::get(rewriter.getContext(), keepDimShape.size());
      auto shapeOp = rewriter.create<tosa::ConstShapeOp>(
          loc, shapeType, rewriter.getIndexTensorAttr(keepDimShape));
      result =
          rewriter.create<tosa::ReshapeOp>(loc, resultType, result, shapeOp);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
// CumSum conversion pattern.
// Cumulative sum computes the running sum along a specified dimension.
// Since this is a scan operation (not a reduction), we cannot use the standard
// TOSA reduce operations. Instead, we unroll the scan at compile time,
// generating a sequence of slice+add+insert operations for each position
// along the scan dimension.
class CumSumOpConversionPattern : public OpConversionPattern<ttir::CumSumOp> {
public:
  using OpConversionPattern<ttir::CumSumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::CumSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Normalize dimension to be positive.
    int64_t dim = normalizeDim(op.getDim(), rank);

    int64_t dimSize = inputType.getShape()[dim];
    Type elementType = inputType.getElementType();

    // Create an initial output tensor filled with zeros.
    DenseElementsAttr zeroAttr = createDenseElementsAttr(resultType, 0.0);
    if (!zeroAttr) {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported element type for cumsum");
    }
    Value output =
        rewriter.create<arith::ConstantOp>(loc, resultType, zeroAttr);

    // Compute the slice type (same shape but with dim size = 1).
    SmallVector<int64_t> sliceShape(inputType.getShape());
    sliceShape[dim] = 1;
    auto sliceType = RankedTensorType::get(sliceShape, elementType);

    // Create a zero-filled tensor for the running sum accumulator.
    DenseElementsAttr zeroSliceAttr = createDenseElementsAttr(sliceType, 0.0);
    Value runningSum =
        rewriter.create<arith::ConstantOp>(loc, sliceType, zeroSliceAttr);

    // Build the static sizes and strides for slice operations.
    SmallVector<OpFoldResult> staticSizes;
    SmallVector<OpFoldResult> staticStrides(rank, rewriter.getIndexAttr(1));
    for (int64_t i = 0; i < rank; ++i) {
      if (i == dim) {
        staticSizes.push_back(rewriter.getIndexAttr(1));
      } else {
        staticSizes.push_back(rewriter.getIndexAttr(inputType.getShape()[i]));
      }
    }

    // Unroll the scan loop at compile time.
    // For each position along the scan dimension, extract the slice, add to
    // running sum, and insert into output.
    for (int64_t idx = 0; idx < dimSize; ++idx) {
      // Build offsets for this iteration.
      SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
      offsets[dim] = rewriter.getIndexAttr(idx);

      // Extract the current slice from input.
      Value inputSlice = rewriter.create<tensor::ExtractSliceOp>(
          loc, sliceType, input, offsets, staticSizes, staticStrides);

      // Add current input slice to running sum.
      auto emptySlice =
          rewriter.create<tensor::EmptyOp>(loc, sliceShape, elementType);
      auto addOp = rewriter.create<linalg::AddOp>(
          loc, sliceType, ValueRange{runningSum, inputSlice},
          emptySlice.getResult());
      runningSum = addOp.getResult(0);

      // Insert the new sum into the output tensor at the current position.
      output = rewriter.create<tensor::InsertSliceOp>(
          loc, runningSum, output, offsets, staticSizes, staticStrides);
    }

    rewriter.replaceOp(op, output);
    return success();
  }
};
} // namespace

namespace {
// ConcatenateHeads conversion pattern.
// This operation concatenates multiple heads of a multi-head attention tensor.
// Input shape: [batch_size, num_heads, sequence_size, head_size]
// Output shape: [batch_size, sequence_size, num_heads * head_size]
// It is equivalent to: permute([0, 2, 1, 3]) + reshape
class ConcatenateHeadsOpConversionPattern
    : public OpConversionPattern<ttir::ConcatenateHeadsOp> {
public:
  using OpConversionPattern<ttir::ConcatenateHeadsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConcatenateHeadsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    Type elementType = inputType.getElementType();

    // ConcatenateHeads expects exactly 4D input:
    // [batch_size, num_heads, sequence_size, head_size].
    if (inputType.getRank() != 4) {
      return rewriter.notifyMatchFailure(
          op, "ConcatenateHeads requires 4D input tensor.");
    }

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Input: [batch_size, num_heads, sequence_size, head_size]
    // Step 1: Transpose to [batch_size, sequence_size, num_heads, head_size]
    SmallVector<int32_t> permutation = {0, 2, 1, 3};
    SmallVector<int64_t> transposedShape = {inputShape[0], inputShape[2],
                                            inputShape[1], inputShape[3]};
    auto transposedType = RankedTensorType::get(transposedShape, elementType);

    auto transposeOp = rewriter.create<tosa::TransposeOp>(loc, transposedType,
                                                          input, permutation);

    // Step 2: Reshape to [batch_size, sequence_size, num_heads * head_size]
    ArrayRef<int64_t> outputShape = resultType.getShape();
    SmallVector<int64_t> newShapeValues(outputShape.begin(), outputShape.end());
    auto shapeType =
        tosa::shapeType::get(rewriter.getContext(), outputShape.size());
    auto attr = rewriter.getIndexTensorAttr(newShapeValues);
    auto shapeOp = rewriter.create<tosa::ConstShapeOp>(loc, shapeType, attr);

    auto reshapeOp =
        rewriter.create<tosa::ReshapeOp>(loc, resultType, transposeOp, shapeOp);

    rewriter.replaceOp(op, reshapeOp);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Linalg Conversions Patterns
//===----------------------------------------------------------------------===//
namespace {
// Conversion pattern of operations which have exactly 2 input and 1 output
// operands.
template <typename TTIROpTy, typename LinalgOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ElementwiseBinaryOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    RankedTensorType lhsType =
        cast<RankedTensorType>(adaptor.getLhs().getType());
    RankedTensorType rhsType =
        cast<RankedTensorType>(adaptor.getRhs().getType());

    // First, compute broadcasted shape from operands.

    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();

    SmallVector<int64_t> broadcastedShape;
    if (!OpTrait::util::getBroadcastedShape(lhsShape, rhsShape,
                                            broadcastedShape)) {
      return rewriter.notifyMatchFailure(op, "Operands are not broadcastable!");
    }

    // Rewrite inputs to target dims with broadcast and collapse shape ops, as
    // needed.
    SmallVector<Value, 2> inputs{adaptor.getLhs(), adaptor.getRhs()};
    SmallVector<Value, 2> broadcastedInputs;
    for (Value input : inputs) {
      auto inputRankedTensorType = dyn_cast<RankedTensorType>(input.getType());
      assert(inputRankedTensorType &&
             "Binary element-wise operations must be ranked tensor types!");

      // Insert and use a broadcast op if input does not perfectly match target
      // shape.
      SmallVector<int64_t, 2> broadcastDims =
          getBroadcastDims(inputRankedTensorType.getShape(), broadcastedShape);

      // If we need to broadcast along all dims, then we need to collapse to a
      // scalar via empty collapseDimGroups.
      SmallVector<SmallVector<int64_t, 2>, 2> collapseDimGroups =
          (broadcastDims.size() != broadcastedShape.size())
              ? getCollapseDims(inputRankedTensorType.getShape(),
                                broadcastedShape)
              : SmallVector<SmallVector<int64_t, 2>, 2>();

      if (!broadcastDims.empty()) {
        Value broadcastInput = input;
        // The broadcast op requires we actually collapse any dimensions with
        // size 1 we want to broadcast along.
        if (collapseDimGroups.size() !=
            inputRankedTensorType.getShape().size()) {
          broadcastInput = rewriter.create<tensor::CollapseShapeOp>(
              loc, input, collapseDimGroups);
        }
        auto initTensor = rewriter.create<ttir::EmptyOp>(
            loc, broadcastedShape, inputRankedTensorType.getElementType());
        auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
            loc, broadcastInput, initTensor.getResult(), broadcastDims);
        broadcastedInputs.push_back(broadcastOp.getResults().front());
      } else {
        broadcastedInputs.push_back(input);
      }
    }

    // Perform the actual op substitution, using broadcasted operands when
    // needed.
    auto resultType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));

    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    rewriter.replaceOpWithNewOp<LinalgOpTy>(op, resultType, broadcastedInputs,
                                            output.getResult());
    return success();
  }
};
} // namespace

namespace {
// General elementwise conversion pattern, without implicit broadcasting etc.
// Appropriate for unary ops, or other ops without broadcasting.
template <typename TTIROpTy, typename LinAlgOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ElementwiseOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));

    auto inputs = adaptor.getOperands();
    auto output = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                   resultType.getElementType());
    rewriter.replaceOpWithNewOp<LinAlgOpTy>(op, resultType, inputs,
                                            output.getResult());
    return success();
  }
};
} // namespace

namespace {
// Decomposes softmax into elementary operations that can be lowered through
// linalg-to-loops. linalg.softmax cannot be lowered directly because it
// implements AggregatedOpInterface rather than LinalgStructuredInterface.
//
// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
//
// Steps:
// 1. max = ReduceMax(input, dim) - for numerical stability
// 2. shifted = input - max (broadcast)
// 3. exp_vals = exp(shifted)
// 4. sum_exp = ReduceSum(exp_vals, dim)
// 5. result = exp_vals / sum_exp (broadcast)
class SoftmaxOpConversionPattern : public OpConversionPattern<ttir::SoftmaxOp> {
public:
  using OpConversionPattern<ttir::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    Type elementType = inputType.getElementType();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Normalize dimension to be positive.
    int64_t dim = normalizeDim(op.getDimension(), rank);

    // Create reduced shape (with dim size = 1 for broadcasting).
    SmallVector<int64_t> reducedShape(inputType.getShape());
    reducedShape[dim] = 1;
    auto reducedType = RankedTensorType::get(reducedShape, elementType);

    // Step 1: Compute max along dimension for numerical stability.
    auto axisAttr = rewriter.getI32IntegerAttr(dim);
    Value maxVal =
        rewriter.create<tosa::ReduceMaxOp>(loc, reducedType, input, axisAttr);

    // Step 2: Subtract max from input (input - max).
    // tosa::SubOp handles broadcasting automatically.
    Value shifted = rewriter.create<tosa::SubOp>(loc, inputType, input, maxVal);

    // Step 3: Compute exp(shifted).
    Value expVals = rewriter.create<tosa::ExpOp>(loc, inputType, shifted);

    // Step 4: Compute sum of exp along dimension.
    Value sumExp =
        rewriter.create<tosa::ReduceSumOp>(loc, reducedType, expVals, axisAttr);

    // Step 5: Divide exp by sum (exp / sum).
    // Use reciprocal and multiply with broadcasting.
    Value reciprocal =
        rewriter.create<tosa::ReciprocalOp>(loc, reducedType, sumExp);

    // tosa::MulOp requires a shift tensor (0 for float ops).
    auto shiftType = RankedTensorType::get({1}, rewriter.getI8Type());
    auto shiftAttr =
        DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
    Value shift = rewriter.create<tosa::ConstOp>(loc, shiftType, shiftAttr);

    Value result = rewriter.create<tosa::MulOp>(loc, resultType, expVals,
                                                reciprocal, shift);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class ReluOpConversionPattern : public OpConversionPattern<ttir::ReluOp> {
public:
  using OpConversionPattern<ttir::ReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getInput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));

    assert(resultType && "Result type must be a ranked tensor type.");

    DenseElementsAttr zeroAttr = createDenseElementsAttr(resultType, 0);
    if (!zeroAttr) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported element type for ReLU zero constant");
    }

    auto zeroes =
        rewriter.create<arith::ConstantOp>(op.getLoc(), resultType, zeroAttr);

    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::MaxOp>(
        op, resultType, ValueRange{input, zeroes.getResult()},
        ValueRange{output});
    return success();
  }
};
} // namespace

namespace {
class Relu6OpConversionPattern : public OpConversionPattern<ttir::Relu6Op> {
public:
  using OpConversionPattern<ttir::Relu6Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::Relu6Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto elementType = resultType.getElementType();
    TypedAttr minAttr, maxAttr;

    if (isa<FloatType>(elementType)) {
      minAttr = rewriter.getFloatAttr(elementType, 0.0);
      maxAttr = rewriter.getFloatAttr(elementType, 6.0);
    } else if (isa<IntegerType>(elementType)) {
      minAttr = rewriter.getIntegerAttr(elementType, 0);
      maxAttr = rewriter.getIntegerAttr(elementType, 6);
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported element type for ReLU6");
    }

    rewriter.replaceOpWithNewOp<tosa::ClampOp>(op, resultType, input, minAttr,
                                               maxAttr);
    return success();
  }
};
} // namespace

namespace {
class EmptyOpConversionPattern : public OpConversionPattern<ttir::EmptyOp> {
public:
  using OpConversionPattern<ttir::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        /*dynamicSizes*/ ValueRange());
    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.permute operation
class PermuteOpConversionPattern : public OpConversionPattern<ttir::PermuteOp> {
public:
  using OpConversionPattern<ttir::PermuteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::PermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    Value input = adaptor.getInput();
    llvm::ArrayRef<int64_t> permutation = op.getPermutation();

    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, input, output.getResult(), permutation);

    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.slice_static operation.
class SliceStaticOpConversionPattern
    : public OpConversionPattern<ttir::SliceStaticOp> {
public:
  using OpConversionPattern<ttir::SliceStaticOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SliceStaticOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    assert(inputType && "Input must be a ranked tensor type.");

    SmallVector<OpFoldResult> offsets, sizes, strides;

    ArrayAttr begins = op.getBegins();
    ArrayAttr ends = op.getEnds();
    ArrayAttr steps = op.getStep();

    assert(begins.size() == ends.size() && begins.size() == steps.size() &&
           "Invalid slice attributes.");

    for (unsigned i = 0; i < begins.size(); ++i) {
      const int32_t beginVal = llvm::cast<IntegerAttr>(begins[i]).getInt();
      const int32_t endVal = llvm::cast<IntegerAttr>(ends[i]).getInt();
      const int32_t stepVal = llvm::cast<IntegerAttr>(steps[i]).getInt();

      offsets.push_back(rewriter.getI64IntegerAttr(beginVal));

      // Calculate size: (end - begin + step - 1) / step
      int64_t size = (endVal - beginVal);
      if (stepVal != 0) {
        size = (size + stepVal - 1) / stepVal;
      }
      sizes.push_back(rewriter.getI64IntegerAttr(size));

      strides.push_back(rewriter.getI64IntegerAttr(stepVal));
    }

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Create the extract_slice operation
    Value extractedSlice = rewriter.create<tensor::ExtractSliceOp>(
        op.getLoc(), resultType, input, offsets, sizes, strides);

    // Since tensor::ExtractSliceOp doesn't support DPS, we need to copy
    // the result into the output buffer
    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    auto copyResult = rewriter.create<linalg::CopyOp>(
        op.getLoc(), extractedSlice, output.getResult());
    rewriter.replaceOp(op, copyResult);

    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.pad operation.
class PadOpConversionPattern : public OpConversionPattern<ttir::PadOp> {
public:
  using OpConversionPattern<ttir::PadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    assert(inputType && "Input must be a ranked tensor type.");

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Get padding attribute: format is [dim0_low, dim0_high, dim1_low,
    // dim1_high, ...]
    ArrayRef<int32_t> paddingArray = op.getPadding();
    int64_t rank = inputType.getRank();
    assert(static_cast<int64_t>(paddingArray.size()) == 2 * rank &&
           "Padding size must be 2 * rank.");

    // Extract low and high padding for each dimension.
    SmallVector<OpFoldResult> lowPad, highPad;
    for (int64_t i = 0; i < rank; ++i) {
      lowPad.push_back(rewriter.getIndexAttr(paddingArray[2 * i]));
      highPad.push_back(rewriter.getIndexAttr(paddingArray[2 * i + 1]));
    }

    // Get the padding value and create a constant.
    float padValue = op.getValue().convertToFloat();
    Type elementType = inputType.getElementType();
    Value padConstant;
    if (isa<FloatType>(elementType)) {
      padConstant = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getFloatAttr(elementType, padValue));
    } else {
      padConstant = rewriter.create<arith::ConstantOp>(
          op.getLoc(),
          rewriter.getIntegerAttr(elementType, static_cast<int64_t>(padValue)));
    }

    // Create tensor::PadOp and replace.
    rewriter.replaceOpWithNewOp<tensor::PadOp>(op, resultType, input, lowPad,
                                               highPad, padConstant);
    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.constant operation
class ConstantOpConversionPattern
    : public OpConversionPattern<ttir::ConstantOp> {
public:
  using OpConversionPattern<ttir::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = adaptor.getValue();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto valueType = dyn_cast<RankedTensorType>(value.getType());
    assert(valueType && "Value type must be a ranked tensor type.");

    ElementsAttr convertedValue;

    // If types already match (e.g., after HoistCPUOps conversion), use value
    // directly.
    if (valueType == resultType) {
      convertedValue = value;
    } else if (auto denseAttr = dyn_cast<DenseElementsAttr>(value)) {
      // Use getFromRawBuffer to reinterpret the raw data with the new type.
      // This handles signedness conversions (si32/ui32 -> i32) which have the
      // same bit width.
      convertedValue = DenseElementsAttr::getFromRawBuffer(
          resultType, denseAttr.getRawData());
    } else if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(value)) {
      convertedValue = DenseResourceElementsAttr::get(
          resultType, resourceAttr.getRawHandle());
    } else {
      return rewriter.notifyMatchFailure(
          op, "Expected DenseElementsAttr or DenseResourceElementsAttr");
    }

    auto newConstant = rewriter.create<arith::ConstantOp>(
        op.getLoc(), resultType, convertedValue);

    rewriter.replaceOp(op, newConstant.getResult());
    return success();
  }
};
} // namespace

namespace {
// Template conversion pattern for constant fill operations (zeros, ones).
template <typename TTIROpTy, int64_t FillValue>
class NamedFillOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    DenseElementsAttr fillAttr =
        createDenseElementsAttr(resultType, static_cast<double>(FillValue));
    if (!fillAttr) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported element type for constant fill");
    }

    auto constOp =
        rewriter.create<arith::ConstantOp>(op.getLoc(), resultType, fillAttr);

    rewriter.replaceOp(op, constOp.getResult());
    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.full operation.
class FullOpConversionPattern : public OpConversionPattern<ttir::FullOp> {
public:
  using OpConversionPattern<ttir::FullOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::FullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Extract numeric value as double.
    Attribute fillValue = adaptor.getFillValue();
    double value;
    if (auto floatAttr = dyn_cast<FloatAttr>(fillValue)) {
      value = floatAttr.getValue().convertToDouble();
    } else if (auto intAttr = dyn_cast<IntegerAttr>(fillValue)) {
      value = static_cast<double>(intAttr.getInt());
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported fill_value type");
    }

    // Create DenseElementsAttr with appropriate element type.
    DenseElementsAttr fillAttr = createDenseElementsAttr(resultType, value);
    if (!fillAttr) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported element type for full fill");
    }

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, fillAttr);
    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.arange operation.
// Generates a tensor with evenly spaced values within a given interval.
class ArangeOpConversionPattern : public OpConversionPattern<ttir::ArangeOp> {
public:
  using OpConversionPattern<ttir::ArangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));

    // ArangeForceLastDimensionPattern ensures arange is always 1D.
    assert(resultType.getRank() == 1 &&
           "Arange must be 1D after decomposition");

    int64_t start = adaptor.getStart();
    int64_t step = adaptor.getStep();
    Type elementType = resultType.getElementType();

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elementType);

    AffineMap outputMap = rewriter.getDimIdentityMap();
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel};

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{}, ValueRange{initTensor},
        SmallVector<AffineMap>{outputMap}, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value idx = b.create<linalg::IndexOp>(loc, 0);

          // Compute: start + idx * step
          Value result;
          if (isa<FloatType>(elementType)) {
            Value idxFloat =
                b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
            Value idxFP = b.create<arith::SIToFPOp>(loc, elementType, idxFloat);
            Value startVal = b.create<arith::ConstantOp>(
                loc, b.getFloatAttr(elementType, static_cast<double>(start)));
            Value stepVal = b.create<arith::ConstantOp>(
                loc, b.getFloatAttr(elementType, static_cast<double>(step)));
            Value scaled = b.create<arith::MulFOp>(loc, idxFP, stepVal);
            result = b.create<arith::AddFOp>(loc, startVal, scaled);
          } else {
            Value idxInt = b.create<arith::IndexCastOp>(loc, elementType, idx);
            Value startVal = b.create<arith::ConstantOp>(
                loc, b.getIntegerAttr(elementType, start));
            Value stepVal = b.create<arith::ConstantOp>(
                loc, b.getIntegerAttr(elementType, step));
            Value scaled = b.create<arith::MulIOp>(loc, idxInt, stepVal);
            result = b.create<arith::AddIOp>(loc, startVal, scaled);
          }

          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};
} // namespace

namespace {
class MeanOpConversionPattern : public OpConversionPattern<ttir::MeanOp> {
public:
  using OpConversionPattern<ttir::MeanOp>::OpConversionPattern;

  // Mean op is a reduction operation that calculates the average value of a
  // tensor along a specified dimension. Tosa has reduction ops to calculate the
  // sum of a tensor along a specified dimension. Sum reduction op can be
  // divided by the number of elements being reduced to get the average.
  LogicalResult
  matchAndRewrite(ttir::MeanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    SmallVector<int64_t> dims = getDimsFromAttribute(op, rank);
    for (size_t i = 0; i < dims.size(); i++) {
      if (dims[i] < 0) {
        dims[i] += inputType.getRank();
      }
    }
    bool keepDim = getKeepDimFromAttribute(op);

    Value sum = createReductionOpChain<tosa::ReduceSumOp>(
        input, resultType, dims, keepDim, op.getLoc(), rewriter);

    int64_t numElements = 1;
    for (int64_t dim : dims) {
      numElements *= inputType.getShape()[dim];
    }

    DenseElementsAttr divisorAttr =
        createDenseElementsAttr(resultType, static_cast<double>(numElements));
    if (!divisorAttr) {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported element type for mean");
    }

    auto divisor =
        rewriter.create<tosa::ConstOp>(op.getLoc(), resultType, divisorAttr);

    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());

    auto divOp = rewriter.create<linalg::DivOp>(
        op.getLoc(), resultType, ValueRange{sum, divisor}, output.getResult());

    rewriter.replaceOp(op, divOp.getResult(0));
    return success();
  }
};
} // namespace

namespace {
// Decomposes LayerNorm into elementary operations that can be lowered through
// linalg-to-loops.
//
// layer_norm(x, weight, bias, epsilon) =
//   ((x - mean(x)) / sqrt(var(x) + epsilon)) * weight + bias
//
// Steps:
// 1. mean = ReduceSum(input, dims) / num_elements
// 2. centered = input - mean (broadcast)
// 3. variance = ReduceSum(centered^2, dims) / num_elements
// 4. inv_std = rsqrt(variance + epsilon)
// 5. normalized = centered * inv_std (broadcast)
// 6. if weight: result = normalized * weight
// 7. if bias: result = result + bias
class LayerNormOpConversionPattern
    : public OpConversionPattern<ttir::LayerNormOp> {
public:
  using OpConversionPattern<ttir::LayerNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LayerNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    Type elementType = inputType.getElementType();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Get normalized_shape to determine which dimensions to reduce over.
    // normalized_shape specifies the shape of the dimensions to normalize,
    // which are always the last N dimensions of the input.
    ArrayRef<int64_t> normalizedShape = op.getNormalizedShape();
    int64_t numNormDims = normalizedShape.size();

    // The reduction dimensions are the last numNormDims dimensions.
    SmallVector<int64_t> reductionDims;
    for (int64_t i = rank - numNormDims; i < rank; ++i) {
      reductionDims.push_back(i);
    }

    // Calculate number of elements being reduced.
    int64_t numElements = 1;
    for (int64_t dim : reductionDims) {
      numElements *= inputType.getShape()[dim];
    }

    // Create reduced shape (with reduced dims = 1 for broadcasting).
    SmallVector<int64_t> reducedShape(inputType.getShape());
    for (int64_t dim : reductionDims) {
      reducedShape[dim] = 1;
    }
    auto reducedType = RankedTensorType::get(reducedShape, elementType);

    // Step 1: Compute mean = sum(input) / num_elements.
    // Use reduction op chain to sum over all reduction dimensions.
    Value sum = createReductionOpChain<tosa::ReduceSumOp>(
        input, reducedType, reductionDims, /*keepDim=*/true, loc, rewriter);

    // Create constant for division by num_elements.
    DenseElementsAttr numElementsAttr =
        createDenseElementsAttr(reducedType, static_cast<double>(numElements));
    if (!numElementsAttr) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported element type for layer norm");
    }
    Value numElementsConst =
        rewriter.create<tosa::ConstOp>(loc, reducedType, numElementsAttr);
    Value reciprocalN =
        rewriter.create<tosa::ReciprocalOp>(loc, reducedType, numElementsConst);

    // mean = sum * (1/N)
    auto shiftType = RankedTensorType::get({1}, rewriter.getI8Type());
    auto shiftAttr =
        DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
    Value shift = rewriter.create<tosa::ConstOp>(loc, shiftType, shiftAttr);

    Value mean =
        rewriter.create<tosa::MulOp>(loc, reducedType, sum, reciprocalN, shift);

    // Step 2: centered = input - mean (tosa broadcasts automatically).
    Value centered = rewriter.create<tosa::SubOp>(loc, inputType, input, mean);

    // Step 3: Compute variance = mean(centered^2).
    // First compute centered^2.
    Value centeredSquared =
        rewriter.create<tosa::MulOp>(loc, inputType, centered, centered, shift);

    // Sum of squared differences.
    Value sumSquared = createReductionOpChain<tosa::ReduceSumOp>(
        centeredSquared, reducedType, reductionDims, /*keepDim=*/true, loc,
        rewriter);

    // variance = sumSquared * (1/N)
    Value variance = rewriter.create<tosa::MulOp>(loc, reducedType, sumSquared,
                                                  reciprocalN, shift);

    // Step 4: Add epsilon for numerical stability.
    float epsilon = op.getEpsilon().convertToFloat();
    DenseElementsAttr epsilonAttr =
        createDenseElementsAttr(reducedType, static_cast<double>(epsilon));
    Value epsilonConst =
        rewriter.create<tosa::ConstOp>(loc, reducedType, epsilonAttr);
    Value variancePlusEps =
        rewriter.create<tosa::AddOp>(loc, reducedType, variance, epsilonConst);

    // Step 5: inv_std = rsqrt(variance + epsilon).
    Value invStd =
        rewriter.create<tosa::RsqrtOp>(loc, reducedType, variancePlusEps);

    // Step 6: normalized = centered * inv_std (tosa broadcasts automatically).
    Value normalized =
        rewriter.create<tosa::MulOp>(loc, resultType, centered, invStd, shift);

    // Step 7: Apply weight (gamma) if present.
    // Weight and bias need to be reshaped to match the input rank for TOSA ops.
    // They have shape [normalized_shape], need to prepend 1s to match input
    // rank.
    Value result = normalized;
    if (adaptor.getWeight()) {
      Value reshapedWeight = reshapeByPrependingOnes(
          adaptor.getWeight(), rank, numNormDims, elementType, loc, rewriter);
      result = rewriter.create<tosa::MulOp>(loc, resultType, result,
                                            reshapedWeight, shift);
    }

    // Step 8: Apply bias (beta) if present.
    if (adaptor.getBias()) {
      Value reshapedBias = reshapeByPrependingOnes(
          adaptor.getBias(), rank, numNormDims, elementType, loc, rewriter);
      result =
          rewriter.create<tosa::AddOp>(loc, resultType, result, reshapedBias);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class SqueezeOpConversionPattern : public OpConversionPattern<ttir::SqueezeOp> {
public:
  using OpConversionPattern<ttir::SqueezeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    int64_t dim = normalizeDim(op.getDim(), inputType.getRank());

    auto inputShape = inputType.getShape();
    SmallVector<int64_t> newShape;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      if (i != dim) {
        newShape.push_back(inputShape[i]);
      }
    }

    auto shapeType =
        tosa::shapeType::get(rewriter.getContext(), newShape.size());
    auto attr = rewriter.getIndexTensorAttr(newShape);
    auto shapeOp =
        rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

    auto reshapeOp = rewriter.create<tosa::ReshapeOp>(op.getLoc(), resultType,
                                                      input, shapeOp);

    // Handle DPS semantics - directly copy to output.
    ttir::EmptyOp output = rewriter.create<ttir::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    auto copyOp = rewriter.create<linalg::CopyOp>(
        op.getLoc(), ValueRange{reshapeOp}, ValueRange{output});
    rewriter.replaceOp(op, copyOp.getResult(0));

    return success();
  }
};
} // namespace

namespace {
class UnsqueezeOpConversionPattern
    : public OpConversionPattern<ttir::UnsqueezeOp> {
public:
  using OpConversionPattern<ttir::UnsqueezeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::UnsqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    SmallVector<int64_t> newShape(resultType.getShape());

    auto shapeType =
        tosa::shapeType::get(rewriter.getContext(), newShape.size());
    auto attr = rewriter.getIndexTensorAttr(newShape);
    auto shapeOp =
        rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(op, resultType, input,
                                                 shapeOp);

    return success();
  }
};
} // namespace

namespace {
class MeshShardOpConversionPattern
    : public OpConversionPattern<ttir::MeshShardOp> {
public:
  using OpConversionPattern<ttir::MeshShardOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MeshShardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle identity MeshShard ops - these are no-ops that just pass
    // through the input.
    if (adaptor.getShardType() != ttcore::MeshShardType::Identity) {
      return rewriter.notifyMatchFailure(
          op, "Only identity MeshShard ops are supported.");
    }

    // Identity MeshShard is a no-op, just forward the input.
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};
} // namespace

namespace {
class ClampScalarOpConversionPattern
    : public OpConversionPattern<ttir::ClampScalarOp> {
public:
  using OpConversionPattern<ttir::ClampScalarOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ClampScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto elementType = resultType.getElementType();
    TypedAttr minAttr, maxAttr;

    if (isa<FloatType>(elementType)) {
      minAttr = rewriter.getFloatAttr(
          elementType, ttmlir::utils::attributeToDouble(op.getMin()));
      maxAttr = rewriter.getFloatAttr(
          elementType, ttmlir::utils::attributeToDouble(op.getMax()));
    } else if (isa<IntegerType>(elementType)) {
      minAttr = rewriter.getIntegerAttr(
          elementType,
          static_cast<int64_t>(ttmlir::utils::attributeToDouble(op.getMin())));
      maxAttr = rewriter.getIntegerAttr(
          elementType,
          static_cast<int64_t>(ttmlir::utils::attributeToDouble(op.getMax())));
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported element type for clamp");
    }

    rewriter.replaceOpWithNewOp<tosa::ClampOp>(op, resultType, input, minAttr,
                                               maxAttr);
    return success();
  }
};
} // namespace

namespace {
class LinearOpConversionPattern : public OpConversionPattern<ttir::LinearOp> {
public:
  using OpConversionPattern<ttir::LinearOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getA();
    Value rhs = adaptor.getB();
    Value bias = adaptor.getBias();

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));

    if (!lhsType || !rhsType || !resultType) {
      return rewriter.notifyMatchFailure(
          op, "Operands or result is not a ranked tensor");
    }

    bool transposeA = op.getTransposeA();
    bool transposeB = op.getTransposeB();

    // Handle transposition if needed
    if (transposeA) {
      auto lhsShape = lhsType.getShape();
      if (lhsShape.size() >= 2) {
        SmallVector<int64_t> transposedShape;
        for (size_t i = 0; i < lhsShape.size() - 2; ++i) {
          transposedShape.push_back(lhsShape[i]);
        }
        transposedShape.push_back(lhsShape[lhsShape.size() - 1]);
        transposedShape.push_back(lhsShape[lhsShape.size() - 2]);

        auto transposedType =
            RankedTensorType::get(transposedShape, lhsType.getElementType());

        SmallVector<int32_t> permutation;
        for (size_t i = 0; i < lhsShape.size() - 2; ++i) {
          permutation.push_back(static_cast<int32_t>(i));
        }
        permutation.push_back(static_cast<int32_t>(lhsShape.size() - 1));
        permutation.push_back(static_cast<int32_t>(lhsShape.size() - 2));

        lhs = rewriter.create<tosa::TransposeOp>(op.getLoc(), transposedType,
                                                 lhs, permutation);
        lhsType = transposedType;
      }
    }

    if (transposeB) {
      auto rhsShape = rhsType.getShape();
      if (rhsShape.size() >= 2) {
        SmallVector<int64_t> transposedShape;
        for (size_t i = 0; i < rhsShape.size() - 2; ++i) {
          transposedShape.push_back(rhsShape[i]);
        }
        transposedShape.push_back(rhsShape[rhsShape.size() - 1]);
        transposedShape.push_back(rhsShape[rhsShape.size() - 2]);

        auto transposedType =
            RankedTensorType::get(transposedShape, rhsType.getElementType());

        SmallVector<int32_t> permutation;
        for (size_t i = 0; i < rhsShape.size() - 2; ++i) {
          permutation.push_back(static_cast<int32_t>(i));
        }
        permutation.push_back(static_cast<int32_t>(rhsShape.size() - 1));
        permutation.push_back(static_cast<int32_t>(rhsShape.size() - 2));

        rhs = rewriter.create<tosa::TransposeOp>(op.getLoc(), transposedType,
                                                 rhs, permutation);
        rhsType = transposedType;
      }
    }

    // Ensure both tensors are 3D for tosa.matmul
    unsigned lhsRank = lhsType.getRank();
    unsigned rhsRank = rhsType.getRank();

    Value lhs3D = lhs;
    Value rhs3D = rhs;
    RankedTensorType lhs3DType = lhsType;
    RankedTensorType rhs3DType = rhsType;

    // If LHS is 2D, reshape to 3D with batch size 1
    if (lhsRank == 2) {
      SmallVector<int64_t> newShape = {1, lhsType.getDimSize(0),
                                       lhsType.getDimSize(1)};
      auto newType = RankedTensorType::get(newShape, lhsType.getElementType());

      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      auto attr = rewriter.getIndexTensorAttr(newShape);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      lhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, lhs,
                                               shapeOp.getResult());
      lhs3DType = newType;
    } else if (lhsRank > 3) {
      // Check for dynamic dimensions in batch dimensions.
      for (uint32_t i = 0; i < lhsRank - 2; ++i) {
        if (lhsType.isDynamicDim(i)) {
          return rewriter.notifyMatchFailure(
              op, "Dynamic batch dimensions not supported in LinearOp");
        }
      }

      int64_t collapsedBatchSize = 1;
      for (uint32_t i = 0; i < lhsRank - 2; ++i) {
        collapsedBatchSize *= lhsType.getShape()[i];
      }

      SmallVector<int64_t> newShape = {collapsedBatchSize,
                                       lhsType.getShape()[lhsRank - 2],
                                       lhsType.getShape()[lhsRank - 1]};
      auto newType = RankedTensorType::get(newShape, lhsType.getElementType());

      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      auto attr = rewriter.getIndexTensorAttr(newShape);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      lhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, lhs,
                                               shapeOp.getResult());
      lhs3DType = newType;
    }

    // If RHS is 2D, reshape to 3D with batch size 1
    if (rhsRank == 2) {
      SmallVector<int64_t> newShape = {1, rhsType.getDimSize(0),
                                       rhsType.getDimSize(1)};
      auto newType = RankedTensorType::get(newShape, rhsType.getElementType());

      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      auto attr = rewriter.getIndexTensorAttr(newShape);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      rhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, rhs,
                                               shapeOp.getResult());
      rhs3DType = newType;
    } else if (rhsRank > 3) {
      // Check for dynamic dimensions in batch dimensions.
      for (uint32_t i = 0; i < rhsRank - 2; ++i) {
        if (rhsType.isDynamicDim(i)) {
          return rewriter.notifyMatchFailure(
              op, "Dynamic batch dimensions not supported in LinearOp");
        }
      }

      int64_t collapsedBatchSize = 1;
      for (uint32_t i = 0; i < rhsRank - 2; ++i) {
        collapsedBatchSize *= rhsType.getShape()[i];
      }

      SmallVector<int64_t> newShape = {collapsedBatchSize,
                                       rhsType.getShape()[rhsRank - 2],
                                       rhsType.getShape()[rhsRank - 1]};
      auto newType = RankedTensorType::get(newShape, rhsType.getElementType());

      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      auto attr = rewriter.getIndexTensorAttr(newShape);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      rhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, rhs,
                                               shapeOp.getResult());
      rhs3DType = newType;
    }

    // Check if we need to broadcast batch dimensions
    if (lhs3DType.getShape()[0] != rhs3DType.getShape()[0]) {
      if (lhs3DType.getShape()[0] == 1 && rhs3DType.getShape()[0] > 1) {
        SmallVector<int64_t> multiples = {rhs3DType.getShape()[0], 1, 1};
        auto newType = RankedTensorType::get({rhs3DType.getShape()[0],
                                              lhs3DType.getShape()[1],
                                              lhs3DType.getShape()[2]},
                                             lhs3DType.getElementType());

        auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
        auto multiplesAttr = rewriter.getIndexTensorAttr(multiples);
        auto multiplesOp = rewriter.create<tosa::ConstShapeOp>(
            op.getLoc(), shapeType, multiplesAttr);

        lhs3D = rewriter.create<tosa::TileOp>(op.getLoc(), newType, lhs3D,
                                              multiplesOp);
        lhs3DType = cast<RankedTensorType>(lhs3D.getType());
      } else if (rhs3DType.getShape()[0] == 1 && lhs3DType.getShape()[0] > 1) {
        SmallVector<int64_t> multiples = {lhs3DType.getShape()[0], 1, 1};
        auto newType = RankedTensorType::get({lhs3DType.getShape()[0],
                                              rhs3DType.getShape()[1],
                                              rhs3DType.getShape()[2]},
                                             rhs3DType.getElementType());

        auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
        auto multiplesAttr = rewriter.getIndexTensorAttr(multiples);
        auto multiplesOp = rewriter.create<tosa::ConstShapeOp>(
            op.getLoc(), shapeType, multiplesAttr);

        rhs3D = rewriter.create<tosa::TileOp>(op.getLoc(), newType, rhs3D,
                                              multiplesOp);
        rhs3DType = cast<RankedTensorType>(rhs3D.getType());
      }
    }

    // Perform matrix multiplication
    auto matmulResultType =
        RankedTensorType::get({lhs3DType.getShape()[0], lhs3DType.getShape()[1],
                               rhs3DType.getShape()[2]},
                              resultType.getElementType());

    Value matmulResult = rewriter.create<tosa::MatMulOp>(
        op.getLoc(), matmulResultType, lhs3D, rhs3D);

    // Reshape result back to original rank if needed
    if (resultType.getRank() != matmulResultType.getRank()) {
      auto shapeType =
          tosa::shapeType::get(rewriter.getContext(), resultType.getRank());
      SmallVector<int64_t> shapeValues;
      for (auto dim : resultType.getShape()) {
        shapeValues.push_back(dim);
      }
      auto attr = rewriter.getIndexTensorAttr(shapeValues);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      matmulResult = rewriter.create<tosa::ReshapeOp>(
          op.getLoc(), resultType, matmulResult, shapeOp.getResult());
    }

    // If bias is provided, add it to the result
    if (bias) {
      auto biasType = cast<RankedTensorType>(bias.getType());

      // Reshape bias to match result rank if needed by prepending 1s
      if (biasType.getRank() < resultType.getRank()) {
        SmallVector<int64_t> newBiasShape;
        int64_t rankDiff = resultType.getRank() - biasType.getRank();
        for (int64_t i = 0; i < rankDiff; ++i) {
          newBiasShape.push_back(1);
        }
        for (auto dim : biasType.getShape()) {
          newBiasShape.push_back(dim);
        }

        auto reshapedBiasType =
            RankedTensorType::get(newBiasShape, biasType.getElementType());
        auto shapeType =
            tosa::shapeType::get(rewriter.getContext(), newBiasShape.size());
        auto shapeAttr = rewriter.getIndexTensorAttr(newBiasShape);
        auto shapeOp = rewriter.create<tosa::ConstShapeOp>(
            op.getLoc(), shapeType, shapeAttr);
        bias = rewriter.create<tosa::ReshapeOp>(op.getLoc(), reshapedBiasType,
                                                bias, shapeOp.getResult());
      }

      matmulResult = rewriter.create<tosa::AddOp>(op.getLoc(), resultType,
                                                  matmulResult, bias);
    }

    rewriter.replaceOp(op, matmulResult);
    return success();
  }
};
} // namespace

namespace {
class RepeatOpConversionPattern : public OpConversionPattern<ttir::RepeatOp> {
public:
  using OpConversionPattern<ttir::RepeatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::RepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto repeatDimensions = op.getRepeatDimensions();
    SmallVector<int64_t> multiples(repeatDimensions.begin(),
                                   repeatDimensions.end());

    auto shapeType =
        tosa::shapeType::get(rewriter.getContext(), multiples.size());
    auto multiplesAttr = rewriter.getIndexTensorAttr(multiples);
    auto multiplesOp = rewriter.create<tosa::ConstShapeOp>(
        op.getLoc(), shapeType, multiplesAttr);

    rewriter.replaceOpWithNewOp<tosa::TileOp>(op, resultType,
                                              adaptor.getInput(), multiplesOp);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateTTIRToLinalgPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter) {
  patterns.add<
      ElementwiseBinaryOpConversionPattern<ttir::AddOp, linalg::AddOp>,
      ElementwiseBinaryOpConversionPattern<ttir::SubtractOp, linalg::SubOp>,
      ElementwiseBinaryOpConversionPattern<ttir::MultiplyOp, linalg::MulOp>,
      ElementwiseBinaryOpConversionPattern<ttir::DivOp, linalg::DivOp>,
      ElementwiseBinaryOpConversionPattern<ttir::PowOp, linalg::PowFOp>,
      ElementwiseOpConversionPattern<ttir::SqrtOp, linalg::SqrtOp>,
      SoftmaxOpConversionPattern, EmptyOpConversionPattern,
      PermuteOpConversionPattern, SliceStaticOpConversionPattern,
      PadOpConversionPattern, ConstantOpConversionPattern,
      ReluOpConversionPattern, NamedFillOpConversionPattern<ttir::ZerosOp, 0>,
      NamedFillOpConversionPattern<ttir::OnesOp, 1>, FullOpConversionPattern,
      ArangeOpConversionPattern, MeshShardOpConversionPattern,
      CumSumOpConversionPattern, ConcatenateHeadsOpConversionPattern>(
      typeConverter, ctx);
}

void populateTTIRToTosaPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  // Elementwise unary operations
  patterns.add<
      ElementwiseUnaryOpConversionPattern<ttir::AbsOp, tosa::AbsOp>,
      ElementwiseUnaryOpConversionPattern<ttir::CeilOp, tosa::CeilOp>,
      ElementwiseUnaryOpConversionPattern<ttir::ExpOp, tosa::ExpOp>,
      ElementwiseUnaryOpConversionPattern<ttir::FloorOp, tosa::FloorOp>,
      ElementwiseUnaryOpConversionPattern<ttir::LogOp, tosa::LogOp>,
      ElementwiseUnaryOpConversionPattern<ttir::NegOp, tosa::NegateOp>,
      ElementwiseUnaryOpConversionPattern<ttir::ReciprocalOp,
                                          tosa::ReciprocalOp>,
      ElementwiseUnaryOpConversionPattern<ttir::RsqrtOp, tosa::RsqrtOp>,
      ElementwiseUnaryOpConversionPattern<ttir::SigmoidOp, tosa::SigmoidOp>,
      ElementwiseUnaryOpConversionPattern<ttir::TanhOp, tosa::TanhOp>,
      ElementwiseUnaryOpConversionPattern<ttir::TypecastOp, tosa::CastOp>>(
      typeConverter, ctx);

  // Comparison operations
  patterns.add<
      DirectComparisonOpConversionPattern<ttir::EqualOp, tosa::EqualOp>,
      DirectComparisonOpConversionPattern<ttir::GreaterThanOp, tosa::GreaterOp>,
      DirectComparisonOpConversionPattern<ttir::GreaterEqualOp,
                                          tosa::GreaterEqualOp>,
      SwappedComparisonOpConversionPattern<ttir::LessThanOp, tosa::GreaterOp>,
      SwappedComparisonOpConversionPattern<ttir::LessEqualOp,
                                           tosa::GreaterEqualOp>,
      NegatedComparisonOpConversionPattern<ttir::NotEqualOp, tosa::EqualOp>>(
      typeConverter, ctx);

  // Logical binary operations
  patterns.add<
      LogicalBinaryOpConversionPattern<ttir::LogicalAndOp, tosa::LogicalAndOp>,
      LogicalBinaryOpConversionPattern<ttir::LogicalOrOp, tosa::LogicalOrOp>,
      LogicalBinaryOpConversionPattern<ttir::LogicalXorOp, tosa::LogicalXorOp>>(
      typeConverter, ctx);

  // Elementwise binary operations (via TOSA)
  patterns.add<TosaElementwiseBinaryOpConversionPattern<ttir::MinimumOp,
                                                        tosa::MinimumOp>,
               TosaElementwiseBinaryOpConversionPattern<ttir::MaximumOp,
                                                        tosa::MaximumOp>>(
      typeConverter, ctx);

  patterns.add<BroadcastOpConversionPattern, SinOpConversionPattern,
               CosOpConversionPattern, MatmulOpConversionPattern,
               LinearOpConversionPattern, ClampScalarOpConversionPattern,
               Relu6OpConversionPattern, GatherOpConversionPattern,
               LogicalNotOpConversionPattern, MaxOpConversionPattern,
               MinOpConversionPattern, SumOpConversionPattern,
               ProdOpConversionPattern, ArgMaxOpConversionPattern,
               MeanOpConversionPattern, LayerNormOpConversionPattern,
               SqueezeOpConversionPattern, UnsqueezeOpConversionPattern,
               MaxPool2dOpConversionPattern, AvgPool2dOpConversionPattern,
               GlobalAvgPool2dOpConversionPattern, Conv2dOpConversionPattern>(
      typeConverter, ctx);

  // Special operations
  patterns.add<WhereOpConversionPattern, ReshapeOpConversionPattern,
               TransposeOpConversionPattern, ConcatOpConversionPattern,
               RepeatOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
