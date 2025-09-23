// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/TTIRToLinalg.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

#include "mlir/IR/OpDefinition.h"
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

// Create a tensor of all ones or zeros with the given type
static std::pair<Value, Value>
createTrueAndFalseSplatConstants(RankedTensorType resultType, Location loc,
                                 ConversionPatternRewriter &rewriter) {
  auto elementType = resultType.getElementType();
  assert(elementType.isF32());
  TypedAttr trueAttr = rewriter.getF32FloatAttr(1.0f);
  TypedAttr falseAttr = rewriter.getF32FloatAttr(0.0f);

  // Create constant scalars with the true/false values
  auto trueValue = rewriter.create<arith::ConstantOp>(loc, trueAttr);
  auto falseValue = rewriter.create<arith::ConstantOp>(loc, falseAttr);

  // Create splat tensors with the true/false values
  auto trueValueSplat =
      rewriter.create<tensor::SplatOp>(loc, resultType, trueValue.getResult());

  auto falseValueSplat =
      rewriter.create<tensor::SplatOp>(loc, resultType, falseValue.getResult());

  return {trueValueSplat, falseValueSplat};
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

    // Otherwise, use the provided dimensions.
    SmallVector<int64_t> dims;
    for (auto dim : dimAttr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(dim)) {
        dims.push_back(intAttr.getInt());
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
static Attribute createDenseElementsAttr(RankedTensorType resultType,
                                         int64_t value) {
  auto elementType = resultType.getElementType();
  if (isa<FloatType>(elementType)) {
    return DenseElementsAttr::get(
        resultType,
        APFloat(cast<FloatType>(elementType).getFloatSemantics(), value));
  }
  if (isa<IntegerType>(elementType)) {
    return DenseElementsAttr::get(
        resultType, APInt(cast<IntegerType>(elementType).getWidth(), value));
  }
  return {};
}

// Helper function to calculate extra padding for MaxPool2dOp.
// This function calculates the extra padding needed to make the output size
// divisible by the stride.
static int64_t calculateExtraPadding(int64_t dim, int64_t kernel,
                                     int64_t stride, int64_t padding1,
                                     int64_t padding2) {
  if ((dim + padding1 + padding2 - kernel) % stride != 0) {
    return (stride - (dim + padding1 + padding2 - kernel) % stride);
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
    Value output = adaptor.getOutput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
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

    // Handle DPS semantics - directly copy to output.
    Value output = adaptor.getOutput();
    auto copyOp = rewriter.create<linalg::CopyOp>(
        op.getLoc(), ValueRange{reshapeOp}, output);
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
    static_assert(ttir::utils::has_dps_trait_v<ttir::ConcatOp>);
    auto inputs =
        ttir::utils::getDpsInputsFromAdaptor(adaptor, op.getNumDpsInits());

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    const int64_t dim = op.getDim();

    // TOSA concat requires at least two inputs.
    if (inputs.size() < 2) {
      return failure();
    }

    // Start with the first two inputs.
    Value result = rewriter.create<tosa::ConcatOp>(
        op.getLoc(),
        RankedTensorType::get(resultType.getShape().take_front(inputs.size()),
                              resultType.getElementType()),
        ValueRange{inputs[0], inputs[1]}, dim);
    // Add remaining inputs one by one.
    for (size_t i = 2; i < inputs.size(); ++i) {
      result = rewriter.create<tosa::ConcatOp>(
          op.getLoc(),
          RankedTensorType::get(resultType.getShape().take_front(i + 1),
                                resultType.getElementType()),
          ValueRange{result, inputs[i]}, dim);
    }

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

    // Create true and false constants for the select operation
    auto [trueValueSplat, falseValueSplat] =
        createTrueAndFalseSplatConstants(resultType, op.getLoc(), rewriter);

    // Convert boolean result to original type using select
    auto result = rewriter.create<tosa::SelectOp>(
        op.getLoc(), resultType, boolResult, trueValueSplat, falseValueSplat);

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

    // Create true and false constants for the select operation
    auto [trueValueSplat, falseValueSplat] =
        createTrueAndFalseSplatConstants(resultType, op.getLoc(), rewriter);

    // Convert boolean result to original type using select
    auto result = rewriter.create<tosa::SelectOp>(
        op.getLoc(), resultType, boolResult, trueValueSplat, falseValueSplat);

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

    // Create true and false constants for the select operation
    auto [trueValueSplat, falseValueSplat] =
        createTrueAndFalseSplatConstants(resultType, op.getLoc(), rewriter);

    // Convert boolean result to original type using select
    auto result = rewriter.create<tosa::SelectOp>(
        op.getLoc(), resultType, notResult, trueValueSplat, falseValueSplat);

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
    Value input = adaptor.getInput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // For TOSA, we can use arithmetic operations that have implicit
    // broadcasting

    // Create a tensor of ones with the result shape
    auto elementType = resultType.getElementType();
    assert(elementType.isF32());
    DenseElementsAttr zerosAttr =
        DenseElementsAttr::get(resultType, ArrayRef<float>(0));

    auto zerosConst =
        rewriter.create<tosa::ConstOp>(op.getLoc(), resultType, zerosAttr);

    // Multiply by ones to implicitly broadcast
    auto result = rewriter.create<tosa::AddOp>(op.getLoc(), resultType, input,
                                               zerosConst);

    rewriter.replaceOp(op, result);
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

        // Create multiples using tosa.const for tile operation
        auto multiplesType = RankedTensorType::get({3}, rewriter.getI64Type());
        auto multiplesAttr =
            DenseIntElementsAttr::get(multiplesType, multiples);
        auto multiplesOp = rewriter.create<tosa::ConstOp>(
            op.getLoc(), multiplesType, multiplesAttr);

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

        // Create multiples using tosa.const for tile operation
        auto multiplesType = RankedTensorType::get({3}, rewriter.getI64Type());
        auto multiplesAttr =
            DenseIntElementsAttr::get(multiplesType, multiples);
        auto multiplesOp = rewriter.create<tosa::ConstOp>(
            op.getLoc(), multiplesType, multiplesAttr);

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

    rewriter.replaceOp(op, matmulResult);
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

    SmallVector<int64_t> expandedPadding = {
        std::get<0>(*paddingResult), std::get<2>(*paddingResult),
        std::get<1>(*paddingResult), std::get<3>(*paddingResult)};

    // Expand kernel if it contains only one element.
    auto kernelResult = ttmlir::utils::getPairOfInteger<int32_t>(kernel);
    if (!kernelResult) {
      return rewriter.notifyMatchFailure(
          op, "kernel must be an integer or array attribute");
    }

    if (auto dilationArray = dyn_cast<DenseI32ArrayAttr>(dilation)) {
      assert(dilationArray.size() == 2 && "dilation must be 2 elements");
      if (dilationArray[0] != 1 || dilationArray[1] != 1) {
        return rewriter.notifyMatchFailure(op, "dilation must be 1x1");
      }
    } else if (auto singleDilation = dyn_cast<IntegerAttr>(dilation)) {
      if (singleDilation.getInt() != 1) {
        return rewriter.notifyMatchFailure(op, "dilation must be 1");
      }
    } else {
      return rewriter.notifyMatchFailure(
          op, "dilation must be an integer or array attribute");
    }

    // Update padding and return shape to be used in the TOSA MaxPool2dOp.
    // input_height + pad_top + pad_bottom - kernel_height must be divisible by
    // stride_y. input_width + pad_left + pad_right - kernel_width must be
    // divisible by stride_x. The padding values are updated to ensure this
    // condition is met.
    int64_t inputHeight = cast<RankedTensorType>(input.getType()).getShape()[1];
    int64_t inputWidth = cast<RankedTensorType>(input.getType()).getShape()[2];
    int64_t kernelHeight = kernelResult->first;
    int64_t kernelWidth = kernelResult->second;

    expandedPadding[1] +=
        calculateExtraPadding(inputHeight, kernelHeight, stridesResult->first,
                              expandedPadding[0], expandedPadding[1]);
    expandedPadding[3] +=
        calculateExtraPadding(inputWidth, kernelWidth, stridesResult->second,
                              expandedPadding[2], expandedPadding[3]);

    auto expandedStridesAttr = rewriter.getDenseI64ArrayAttr(
        {stridesResult->first, stridesResult->second});
    auto expandedPaddingAttr = rewriter.getDenseI64ArrayAttr(expandedPadding);
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
    resultShape[1] =
        (inputHeight + expandedPadding[0] + expandedPadding[1] - kernelHeight) /
            stridesResult->first +
        1;
    resultShape[2] =
        (inputWidth + expandedPadding[2] + expandedPadding[3] - kernelWidth) /
            stridesResult->second +
        1;

    auto actualResultType =
        RankedTensorType::get(resultShape, resultType.getElementType());

    // Create the max pool op.
    auto maxPoolOp = rewriter.create<tosa::MaxPool2dOp>(
        op.getLoc(), actualResultType, input, expandedKernelAttr,
        expandedStridesAttr, expandedPaddingAttr);

    // Slice the result back to the original expected shape if needed.
    Value result = maxPoolOp.getResult();
    if (!llvm::equal(resultShape, resultType.getShape())) {
      SmallVector<OpFoldResult> offsets, sizes, strides;
      for (int64_t i = 0; i < resultType.getRank(); ++i) {
        offsets.push_back(rewriter.getI64IntegerAttr(0));
        sizes.push_back(rewriter.getI64IntegerAttr(resultType.getShape()[i]));
        strides.push_back(rewriter.getI64IntegerAttr(1));
      }
      result = rewriter.create<tensor::ExtractSliceOp>(
          op.getLoc(), resultType, result, offsets, sizes, strides);
    }

    rewriter.replaceOp(op, result);
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

    // Create true and false constants for the select operation
    auto [trueValueSplat, falseValueSplat] =
        createTrueAndFalseSplatConstants(resultType, op.getLoc(), rewriter);

    // Convert boolean result back to original type using select
    auto result = rewriter.create<tosa::SelectOp>(
        op.getLoc(), resultType, notResult, trueValueSplat, falseValueSplat);

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
class ReduceOrOpConversionPattern
    : public OpConversionPattern<ttir::ReduceOrOp> {
public:
  using OpConversionPattern<ttir::ReduceOrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReduceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    // Convert input to boolean tensor if needed
    input = convertToBooleanTensor(input, op.getLoc(), rewriter);

    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Get dimensions to reduce and keep_dim attribute
    SmallVector<int64_t> dims = getDimsFromAttribute(op, rank);
    bool keepDim = getKeepDimFromAttribute(op);

    // Create a chain of reduction operations
    Value result = createReductionOpChain<tosa::ReduceAnyOp>(
        input, resultType, dims, keepDim, op.getLoc(), rewriter);

    rewriter.replaceOp(op, result);
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
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return failure();
    }

    static_assert(ttir::utils::has_dps_trait_v<TTIROpTy>);
    auto outputs = adaptor.getOperands().take_back(op.getNumDpsInits());
    rewriter.replaceOpWithNewOp<LinalgOpTy>(op, resultTypes, broadcastedInputs,
                                            outputs);
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
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return failure();
    }

    static_assert(ttir::utils::has_dps_trait_v<TTIROpTy>);
    auto inputs =
        ttir::utils::getDpsInputsFromAdaptor(adaptor, op.getNumDpsInits());
    auto outputs =
        ttir::utils::getDpsOutputsFromAdaptor(adaptor, op.getNumDpsInits());
    rewriter.replaceOpWithNewOp<LinAlgOpTy>(op, resultTypes, inputs, outputs);
    return success();
  }
};
} // namespace

namespace {
class SoftmaxOpConversionPattern : public OpConversionPattern<ttir::SoftmaxOp> {
public:
  using OpConversionPattern<ttir::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getInput();
    const size_t inputSize =
        dyn_cast<RankedTensorType>(input.getType()).getShape().size();
    const int32_t dimension = (op.getDimension() < 0)
                                  ? op.getDimension() + inputSize
                                  : op.getDimension();

    rewriter.replaceOpWithNewOp<linalg::SoftmaxOp>(
        op, this->getTypeConverter()->convertType(op.getType()), input,
        adaptor.getOutput(), dimension);
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

    Attribute zeroAttr = createDenseElementsAttr(resultType, 0);
    if (!zeroAttr) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported element type for ReLU zero constant");
    }

    auto zeroes = rewriter.create<arith::ConstantOp>(
        op.getLoc(), resultType, cast<DenseElementsAttr>(zeroAttr));

    rewriter.replaceOpWithNewOp<linalg::MaxOp>(
        op, resultType, ValueRange{input, zeroes.getResult()},
        ValueRange{adaptor.getOutput()});
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
    Value input = adaptor.getInput();
    llvm::ArrayRef<int64_t> permutation = op.getPermutation();

    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, input, adaptor.getOutput(), permutation);

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
    Value output = adaptor.getOutput(); // Get the output buffer
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
    auto copyResult =
        rewriter.create<linalg::CopyOp>(op.getLoc(), extractedSlice, output);
    rewriter.replaceOp(op, copyResult);

    return success();
  }
};
} // namespace

namespace {
class EmbeddingOpConversionPattern
    : public OpConversionPattern<ttir::EmbeddingOp> {
public:
  using OpConversionPattern<ttir::EmbeddingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmbeddingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input tensor and weight
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto weightType = dyn_cast<RankedTensorType>(weight.getType());
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));

    if (!inputType || !weightType || !resultType) {
      return rewriter.notifyMatchFailure(
          op, "Input, weight, or result is not a ranked tensor");
    }

    // For embedding, we're gathering along dimension 0
    SmallVector<int64_t> dims{
        0}; // Always gather along dimension 0 for embedding
    auto dimsAttr = rewriter.getDenseI64ArrayAttr(dims);

    // tensor.gather requires integer indices, so we need to cast if the input
    // is not an integer type
    if (!inputType.getElementType().isIntOrIndex()) {
      // Create a new type with the same shape but i64 element type
      auto newInputType =
          RankedTensorType::get(inputType.getShape(), rewriter.getI64Type());

      // Convert the input tensor to integer type
      input =
          rewriter.create<arith::FPToSIOp>(op.getLoc(), newInputType, input);

      // Update inputType to reflect the new type
      inputType = dyn_cast<RankedTensorType>(input.getType());
    }

    // For tensor.gather, the last dimension of indices must match the length of
    // gather_dims If indices is 1D and gather_dims has one element, reshape
    // indices to add a dimension
    auto indicesShape = inputType.getShape();
    if (indicesShape.size() == 1 && dims.size() == 1) {
      // Create a new shape with an additional dimension of size 1
      SmallVector<int64_t> newShape(indicesShape.begin(), indicesShape.end());
      newShape.push_back(1);

      // Create a new type with the additional dimension
      auto reshapedType =
          RankedTensorType::get(newShape, inputType.getElementType());

      // Create a reshape operation to add the dimension
      input = rewriter.create<tensor::ExpandShapeOp>(
          op.getLoc(), reshapedType, input,
          ArrayRef<ReassociationIndices>{{0, 1}});
    }

    // Create the tensor.gather operation
    auto result = rewriter.create<tensor::GatherOp>(op.getLoc(), resultType,
                                                    weight, input, dimsAttr);

    rewriter.replaceOp(op, result);
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

    auto newConstant =
        rewriter.create<arith::ConstantOp>(op.getLoc(), resultType, value);

    rewriter.replaceOp(op, newConstant.getResult());
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

    Attribute divisorAttr = createDenseElementsAttr(resultType, numElements);
    if (!divisorAttr) {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported element type for mean");
    }

    auto divisor = rewriter.create<tosa::ConstOp>(
        op.getLoc(), resultType, cast<DenseElementsAttr>(divisorAttr));

    auto divOp = rewriter.create<linalg::DivOp>(
        op.getLoc(), resultType, ValueRange{sum, divisor},
        ValueRange{adaptor.getOutput()});

    rewriter.replaceOp(op, divOp.getResult(0));
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

    int32_t dim = op.getDim();
    if (dim < 0) {
      dim += inputType.getRank();
    }

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
    Value output = adaptor.getOutput();
    auto copyOp = rewriter.create<linalg::CopyOp>(
        op.getLoc(), ValueRange{reshapeOp}, output);
    rewriter.replaceOp(op, copyOp.getResult(0));

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
      ConstantOpConversionPattern, EmbeddingOpConversionPattern,
      ReluOpConversionPattern>(typeConverter, ctx);
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
      ElementwiseUnaryOpConversionPattern<ttir::TanhOp, tosa::TanhOp>>(
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

  patterns.add<BroadcastOpConversionPattern, SinOpConversionPattern,
               CosOpConversionPattern, MatmulOpConversionPattern,
               GatherOpConversionPattern, LogicalNotOpConversionPattern,
               MaxOpConversionPattern, SumOpConversionPattern,
               ReduceOrOpConversionPattern, MeanOpConversionPattern,
               SqueezeOpConversionPattern, MaxPool2dOpConversionPattern>(typeConverter, ctx);
               

  // Special operations
  patterns.add<WhereOpConversionPattern, ReshapeOpConversionPattern,
               TransposeOpConversionPattern, ConcatOpConversionPattern>(
      typeConverter, ctx);
}

} // namespace mlir::tt
