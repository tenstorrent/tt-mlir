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

#include <cstdint>

namespace mlir::tt {
//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//
namespace {
// Convert a tensor of floating-point values to a tensor of boolean values
// by comparing with zero
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
  TypedAttr zeroAttr;

  if (elementType.isF32()) {
    zeroAttr = rewriter.getF32FloatAttr(0.0f);
  } else if (elementType.isF64()) {
    zeroAttr = rewriter.getF64FloatAttr(0.0);
  } else if (elementType.isF16()) {
    zeroAttr = rewriter.getF16FloatAttr(0.0);
  } else if (elementType.isBF16()) {
    zeroAttr = rewriter.getFloatAttr(elementType, 0.0);
  } else if (elementType.isInteger(32)) {
    zeroAttr = rewriter.getI32IntegerAttr(0);
  } else if (elementType.isInteger(64)) {
    zeroAttr = rewriter.getI64IntegerAttr(0);
  } else if (elementType.isInteger(16)) {
    zeroAttr = rewriter.getI16IntegerAttr(0);
  } else if (elementType.isInteger(8)) {
    zeroAttr = rewriter.getI8IntegerAttr(0);
  } else {
    // Default to i32 for unsupported types
    zeroAttr = rewriter.getI32IntegerAttr(0);
  }

  // Create a constant scalar with the zero value
  auto zeroValue = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

  // Create a splat tensor with the zero value
  auto zeroSplat =
      rewriter.create<tensor::SplatOp>(loc, inputType, zeroValue.getResult());

  // Compare input with zero to get a boolean tensor
  auto boolType =
      RankedTensorType::get(inputType.getShape(), rewriter.getIntegerType(1));
  return rewriter.create<tosa::GreaterOp>(loc, boolType, input, zeroSplat);
}

// Create a tensor of all ones or zeros with the given type
static std::pair<Value, Value>
createTrueAndFalseSplatConstants(RankedTensorType resultType, Location loc,
                                 ConversionPatternRewriter &rewriter) {
  auto elementType = resultType.getElementType();
  TypedAttr trueAttr, falseAttr;

  if (elementType.isF32()) {
    trueAttr = rewriter.getF32FloatAttr(1.0f);
    falseAttr = rewriter.getF32FloatAttr(0.0f);
  } else if (elementType.isF64()) {
    trueAttr = rewriter.getF64FloatAttr(1.0);
    falseAttr = rewriter.getF64FloatAttr(0.0);
  } else if (elementType.isF16()) {
    trueAttr = rewriter.getF16FloatAttr(1.0);
    falseAttr = rewriter.getF16FloatAttr(0.0);
  } else if (elementType.isBF16()) {
    // Use FloatAttr with BF16 type instead of getBF16FloatAttr
    trueAttr = rewriter.getFloatAttr(elementType, 1.0);
    falseAttr = rewriter.getFloatAttr(elementType, 0.0);
  } else if (elementType.isInteger(32)) {
    trueAttr = rewriter.getI32IntegerAttr(1);
    falseAttr = rewriter.getI32IntegerAttr(0);
  } else if (elementType.isInteger(64)) {
    trueAttr = rewriter.getI64IntegerAttr(1);
    falseAttr = rewriter.getI64IntegerAttr(0);
  } else if (elementType.isInteger(16)) {
    trueAttr = rewriter.getI16IntegerAttr(1);
    falseAttr = rewriter.getI16IntegerAttr(0);
  } else if (elementType.isInteger(8)) {
    trueAttr = rewriter.getI8IntegerAttr(1);
    falseAttr = rewriter.getI8IntegerAttr(0);
  } else if (elementType.isInteger(1)) {
    trueAttr = rewriter.getBoolAttr(true);
    falseAttr = rewriter.getBoolAttr(false);
  } else {
    // Default to i32 for unsupported types
    trueAttr = rewriter.getI32IntegerAttr(1);
    falseAttr = rewriter.getI32IntegerAttr(0);
  }

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
// empty, return all dimensions
static SmallVector<int64_t> getDimsFromAttribute(Operation *op, int64_t rank) {
  if (auto dimAttr = op->getAttrOfType<ArrayAttr>("dim_arg")) {
    if (dimAttr.size() == 0) {
      // If dim_arg is present but empty, reduce along all dimensions
      SmallVector<int64_t> allDims(rank);
      std::iota(allDims.begin(), allDims.end(), 0);
      return allDims;
    }

    // Otherwise, use the provided dimensions
    SmallVector<int64_t> dims;
    for (auto dim : dimAttr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(dim)) {
        dims.push_back(intAttr.getInt());
      }
    }
    return dims;
  }

  // If no dim_arg attribute, reduce along all dimensions
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

    // Create the TOSA operation
    auto result = rewriter.create<TosaOpTy>(op.getLoc(), resultType,
                                            ValueRange{lhs, rhs});

    // Replace the original operation with the result
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

    // Convert the condition to a boolean tensor
    condition = convertToBooleanTensor(condition, op.getLoc(), rewriter);

    // Create the TOSA select operation (equivalent to where)
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

    // Create the shape type
    auto shapeType =
        mlir::tosa::shapeType::get(rewriter.getContext(), newShape.size());

    // Create the index tensor attribute for the shape
    auto attr = rewriter.getIndexTensorAttr(newShapeValues);

    // Create the tosa.const_shape operation
    auto shapeOp =
        rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

    // Create the reshape operation with the shape operand
    auto result = rewriter.create<tosa::ReshapeOp>(op.getLoc(), resultType,
                                                   adaptor.getInput(), shapeOp);

    rewriter.replaceOp(op, result);
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

    // Calculate the transposed shape
    SmallVector<int64_t> transposedShape;
    for (size_t i = 0; i < permutation.size(); ++i) {
      transposedShape.push_back(inputType.getShape()[permutation[i]]);
    }

    // Create TransposeOp directly with the permutation array - it expects an
    // ArrayRef<int32_t>
    auto transposedType =
        RankedTensorType::get(transposedShape, inputType.getElementType());
    auto result = rewriter.create<tosa::TransposeOp>(
        op.getLoc(), transposedType, input, permutation);

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

    // Get the dimension to concatenate along
    int64_t dim = op.getDim();

    // TOSA concat requires at least two inputs
    if (inputs.size() < 2) {
      return failure();
    }

    // Start with the first two inputs
    Value result = rewriter.create<tosa::ConcatOp>(
        op.getLoc(),
        RankedTensorType::get(resultType.getShape().take_front(inputs.size()),
                              resultType.getElementType()),
        ValueRange{inputs[0], inputs[1]}, dim);

    // Add remaining inputs one by one
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

// Swapped comparison operations (where TTIR and TOSA ops have swapped operands)
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

// Negated comparison operations (where TTIR op is the negation of a TOSA op)
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
    DenseElementsAttr onesAttr;

    if (elementType.isF32()) {
      // Create a splat attribute with value 1.0f
      float oneValue = 1.0f;
      onesAttr = DenseElementsAttr::get(resultType, ArrayRef<float>(oneValue));
    } else if (elementType.isF64()) {
      // Create a splat attribute with value 1.0
      double oneValue = 1.0;
      onesAttr = DenseElementsAttr::get(resultType, ArrayRef<double>(oneValue));
    } else if (elementType.isInteger(32)) {
      // Create a splat attribute with value 1
      int32_t oneValue = 1;
      onesAttr =
          DenseElementsAttr::get(resultType, ArrayRef<int32_t>(oneValue));
    } else {
      // Default to F32 for other types
      float oneValue = 1.0f;
      onesAttr = DenseElementsAttr::get(resultType, ArrayRef<float>(oneValue));
    }

    auto onesConst =
        rewriter.create<tosa::ConstOp>(op.getLoc(), resultType, onesAttr);

    // Multiply by ones to implicitly broadcast
    auto result = rewriter.create<tosa::MulOp>(op.getLoc(), resultType, input,
                                               onesConst, /*shift=*/nullptr);

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
class DotGeneralOpConversionPattern
    : public OpConversionPattern<ttir::DotGeneralOp> {
public:
  using OpConversionPattern<ttir::DotGeneralOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::DotGeneralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // For simple matrix multiplication
    auto result =
        rewriter.create<tosa::MatMulOp>(op.getLoc(), resultType, lhs, rhs);

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
    Value input = adaptor.getInput();
    Value indices = adaptor.getOperands()[1];

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto result = rewriter.create<tosa::GatherOp>(op.getLoc(), resultType,
                                                  input, indices);

    rewriter.replaceOp(op, result);
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

    auto result =
        rewriter.create<tosa::LogicalNotOp>(op.getLoc(), resultType, input);

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

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Create a constant for the axis - assuming 0 as default if not specified
    int64_t axis = 0;
    // Check if the op has an axis attribute
    if (auto axisAttr = op->getAttrOfType<IntegerAttr>("axis")) {
      axis = axisAttr.getInt();
    }

    // TOSA's ReduceMaxOp takes an integer attribute for the axis
    auto axisAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(axis));

    // Create the ReduceMaxOp with the axis attribute
    auto result = rewriter.create<tosa::ReduceMaxOp>(op.getLoc(), resultType,
                                                     input, axisAttr);

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

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Create a constant for the axis - assuming 0 as default if not specified
    int64_t axis = 0;
    // Check if the op has an axis attribute
    if (auto axisAttr = op->getAttrOfType<IntegerAttr>("axis")) {
      axis = axisAttr.getInt();
    }

    // TOSA's ReduceSumOp takes an integer attribute for the axis
    auto axisAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(axis));

    // Create the ReduceSumOp with the axis attribute
    auto result = rewriter.create<tosa::ReduceSumOp>(op.getLoc(), resultType,
                                                     input, axisAttr);

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

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Create a constant for the axis - assuming 0 as default if not specified
    int64_t axis = 0;
    // Check if the op has an axis attribute
    if (auto axisAttr = op->getAttrOfType<IntegerAttr>("axis")) {
      axis = axisAttr.getInt();
    }

    // TOSA's ReduceAnyOp takes an integer attribute for the axis
    auto axisAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(axis));

    // Create the ReduceAnyOp with the axis attribute
    auto result = rewriter.create<tosa::ReduceAnyOp>(op.getLoc(), resultType,
                                                     input, axisAttr);

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
// Conversion pattern for ttir.slice operation
class SliceOpConversionPattern : public OpConversionPattern<ttir::SliceOp> {
public:
  using OpConversionPattern<ttir::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SliceOp op, OpAdaptor adaptor,
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

      // Calculate size: (end - begin) / step.
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

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resultType, input, offsets, sizes, strides);

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
    auto value = op.getValue();

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

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateTTIRToLinalgPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter) {
  patterns.add<
      ElementwiseBinaryOpConversionPattern<ttir::MultiplyOp, linalg::MulOp>,
      ElementwiseBinaryOpConversionPattern<ttir::DivOp, linalg::DivOp>,
      ElementwiseBinaryOpConversionPattern<ttir::PowOp, linalg::PowFOp>,
      ElementwiseOpConversionPattern<ttir::SqrtOp, linalg::SqrtOp>,
      SoftmaxOpConversionPattern, EmptyOpConversionPattern,
      PermuteOpConversionPattern, SliceOpConversionPattern,
      ConstantOpConversionPattern, EmbeddingOpConversionPattern>(typeConverter,
                                                                 ctx);
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

  // Elementwise binary operations
  patterns.add<
      TosaElementwiseBinaryOpConversionPattern<ttir::AddOp, tosa::AddOp>,
      TosaElementwiseBinaryOpConversionPattern<ttir::SubtractOp, tosa::SubOp>>(
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
               CosOpConversionPattern, DotGeneralOpConversionPattern,
               GatherOpConversionPattern, LogicalNotOpConversionPattern,
               MaxOpConversionPattern, SumOpConversionPattern,
               ReduceOrOpConversionPattern>(typeConverter, ctx);

  // Special operations
  patterns.add<WhereOpConversionPattern, ReshapeOpConversionPattern,
               TransposeOpConversionPattern, ConcatOpConversionPattern>(
      typeConverter, ctx);
}

} // namespace mlir::tt
