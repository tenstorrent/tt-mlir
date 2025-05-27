// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTosa/TTIRToTosa.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

namespace mlir::tt {
//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//
namespace {
// Convert a tensor of floating-point values to a tensor of boolean values
// by comparing with zero
Value convertToBooleanTensor(Value input, Location loc,
                             ConversionPatternRewriter &rewriter) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return input;

  // If it's already a boolean tensor, return it as is
  if (inputType.getElementType().isInteger(1))
    return input;

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
std::pair<Value, Value>
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

// Helper function to handle common logic for reduction operations
template <typename ReductionOp>
Value createReductionOpChain(Operation *op, Value input,
                             SmallVector<int64_t> &dims, bool keepDim,
                             ConversionPatternRewriter &rewriter) {
  // Get the expected output shape from the original operation
  auto expectedResultType =
      dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto expectedShape = expectedResultType.getShape();

  // Debug prints
  llvm::errs() << "DEBUG: Operation: " << op->getName() << "\n";
  llvm::errs() << "DEBUG: Expected output shape: [";
  for (auto dim : expectedShape) {
    llvm::errs() << dim << " ";
  }
  llvm::errs() << "]\n";

  llvm::errs() << "DEBUG: Dimensions to reduce: [";
  for (auto dim : dims) {
    llvm::errs() << dim << " ";
  }
  llvm::errs() << "]\n";

  llvm::errs() << "DEBUG: Keep dimensions: " << (keepDim ? "true" : "false")
               << "\n";

  // Special case: if no dimensions are specified, directly create the result
  // with the expected shape
  if (dims.empty()) {
    // Get input shape to determine all dimensions
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto inputRank = inputType.getRank();

    // Reduce along all dimensions
    for (int64_t i = 0; i < inputRank; ++i) {
      dims.push_back(i);
    }

    llvm::errs()
        << "DEBUG: No dimensions specified, reducing along all dimensions: [";
    for (auto dim : dims) {
      llvm::errs() << dim << " ";
    }
    llvm::errs() << "]\n";
  }

  // Sort dimensions in descending order to avoid changing the indices of
  // earlier dimensions
  std::sort(dims.begin(), dims.end(), std::greater<int64_t>());

  // Create a chain of reduction ops for each dimension
  Value result = input;

  // For each dimension to reduce, create a reduction operation
  for (size_t i = 0; i < dims.size(); ++i) {
    int64_t dim = dims[i];
    auto axisAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(dim));

    // Create a new result type for this reduction
    auto inputType = mlir::cast<RankedTensorType>(result.getType());
    auto inputShape = inputType.getShape();
    auto elementType = inputType.getElementType();

    llvm::errs() << "DEBUG: Reducing dimension " << dim << " of input shape: [";
    for (auto d : inputShape) {
      llvm::errs() << d << " ";
    }
    llvm::errs() << "]\n";

    // Determine the output shape for this reduction
    SmallVector<int64_t> newShape;

    // For all reductions, follow standard reduction rules
    for (size_t j = 0; j < inputShape.size(); ++j) {
      if (static_cast<int64_t>(j) == dim) {
        if (keepDim) {
          newShape.push_back(1);
        }
      } else {
        newShape.push_back(inputShape[j]);
      }
    }

    // If this is the last dimension to reduce, check if we need to reshape to
    // match expected output
    if (i == dims.size() - 1) {
      // Check if the shape after reduction matches the expected shape
      bool shapesMatch = newShape.size() == expectedShape.size();
      if (shapesMatch) {
        for (size_t j = 0; j < newShape.size(); ++j) {
          if (newShape[j] != expectedShape[j]) {
            shapesMatch = false;
            break;
          }
        }
      }

      // If shapes don't match, use the expected shape directly
      if (!shapesMatch) {
        newShape.assign(expectedShape.begin(), expectedShape.end());
      }

      llvm::errs() << "DEBUG: Final reduction, output shape: [";
      for (auto d : newShape) {
        llvm::errs() << d << " ";
      }
      llvm::errs() << "]\n";
    } else {
      llvm::errs() << "DEBUG: Intermediate reduction, output shape: [";
      for (auto d : newShape) {
        llvm::errs() << d << " ";
      }
      llvm::errs() << "]\n";
    }

    auto resultType = RankedTensorType::get(newShape, elementType);
    result = rewriter.create<ReductionOp>(op->getLoc(), resultType, result,
                                          axisAttr);

    // Debug print the created operation
    llvm::errs() << "DEBUG: Created reduction operation: "
                 << *result.getDefiningOp() << "\n";
  }

  return result;
}

// Helper function to extract dimensions from dim_arg attribute
SmallVector<int64_t> getDimsFromAttribute(Operation *op, Value input) {
  SmallVector<int64_t> dims;

  // Get the dim_arg attribute
  if (auto dimAttr = op->getAttrOfType<ArrayAttr>("dim_arg")) {
    llvm::errs() << "DEBUG: Found dim_arg attribute with size: "
                 << dimAttr.size() << "\n";

    // If the attribute is empty, reduce along all dimensions
    if (dimAttr.size() == 0) {
      auto inputType = mlir::cast<RankedTensorType>(input.getType());
      auto inputRank = inputType.getRank();
      for (int64_t i = 0; i < inputRank; ++i) {
        dims.push_back(i);
      }
      llvm::errs() << "DEBUG: Empty dim_arg, reducing all dimensions\n";
    } else {
      // Otherwise use the provided dimensions
      for (auto dimAttrValue : dimAttr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(dimAttrValue)) {
          int64_t dim = intAttr.getInt();
          dims.push_back(dim);
          llvm::errs() << "DEBUG: Adding dimension: " << dim << "\n";
        }
      }
    }
  } else {
    // If no dim_arg is provided, reduce along all dimensions
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto inputRank = inputType.getRank();
    for (int64_t i = 0; i < inputRank; ++i) {
      dims.push_back(i);
    }
    llvm::errs() << "DEBUG: No dim_arg attribute, reducing all dimensions\n";
  }

  // Debug print the final dimensions to reduce
  llvm::errs() << "DEBUG: Final dimensions to reduce: [";
  for (auto dim : dims) {
    llvm::errs() << dim << " ";
  }
  llvm::errs() << "]\n";

  return dims;
}

// Helper function to extract keep_dim attribute
bool getKeepDimFromAttribute(Operation *op) {
  bool keepDim = false;
  if (auto keepDimAttr = op->getAttrOfType<BoolAttr>("keep_dim")) {
    keepDim = keepDimAttr.getValue();
  }
  return keepDim;
}

} // namespace

//===----------------------------------------------------------------------===//
// Elementwise Unary Operation Conversion Patterns
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

//===----------------------------------------------------------------------===//
// Elementwise Binary Operation Conversion Patterns
//===----------------------------------------------------------------------===//
namespace {
template <typename TTIROpTy, typename TosaOpTy>
class ElementwiseBinaryOpConversionPattern
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

//===----------------------------------------------------------------------===//
// WhereOp Conversion Pattern
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// Reshape Operation Conversion Pattern
//===----------------------------------------------------------------------===//
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
    SmallVector<int32_t> newShapeValues(newShape.begin(), newShape.end());

    // Create the shape type
    auto shapeType =
        mlir::tosa::shapeType::get(rewriter.getContext(), newShape.size());

    // Convert int32_t values to int64_t for getIndexTensorAttr
    SmallVector<int64_t> newShapeInt64;
    for (auto dim : newShapeValues) {
      newShapeInt64.push_back(static_cast<int64_t>(dim));
    }

    // Create the index tensor attribute for the shape
    auto attr = rewriter.getIndexTensorAttr(newShapeInt64);

    // Create the tosa.const_shape operation
    auto shapeOp =
        rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

    // Create the reshape operation with the shape operand
    auto result = rewriter.create<tosa::ReshapeOp>(
        op.getLoc(), resultType, adaptor.getInput(), shapeOp.getResult());

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Transpose Operation Conversion Pattern
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// Constant Operation Conversion Pattern
//===----------------------------------------------------------------------===//
namespace {
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

    auto result =
        rewriter.create<tosa::ConstOp>(op.getLoc(), resultType, value);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Concat Operation Conversion Pattern
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// Comparison Operation Conversion Patterns
//===----------------------------------------------------------------------===//

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

    auto resultType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Create a scalar zero attribute
    auto elementType = resultType.getElementType();
    TypedAttr zeroAttr;

    if (elementType.isF32()) {
      zeroAttr = rewriter.getF32FloatAttr(0.0f);
    } else if (elementType.isF64()) {
      zeroAttr = rewriter.getF64FloatAttr(0.0);
    } else if (elementType.isInteger(32)) {
      zeroAttr = rewriter.getI32IntegerAttr(0);
    } else if (elementType.isInteger(64)) {
      zeroAttr = rewriter.getI64IntegerAttr(0);
    } else if (elementType.isInteger(16)) {
      zeroAttr = rewriter.getI16IntegerAttr(0);
    } else if (elementType.isInteger(8)) {
      zeroAttr = rewriter.getI8IntegerAttr(0);
    } else if (elementType.isInteger(1)) {
      zeroAttr = rewriter.getBoolAttr(false);
    } else {
      // Default to F32 for other types
      zeroAttr = rewriter.getF32FloatAttr(0.0f);
    }

    // Create a scalar value using arith.constant
    auto scalarZero =
        rewriter.create<arith::ConstantOp>(op.getLoc(), elementType, zeroAttr);

    // Create a tensor filled with zeros of the result shape
    auto zeroTensor = rewriter.create<tensor::SplatOp>(op.getLoc(), resultType,
                                                       scalarZero.getResult());

    // Use addition with the zero tensor to achieve broadcasting
    auto result = rewriter.create<tosa::AddOp>(op.getLoc(), resultType, input,
                                               zeroTensor);

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
class LogicalNotOpConversionPattern
    : public OpConversionPattern<ttir::LogicalNotOp> {
public:
  using OpConversionPattern<ttir::LogicalNotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LogicalNotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto resultType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Convert input to boolean tensor
    input = convertToBooleanTensor(input, op.getLoc(), rewriter);

    // Create the logical not operation with the boolean input
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));
    auto boolResult =
        rewriter.create<tosa::LogicalNotOp>(op.getLoc(), boolType, input);

    // If the result type is not boolean, convert back to the original type
    if (resultType.getElementType() != rewriter.getIntegerType(1)) {
      // Create true and false constants with the result type
      auto [trueValue, falseValue] =
          createTrueAndFalseSplatConstants(resultType, op.getLoc(), rewriter);

      // Use select to convert boolean result back to the original type
      auto result = rewriter.create<tosa::SelectOp>(
          op.getLoc(), resultType, boolResult, falseValue, trueValue);

      rewriter.replaceOp(op, result);
    } else {
      // If the result type is already boolean, use the boolean result directly
      rewriter.replaceOp(op, boolResult);
    }

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

    // Get dimensions to reduce and keep_dim attribute
    SmallVector<int64_t> dims = getDimsFromAttribute(op, input);
    bool keepDim = getKeepDimFromAttribute(op);

    // Create the chain of reduction operations
    Value result = createReductionOpChain<tosa::ReduceMaxOp>(op, input, dims,
                                                             keepDim, rewriter);

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

    // Get dimensions to reduce and keep_dim attribute
    SmallVector<int64_t> dims = getDimsFromAttribute(op, input);
    bool keepDim = getKeepDimFromAttribute(op);

    // Create the chain of reduction operations
    Value result = createReductionOpChain<tosa::ReduceSumOp>(op, input, dims,
                                                             keepDim, rewriter);

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

    // Get dimensions to reduce and keep_dim attribute
    SmallVector<int64_t> dims = getDimsFromAttribute(op, input);
    bool keepDim = getKeepDimFromAttribute(op);

    // Convert input to boolean tensor
    input = convertToBooleanTensor(input, op.getLoc(), rewriter);

    // Create the chain of reduction operations
    Value result = createReductionOpChain<tosa::ReduceAnyOp>(op, input, dims,
                                                             keepDim, rewriter);

    // Convert the boolean result back to the original type
    // First create the true and false constants with the result type
    auto [trueVal, falseVal] =
        createTrueAndFalseSplatConstants(resultType, op.getLoc(), rewriter);

    // Use a select operation to convert from boolean to the expected type
    result = rewriter.create<tosa::SelectOp>(op.getLoc(), resultType, result,
                                             trueVal, falseVal);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// MatmulOp Conversion Pattern
//===----------------------------------------------------------------------===//
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

    // First, reshape both tensors to 3D if needed
    Value lhs3D = lhs;
    Value rhs3D = rhs;

    // Reshape LHS to 3D if needed
    if (lhsType.getRank() < 3) {
      SmallVector<int64_t> newShape;
      if (lhsType.getRank() == 1) {
        // For 1D tensor [K], reshape to [1, 1, K]
        newShape = {1, 1, lhsType.getShape()[0]};
      } else if (lhsType.getRank() == 2) {
        // For 2D tensor [M, K], reshape to [1, M, K]
        newShape = {1, lhsType.getShape()[0], lhsType.getShape()[1]};
      }

      // Create the reshape op
      auto newType = RankedTensorType::get(newShape, lhsType.getElementType());

      // Create a shape value using tosa::ConstShapeOp
      auto shapeType =
          mlir::tosa::shapeType::get(rewriter.getContext(), newShape.size());

      // Convert to int64_t for getIndexTensorAttr
      SmallVector<int64_t> newShapeInt64;
      for (auto dim : newShape) {
        newShapeInt64.push_back(static_cast<int64_t>(dim));
      }

      // Create the index tensor attribute for the shape
      auto attr = rewriter.getIndexTensorAttr(newShapeInt64);

      // Create the tosa.const_shape operation
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      // Reshape lhs to add the batch dimensions
      lhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, lhs,
                                               shapeOp.getResult());

      // Update lhsType
      lhsType = newType;
    } else if (lhsType.getRank() > 3) {
      // For tensors with rank > 3, collapse all but the last two dimensions
      int64_t collapsedBatchSize = 1;
      for (int i = 0; i < lhsType.getRank() - 2; ++i) {
        collapsedBatchSize *= lhsType.getShape()[i];
      }

      SmallVector<int64_t> newShape = {
          collapsedBatchSize, lhsType.getShape()[lhsType.getRank() - 2],
          lhsType.getShape()[lhsType.getRank() - 1]};

      // Create the reshape op
      auto newType = RankedTensorType::get(newShape, lhsType.getElementType());

      // Create a shape value using tosa::ConstShapeOp
      auto shapeType =
          mlir::tosa::shapeType::get(rewriter.getContext(), newShape.size());

      // Convert to int64_t for getIndexTensorAttr
      SmallVector<int64_t> newShapeInt64;
      for (auto dim : newShape) {
        newShapeInt64.push_back(static_cast<int64_t>(dim));
      }

      // Create the index tensor attribute for the shape
      auto attr = rewriter.getIndexTensorAttr(newShapeInt64);

      // Create the tosa.const_shape operation
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      // Reshape lhs to add the batch dimensions
      lhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, lhs,
                                               shapeOp.getResult());

      // Update lhsType
      lhsType = newType;
    } else {
      lhs3D = lhs;
    }

    // Reshape RHS to 3D if needed
    if (rhsType.getRank() < 3) {
      SmallVector<int64_t> newShape;
      if (rhsType.getRank() == 1) {
        // For 1D tensor [K], reshape to [1, K, 1]
        newShape = {1, rhsType.getShape()[0], 1};
      } else if (rhsType.getRank() == 2) {
        // For 2D tensor [K, N], reshape to [1, K, N]
        newShape = {1, rhsType.getShape()[0], rhsType.getShape()[1]};
      }

      // Create the reshape op
      auto newType = RankedTensorType::get(newShape, rhsType.getElementType());

      // Create a shape value using tosa::ConstShapeOp
      auto shapeType =
          mlir::tosa::shapeType::get(rewriter.getContext(), newShape.size());

      // Convert to int64_t for getIndexTensorAttr
      SmallVector<int64_t> newShapeInt64;
      for (auto dim : newShape) {
        newShapeInt64.push_back(static_cast<int64_t>(dim));
      }

      // Create the index tensor attribute for the shape
      auto attr = rewriter.getIndexTensorAttr(newShapeInt64);

      // Create the tosa.const_shape operation
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      // Reshape rhs to add the batch dimensions
      rhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, rhs,
                                               shapeOp.getResult());

      // Update rhsType
      rhsType = newType;
    } else if (rhsType.getRank() > 3) {
      // For tensors with rank > 3, collapse all but the last two dimensions
      int64_t collapsedBatchSize = 1;
      for (int i = 0; i < rhsType.getRank() - 2; ++i) {
        collapsedBatchSize *= rhsType.getShape()[i];
      }

      SmallVector<int64_t> newShape = {
          collapsedBatchSize, rhsType.getShape()[rhsType.getRank() - 2],
          rhsType.getShape()[rhsType.getRank() - 1]};

      // Create the reshape op
      auto newType = RankedTensorType::get(newShape, rhsType.getElementType());

      // Create a shape value using tosa::ConstShapeOp
      auto shapeType =
          mlir::tosa::shapeType::get(rewriter.getContext(), newShape.size());

      // Convert to int64_t for getIndexTensorAttr
      SmallVector<int64_t> newShapeInt64;
      for (auto dim : newShape) {
        newShapeInt64.push_back(static_cast<int64_t>(dim));
      }

      // Create the index tensor attribute for the shape
      auto attr = rewriter.getIndexTensorAttr(newShapeInt64);

      // Create the tosa.const_shape operation
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      // Reshape rhs to add the batch dimensions
      rhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, rhs,
                                               shapeOp.getResult());

      // Update rhsType
      rhsType = newType;
    } else {
      rhs3D = rhs;
    }

    // Get the types after reshaping
    auto lhs3DType = cast<RankedTensorType>(lhs3D.getType());
    auto rhs3DType = cast<RankedTensorType>(rhs3D.getType());

    // Handle transpose for LHS if needed
    if (transposeA) {
      // Create permutation for transposing the last two dimensions
      SmallVector<int32_t> perm;
      for (int i = 0; i < lhs3DType.getRank() - 2; ++i) {
        perm.push_back(i);
      }
      perm.push_back(lhs3DType.getRank() - 1);
      perm.push_back(lhs3DType.getRank() - 2);

      // Calculate the transposed shape
      SmallVector<int64_t> transposedShape;
      for (size_t i = 0; i < perm.size(); ++i) {
        transposedShape.push_back(lhs3DType.getShape()[perm[i]]);
      }

      // Create the transpose op
      auto transposedType =
          RankedTensorType::get(transposedShape, lhs3DType.getElementType());

      lhs3D = rewriter.create<tosa::TransposeOp>(op.getLoc(), transposedType,
                                                 lhs3D, perm);
    }

    // Handle transpose for RHS if needed
    if (transposeB) {
      // Create permutation for transposing the last two dimensions
      SmallVector<int32_t> perm;
      for (int i = 0; i < rhs3DType.getRank() - 2; ++i) {
        perm.push_back(i);
      }
      perm.push_back(rhs3DType.getRank() - 1);
      perm.push_back(rhs3DType.getRank() - 2);

      // Calculate the transposed shape
      SmallVector<int64_t> transposedShape;
      for (size_t i = 0; i < perm.size(); ++i) {
        transposedShape.push_back(rhs3DType.getShape()[perm[i]]);
      }

      // Create the transpose op
      auto transposedType =
          RankedTensorType::get(transposedShape, rhs3DType.getElementType());

      rhs3D = rewriter.create<tosa::TransposeOp>(op.getLoc(), transposedType,
                                                 rhs3D, perm);
    }

    // Check if we need to broadcast batch dimensions
    if (lhs3DType.getShape()[0] != rhs3DType.getShape()[0]) {
      // We need to broadcast one of the inputs to match the other's batch
      // dimension
      if (lhs3DType.getShape()[0] == 1 && rhs3DType.getShape()[0] > 1) {
        // Create a tensor of zeros with the target batch size for broadcasting
        auto zeroType = RankedTensorType::get({rhs3DType.getShape()[0],
                                               lhs3DType.getShape()[1],
                                               lhs3DType.getShape()[2]},
                                              lhs3DType.getElementType());

        // Create a scalar zero attribute
        auto elementType = lhs3DType.getElementType();
        TypedAttr zeroAttr;

        if (elementType.isF32()) {
          zeroAttr = rewriter.getF32FloatAttr(0.0f);
        } else if (elementType.isF64()) {
          zeroAttr = rewriter.getF64FloatAttr(0.0);
        } else if (elementType.isInteger(32)) {
          zeroAttr = rewriter.getI32IntegerAttr(0);
        } else if (elementType.isInteger(64)) {
          zeroAttr = rewriter.getI64IntegerAttr(0);
        } else if (elementType.isInteger(16)) {
          zeroAttr = rewriter.getI16IntegerAttr(0);
        } else if (elementType.isInteger(8)) {
          zeroAttr = rewriter.getI8IntegerAttr(0);
        } else if (elementType.isInteger(1)) {
          zeroAttr = rewriter.getBoolAttr(false);
        } else {
          // Default to F32 for other types
          zeroAttr = rewriter.getF32FloatAttr(0.0f);
        }

        // Create a scalar value using arith.constant
        auto scalarZero = rewriter.create<arith::ConstantOp>(
            op.getLoc(), elementType, zeroAttr);

        // Create a tensor filled with zeros of the result shape
        auto zeroTensor = rewriter.create<tensor::SplatOp>(
            op.getLoc(), zeroType, scalarZero.getResult());

        // Reshape lhs to [1, M, K] if needed
        Value lhsReshaped = lhs3D;
        if (lhs3DType.getRank() != 3) {
          auto reshapeType = RankedTensorType::get(
              {1, lhs3DType.getDimSize(0), lhs3DType.getDimSize(1)},
              lhs3DType.getElementType());

          // Create a shape value using tosa::ConstShapeOp
          auto shapeType = mlir::tosa::shapeType::get(rewriter.getContext(), 3);
          SmallVector<int64_t> shapeValues = {1, lhs3DType.getDimSize(0),
                                              lhs3DType.getDimSize(1)};
          auto attr = rewriter.getIndexTensorAttr(shapeValues);
          auto shapeOp =
              rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

          lhsReshaped = rewriter.create<tosa::ReshapeOp>(
              op.getLoc(), reshapeType, lhs3D, shapeOp.getResult());
        }

        // Use addition with the zero tensor to achieve broadcasting
        lhs3D = rewriter.create<tosa::AddOp>(op.getLoc(), zeroType, lhsReshaped,
                                             zeroTensor);

        // Update lhs3DType
        lhs3DType = cast<RankedTensorType>(lhs3D.getType());
      } else if (rhs3DType.getShape()[0] == 1 && lhs3DType.getShape()[0] > 1) {
        // Create a tensor of zeros with the target batch size for broadcasting
        auto zeroType = RankedTensorType::get({lhs3DType.getShape()[0],
                                               rhs3DType.getShape()[1],
                                               rhs3DType.getShape()[2]},
                                              rhs3DType.getElementType());

        // Create a scalar zero attribute
        auto elementType = rhs3DType.getElementType();
        TypedAttr zeroAttr;

        if (elementType.isF32()) {
          zeroAttr = rewriter.getF32FloatAttr(0.0f);
        } else if (elementType.isF64()) {
          zeroAttr = rewriter.getF64FloatAttr(0.0);
        } else if (elementType.isInteger(32)) {
          zeroAttr = rewriter.getI32IntegerAttr(0);
        } else if (elementType.isInteger(64)) {
          zeroAttr = rewriter.getI64IntegerAttr(0);
        } else if (elementType.isInteger(16)) {
          zeroAttr = rewriter.getI16IntegerAttr(0);
        } else if (elementType.isInteger(8)) {
          zeroAttr = rewriter.getI8IntegerAttr(0);
        } else if (elementType.isInteger(1)) {
          zeroAttr = rewriter.getBoolAttr(false);
        } else {
          // Default to F32 for other types
          zeroAttr = rewriter.getF32FloatAttr(0.0f);
        }

        // Create a scalar value using arith.constant
        auto scalarZero = rewriter.create<arith::ConstantOp>(
            op.getLoc(), elementType, zeroAttr);

        // Create a tensor filled with zeros of the result shape
        auto zeroTensor = rewriter.create<tensor::SplatOp>(
            op.getLoc(), zeroType, scalarZero.getResult());

        // Reshape rhs to [1, K, N] if needed
        Value rhsReshaped = rhs3D;
        if (rhs3DType.getRank() != 3) {
          auto reshapeType = RankedTensorType::get(
              {1, rhs3DType.getDimSize(0), rhs3DType.getDimSize(1)},
              rhs3DType.getElementType());

          // Create a shape value using tosa::ConstShapeOp
          auto shapeType = mlir::tosa::shapeType::get(rewriter.getContext(), 3);
          SmallVector<int64_t> shapeValues = {1, rhs3DType.getDimSize(0),
                                              rhs3DType.getDimSize(1)};
          auto attr = rewriter.getIndexTensorAttr(shapeValues);
          auto shapeOp =
              rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

          rhsReshaped = rewriter.create<tosa::ReshapeOp>(
              op.getLoc(), reshapeType, rhs3D, shapeOp.getResult());
        }

        // Use addition with the zero tensor to achieve broadcasting
        rhs3D = rewriter.create<tosa::AddOp>(op.getLoc(), zeroType, rhsReshaped,
                                             zeroTensor);

        // Update rhs3DType
        rhs3DType = cast<RankedTensorType>(rhs3D.getType());
      }
    }

    // Now both tensors should have the same batch dimension
    auto matmulResultType = RankedTensorType::get(
        {std::max(lhs3DType.getShape()[0], rhs3DType.getShape()[0]),
         lhs3DType.getShape()[1], rhs3DType.getShape()[2]},
        resultType.getElementType());

    Value matmulResult = rewriter.create<tosa::MatMulOp>(
        op.getLoc(), matmulResultType, lhs3D, rhs3D);

    // Reshape the result back to the expected shape if needed
    if (resultType != matmulResultType) {
      // Create a shape value using tosa::ConstShapeOp
      auto shapeType = mlir::tosa::shapeType::get(rewriter.getContext(),
                                                  resultType.getShape().size());

      // Convert to int64_t for getIndexTensorAttr
      SmallVector<int64_t> resultShapeInt64;
      for (auto dim : resultType.getShape()) {
        resultShapeInt64.push_back(static_cast<int64_t>(dim));
      }

      // Create the index tensor attribute for the shape
      auto attr = rewriter.getIndexTensorAttr(resultShapeInt64);

      // Create the tosa.const_shape operation
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      matmulResult = rewriter.create<tosa::ReshapeOp>(
          op.getLoc(), resultType, matmulResult, shapeOp.getResult());
    }

    rewriter.replaceOp(op, matmulResult);
    return success();
  }

private:
  // Helper function to transpose a shape according to permutation
  SmallVector<int64_t> transposeShape(ArrayRef<int64_t> shape,
                                      ArrayRef<int64_t> permutation) const {
    SmallVector<int64_t> result(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      result[i] = shape[permutation[i]];
    }
    return result;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

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
  // patterns
  //     .add<ElementwiseBinaryOpConversionPattern<ttir::AddOp, tosa::AddOp>,
  //          ElementwiseBinaryOpConversionPattern<ttir::SubtractOp,
  //          tosa::SubOp>,
  //          ElementwiseBinaryOpConversionPattern<ttir::MultiplyOp,
  //          tosa::MulOp>>(
  //         typeConverter, ctx);

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
               CosOpConversionPattern, LogicalNotOpConversionPattern,
               MaxOpConversionPattern, SumOpConversionPattern>(typeConverter,
                                                               ctx);

  // Special operations
  patterns.add<WhereOpConversionPattern, ReshapeOpConversionPattern,
               TransposeOpConversionPattern, MatmulOpConversionPattern,
               ConcatOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
