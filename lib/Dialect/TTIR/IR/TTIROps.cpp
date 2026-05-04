// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.cpp.inc"
#include "ttmlir/Dialect/TTIR/Utils/QuantUtils.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTIR/Utils/VerificationUtils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <utility>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROps.cpp.inc"

namespace mlir::tt::ttir {

//===----------------------------------------------------------------------===//
// Constant folding helpers
//===----------------------------------------------------------------------===//

// Reshape attribute if it is splat otherwise return nullptr.
static DenseElementsAttr reshapeIfSplat(ShapedType type, Attribute attr) {
  if (auto splat = llvm::dyn_cast<SplatElementsAttr>(attr)) {
    return splat.resizeSplat(type);
  }
  return nullptr;
}

// Heuristic for whether constant folding should run when the input is not a
// splat, based on the output size. Folding is skipped for very large tensors
// to avoid dramatically increasing compile time and memory usage.
static bool shouldFold(mlir::Operation *op) {
  constexpr int64_t foldLimit = 1'000'000;
  mlir::Type resultType = op->getResult(0).getType();
  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(resultType);
  if (!shapedType) {
    return false;
  }
  return ttmlir::utils::volume(shapedType.getShape()) <= foldLimit;
}

// Helper to fold a tensor manipulation operation with a non-splat constant
// argument using an index mapping function. The index mapping function takes
// output coordinates and returns input coordinates.
template <typename ElementType, typename Fun>
static ::mlir::OpFoldResult foldNonSplatTM(ShapedType resultType,
                                           DenseElementsAttr inputAttr,
                                           Fun indexMap) {
  auto inputValues = inputAttr.getValues<ElementType>();
  llvm::SmallVector<ElementType> outputValues;
  outputValues.reserve(resultType.getNumElements());

  auto inputStrides = mlir::computeStrides(inputAttr.getType().getShape());
  auto outputShape = resultType.getShape();

  llvm::SmallVector<int64_t> outputCoord(outputShape.size(), 0);
  for (int64_t i = 0; i != resultType.getNumElements(); ++i) {
    llvm::SmallVector<int64_t> inputCoord = indexMap(outputCoord);
    int64_t inputIndex = mlir::linearize(inputCoord, inputStrides);
    outputValues.push_back(inputValues[inputIndex]);

    // Increment output coordinates in row-major order. Increment the innermost
    // dimension; if it reaches the dimension size, reset it to 0 and carry into
    // the next outer dimension.
    for (int64_t dim = outputShape.size() - 1; dim >= 0; --dim) {
      if (++outputCoord[dim] < outputShape[dim]) {
        break;
      }
      outputCoord[dim] = 0;
    }
  }

  return mlir::DenseElementsAttr::get(resultType, outputValues);
}

// Helper to fold a constant tensor manipulation operation using an index
// mapping function. The index mapping function takes output coordinates and
// returns input coordinates.
template <typename Fun>
static ::mlir::OpFoldResult
constantFoldTM(mlir::Operation *op, mlir::Attribute inputAttr, Fun indexMap) {
  if (!inputAttr) {
    return nullptr;
  }

  ShapedType resultType = mlir::cast<ShapedType>(op->getResult(0).getType());
  if (auto foldResult = reshapeIfSplat(resultType, inputAttr)) {
    return foldResult;
  }

  if (!shouldFold(op)) {
    return nullptr;
  }

  if (auto denseAttr =
          llvm::dyn_cast_if_present<mlir::DenseElementsAttr>(inputAttr)) {
    if (resultType.getElementType().isFloat()) {
      return foldNonSplatTM<llvm::APFloat>(resultType, denseAttr, indexMap);
    }
    if (resultType.getElementType().isInteger()) {
      return foldNonSplatTM<llvm::APInt>(resultType, denseAttr, indexMap);
    }
  }
  return nullptr;
}

// Wrapper to avoid repetitive writing of template arguments of constFoldCastOp.
template <
    class AttrElementT, class TargetAttrElementT,
    class ElementValueT = typename AttrElementT::ValueType,
    class TargetElementValueT = typename TargetAttrElementT::ValueType,
    class CalculationT = function_ref<TargetElementValueT(ElementValueT, bool)>>
Attribute foldCast(ArrayRef<Attribute> operands, Type resType,
                   CalculationT &&calculate) {
  return mlir::constFoldCastOp<AttrElementT, TargetAttrElementT, ElementValueT,
                               TargetElementValueT, void, CalculationT>(
      operands, resType, std::forward<CalculationT>(calculate));
}

// Helper to perform constant folding of elementwise unary operators when
// element type is float. `floatMap` should perform a unary operation on
// `APFloat` values respectively.
template <typename Fun>
static ::mlir::Attribute
constantFoldEltwiseUnaryFloat(mlir::Operation *op, mlir::Attribute inputAttr,
                              Fun floatMap) {
  mlir::DenseElementsAttr input =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(inputAttr);
  if (!input || !input.getElementType().isFloat()) {
    return nullptr;
  }

  if (op->getOperand(0).getType() != op->getResult(0).getType()) {
    return nullptr;
  }
  if (!input.isSplat() && !shouldFold(op)) {
    return nullptr;
  }

  return input.mapValues(input.getElementType(),
                         [floatMap](const llvm::APFloat &value) {
                           llvm::APFloat result = floatMap(value);
                           // `mapValues` expects the lambda to return an APInt,
                           // so we reinterpret the bits of the APFloat result
                           // as an APInt before returning.
                           return result.bitcastToAPInt();
                         });
}

// Helper to perform constant folding of elementwise unary operators when
// element type is integer. `intMap` should perform a unary operation on `APInt`
// values respectively.
template <typename Fun>
static ::mlir::Attribute constantFoldEltwiseUnaryInt(mlir::Operation *op,
                                                     mlir::Attribute inputAttr,
                                                     Fun intMap) {
  mlir::DenseElementsAttr input =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(inputAttr);
  if (!input || !input.getElementType().isInteger()) {
    return nullptr;
  }

  if (op->getOperand(0).getType() != op->getResult(0).getType()) {
    // Avoid implicit type conversion in folders since it does not happen often.
    return nullptr;
  }
  if (!input.isSplat() && !shouldFold(op)) {
    return nullptr;
  }

  return input.mapValues(input.getElementType(), intMap);
}

// Helper to perform constant folding of elementwise unary operators. `floatMap`
// and `intMap` should perform a unary operation on `APFloat` and `APInt` values
// respectively.
template <typename FloatMap, typename IntMap>
static ::mlir::OpFoldResult
constantFoldEltwiseUnary(mlir::Operation *op, mlir::Attribute inputAttr,
                         FloatMap floatMap, IntMap intMap) {
  mlir::DenseElementsAttr input =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(inputAttr);
  if (!input) {
    return nullptr;
  }

  if (input.getElementType().isFloat()) {
    return constantFoldEltwiseUnaryFloat(op, inputAttr, floatMap);
  }
  if (input.getElementType().isInteger()) {
    return constantFoldEltwiseUnaryInt(op, inputAttr, intMap);
  }
  return nullptr;
}

// Callable that maps a C++ float function over an APFloat by converting it to
// f32 and back to the original type. This allows folding with non standard
// float types like bf16 using functions from C++ standard library.
template <typename Fun>
class ApplyToAPFloat {
public:
  explicit ApplyToAPFloat(Fun fn) : fn{fn} {}
  llvm::APFloat operator()(const llvm::APFloat &value) const {
    // If the input is already a float32, apply the function directly.
    if (&value.getSemantics() == &llvm::APFloat::IEEEsingle()) {
      float nativeFloat = value.convertToFloat();
      float result = fn(nativeFloat);
      return llvm::APFloat(result);
    }

    // Else, convert to float32, apply the function, and convert back.
    // Note that we don't support f64, so this won't lower precision.
    llvm::APFloat floatVal = value;
    bool losesInfo{};
    floatVal.convert(llvm::APFloat::IEEEsingle(),
                     llvm::APFloat::rmNearestTiesToEven, &losesInfo);

    float nativeFloat = floatVal.convertToFloat();
    float result = fn(nativeFloat);

    llvm::APFloat finalResult(result);
    finalResult.convert(value.getSemantics(),
                        llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    return finalResult;
  }

private:
  Fun fn;
};

// Deduction guide to avoid having to write the template parameter at the
// call site.
template <typename Fun>
ApplyToAPFloat(Fun) -> ApplyToAPFloat<Fun>;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

bool mlir::tt::ttir::AddOp::isQuantizedRewriteFavorable(
    mlir::ArrayRef<mlir::Value> sourceOperands) {
  // If the operands are both quantized but the types do not align, return
  // false.
  return mlir::tt::ttir::utils::areQuantizationParamsAligned(sourceOperands);
}

mlir::Operation *mlir::tt::ttir::AddOp::rewriteWithQuantizedInputs(
    mlir::PatternRewriter &rewriter,
    mlir::ArrayRef<mlir::Value> sourceOperands) {
  // Two cases:
  // 1. One operand is quantized and the other is not: apply quantization and
  //    proceed to case two.
  // 2. Both operands are quantized: supported, return quantized add.
  assert(sourceOperands.size() == 2 && "AddOp should have two operands.");
  auto lhs = sourceOperands[0];
  auto rhs = sourceOperands[1];

  RankedTensorType lhsType = mlir::cast<RankedTensorType>(lhs.getType());
  RankedTensorType rhsType = mlir::cast<RankedTensorType>(rhs.getType());

  auto lhsElemQ =
      mlir::dyn_cast<mlir::quant::QuantizedType>(lhsType.getElementType());
  auto rhsElemQ =
      mlir::dyn_cast<mlir::quant::QuantizedType>(rhsType.getElementType());

  // One operand is dequantized, one is quantized — try to quantize the
  // dequantized one.
  if ((lhsElemQ && !rhsElemQ) || (!lhsElemQ && rhsElemQ)) {
    Value quantVal = lhsElemQ ? lhs : rhs;
    Value dequantVal = lhsElemQ ? rhs : lhs;
    auto quantElemQ = lhsElemQ ? lhsElemQ : rhsElemQ;
    auto quantType = mlir::cast<mlir::RankedTensorType>(quantVal.getType());
    auto expressedType =
        mlir::cast<mlir::RankedTensorType>(dequantVal.getType())
            .getElementType();

    // Insert quantize op for the dequantized value (the types must be
    // compatible).
    if (!isa<mlir::quant::UniformQuantizedType,
             mlir::quant::UniformQuantizedPerAxisType>(quantElemQ)) {
      return nullptr;
    }
    if (expressedType != quantElemQ.getExpressedType()) {
      return nullptr;
    }

    RankedTensorType newType = RankedTensorType::get(
        mlir::cast<mlir::RankedTensorType>(dequantVal.getType()).getShape(),
        quantElemQ, quantType.getEncoding());

    auto quantizedInput =
        rewriter.create<ttir::QuantizeOp>(getLoc(), newType, dequantVal);

    // Update operands.
    if (lhsElemQ) {
      rhs = quantizedInput;
    } else {
      lhs = quantizedInput;
      lhsElemQ = quantElemQ;
    }
  }
  // Now both values are quantized (and are equivalent).
  RankedTensorType oldType = mlir::cast<RankedTensorType>(getType());
  RankedTensorType newResultType = RankedTensorType::get(
      oldType.getShape(), lhsElemQ, oldType.getEncoding());

  // Emit new AddOp with quantized types.
  auto newAdd = rewriter.create<ttir::AddOp>(getLoc(), newResultType, lhs, rhs);
  return newAdd.getOperation();
}

//===----------------------------------------------------------------------===//
// BitwiseXorOp
//===----------------------------------------------------------------------===//

// BitwiseXorOp canonicalization
void mlir::tt::ttir::BitwiseXorOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  // x ^ x == 0
  patterns.add(
      +[](mlir::tt::ttir::BitwiseXorOp op, mlir::PatternRewriter &rewriter) {
        if (op.getLhs() != op.getRhs()) {
          return mlir::failure();
        }

        mlir::RankedTensorType tensorType = op.getResult().getType();
        auto elementType = tensorType.getElementType();
        Attribute zeroAttr;
        if (mlir::isa<mlir::FloatType>(elementType)) {
          zeroAttr = mlir::FloatAttr::get(elementType, 0.0);
        } else if (mlir::isa<mlir::IntegerType>(elementType)) {
          zeroAttr = mlir::IntegerAttr::get(elementType, 0);
        } else {
          return mlir::failure();
        }
        auto resultType = mlir::SplatElementsAttr::get(tensorType, zeroAttr);

        rewriter.replaceOpWithNewOp<ttir::ConstantOp>(
            op, op->getOperand(0).getType(), resultType);
        return mlir::success();
      });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// LessEqualOp
//===----------------------------------------------------------------------===//

// LessEqualOp canonicalization: le(a, b) -> ge(b, a)
void mlir::tt::ttir::LessEqualOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  patterns.add(
      +[](mlir::tt::ttir::LessEqualOp op, mlir::PatternRewriter &rewriter) {
        rewriter.replaceOpWithNewOp<mlir::tt::ttir::GreaterEqualOp>(
            op, op.getResult().getType(), op.getRhs(), op.getLhs());
        return mlir::success();
      });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// LessThanOp
//===----------------------------------------------------------------------===//

// LessThanOp canonicalization: lt(a, b) -> gt(b, a)
void mlir::tt::ttir::LessThanOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  patterns.add(
      +[](mlir::tt::ttir::LessThanOp op, mlir::PatternRewriter &rewriter) {
        rewriter.replaceOpWithNewOp<mlir::tt::ttir::GreaterThanOp>(
            op, op.getResult().getType(), op.getRhs(), op.getLhs());
        return mlir::success();
      });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// Constant value helpers
//===----------------------------------------------------------------------===//

// Helper to create a scalar attribute from an element type and a double value.
static mlir::Attribute makeScalarAttr(mlir::Type elemType, double val) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elemType)) {
    return mlir::FloatAttr::get(floatType, val);
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
    return mlir::IntegerAttr::get(intType, static_cast<int64_t>(val));
  }
  llvm_unreachable("Expected a FloatType or IntegerType");
}

// Extract constant fill value from FullOp, ZerosOp, or OnesOp.
static mlir::Attribute getConstantValue(mlir::Value value) {
  mlir::Operation *op = value.getDefiningOp();

  if (auto fullOp = mlir::dyn_cast_if_present<mlir::tt::ttir::FullOp>(op)) {
    return fullOp.getFillValueAttr();
  }

  if (mlir::isa_and_present<mlir::tt::ttir::ZerosOp>(op)) {
    return makeScalarAttr(
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
            .getElementType(),
        0.0);
  }
  if (mlir::isa_and_present<mlir::tt::ttir::OnesOp>(op)) {
    return makeScalarAttr(
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
            .getElementType(),
        1.0);
  }

  return {};
}

// Check if the attribute represents zero.
static bool isZeroAttr(mlir::Attribute attr) {
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
    return floatAttr.getValue().isZero();
  }
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    return intAttr.getValue().isZero();
  }
  return false;
}

static bool isConstantZero(mlir::Value value) {
  mlir::Attribute attr = getConstantValue(value);
  return attr && isZeroAttr(attr);
}

static bool isConstantNonZero(mlir::Value value) {
  mlir::Attribute attr = getConstantValue(value);
  return attr && !isZeroAttr(attr);
}

// Check if the attribute represents one.
static bool isOneAttr(mlir::Attribute attr) {
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
    return floatAttr.getValue().isExactlyValue(1.0);
  }
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    return intAttr.getValue().isOne();
  }
  return false;
}

static bool isConstantOne(mlir::Value value) {
  mlir::Attribute attr = getConstantValue(value);
  return attr && isOneAttr(attr);
}

// Helper to extract the shape of a RankedTensorType as a vector of i32.
static llvm::SmallVector<int32_t>
getShapeAsI32(mlir::RankedTensorType tensorType) {
  return llvm::to_vector_of<int32_t>(tensorType.getShape());
}

//===----------------------------------------------------------------------===//
// LogicalAndOp
//===----------------------------------------------------------------------===//

// Check if a value is known to be boolean-valued (exactly 0 or 1).
// True for: i1 types, constants that are exactly 0 or 1 and results of
// comparison/logical ops.
static bool isBooleanValued(mlir::Value value) {
  auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
  if (type.getElementType().isInteger(1)) {
    return true;
  }

  if (isConstantZero(value) || isConstantOne(value)) {
    return true;
  }

  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }

  // Comparison and logical ops always produce 0/1.
  return llvm::isa<mlir::tt::ttir::EqualOp, mlir::tt::ttir::NotEqualOp,
                   mlir::tt::ttir::GreaterEqualOp,
                   mlir::tt::ttir::GreaterThanOp, mlir::tt::ttir::LessEqualOp,
                   mlir::tt::ttir::LessThanOp, mlir::tt::ttir::LogicalAndOp,
                   mlir::tt::ttir::LogicalOrOp, mlir::tt::ttir::LogicalXorOp,
                   mlir::tt::ttir::LogicalNotOp, mlir::tt::ttir::IsFiniteOp>(
      defOp);
}

// LogicalAndOp canonicalization:
//   and(zero, x)    -> ZerosOp   (absorbing)
//   and(nonzero, x) -> x         (identity, when x is boolean-valued)
//   and(nonzero, nonzero) -> OnesOp (both constant nonzero)
void mlir::tt::ttir::LogicalAndOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  patterns.add(
      +[](mlir::tt::ttir::LogicalAndOp op, mlir::PatternRewriter &rewriter) {
        auto resultType =
            mlir::cast<mlir::RankedTensorType>(op.getResult().getType());

        // Absorbing: and(zero, x) -> 0
        if (isConstantZero(op.getLhs()) || isConstantZero(op.getRhs())) {
          rewriter.replaceOpWithNewOp<mlir::tt::ttir::ZerosOp>(
              op, resultType,
              rewriter.getDenseI32ArrayAttr(getShapeAsI32(resultType)));
          return mlir::success();
        }

        // Identity: and(nonzero, x) -> x when x is boolean-valued
        if (isConstantNonZero(op.getLhs()) && isBooleanValued(op.getRhs())) {
          rewriter.replaceOp(op, op.getRhs());
          return mlir::success();
        }
        if (isConstantNonZero(op.getRhs()) && isBooleanValued(op.getLhs())) {
          rewriter.replaceOp(op, op.getLhs());
          return mlir::success();
        }

        // Both constant nonzero -> OnesOp
        if (isConstantNonZero(op.getLhs()) && isConstantNonZero(op.getRhs())) {
          rewriter.replaceOpWithNewOp<mlir::tt::ttir::OnesOp>(
              op, resultType,
              rewriter.getDenseI32ArrayAttr(getShapeAsI32(resultType)));
          return mlir::success();
        }

        return mlir::failure();
      });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// LogicalOrOp
//===----------------------------------------------------------------------===//

// LogicalOrOp canonicalization:
//   or(nonzero, x) -> OnesOp   (absorbing)
//   or(zero, x)    -> x        (identity, when x is boolean-valued)
//   or(zero, zero)  -> ZerosOp  (both constant zero)
void mlir::tt::ttir::LogicalOrOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  patterns.add(
      +[](mlir::tt::ttir::LogicalOrOp op, mlir::PatternRewriter &rewriter) {
        auto resultType =
            mlir::cast<mlir::RankedTensorType>(op.getResult().getType());

        // Absorbing: or(nonzero, x) -> 1
        if (isConstantNonZero(op.getLhs()) || isConstantNonZero(op.getRhs())) {
          rewriter.replaceOpWithNewOp<mlir::tt::ttir::OnesOp>(
              op, resultType,
              rewriter.getDenseI32ArrayAttr(getShapeAsI32(resultType)));
          return mlir::success();
        }

        // Identity: or(zero, x) -> x when x is boolean-valued
        if (isConstantZero(op.getLhs()) && isBooleanValued(op.getRhs())) {
          rewriter.replaceOp(op, op.getRhs());
          return mlir::success();
        }
        if (isConstantZero(op.getRhs()) && isBooleanValued(op.getLhs())) {
          rewriter.replaceOp(op, op.getLhs());
          return mlir::success();
        }

        // Both constant zero -> ZerosOp
        if (isConstantZero(op.getLhs()) && isConstantZero(op.getRhs())) {
          rewriter.replaceOpWithNewOp<mlir::tt::ttir::ZerosOp>(
              op, resultType,
              rewriter.getDenseI32ArrayAttr(getShapeAsI32(resultType)));
          return mlir::success();
        }

        return mlir::failure();
      });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// ClampScalarOp
//===----------------------------------------------------------------------===//

// ClampScalarOp verifier
::mlir::LogicalResult mlir::tt::ttir::ClampScalarOp::verify() {
  const RankedTensorType inputTensorType =
      mlir::cast<RankedTensorType>(getInput().getType());

  const RankedTensorType outputTensorType =
      mlir::cast<RankedTensorType>(getResult().getType());

  if (inputTensorType.getShape() != outputTensorType.getShape()) {
    return emitOpError("input and output must have same shape.");
  }

  return success();
}

// ClampScalarOp folder
::mlir::OpFoldResult mlir::tt::ttir::ClampScalarOp::fold(FoldAdaptor adaptor) {
  auto input =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(adaptor.getInput());
  if (!input) {
    return nullptr;
  }

  if (auto floatType =
          mlir::dyn_cast<mlir::FloatType>(input.getElementType())) {
    auto min = ttmlir::utils::attributeToAPFloat(getMin());
    auto max = ttmlir::utils::attributeToAPFloat(getMax());
    if (&floatType.getFloatSemantics() != &llvm::APFloat::IEEEsingle()) {
      bool losesInfo{};
      min.convert(floatType.getFloatSemantics(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInfo);
      max.convert(floatType.getFloatSemantics(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    }
    return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                         [min, max](const llvm::APFloat &val) {
                                           return std::clamp(val, min, max);
                                         });
  }

  if (auto intType =
          mlir::dyn_cast<mlir::IntegerType>(input.getElementType())) {
    double minDouble = ttmlir::utils::attributeToDouble(getMin());
    double maxDouble = ttmlir::utils::attributeToDouble(getMax());
    auto min = llvm::APInt(intType.getWidth(), static_cast<int64_t>(minDouble));
    auto max = llvm::APInt(intType.getWidth(), static_cast<int64_t>(maxDouble));
    bool isUnsigned = intType.isUnsigned();
    return constantFoldEltwiseUnaryInt(
        *this, adaptor.getInput(),
        [min, max, isUnsigned](const llvm::APInt &val) {
          if (isUnsigned) {
            return llvm::APIntOps::umax(min, llvm::APIntOps::umin(val, max));
          }
          return llvm::APIntOps::smax(min, llvm::APIntOps::smin(val, max));
        });
  }

  return nullptr;
}

// ClampScalarOp canonicalization
void mlir::tt::ttir::ClampScalarOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // Fold two consecutive ClampScalarOp into a single one with tighter bounds
  patterns.add(+[](mlir::tt::ttir::ClampScalarOp op,
                   mlir::PatternRewriter &rewriter) {
    auto producerOp = op.getInput().getDefiningOp<ClampScalarOp>();
    if (!producerOp || !producerOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    auto newMinVal =
        llvm::maximum(ttmlir::utils::attributeToAPFloat(producerOp.getMin()),
                      ttmlir::utils::attributeToAPFloat(op.getMin()));
    auto newMaxVal =
        llvm::minimum(ttmlir::utils::attributeToAPFloat(producerOp.getMax()),
                      ttmlir::utils::attributeToAPFloat(op.getMax()));

    mlir::Attribute newMin, newMax;
    if (mlir::isa<mlir::FloatAttr>(op.getMin())) {
      newMin = mlir::FloatAttr::get(
          mlir::cast<mlir::FloatAttr>(op.getMin()).getType(), newMinVal);
      newMax = mlir::FloatAttr::get(
          mlir::cast<mlir::FloatAttr>(op.getMax()).getType(), newMaxVal);
    } else {
      newMin = mlir::IntegerAttr::get(
          mlir::cast<mlir::IntegerAttr>(op.getMin()).getType(),
          static_cast<int32_t>(newMinVal.convertToDouble()));
      newMax = mlir::IntegerAttr::get(
          mlir::cast<mlir::IntegerAttr>(op.getMax()).getType(),
          static_cast<int32_t>(newMaxVal.convertToDouble()));
    }

    rewriter.replaceOpWithNewOp<ClampScalarOp>(
        op, op.getResult().getType(), producerOp.getInput(), newMin, newMax);
    return mlir::success();
  });

  // Fold clamp with min=0 and max=6 to relu6
  patterns.add(+[](mlir::tt::ttir::ClampScalarOp op,
                   mlir::PatternRewriter &rewriter) {
    double minVal = ttmlir::utils::attributeToDouble(op.getMin());
    double maxVal = ttmlir::utils::attributeToDouble(op.getMax());

    if (minVal == 0.0 && maxVal == 6.0) {
      rewriter.replaceOpWithNewOp<ttir::Relu6Op>(op, op.getResult().getType(),
                                                 op.getInput());
      return mlir::success();
    }

    return mlir::failure();
  });
}

//===----------------------------------------------------------------------===//
// LogicalRightShiftOp
//===----------------------------------------------------------------------===//

// LogicalRightShiftOp verifier
::mlir::LogicalResult mlir::tt::ttir::LogicalRightShiftOp::verify() {
  RankedTensorType lhsTensorType = getLhs().getType();
  RankedTensorType rhsTensorType = getRhs().getType();
  RankedTensorType outputTensorType = getResult().getType();

  // Check that left operand (value to be shifted) has integer element type.
  auto lhsElemType = lhsTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(lhsElemType)) {
    return emitOpError()
           << "Left operand element type must be integer, but got "
           << lhsElemType;
  }

  // Check that right operand (shift amount) has integer element type.
  auto rhsElemType = rhsTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(rhsElemType)) {
    return emitOpError()
           << "Right operand element type must be integer, but got "
           << rhsElemType;
  }

  // Check that output has integer element type.
  auto outputElemType = outputTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(outputElemType)) {
    return emitOpError() << "Output element type must be integer, but got "
                         << outputElemType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LogicalLeftShiftOp
//===----------------------------------------------------------------------===//

// LogicalLeftShiftOp verifier
::mlir::LogicalResult mlir::tt::ttir::LogicalLeftShiftOp::verify() {
  RankedTensorType lhsTensorType = getLhs().getType();
  RankedTensorType rhsTensorType = getRhs().getType();
  RankedTensorType outputTensorType = getResult().getType();

  // Check that left operand (value to be shifted) has integer element type.
  auto lhsElemType = lhsTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(lhsElemType)) {
    return emitOpError()
           << "Left operand element type must be integer, but got "
           << lhsElemType;
  }

  // Check that right operand (shift amount) has integer element type.
  auto rhsElemType = rhsTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(rhsElemType)) {
    return emitOpError()
           << "Right operand element type must be integer, but got "
           << rhsElemType;
  }

  // Check that output has integer element type.
  auto outputElemType = outputTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(outputElemType)) {
    return emitOpError() << "Output element type must be integer, but got "
                         << outputElemType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RightShiftOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::RightShiftOp::verify() {
  RankedTensorType lhsTensorType = getLhs().getType();
  RankedTensorType rhsTensorType = getRhs().getType();
  RankedTensorType outputTensorType = getResult().getType();

  // Check that left operand (value to be shifted) has integer element type.
  auto lhsElemType = lhsTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(lhsElemType)) {
    return emitOpError()
           << "Left operand element type must be integer, but got "
           << lhsElemType;
  }

  // Check that right operand (shift amount) has integer element type.
  auto rhsElemType = rhsTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(rhsElemType)) {
    return emitOpError()
           << "Right operand element type must be integer, but got "
           << rhsElemType;
  }

  // Check that output has integer element type.
  auto outputElemType = outputTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(outputElemType)) {
    return emitOpError() << "Output element type must be integer, but got "
                         << outputElemType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ClampTensorOp
//===----------------------------------------------------------------------===//

// ClampTensorOp verifier
::mlir::LogicalResult mlir::tt::ttir::ClampTensorOp::verify() {
  llvm::ArrayRef<int64_t> minShape = getMin().getType().getShape();

  llvm::ArrayRef<int64_t> outputShape = getResult().getType().getShape();

  llvm::SmallVector<int64_t, 4> broadcastedShape;
  if (!mlir::OpTrait::util::getBroadcastedShape(minShape, outputShape,
                                                broadcastedShape)) {
    return emitOpError("Min attribute shape (" +
                       ttmlir::utils::join(minShape, ",") +
                       ") cannot be broadcasted to output shape (" +
                       ttmlir::utils::join(outputShape, ",") + ").");
  }

  llvm::ArrayRef<int64_t> maxShape = getMax().getType().getShape();
  if (!mlir::OpTrait::util::getBroadcastedShape(maxShape, outputShape,
                                                broadcastedShape)) {
    return emitOpError("Max attribute shape (" +
                       ttmlir::utils::join(maxShape, ",") +
                       ") cannot be broadcasted to output shape (" +
                       ttmlir::utils::join(outputShape, ",") + ").");
  }

  return success();
}

// ClampTensorOp canonicalization
void mlir::tt::ttir::ClampTensorOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add(+[](mlir::tt::ttir::ClampTensorOp op,
                   mlir::PatternRewriter &rewriter) {
    RankedTensorType outputType = op.getResult().getType();

    Attribute minValue = getConstantValue(op.getMin());
    Attribute maxValue = getConstantValue(op.getMax());
    if (minValue && maxValue) {
      // Cast min and max attributes to match the output element type
      Type elementType = outputType.getElementType();

      Attribute castedMin, castedMax;
      if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
        // For float types, convert to double then create attribute
        // This ensures proper conversion regardless of source type
        double minVal = ttmlir::utils::attributeToDouble(minValue);
        double maxVal = ttmlir::utils::attributeToDouble(maxValue);
        castedMin = rewriter.getFloatAttr(
            mlir::Float32Type::get(rewriter.getContext()), minVal);
        castedMax = rewriter.getFloatAttr(
            mlir::Float32Type::get(rewriter.getContext()), maxVal);
      } else {
        // For integer types, convert via APFloat for proper rounding
        llvm::APFloat minAPFloat = ttmlir::utils::attributeToAPFloat(minValue);
        llvm::APFloat maxAPFloat = ttmlir::utils::attributeToAPFloat(maxValue);
        bool ignored;
        llvm::APSInt minInt(32, false);
        llvm::APSInt maxInt(32, false);
        minAPFloat.convertToInteger(minInt, llvm::APFloat::rmTowardZero,
                                    &ignored);
        maxAPFloat.convertToInteger(maxInt, llvm::APFloat::rmTowardZero,
                                    &ignored);
        castedMin = rewriter.getI32IntegerAttr(minInt.getExtValue());
        castedMax = rewriter.getI32IntegerAttr(maxInt.getExtValue());
      }

      rewriter.replaceOpWithNewOp<ttir::ClampScalarOp>(
          op, outputType, op.getInput(), castedMin, castedMax);

      return success();
    }

    if (outputType.getShape() == op.getMin().getType().getShape() &&
        outputType.getShape() == op.getMax().getType().getShape()) {
      return failure();
    }

    Location loc = op->getLoc();
    mlir::Value minTensor;
    LogicalResult legalityResult = ttir::utils::broadcastValue(
        rewriter, op.getMin(), outputType, minTensor, loc,
        /*frontUnsqueeze=*/false);
    assert(legalityResult.succeeded() &&
           "Min attribute cannot be broadcasted to provided dimensions.");

    mlir::Value maxTensor;
    legalityResult = ttir::utils::broadcastValue(rewriter, op.getMax(),
                                                 outputType, maxTensor, loc,
                                                 /*frontUnsqueeze=*/false);
    assert(legalityResult.succeeded() &&
           "Max attribute cannot be broadcasted to provided dimensions.");

    rewriter.replaceOpWithNewOp<ttir::ClampTensorOp>(
        op, outputType, op.getInput(), minTensor, maxTensor);
    return success();
  });
}

//===----------------------------------------------------------------------===//
// ArangeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::ArangeOp::verify() {
  int64_t start = getStart();
  int64_t end = getEnd();
  int64_t step = getStep();

  if (step == 0) {
    return emitOpError("Step value cannot be zero");
  }

  int64_t numValues = (end - start) / step;

  if (numValues <= 0) {
    return emitOpError() << "Invalid range: start=" << start << ", end=" << end
                         << ", step=" << step;
  }

  if (numValues != getType().getDimSize(getArangeDimension())) {
    return emitOpError() << "Output tensor shape must be " << numValues
                         << " at dim " << getArangeDimension()
                         << " (since start=" << start << ", end=" << end
                         << ", step=" << step << "), but got "
                         << getType().getDimSize(getArangeDimension());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EmptyOp
//===----------------------------------------------------------------------===//

// EmptyOp models allocation semantics: each empty() is a distinct allocation
// that gets written into (e.g. DPS output buffer for ttir.ToLayoutOp). The
// Allocate effect prevents CSE from merging identical empty ops while still
// allowing DCE to remove unused ones.
void EmptyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       cast<OpResult>(getResult()));
}

//===----------------------------------------------------------------------===//
// RandOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::RandOp::verify() {
  auto dtype = getDtype();
  auto outputType = getResult().getType().getElementType();

  if (dtype != outputType) {
    return emitOpError()
           << "dtype does not match with output tensor type [dtype = " << dtype
           << ", output tensor type = " << outputType << "].";
  }

  float low = getLow().convertToFloat();
  float high = getHigh().convertToFloat();
  if (low >= high) {
    return emitOpError() << "'low' value must be < 'high' value.";
  }

  llvm::SmallVector<int64_t> sizeVec;
  for (auto size : getSize()) {
    sizeVec.push_back(mlir::cast<mlir::IntegerAttr>(size).getInt());
  }
  if (!llvm::equal(getResult().getType().getShape(), sizeVec)) {
    return emitOpError()
           << "Size argument does not match with output tensor shape. [Size = "
           << getSize() << ", output tensor shape = ("
           << getResult().getType().getShape() << ")].";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DropoutOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::DropoutOp::verify() {
  auto inputType = getInput().getType();
  auto outputType = getResult().getType();

  if (inputType != outputType) {
    return emitOpError() << "Input tensor type does not match with output "
                            "tensor type. [Input tensor type = "
                         << inputType << ", output tensor type = " << outputType
                         << "].";
  }

  float prob = getProb().convertToFloat();
  if (prob < 0.0f || prob > 1.0f) {
    return emitOpError() << "Probability must be in range [0.0, 1.0], but got "
                         << prob;
  }

  float scale = getScale().convertToFloat();
  if (scale <= 0.0f) {
    return emitOpError() << "Scale must be positive, but got " << scale;
  }

  int64_t rank = inputType.getRank();
  if (rank < 2 || rank > 4) {
    return emitOpError() << "Input tensor rank must be 2, 3, or 4, but got "
                         << rank;
  }

  return success();
}
//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// ConstantOp folder
::mlir::OpFoldResult mlir::tt::ttir::ConstantOp::fold(FoldAdaptor) {
  return getValueAttr();
}

// ConstantOp canonicalization
void mlir::tt::ttir::ConstantOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *) {

  // Canonicalize ConstantOp to FullOp when the value is a splat value (i.e. all
  // elements are the same).
  patterns.add(
      +[](mlir::tt::ttir::ConstantOp op, mlir::PatternRewriter &rewriter) {
        auto valueAttr = op.getValueAttr();
        if (!valueAttr.isSplat()) {
          return failure();
        }

        mlir::Attribute fillValueAttr =
            utils::splatToFillValue(rewriter, valueAttr);
        if (!fillValueAttr) {
          return failure();
        }

        rewriter.replaceOpWithNewOp<mlir::tt::ttir::FullOp>(
            op, op.getType(),
            rewriter.getDenseI32ArrayAttr(
                llvm::to_vector_of<int32_t>(op.getType().getShape())),
            fillValueAttr);

        return success();
      });
}

::mlir::LogicalResult mlir::tt::ttir::ConstantOp::verify() {
  if (!isa<DenseResourceElementsAttr, DenseElementsAttr>(getValue())) {
    return emitOpError("value attribute must be one of "
                       "DenseResourceElementsAttr or DenseElementsAttr.");
  }

  if (!getValue().getElementType().isIntOrFloat()) {
    return emitOpError("value attribute must be of int or float type.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//

// GetDimensionSizeOp verification
::mlir::LogicalResult mlir::tt::ttir::GetDimensionSizeOp::verify() {
  RankedTensorType inputTensorType = getOperand().getType();

  int64_t dimensionIndex = getDimension();

  if (dimensionIndex >=
      static_cast<int64_t>(inputTensorType.getShape().size())) {
    return failure();
  };

  return success();
}

// GetDimensionSizeOp folder
::mlir::OpFoldResult
mlir::tt::ttir::GetDimensionSizeOp::fold(FoldAdaptor adaptor) {
  RankedTensorType inputTensorType = getOperand().getType();
  uint32_t dimensionIndex = getDimension();
  uint32_t dimSize = inputTensorType.getShape()[dimensionIndex];

  auto resultElType = IntegerType::get(
      getContext(), 32, IntegerType::SignednessSemantics::Unsigned);
  auto resultType = RankedTensorType::get(/*shape=*/{1}, resultElType);
  return mlir::DenseElementsAttr::get<uint32_t>(resultType, dimSize);
}

//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//

bool mlir::tt::ttir::Conv2dOp::isQuantizedRewriteFavorable(
    mlir::ArrayRef<mlir::Value> sourceOperands) {
  // Convolution op requires both input and weight to be quantized.
  // Bias (if present) can be float - it will be quantized in
  // rewriteWithQuantizedInputs.
  assert(sourceOperands.size() >= 2 &&
         "Conv2dOp should have at least two operands (input and weight).");
  // Only check input and weight operands. If bias exists, it can remain float.
  size_t numToCheck = getBias() ? 2 : sourceOperands.size();
  for (size_t i = 0; i < numToCheck; ++i) {
    auto type =
        mlir::dyn_cast<mlir::RankedTensorType>(sourceOperands[i].getType());
    if (!type) {
      return false;
    }
    auto qType =
        mlir::dyn_cast<mlir::quant::QuantizedType>(type.getElementType());
    if (!qType || qType.getStorageType().getIntOrFloatBitWidth() != 8) {
      return false;
    }
  }
  return true;
}

mlir::Operation *mlir::tt::ttir::Conv2dOp::rewriteWithQuantizedInputs(
    mlir::PatternRewriter &rewriter, mlir::ArrayRef<Value> sourceOperands) {
  // rewrite the convolution op to be quantized.
  // create the output quantized type, whose scale is input * weight.
  // If bias is provided, output storage type is i8 (bias addition handles
  // requantization internally). Otherwise, storage type is i32 (accumulator).
  auto storageType =
      getBias()
          ? IntegerType::get(rewriter.getContext(), 8, IntegerType::Signed)
          : IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed);
  auto quantInputType = mlir::cast<mlir::quant::QuantizedType>(
      mlir::cast<RankedTensorType>(sourceOperands[0].getType())
          .getElementType());
  auto quantWeightType = mlir::cast<mlir::quant::QuantizedType>(
      mlir::cast<RankedTensorType>(sourceOperands[1].getType())
          .getElementType());
  auto oldConvOutputType = cast<RankedTensorType>(getResult().getType());

  // Pass back axes needed for computation of output scale and zero point.
  // Use the channel dimension from the op's attributes.
  const int64_t outFeatAxis = getChannelDim();
  // Weight is always in OIHW format, so output channel is at dim 0.
  const int64_t weightOcAxis = 0;
  const int64_t ocSize = oldConvOutputType.getDimSize(outFeatAxis);

  mlir::quant::QuantizedType quantOutputType =
      mlir::tt::ttir::utils::computeOutputScalesAndZeroPoint(
          quantInputType, quantWeightType, storageType, getLoc(), outFeatAxis,
          weightOcAxis, ocSize);
  if (!quantOutputType) {
    return nullptr;
  }
  auto quantConvOutputType =
      quantOutputType.castFromExpressedType(oldConvOutputType.getElementType());
  if (!quantConvOutputType) {
    return nullptr;
  }
  RankedTensorType newType =
      RankedTensorType::get(oldConvOutputType.getShape(), quantConvOutputType,
                            oldConvOutputType.getEncoding());
  // we need to quantize the bias if it is provided, the new bias type should be
  // the same as the quantconvoutput type
  auto quantBias = getBias();
  if (quantBias) {
    RankedTensorType quantBiasType = RankedTensorType::get(
        getBias().getType().getShape(), quantConvOutputType,
        getBias().getType().getEncoding());
    quantBias = rewriter.create<mlir::tt::ttir::QuantizeOp>(
        getLoc(), quantBiasType, quantBias);
  }
  auto quantConv = rewriter.create<mlir::tt::ttir::Conv2dOp>(
      getLoc(), newType, sourceOperands[0], sourceOperands[1], quantBias,
      getStrideAttr(), getPaddingAttr(), getDilationAttr(),
      rewriter.getI32IntegerAttr(getGroups()), getBatchDimAttr(),
      getHeightDimAttr(), getWidthDimAttr(), getChannelDimAttr(),
      /*flattenedCompatInfo=*/nullptr);

  return quantConv.getOperation();
}

// Conv2dOp verification
::mlir::LogicalResult mlir::tt::ttir::Conv2dOp::verify() {
  // Verify tensor ranks.
  if (verification_utils::verifyTensorRanks<Conv2dOp, true>(this).failed()) {
    return mlir::failure();
  }

  auto flatInfo = getFlattenedCompatInfoAttr();
  if (flatInfo &&
      flatInfo.getBatchSize() * flatInfo.getInputHeight() *
              flatInfo.getInputWidth() !=
          getInput().getType().getDimSize(verification_utils::FLATTENED_DIM)) {
    int64_t expectedSize = flatInfo.getBatchSize() * flatInfo.getInputHeight() *
                           flatInfo.getInputWidth();
    int64_t actualSize =
        getInput().getType().getDimSize(verification_utils::FLATTENED_DIM);
    return emitOpError()
           << "The input tensor's flattened dimension (" << actualSize
           << ") does not match the product of batch_size * input_height * "
              "input_width from FlattenedCompatInfo ("
           << flatInfo.getBatchSize() << " * " << flatInfo.getInputHeight()
           << " * " << flatInfo.getInputWidth() << " = " << expectedSize
           << ").";
  }

  auto [inputDims, weightDims, biasDims] =
      verification_utils::conv2d::getConv2dInputDims(this);
  verification_utils::OutputTensorDims outputDims =
      verification_utils::conv2d::getConv2dOutputDims(this);
  auto expectedParams = verification_utils::conv2d::getConv2dParams(this);
  if (auto error = expectedParams.takeError()) {
    return emitOpError() << llvm::toString(std::move(error));
  }
  verification_utils::conv2d::Conv2dParams params = *expectedParams;

  if (verification_utils::conv2d::verifyConv2dParams(this, params).failed()) {
    return mlir::failure();
  }

  if (verification_utils::conv2d::verifyConv2dInputDims(
          this, inputDims, weightDims, biasDims, params)
          .failed()) {
    return mlir::failure();
  }

  if (verification_utils::conv2d::verifyOutputDimensions(
          this, inputDims, weightDims, biasDims, outputDims, params)
          .failed()) {
    return mlir::failure();
  }

  return mlir::success();
}

// Get number of output channels
int64_t mlir::tt::ttir::Conv2dOp::getOutputChannelSize() {
  RankedTensorType weightTy = getWeight().getType();
  return weightTy.getShape()[0];
}

// Verify that bias dimensions are compatible with conv2d operation
bool mlir::tt::ttir::Conv2dOp::isBiasCompatible(llvm::ArrayRef<int64_t> bias) {
  return bias[0] == 1 && bias[1] == 1 && bias[2] == 1 &&
         bias[3] == getOutputChannelSize();
}

//===----------------------------------------------------------------------===//
// Conv3dOp
//===----------------------------------------------------------------------===//

// Conv3dOp verification
::mlir::LogicalResult mlir::tt::ttir::Conv3dOp::verify() {
  // Verify tensor ranks
  if (this->getInput().getType().getRank() != 5) {
    return this->emitOpError("input must be a 5D tensor");
  }

  if (this->getWeight().getType().getRank() != 5) {
    return this->emitOpError("weight must be a 5D tensor");
  }

  // Bias is optional
  if (this->getBias() && this->getBias().getType().getRank() != 5) {
    return this->emitOpError(
        "bias must be a 5D tensor with shape (1,1,1,1,C_out)");
  }

  if (this->getResult().getType().getRank() != 5) {
    return this->emitOpError("output must be a 5D tensor");
  }

  auto [inputDims, weightDims, biasDims] =
      verification_utils::conv3d::getConv3dInputDims(this);

  verification_utils::conv3d::OutputTensorDims3d outputDims =
      verification_utils::conv3d::getConv3dOutputDims(this);

  auto expectedParams = verification_utils::conv3d::getConv3dParams(this);

  if (auto error = expectedParams.takeError()) {
    return emitOpError() << llvm::toString(std::move(error));
  }

  verification_utils::conv3d::Conv3dParams params = *expectedParams;
  if (verification_utils::conv3d::verifyConv3dParams(this, params).failed()) {
    return mlir::failure();
  }

  if (verification_utils::conv3d::verifyConv3dInputDims(
          this, inputDims, weightDims, biasDims, params)
          .failed()) {
    return mlir::failure();
  }

  if (verification_utils::conv3d::verifyOutputDimensions(
          this, inputDims, weightDims, biasDims, outputDims, params)
          .failed()) {
    return mlir::failure();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Quantize ops
//===----------------------------------------------------------------------===//

// Helper function to verify that a zero point is within the range of the
// storage type.
static ::mlir::LogicalResult verifyZeroPointInRange(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitOpError,
    int64_t zeroPoint, int64_t min, int64_t max, mlir::Type storageType) {
  if (zeroPoint < min || zeroPoint > max) {
    return emitOpError() << "Zero point " << zeroPoint
                         << " is out of the range for storage type "
                         << storageType;
  }
  return ::mlir::success();
}

// Common verifier for all Quantize ops.
static ::mlir::LogicalResult verifyQuantizeOpCommon(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitOpError,
    ::mlir::RankedTensorType inputType, ::mlir::RankedTensorType outputType,
    std::optional<uint32_t> axis = std::nullopt, bool isUnrolled = false) {
  // Sanity check to make sure that input rank matches the rank of the output
  // tensor.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "Input tensor rank of " << inputType.getRank()
                         << " does not match the output tensor rank of "
                         << outputType.getRank();
  }

  // Shapes of input and output of a quantize operation must be the same.
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError() << "Output tensor shape ("
                         << ttmlir::utils::join(outputType.getShape(), ",") +
                                ") must match the inferred shape: (" +
                                ttmlir::utils::join(inputType.getShape(), ",") +
                                ")";
  }

  if (!isUnrolled) {
    return ::mlir::success();
  }

  if (axis.has_value()) {
    int32_t axisValue = axis.value();
    if (axisValue < 0 || axisValue >= inputType.getRank()) {
      return emitOpError() << "Axis value " << axisValue
                           << " is out of the range [0, " << inputType.getRank()
                           << ") for the input tensor of rank "
                           << inputType.getRank();
    }
  }
  for (auto tensorType : {inputType, outputType}) {
    auto elemType = tensorType.getElementType();
    if (auto quantPerAxisType =
            mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
                elemType)) {
      // Verify that the scales size matches the axis size for per-axis
      // quantization on both input and output types. This aligns with the
      // runtime's behavior.
      int64_t axis = quantPerAxisType.getQuantizedDimension();
      auto shape = tensorType.getShape();
      auto scales = quantPerAxisType.getScales();
      if (scales.size() != static_cast<size_t>(shape[axis])) {
        return emitOpError()
               << "Number of scales (" << scales.size()
               << ") does not match the size of the quantized axis ("
               << shape[axis] << ")";
      }
      // Verify that the zero point is in the range of the storage type.
      // This aligns with the frontends' behavior.
      llvm::ArrayRef<int64_t> zeroPoints = quantPerAxisType.getZeroPoints();
      int64_t min = quantPerAxisType.getStorageTypeMin();
      int64_t max = quantPerAxisType.getStorageTypeMax();
      for (int64_t curZeroPoint : zeroPoints) {
        if (auto result =
                verifyZeroPointInRange(emitOpError, curZeroPoint, min, max,
                                       quantPerAxisType.getStorageType());
            failed(result)) {
          return result;
        }
      }
    }
    if (auto quantType =
            mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType)) {
      // Verify that the zero point is in the range of the storage type
      // (per-tensor). This aligns with the frontends' behavior.
      int64_t curZeroPoint = quantType.getZeroPoint();
      int64_t min = quantType.getStorageTypeMin();
      int64_t max = quantType.getStorageTypeMax();
      return verifyZeroPointInRange(emitOpError, curZeroPoint, min, max,
                                    quantType.getStorageType());
    }
  }

  return ::mlir::success();
}

// QuantizeOp verification.
::mlir::LogicalResult mlir::tt::ttir::QuantizeOp::verify() {
  auto inputElemType = getInput().getType().getElementType();
  auto outputElemType = getResult().getType().getElementType();

  if (!mlir::isa<mlir::FloatType>(inputElemType)) {
    return emitOpError() << "Input element type must be float, but got "
                         << inputElemType;
  }

  if (!mlir::isa<mlir::quant::UniformQuantizedType,
                 mlir::quant::UniformQuantizedPerAxisType>(outputElemType)) {
    return emitOpError()
           << "Output element type must be UniformQuantizedType or "
              "UniformQuantizedPerAxisType, but got "
           << outputElemType;
  }

  return verifyQuantizeOpCommon([&]() { return emitOpError(); },
                                getInput().getType(), getType(),
                                /*axis=*/std::nullopt, /*isUnrolled=*/false);
}

// QuantizeUnrolledOp verification.
::mlir::LogicalResult mlir::tt::ttir::QuantizeUnrolledOp::verify() {
  auto inputElemType = getInput().getType().getElementType();
  auto outputElemType = getResult().getType().getElementType();

  if (!mlir::isa<mlir::FloatType>(inputElemType)) {
    return emitOpError() << "Input element type must be float, but got "
                         << inputElemType;
  }

  if (!mlir::isa<mlir::quant::UniformQuantizedType,
                 mlir::quant::UniformQuantizedPerAxisType>(outputElemType)) {
    return emitOpError()
           << "Output element type must be UniformQuantizedType or "
              "UniformQuantizedPerAxisType, but got "
           << outputElemType;
  }

  return verifyQuantizeOpCommon([&]() { return emitOpError(); },
                                getInput().getType(), getType(), getAxis(),
                                /*isUnrolled=*/true);
}

// DequantizeOp verification.
::mlir::LogicalResult mlir::tt::ttir::DequantizeOp::verify() {
  auto inputElemType = getInput().getType().getElementType();
  auto outputElemType = getResult().getType().getElementType();

  if (!mlir::isa<mlir::quant::UniformQuantizedType,
                 mlir::quant::UniformQuantizedPerAxisType>(inputElemType)) {
    return emitOpError() << "Input element type must be UniformQuantizedType "
                            "or UniformQuantizedPerAxisType, but got "
                         << inputElemType;
  }

  if (!mlir::isa<mlir::FloatType>(outputElemType)) {
    return emitOpError() << "Output element type must be float, but got "
                         << outputElemType;
  }

  return verifyQuantizeOpCommon([&]() { return emitOpError(); },
                                getInput().getType(), getType(),
                                /*axis=*/std::nullopt, /*isUnrolled=*/false);
}

// DequantizeUnrolledOp verification.
::mlir::LogicalResult mlir::tt::ttir::DequantizeUnrolledOp::verify() {
  RankedTensorType inputTensorType = getInput().getType();
  RankedTensorType outputTensorType = getResult().getType();

  auto inputElemType = inputTensorType.getElementType();
  auto outputElemType = outputTensorType.getElementType();

  if (!mlir::isa<mlir::quant::UniformQuantizedType,
                 mlir::quant::UniformQuantizedPerAxisType>(inputElemType)) {
    return emitOpError() << "Input element type must be UniformQuantizedType "
                            "or UniformQuantizedPerAxisType, but got "
                         << inputElemType;
  }

  if (!mlir::isa<mlir::FloatType>(outputElemType)) {
    return emitOpError() << "Output element type must be float, but got "
                         << outputElemType;
  }

  return verifyQuantizeOpCommon([&]() { return emitOpError(); },
                                inputTensorType, outputTensorType, getAxis(),
                                /*isUnrolled=*/true);
}

// RequantizeOp folder for identity requantize.
::mlir::OpFoldResult mlir::tt::ttir::RequantizeOp::fold(FoldAdaptor adaptor) {
  // if types of input and output are equivalent, return input.
  if (getInput().getType() == getType()) {
    return getInput();
  }
  return nullptr;
}

// RequantizeOp verification.
::mlir::LogicalResult mlir::tt::ttir::RequantizeOp::verify() {
  auto inputElemType = getInput().getType().getElementType();
  auto outputElemType = getResult().getType().getElementType();

  if (!mlir::isa<mlir::quant::UniformQuantizedType,
                 mlir::quant::UniformQuantizedPerAxisType>(inputElemType)) {
    return emitOpError() << "Input element type must be UniformQuantizedType "
                            "or UniformQuantizedPerAxisType, but got "
                         << inputElemType;
  }

  if (!mlir::isa<mlir::quant::UniformQuantizedType,
                 mlir::quant::UniformQuantizedPerAxisType>(outputElemType)) {
    return emitOpError() << "Output element type must be UniformQuantizedType "
                            "or UniformQuantizedPerAxisType, but got "
                         << outputElemType;
  }

  return verifyQuantizeOpCommon([&]() { return emitOpError(); },
                                getInput().getType(), getType(),
                                /*axis=*/std::nullopt, /*isUnrolled=*/false);
}

// RequantizeUnrolledOp verification.
::mlir::LogicalResult mlir::tt::ttir::RequantizeUnrolledOp::verify() {
  auto inputElemType = getInput().getType().getElementType();
  auto outputElemType = getResult().getType().getElementType();

  auto inputIsPerAxis =
      mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputElemType);
  auto outputIsPerAxis =
      mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType);
  auto inputIsPerTensor =
      mlir::isa<mlir::quant::UniformQuantizedType>(inputElemType);
  auto outputIsPerTensor =
      mlir::isa<mlir::quant::UniformQuantizedType>(outputElemType);

  if (!((inputIsPerAxis && outputIsPerAxis) ||
        (inputIsPerTensor && outputIsPerTensor))) {
    return emitOpError()
           << "Input and output element types must both be per-axis "
              "or both be per-tensor quantized types, but got "
           << inputElemType << " and " << outputElemType;
  }

  return verifyQuantizeOpCommon([&]() { return emitOpError(); },
                                getInput().getType(), getType(),
                                /*axis=*/getAxis(), /*isUnrolled=*/true);
}

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp
//===----------------------------------------------------------------------===//

bool mlir::tt::ttir::ConvTranspose2dOp::isQuantizedRewriteFavorable(
    mlir::ArrayRef<mlir::Value> sourceOperands) {
  // convolution op currently requires both input and weight to be quantized
  // TODO(anuragsingh): enable float bias support
  assert(sourceOperands.size() == 2 &&
         "Quantized ConvolutionOp should have two operands (only input and "
         "weight).");
  return llvm::all_of(sourceOperands, [](mlir::Value val) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
    if (!type) {
      return false;
    }
    auto qType =
        mlir::dyn_cast<mlir::quant::QuantizedType>(type.getElementType());
    return qType && qType.getStorageType().getIntOrFloatBitWidth() == 8;
  });
}

mlir::Operation *mlir::tt::ttir::ConvTranspose2dOp::rewriteWithQuantizedInputs(
    mlir::PatternRewriter &rewriter, mlir::ArrayRef<Value> sourceOperands) {
  // rewrite the convolution op to be quantized.
  // create the output quantized type, whose scale is input * weight and
  // storage type is i32.
  auto storageType =
      IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed);
  auto quantInputType = mlir::cast<mlir::quant::QuantizedType>(
      mlir::cast<RankedTensorType>(sourceOperands[0].getType())
          .getElementType());
  auto quantWeightType = mlir::cast<mlir::quant::QuantizedType>(
      mlir::cast<RankedTensorType>(sourceOperands[1].getType())
          .getElementType());
  auto oldConvOutputType = cast<RankedTensorType>(getResult().getType());

  // Pass back axes needed for computation of output scale and zero point.
  // ConvTranspose2d op has output in NHWC format and weight in OIHW format.
  const int64_t outFeatAxis = 3;
  const int64_t weightOcAxis = 0;
  const int64_t ocSize = oldConvOutputType.getDimSize(outFeatAxis);

  mlir::quant::QuantizedType quantOutputType =
      mlir::tt::ttir::utils::computeOutputScalesAndZeroPoint(
          quantInputType, quantWeightType, storageType, getLoc(), outFeatAxis,
          weightOcAxis, ocSize);
  if (!quantOutputType) {
    return nullptr;
  }
  auto quantConvOutputType =
      quantOutputType.castFromExpressedType(oldConvOutputType.getElementType());
  if (!quantConvOutputType) {
    return nullptr;
  }
  RankedTensorType newType =
      RankedTensorType::get(oldConvOutputType.getShape(), quantConvOutputType,
                            oldConvOutputType.getEncoding());
  auto quantConv = rewriter.create<mlir::tt::ttir::ConvTranspose2dOp>(
      getLoc(), newType, sourceOperands[0], sourceOperands[1], getBias(),
      getStrideAttr(), getPaddingAttr(), getOutputPaddingAttr(),
      getDilationAttr(), getGroupsAttr(), /*flattenedCompatInfo=*/nullptr);

  return quantConv.getOperation();
}

// ConvTranspose2dOp verification
mlir::LogicalResult mlir::tt::ttir::ConvTranspose2dOp::verify() {
  mlir::RankedTensorType inputType = getInput().getType();
  mlir::RankedTensorType weightType = getWeight().getType();
  mlir::RankedTensorType outputType = getType();
  std::optional<mlir::RankedTensorType> bias =
      getBias().getImpl() ? std::make_optional(getBias().getType())
                          : std::nullopt;

  auto flatInfo = getFlattenedCompatInfoAttr();
  if (flatInfo &&
      flatInfo.getBatchSize() * flatInfo.getInputHeight() *
              flatInfo.getInputWidth() !=
          getInput().getType().getDimSize(verification_utils::FLATTENED_DIM)) {
    int64_t expectedSize = flatInfo.getBatchSize() * flatInfo.getInputHeight() *
                           flatInfo.getInputWidth();
    int64_t actualSize =
        getInput().getType().getDimSize(verification_utils::FLATTENED_DIM);
    return emitOpError()
           << "The input tensor's flattened dimension (" << actualSize
           << ") does not match the product of batch_size * input_height * "
              "input_width from FlattenedCompatInfo ("
           << flatInfo.getBatchSize() << " * " << flatInfo.getInputHeight()
           << " * " << flatInfo.getInputWidth() << " = " << expectedSize
           << ").";
  }

  if (inputType.getRank() != 4) {
    return emitOpError("Input must be a 4D tensor");
  }

  if (outputType.getRank() != 4) {
    return emitOpError("Output must be a 4D tensor");
  }

  if (weightType.getRank() != 4) {
    return emitOpError("Weight must be a 4D tensor");
  }

  if (bias.has_value()) {
    if (bias->getRank() != 4) {
      return emitOpError("Bias must be a 4D tensor");
    }
  }

  int64_t inputBatchSize = inputType.getDimSize(0);
  int64_t outputBatchSize = outputType.getDimSize(0);
  if (flatInfo) {
    inputBatchSize = flatInfo.getBatchSize();
    outputBatchSize = flatInfo.getBatchSize();
  }

  if (inputBatchSize != outputBatchSize) {
    return emitOpError("Batch size of input and output tensors must match");
  }

  auto stride = ttmlir::utils::getPairOfInteger<int32_t>(getStride());
  if (auto error = stride.takeError()) {
    return emitOpError() << llvm::toString(std::move(error)) << " for stride";
  }
  if (stride->first < 1 || stride->second < 1) {
    return emitOpError("Stride values must be greater than 0");
  }

  auto padding = ttmlir::utils::getQuadrupleOfInteger<int32_t>(getPadding());
  if (auto error = padding.takeError()) {
    return emitOpError() << llvm::toString(std::move(error)) << " for padding";
  }

  auto [paddingTop, paddingLeft, paddingBottom, paddingRight] = *padding;
  if (paddingTop < 0 || paddingBottom < 0 || paddingLeft < 0 ||
      paddingRight < 0) {
    return emitOpError("Padding values must be greater or equal than 0");
  }
  int32_t verticalPadding = paddingTop + paddingBottom;
  int32_t horizontalPadding = paddingLeft + paddingRight;

  auto outputPadding =
      ttmlir::utils::getPairOfInteger<int32_t>(getOutputPadding());
  if (auto error = outputPadding.takeError()) {
    return emitOpError() << llvm::toString(std::move(error))
                         << " for output padding";
  }
  if (outputPadding->first < 0 || outputPadding->second < 0) {
    return emitOpError("Output padding values must be greater or equal than 0");
  }

  auto dilation = ttmlir::utils::getPairOfInteger<int32_t>(getDilation());
  if (auto error = dilation.takeError()) {
    return emitOpError() << llvm::toString(std::move(error)) << " for dilation";
  }
  if (dilation->first < 1 || dilation->second < 1) {
    return emitOpError("Dilation values must be greater than 0");
  }

  llvm::ArrayRef<std::int64_t> kernelShape = weightType.getShape();

  int32_t inputChannels = inputType.getDimSize(getChannelDim());
  int32_t outputChannels = outputType.getDimSize(getChannelDim());
  uint32_t groups = getGroups();

  if (inputChannels % groups != 0) {
    return emitOpError() << "Number of input channels from input tensor must "
                            "be divisible by the number of groups. "
                         << "Got " << inputChannels << " input channels and "
                         << groups << " groups.";
  }

  if (outputChannels % groups != 0) {
    return emitOpError() << "Number of output channels from output tensor must "
                            "be divisible by the number of groups. "
                         << "Got " << outputChannels << " output channels and "
                         << groups << " groups.";
  }

  if (inputChannels != kernelShape[0]) {
    return emitOpError() << "Number of input channels from input tensor must "
                            "match the first dimension of the weight tensor. "
                         << "Got " << inputChannels << " input channels and "
                         << kernelShape[0] << " in the weight tensor.";
  }

  if (outputChannels / groups != kernelShape[1]) {
    return emitOpError() << "Number of output channels per group must match "
                            "the second dimension of the weight tensor. "
                         << "Got " << (outputChannels / groups)
                         << " output channels per group and " << kernelShape[1]
                         << " in the weight tensor.";
  }

  if (bias) {
    if (bias->getDimSize(getChannelDim()) != outputChannels) {
      return emitOpError() << "Mismatch in bias tensor dimensions. "
                           << "Bias tensor has "
                           << bias->getDimSize(getChannelDim()) << " channels, "
                           << "but the output tensor has " << outputChannels
                           << " channels.";
    }
  }

  int32_t kernelHeight = kernelShape[2];
  int32_t kernelWidth = kernelShape[3];

  int32_t Hin = inputType.getDimSize(getHeightDim());
  int32_t Win = inputType.getDimSize(getWidthDim());
  if (flatInfo) {
    Hin = flatInfo.getInputHeight();
    Win = flatInfo.getInputWidth();
  }

  int32_t expectedHOut = (Hin - 1) * stride->first - verticalPadding +
                         dilation->first * (kernelHeight - 1) +
                         outputPadding->first + 1;
  int32_t expectedWOut = (Win - 1) * stride->second - horizontalPadding +
                         dilation->second * (kernelWidth - 1) +
                         outputPadding->second + 1;
  if (expectedHOut < 0 || expectedWOut < 0) {
    return emitOpError() << "Given input size per channel: (" << Hin << " x "
                         << Win << "). "
                         << "Calculated output size per channel: ("
                         << expectedHOut << " x " << expectedWOut << "). "
                         << "Output size is too small";
  }

  int32_t HOut = outputType.getDimSize(getHeightDim());
  int32_t WOut = outputType.getDimSize(getWidthDim());
  if (!flatInfo && (HOut != expectedHOut || WOut != expectedWOut)) {
    return emitOpError() << "Mismatch between expected output size per channel "
                            "and got output tensor dimensions. "
                         << "Expected: (" << expectedHOut << " x "
                         << expectedWOut << "), "
                         << "got: (" << HOut << " x " << WOut << ").";
  }

  if (flatInfo &&
      inputBatchSize * expectedHOut * expectedWOut !=
          outputType.getDimSize(verification_utils::FLATTENED_DIM)) {
    return emitOpError() << "Mismatch between expected flattened NHW dim size. "
                         << "Expected: "
                         << inputBatchSize * expectedHOut * expectedWOut << ", "
                         << "got: "
                         << outputType.getDimSize(
                                verification_utils::FLATTENED_DIM)
                         << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pooling helper functions
//===----------------------------------------------------------------------===//

// Checks if a AvgPool2dOp or MaxPool2dOp operation is an identity operation.
// Identity operations can be folded away when kernel=[1,1], stride=[1,1],
// dilation=[1,1], and padding=[0,0,0,0].
template <typename Pool2dOp>
static bool isIdentityPool2d(Pool2dOp op) {
  auto kernel = ttmlir::utils::getPairOfInteger<int32_t>(op.getKernel());
  auto stride = ttmlir::utils::getPairOfInteger<int32_t>(op.getStride());
  auto dilation = ttmlir::utils::getPairOfInteger<int32_t>(op.getDilation());
  auto padding = ttmlir::utils::getQuadrupleOfInteger<int32_t>(op.getPadding());

  auto tupleToArray = [](const auto &t) {
    return std::apply([](auto... args) { return std::array{args...}; }, t);
  };

  return kernel && stride && dilation && padding &&
         llvm::all_of(tupleToArray(*kernel),
                      [](int32_t v) { return v == 1; }) &&
         llvm::all_of(tupleToArray(*stride),
                      [](int32_t v) { return v == 1; }) &&
         llvm::all_of(tupleToArray(*dilation),
                      [](int32_t v) { return v == 1; }) &&
         llvm::all_of(tupleToArray(*padding), [](int32_t v) { return v == 0; });
}

//===----------------------------------------------------------------------===//
// Generic Pool2dOp verification
//===----------------------------------------------------------------------===//

template <typename Pool2dOp>
static mlir::LogicalResult verifyPooling2dOp(Pool2dOp *op) {

  // Verify tensor ranks.
  if (verification_utils::verifyTensorRanks(op).failed()) {
    return mlir::failure();
  }

  // Verify flattened compatibility info if present.
  if (mlir::failed(verification_utils::pool2d::verifyFlattenedCompatInfo(op))) {
    return mlir::failure();
  }

  // Get input and output dimensions with flattened support.
  verification_utils::InputTensorDims inputDims =
      verification_utils::pool2d::getPool2dInputDims(op);
  verification_utils::OutputTensorDims outputDims =
      verification_utils::pool2d::getPool2dOutputDims(op);
  auto expectedParams = verification_utils::pool2d::getPool2dParams(op);
  if (auto error = expectedParams.takeError()) {
    return op->emitOpError() << llvm::toString(std::move(error));
  }
  verification_utils::pool2d::Pool2dParams params = *expectedParams;

  // Verify pooling parameters.
  if (mlir::failed(
          verification_utils::pool2d::verifyPool2dParams(op, params))) {
    return mlir::failure();
  }

  // Verify input dimensions constraints.
  if (mlir::failed(verification_utils::pool2d::verifyPool2dInputDims(
          op, inputDims, params))) {
    return mlir::failure();
  }

  // Verify output dimensions match expected calculations.
  if (mlir::failed(verification_utils::pool2d::verifyPool2dOutputDims(
          op, inputDims, outputDims, params))) {
    return mlir::failure();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AvgPool2dOp
//===----------------------------------------------------------------------===//

// AvgPool2dOp verification.
::mlir::LogicalResult mlir::tt::ttir::AvgPool2dOp::verify() {
  return verifyPooling2dOp(this);
}

// Folds AvgPool2dOp when it is an identity operation.
::mlir::OpFoldResult mlir::tt::ttir::AvgPool2dOp::fold(FoldAdaptor adaptor) {
  if (isIdentityPool2d(*this)) {
    return getInput();
  }
  return {};
}

// Rewrites operation with quantization, if possible.
::mlir::Operation *mlir::tt::ttir::AvgPool2dOp::rewriteWithQuantizedInputs(
    mlir::PatternRewriter &rewriter,
    mlir::ArrayRef<mlir::Value> sourceOperands) {
  // Unlike MaxPool2dOp which only compares values (scale-invariant), AvgPool2d
  // performs a calculation which causes precision loss due to integer division
  // truncation and scale parameter mismatch.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// MaxPool2dOp
//===----------------------------------------------------------------------===//

// MaxPool2dOp verification.
::mlir::LogicalResult mlir::tt::ttir::MaxPool2dOp::verify() {
  return verifyPooling2dOp(this);
}

// Folds MaxPool2dOp when it is an identity operation.
::mlir::OpFoldResult mlir::tt::ttir::MaxPool2dOp::fold(FoldAdaptor adaptor) {
  if (isIdentityPool2d(*this)) {
    return getInput();
  }
  return {};
}

// Rewrites operation with quantization, if possible.
::mlir::Operation *mlir::tt::ttir::MaxPool2dOp::rewriteWithQuantizedInputs(
    mlir::PatternRewriter &rewriter,
    mlir::ArrayRef<mlir::Value> sourceOperands) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  using mlir::quant::UniformQuantizedPerAxisType;

  mlir::Value input = sourceOperands[0];
  if (mlir::dyn_cast<UniformQuantizedPerAxisType>(input.getType())) {
    return nullptr;
  }

  RankedTensorType outType =
      mlir::cast<RankedTensorType>(getResult().getType());

  RankedTensorType newOutType = RankedTensorType::get(
      outType.getShape(),
      mlir::cast<RankedTensorType>(input.getType()).getElementType(),
      outType.getEncoding());

  return rewriter
      .create<mlir::tt::ttir::MaxPool2dOp>(
          getLoc(), newOutType, input, getKernelAttr(), getStrideAttr(),
          getDilationAttr(), getPaddingAttr(), getCeilModeAttr())
      .getOperation();
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// MaxPool2dWithIndicesOp
//===----------------------------------------------------------------------===//

// MaxPool2dWithIndicesOp verification
::mlir::LogicalResult mlir::tt::ttir::MaxPool2dWithIndicesOp::verify() {
  // First verify the pooling operation itself
  if (mlir::failed(verifyPooling2dOp(this))) {
    return mlir::failure();
  }

  // Verify that both results have the same shape
  auto pooledShape = this->getResult().getType().getShape();
  auto indicesShape = this->getResultIndices().getType().getShape();

  if (pooledShape != indicesShape) {
    return emitOpError("Pooled values and indices must have the same shape");
  }

  // Verify that indices have integer element type
  auto indicesElementType = this->getResultIndices().getType().getElementType();
  if (!indicesElementType.isInteger()) {
    return emitOpError("Indices result must have integer element type");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// ConcatOp verification
::mlir::LogicalResult mlir::tt::ttir::ConcatOp::verify() {
  mlir::OperandRange inputs = getInputs();
  int32_t dim = getDim();
  mlir::RankedTensorType firstTensor =
      mlir::cast<mlir::RankedTensorType>(inputs.front().getType());
  int64_t firstTensorRank = firstTensor.getRank();

  if (dim < 0) {
    dim += firstTensorRank;
  }

  // Check that the dimension `dim` is valid.
  if (dim < 0 || dim >= firstTensor.getRank()) {
    return emitOpError() << "Invalid dimension " << getDim()
                         << " for concatenation.";
  }

  // Get the rank of the first input tensor
  // and check that all input tensors have the same rank
  // and that all dimensions except `dim` are the same.
  int64_t dimSizeSum = firstTensor.getDimSize(dim);
  for (auto input : inputs.drop_front()) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());

    // Check if all inputs have the same rank.
    if (inputType.getRank() != firstTensorRank) {
      return emitOpError("All input tensors must have the same rank.");
    }

    // Check that dimensions (except `dim`) are the same.
    for (int64_t i = 0; i < firstTensorRank; ++i) {
      if (i == dim) {
        dimSizeSum += inputType.getDimSize(i);
        continue;
      }
      if (inputType.getDimSize(i) != firstTensor.getDimSize(i)) {
        return emitOpError() << "All input tensors must have the same "
                                "dimensions, except for dimension "
                             << dim << ".";
      }
    }
  }

  auto outputType = getType();
  if (outputType.getDimSize(dim) != dimSizeSum) {
    return emitOpError()
           << "Output tensor dimension " << dim
           << " does not match the sum of input tensor dimensions: "
           << outputType.getDimSize(dim) << " vs. " << dimSizeSum << ".";
  }

  return success();
}

// ConcatOp with single input is a no-op.
// Replace the op with input.
mlir::OpFoldResult foldUnitConcatOp(ttir::ConcatOp op) {
  mlir::ValueRange inputs = op.getInputs();
  if (inputs.size() == 1) {
    return inputs.front();
  }
  return nullptr;
}

// Empty tensor(s) act as neutral/identity element for ConcatOp.
// Remove empty tensors from ConcatOp operands.
mlir::OpFoldResult foldEmptyTensorsConcatOp(ttir::ConcatOp op) {
  RankedTensorType outputType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  mlir::ValueRange inputs = op.getInputs();
  int32_t dim = op.getDim();
  int32_t rank = outputType.getRank();
  int32_t adjustedDim = dim < 0 ? (dim + rank) : dim;
  llvm::SmallVector<mlir::Value> nonEmptyInputs;

  for (auto input : inputs) {
    auto shape = mlir::cast<RankedTensorType>(input.getType()).getShape();
    if (shape[adjustedDim] == 0) {
      continue;
    }
    nonEmptyInputs.push_back(input);
  }

  // No empty tensors to remove; Folding not applicable.
  if (inputs.size() == nonEmptyInputs.size()) {
    return nullptr;
  }

  // All inputs are empty tensors; returning first input (it can be any input).
  if (nonEmptyInputs.empty()) {
    return inputs.front();
  }

  // Update the operands with non empty inputs.
  op.getInputsMutable().assign(nonEmptyInputs);

  return op.getResult();
}

// ConcatOp Folder
mlir::OpFoldResult mlir::tt::ttir::ConcatOp::fold(FoldAdaptor adaptor) {
  if (auto foldResult = foldUnitConcatOp(*this)) {
    return foldResult;
  }
  if (auto foldResult = foldEmptyTensorsConcatOp(*this)) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

// PadOp verification
::mlir::LogicalResult mlir::tt::ttir::PadOp::verify() {

  ::mlir::RankedTensorType inputType = getInput().getType();

  // Check that size of padding is correct
  if (static_cast<int64_t>(getPadding().size()) != 2 * inputType.getRank()) {
    return emitOpError("Padding must have the same number of elements as twice "
                       "the rank of the input tensor");
  }

  std::vector<int64_t> inferredShapeVec = inputType.getShape().vec();
  llvm::ArrayRef<int32_t> padding = getPadding();
  for (int64_t i = 0; i < inputType.getRank(); i++) {
    inferredShapeVec[i] += padding[2 * i];
    inferredShapeVec[i] += padding[2 * i + 1];
  }
  llvm::ArrayRef<int64_t> inferredShape = inferredShapeVec;

  // Check that the output tensor shape is correct
  ::mlir::RankedTensorType resultType = getResult().getType();
  llvm::ArrayRef<int64_t> resultShape = resultType.getShape();
  if (resultShape != inferredShape) {
    return emitOpError("Output tensor shape (" +
                       ttmlir::utils::join(resultShape, ",") +
                       ") must match the inferred shape: (" +
                       ttmlir::utils::join(inferredShape, ",") + ")");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// ReshapeOp verification
::mlir::LogicalResult mlir::tt::ttir::ReshapeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();
  auto shape = getShape();
  int64_t shapeSize = static_cast<int64_t>(shape.size());

  // Check that the shape size matches the rank of the output tensor.
  if (shapeSize != static_cast<int64_t>(outputType.getRank())) {
    return emitOpError() << "Shape attribute size " << shapeSize
                         << " must match output tensor rank "
                         << outputType.getRank();
  }

  // Cardinality of the input and output tensors must be the same.
  if (inputType.getNumElements() != outputType.getNumElements()) {
    return emitOpError() << "Input tensor number of elements "
                         << inputType.getNumElements()
                         << " and output tensor number of elements "
                         << outputType.getNumElements() << " must be the same";
  }

  bool hasNegative = false;
  auto outputShape = outputType.getShape();

  // Check that all dimensions are positive except for at most one -1.
  // Check that the non-negative dimensions match the output tensor shape.
  // Calculate the product of the known dimensions.
  for (int64_t i = 0; i < shapeSize; i++) {
    int64_t dimValue = mlir::cast<IntegerAttr>(shape[i]).getInt();

    if (dimValue == -1) {
      if (hasNegative) {
        return emitOpError("Shape attribute must have at most one -1 element");
      }
      hasNegative = true;
    } else {
      if (dimValue < 0) {
        return emitOpError(
            "All dimensions must be >= 0 except the one with -1");
      }

      // Ensure that the non-negative dimensions match the output tensor shape.
      if (dimValue != outputShape[i]) {
        return emitOpError()
               << "Shape attribute " << dimValue
               << " must match the output tensor shape " << outputShape[i]
               << " at index " << i << " for dimension that is not -1";
      }
    }
  }

  return success();
}

// Fold the operation if the type of the input and output types are the same.
static mlir::OpFoldResult foldIdentityReshape(mlir::tt::ttir::ReshapeOp op) {
  if (op.getType() == op.getInput().getType()) {
    return op.getInput();
  }
  return nullptr;
}

// Back to back reshapes can be replaced with the final reshape.
static mlir::OpFoldResult foldConsecutiveReshape(mlir::tt::ttir::ReshapeOp op) {
  if (auto reshapeOperand =
          op.getInput().getDefiningOp<mlir::tt::ttir::ReshapeOp>()) {
    op.getInputMutable().assign(reshapeOperand.getInput());
    return op.getResult();
  }
  return nullptr;
}

// Fold reshape if input is constant
static mlir::OpFoldResult constFoldReshape(mlir::tt::ttir::ReshapeOp op,
                                           Attribute constInput) {
  if (auto denseAttr = dyn_cast_if_present<DenseElementsAttr>(constInput)) {
    RankedTensorType type = op.getResult().getType();
    return denseAttr.reshape(type);
  }
  return nullptr;
}

// ReshapeOp folder
::mlir::OpFoldResult mlir::tt::ttir::ReshapeOp::fold(FoldAdaptor adaptor) {
  if (auto foldResult = foldIdentityReshape(*this)) {
    return foldResult;
  }

  if (auto foldResult = foldConsecutiveReshape(*this)) {
    return foldResult;
  }

  if (auto foldResult = constFoldReshape(*this, adaptor.getInput())) {
    return foldResult;
  }

  return nullptr;
}

// ReshapeOp canonicalization
//
// Fold Reshape(Permute(Reshape(x))) → Permute(Reshape(x)) when the trailing
// reshape only removes leading unit dimensions. The permutation is adjusted to
// operate at the lower rank.
//
// Example:
//   reshape: 256x32 → 1x1x256x32
//   permute [0,1,3,2]: 1x1x256x32 → 1x1x32x256
//   reshape: 1x1x32x256 → 1x32x256
// Becomes:
//   reshape: 256x32 → 1x256x32
//   permute [0,2,1]: 1x256x32 → 1x32x256
//
void mlir::tt::ttir::ReshapeOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add(+[](mlir::tt::ttir::ReshapeOp trailingReshape,
                   mlir::PatternRewriter &rewriter) -> LogicalResult {
    auto permuteOp =
        trailingReshape.getInput().getDefiningOp<mlir::tt::ttir::PermuteOp>();
    if (!permuteOp || !permuteOp->hasOneUse()) {
      return failure();
    }

    auto leadingReshape =
        permuteOp.getInput().getDefiningOp<mlir::tt::ttir::ReshapeOp>();
    if (!leadingReshape) {
      return failure();
    }

    // Check that the trailing reshape only removes leading 1s.
    auto permuteOutShape = permuteOp.getType().getShape();
    auto outShape = trailingReshape.getType().getShape();
    if (outShape.size() >= permuteOutShape.size()) {
      return failure();
    }
    int64_t n = permuteOutShape.size() - outShape.size();
    if (!llvm::all_of(permuteOutShape.take_front(n),
                      [](int64_t d) { return d == 1; })) {
      return failure();
    }
    if (permuteOutShape.drop_front(n) != outShape) {
      return failure();
    }

    // Check that the first n permuted dims come from the first n input dims
    // (all unit dims mapping to unit dims).
    auto perm = permuteOp.getPermutation();
    for (int64_t i = 0; i < n; ++i) {
      if (perm[i] >= n) {
        return failure();
      }
    }

    // Build the new lower-rank permutation.
    SmallVector<int64_t> newPerm;
    for (int64_t i = n; i < static_cast<int64_t>(perm.size()); ++i) {
      newPerm.push_back(perm[i] - n);
    }

    // Build the new input reshape shape (drop leading 1s from permute input).
    auto permuteInType = permuteOp.getInput().getType();
    auto permuteInShape = permuteInType.getShape();
    SmallVector<int64_t> newMidShape(permuteInShape.drop_front(n));

    // Create new reshape: original input → reduced rank.
    auto newMidType =
        RankedTensorType::get(newMidShape, permuteInType.getElementType(),
                              permuteInType.getEncoding());
    SmallVector<int32_t> midShapeAttr(newMidShape.begin(), newMidShape.end());
    auto newReshape = rewriter.create<mlir::tt::ttir::ReshapeOp>(
        leadingReshape.getLoc(), newMidType, leadingReshape.getInput(),
        rewriter.getI32ArrayAttr(midShapeAttr));

    // Create new permute at reduced rank.
    SmallVector<int64_t> newOutShape;
    for (int64_t i : newPerm) {
      newOutShape.push_back(newMidShape[i]);
    }
    auto trailingType = trailingReshape.getType();
    auto newOutType = RankedTensorType::get(
        newOutShape, trailingType.getElementType(), trailingType.getEncoding());
    auto newPermute = rewriter.create<mlir::tt::ttir::PermuteOp>(
        permuteOp.getLoc(), newOutType, newReshape.getResult(), newPerm);

    rewriter.replaceOp(trailingReshape, newPermute.getResult());
    return success();
  });
}

//===----------------------------------------------------------------------===//
// RearrangeOp
//===----------------------------------------------------------------------===//

// RearrangeOp verification
::mlir::LogicalResult mlir::tt::ttir::RearrangeOp::verify() {
  llvm::StringRef patternStr = getPattern();
  llvm::SmallVector<llvm::StringRef> parts;
  patternStr.split(parts, "->");

  if (parts.size() != 2) {
    return emitOpError() << "pattern must contain exactly one '->' separator.";
  }

  mlir::FailureOr<AffineMap> failureOrInvMap = getInvPatternMap();
  if (failed(failureOrInvMap)) {
    return emitOpError() << "failed to parse pattern " << patternStr << ".";
  }

  auto invMap = *failureOrInvMap;
  if (getInput().getType().getRank() != invMap.getNumResults()) {
    return emitOpError() << "number of dimensions in the pattern's input ("
                         << invMap.getNumResults()
                         << ") must match the rank of the input tensor ("
                         << getInput().getType().getRank() << ").";
  }

  SmallVector<int64_t> expectedInputShape =
      ttmlir::utils::evalShape(invMap, getResult().getType().getShape());
  if (getInput().getType().getShape() !=
      ArrayRef<int64_t>(expectedInputShape)) {
    return emitOpError() << "input tensor shape ("
                         << ttmlir::utils::join(
                                getResult().getType().getShape(), ",")
                         << ") does not match the expected shape ("
                         << ttmlir::utils::join(expectedInputShape, ",")
                         << ").";
  }

  return success();
}

mlir::FailureOr<::mlir::AffineMap>
mlir::tt::ttir::RearrangeOp::getInvPatternMap(mlir::MLIRContext *context,
                                              StringRef pattern,
                                              ArrayRef<int64_t> shape) {
  // We need to write a routine to convert the pattern string to an affine map.
  // Example patterns:
  // >>> rearrange(images, 'b h w c -> b h w c').shape
  // (32, 30, 40, 3)
  //
  // # stacked and reordered axes to "b c h w" format
  // >>> rearrange(images, 'b h w c -> b c h w').shape
  // (32, 3, 30, 40)
  //
  // # concatenate images along height (vertical axis), 960 = 32 * 30
  // >>> rearrange(images, 'b h w c -> (b h) w c').shape
  // (960, 40, 3)
  //
  // # concatenated images along horizontal axis, 1280 = 32 * 40
  // >>> rearrange(images, 'b h w c -> h (b w) c').shape
  // (30, 1280, 3)
  //
  // # flattened each image into a vector, 3600 = 30 * 40 * 3
  // >>> rearrange(images, 'b h w c -> b (c h w)').shape
  // (32, 3600)
  //
  // # split each image into 4 smaller (top-left, top-right, bottom-left,
  // bottom-right), 128 = 32 * 2 * 2
  // >>> rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2,
  // w1=2).shape (128, 15, 20, 3)
  //
  // # space-to-depth operation
  // >>> rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2,
  // w1=2).shape (32, 15, 20, 12)

  llvm::SmallVector<llvm::StringRef> parts;
  pattern.split(parts, "->");

  assert(parts.size() == 2 &&
         "RearrangeOp pattern must contain exactly one '->' separator.");

  llvm::StringRef inputPattern = parts[0].trim();
  llvm::StringRef outputPattern = parts[1].trim();

  // Helper lambda to parse dimension names from a pattern.
  auto parseDims = [](llvm::StringRef pattern)
      -> llvm::SmallVector<llvm::SmallVector<llvm::StringRef>> {
    llvm::SmallVector<llvm::SmallVector<llvm::StringRef>> result;
    llvm::SmallVector<llvm::StringRef> currentGroup;
    bool inParens = false;
    size_t i = 0;

    while (i < pattern.size()) {
      char c = pattern[i];

      if (c == '(') {
        inParens = true;
        i++;
        continue;
      }

      if (c == ')') {
        if (!currentGroup.empty()) {
          result.push_back(currentGroup);
          currentGroup.clear();
        }
        inParens = false;
        i++;
        continue;
      }

      if (c == ' ' && !inParens && !currentGroup.empty()) {
        result.push_back(currentGroup);
        currentGroup.clear();
        i++;
        continue;
      }

      if (c == ' ') {
        i++;
        continue;
      }

      // Parse dimension name.
      size_t start = i;
      while (i < pattern.size() && pattern[i] != ' ' && pattern[i] != ')' &&
             pattern[i] != '(') {
        i++;
      }

      if (i > start) {
        currentGroup.push_back(pattern.substr(start, i - start));
      }
    }

    if (!currentGroup.empty()) {
      result.push_back(currentGroup);
    }

    return result;
  };

  auto inputDims = parseDims(inputPattern);
  auto outputDims = parseDims(outputPattern);

  // Build a map from dimension name to input position.
  llvm::DenseMap<llvm::StringRef, unsigned> dimToInputPos;
  unsigned pos = 0;
  for (const auto &group : inputDims) {
    if (group.size() > 1) {
      // Input groups are currently unsupported.
      return failure();
    }

    if (pos >= shape.size()) {
      // OOB dimension position for provided shape.
      return failure();
    }

    for (llvm::StringRef dim : group) {
      dimToInputPos[dim] = pos;
    }

    pos++;
  }

  // Build the affine expressions for the output.
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.resize(inputDims.size(), nullptr);
  for (const auto [groupPos, group] : llvm::enumerate(outputDims)) {
    assert(!group.empty());
    // For flattening like b h -> (b h)@d, we create inverse map: (d / h_size, d
    // % h_size).
    int64_t stride = 1;
    mlir::AffineExpr expr = mlir::getAffineConstantExpr(0, context);
    for (int64_t i = static_cast<int64_t>(group.size()) - 1; i >= 0; --i) {
      unsigned dimPos = dimToInputPos[group[i]];
      expr = mlir::getAffineDimExpr(groupPos, context).floorDiv(stride);
      if (i > 0) {
        expr = expr % shape[dimPos];
      }
      stride *= shape[dimPos];
      exprs[dimPos] = expr;
    }
  }

  return mlir::AffineMap::get(outputDims.size(), 0, exprs, context);
}

mlir::FailureOr<::mlir::AffineMap>
mlir::tt::ttir::RearrangeOp::getInvPatternMap() {
  return getInvPatternMap(
      getContext(), getPattern(),
      mlir::cast<ShapedType>(getInput().getType()).getShape());
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// BroadcastOp verification
::mlir::LogicalResult mlir::tt::ttir::BroadcastOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();

  // Sanity check to make sure that input rank matches the rank of the output
  // tensor.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "Input tensor rank of " << inputType.getRank()
                         << " does not match output tensor rank of "
                         << outputType.getRank();
  }

  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  // Verify that inputShape can be legally broadcasted to outputShape.
  llvm::SmallVector<int64_t> broadcastedShape;
  if (!mlir::OpTrait::util::getBroadcastedShape(inputShape, outputShape,
                                                broadcastedShape)) {
    return emitOpError() << "Input tensor shape ("
                         << ttmlir::utils::join(inputShape, ",")
                         << ") is not broadcastable to output shape ("
                         << ttmlir::utils::join(outputShape, ",") << ")";
  }

  auto broadcastDimensions = getBroadcastDimensions();

  // Check that the shape size matches the rank of the output tensor.
  if (static_cast<int64_t>(broadcastDimensions.size()) != inputType.getRank()) {
    return emitOpError("Input tensor rank should match output tensor rank.");
  }

  // Verify that each dimension of the inputShape multiplied by corresponding
  // broadcast dimension is equal to the outputShape dimension.
  for (size_t i = 0; i < broadcastDimensions.size(); i++) {
    int64_t dimValue = broadcastDimensions[i];
    if (inputShape[i] * dimValue != outputShape[i]) {
      return emitOpError() << "Input tensor shape ("
                           << ttmlir::utils::join(inputShape, ",") << ") index "
                           << i << " does not broadcast to output ("
                           << ttmlir::utils::join(outputShape, ",")
                           << ") using broadcast value " << dimValue;
    }
  }

  return success();
}

::mlir::OpFoldResult broadcastIdentityFold(mlir::tt::ttir::BroadcastOp op) {
  if (llvm::all_of(op.getBroadcastDimensions(),
                   [](const int32_t dim) { return dim == 1; })) {
    return op.getInput();
  }
  return nullptr;
}

::mlir::OpFoldResult constFoldBroadcast(mlir::tt::ttir::BroadcastOp op,
                                        Attribute constInput) {
  if (!constInput) {
    return nullptr;
  }
  RankedTensorType resultType = op.getResult().getType();
  if (auto foldResult = reshapeIfSplat(resultType, constInput)) {
    return foldResult;
  }
  return nullptr;
}

// BroadcastOp folder
::mlir::OpFoldResult mlir::tt::ttir::BroadcastOp::fold(FoldAdaptor adaptor) {
  if (auto foldResult = broadcastIdentityFold(*this)) {
    return foldResult;
  }

  if (auto foldResult = constFoldBroadcast(*this, adaptor.getInput())) {
    return foldResult;
  }

  return {};
}

//===----------------------------------------------------------------------===//
// SliceStaticOp
//===----------------------------------------------------------------------===//

// SliceStaticOp verification
::mlir::LogicalResult mlir::tt::ttir::SliceStaticOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::mlir::ArrayAttr begins = getBeginsAttr();
  ::mlir::ArrayAttr ends = getEndsAttr();
  ::mlir::ArrayAttr stepAttr = getStepAttr();
  ::mlir::RankedTensorType outputType = getType();

  // Verify that the input is at least 1D tensor
  if (inputType.getRank() < 1) {
    return emitOpError("Input must be at least a 1D tensor");
  }

  // Verify that the input rank matches number of elements in begins, ends, and
  // step
  size_t input_rank = static_cast<size_t>(inputType.getRank());
  if (input_rank != begins.size() || input_rank != ends.size() ||
      input_rank != stepAttr.size()) {
    return emitOpError("Begins, ends, and step attributes must have the same "
                       "number of elements as the input tensor rank");
  }

  // Validate that the output tensor has the same element type as the input
  // tensor
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError(
        "Output tensor must have the same element type as the input tensor");
  }

  // Verify the output tensor rank
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError(
        "Output tensor must have the same rank as the input tensor");
  }

  // Verify begin, end, step and the output tensor dimensions
  for (size_t i = 0; i < input_rank; ++i) {
    int64_t dimSize = inputShape[i];

    int32_t begin = ::mlir::cast<::mlir::IntegerAttr>(begins[i]).getInt();
    int32_t end = ::mlir::cast<::mlir::IntegerAttr>(ends[i]).getInt();
    int32_t step = ::mlir::cast<::mlir::IntegerAttr>(stepAttr[i]).getInt();

    // Adjust negative begin and end
    int32_t adjustedBegin = (begin < 0) ? (begin + dimSize) : begin;
    int32_t adjustedEnd = (end < 0) ? (end + dimSize) : end;

    std::ostringstream inputShapeStream;
    inputShapeStream << "(";
    for (size_t i = 0; i < inputShape.size(); ++i) {
      inputShapeStream << inputShape[i];
      if (i != inputShape.size() - 1) {
        inputShapeStream << ", ";
      }
    }
    inputShapeStream << ")";
    std::string inputShapeStr = inputShapeStream.str();
    bool isEmptySliceOp = adjustedEnd == adjustedBegin;

    if (!isEmptySliceOp && (adjustedBegin < 0 || adjustedBegin >= dimSize)) {
      return emitOpError() << "Invalid begin index for dimension "
                           << std::to_string(i) << ". Expected value in range ["
                           << std::to_string(-dimSize) << ", " << dimSize
                           << "), got " << begin
                           << ". Input shape: " << inputShapeStr;
    }
    if (!isEmptySliceOp && (adjustedEnd < 0 || adjustedEnd > dimSize)) {
      return emitOpError() << "Invalid end index for dimension "
                           << std::to_string(i) << ". Expected value in range ["
                           << std::to_string(-dimSize) << ", " << dimSize
                           << "], got " << end
                           << ". Input shape: " << inputShapeStr;
    }

    auto formatValueMessage = [](int value, int adjustedValue) {
      return value < 0 ? std::to_string(adjustedValue) + " (" +
                             std::to_string(value) + ")"
                       : std::to_string(value);
    };
    std::string beginValueMessage = formatValueMessage(begin, adjustedBegin);
    std::string endValueMessage = formatValueMessage(end, adjustedEnd);

    if (step == 0) {
      return emitOpError("Step value for dimension " + std::to_string(i) +
                         " cannot be zero");
    }

    if (step > 0 && adjustedBegin > adjustedEnd) {
      return emitOpError() << "For positive step, begin index must be less "
                              "than or equal to end index for dimension "
                           << i << ". Got begin: " << beginValueMessage
                           << ", end: " << endValueMessage << ", step: " << step
                           << ", input shape: " << inputShapeStr;
    }

    if (step < 0 && adjustedBegin < adjustedEnd) {
      return emitOpError() << "For negative step, begin index must be greater "
                              "than or equal to end index for dimension "
                           << i << ". Got begin: " << beginValueMessage
                           << ", end: " << endValueMessage << ", step: " << step
                           << ", input shape: " << inputShapeStr;
    }

    // Calculate the expected size of the output dimension
    int32_t expectedDimSize =
        (std::abs(adjustedEnd - adjustedBegin) + std::abs(step) - 1) /
        std::abs(step);
    if (outputType.getDimSize(i) != expectedDimSize) {
      return emitOpError() << "Mismatch in dimension " << std::to_string(i)
                           << " of the output tensor: expected size "
                           << expectedDimSize << ", but got "
                           << outputType.getDimSize(i);
    }
  }

  return success();
}

// back to back producer-consumer SliceStaticOp can be folded
// into a single SliceStaticOp.
static mlir::OpFoldResult
foldConsecutiveSliceStatic(mlir::tt::ttir::SliceStaticOp consumerOp) {

  if (auto producerSliceOp =
          consumerOp.getInput()
              .getDefiningOp<mlir::tt::ttir::SliceStaticOp>()) {

    // If producerOp has multiple uses, do not fold
    if (!producerSliceOp->hasOneUse()) {
      return nullptr;
    }

    mlir::RankedTensorType inputType = consumerOp.getInput().getType();
    size_t input_rank = static_cast<size_t>(inputType.getRank());

    // Producer begins and step arrays
    mlir::ArrayAttr begins1 = producerSliceOp.getBeginsAttr();
    mlir::ArrayAttr stepAttr1 = producerSliceOp.getStepAttr();

    // Consumer begins, end and step arrays
    mlir::ArrayAttr begins2 = consumerOp.getBeginsAttr();
    mlir::ArrayAttr ends2 = consumerOp.getEndsAttr();
    mlir::ArrayAttr stepAttr2 = consumerOp.getStepAttr();

    llvm::SmallVector<mlir::Attribute> begins, ends, step;
    mlir::MLIRContext *ctx = consumerOp->getContext();
    auto i32Ty = mlir::IntegerType::get(ctx, 32);
    for (size_t i = 0; i < input_rank; ++i) {
      // ith dimension in Producer begins and step arrays
      int32_t b1_i = mlir::cast<::mlir::IntegerAttr>(begins1[i]).getInt();
      int32_t s1_i = mlir::cast<::mlir::IntegerAttr>(stepAttr1[i]).getInt();

      // ith dimension in Consumer begins, end and step arrays
      int32_t b2_i = mlir::cast<::mlir::IntegerAttr>(begins2[i]).getInt();
      int32_t e2_i = mlir::cast<::mlir::IntegerAttr>(ends2[i]).getInt();
      int32_t s2_i = mlir::cast<::mlir::IntegerAttr>(stepAttr2[i]).getInt();

      int32_t b_i = b1_i + s1_i * b2_i;
      int32_t e_i = b1_i + s1_i * (e2_i - 1) + 1;
      int32_t s_i = s1_i * s2_i;

      begins.push_back(mlir::IntegerAttr::get(i32Ty, b_i));
      ends.push_back(mlir::IntegerAttr::get(i32Ty, e_i));
      step.push_back(mlir::IntegerAttr::get(i32Ty, s_i));
    }

    mlir::ArrayAttr beginsArrayAttr = mlir::ArrayAttr::get(ctx, begins);
    mlir::ArrayAttr endsArrayAttr = mlir::ArrayAttr::get(ctx, ends);
    mlir::ArrayAttr stepArrayAttr = mlir::ArrayAttr::get(ctx, step);

    consumerOp.setBeginsAttr(beginsArrayAttr);
    consumerOp.setEndsAttr(endsArrayAttr);
    consumerOp.setStepAttr(stepArrayAttr);
    consumerOp->setOperand(0, producerSliceOp.getInput());
    return consumerOp.getResult();
  }

  return nullptr;
}

// Fold slice of concat when taking an entire input tensor along the concat
// dimension Pattern: slice(concat(t1, t2, ..., tn), dim=concat_dim,
// begins=[..., offset, ...], ends=[..., offset + size_of_ti, ...]) -> ti This
// eliminates unnecessary concatenation when only one input tensor is needed.
static mlir::OpFoldResult
foldSliceOfConcat(mlir::tt::ttir::SliceStaticOp sliceOp) {
  auto concatOp = sliceOp.getInput().getDefiningOp<mlir::tt::ttir::ConcatOp>();
  if (!concatOp) {
    return nullptr;
  }

  mlir::ArrayAttr beginsAttr = sliceOp.getBeginsAttr();
  mlir::ArrayAttr endsAttr = sliceOp.getEndsAttr();
  mlir::ArrayAttr stepsAttr = sliceOp.getStepAttr();

  if (!beginsAttr || !endsAttr || !stepsAttr ||
      beginsAttr.size() != endsAttr.size() ||
      beginsAttr.size() != stepsAttr.size()) {
    return nullptr;
  }

  int32_t concatDim = concatOp.getDim();
  // Normalize negative concat dimension
  if (concatDim < 0) {
    concatDim += beginsAttr.size();
  }

  // Track offset along the concat dimension
  int64_t offset = 0;

  // Try every concat input to find one that matches the slice pattern
  for (auto curInput : concatOp.getInputs()) {
    auto curInputType =
        mlir::dyn_cast<mlir::RankedTensorType>(curInput.getType());
    if (!curInputType ||
        curInputType.getRank() != static_cast<int64_t>(beginsAttr.size()) ||
        concatDim >= curInputType.getRank()) {
      continue;
    }

    int64_t curInputSize = curInputType.getShape()[concatDim];
    if (curInputSize == mlir::ShapedType::kDynamic) {
      continue;
    }

    // Check if the slice covers this input entirely
    bool matches = llvm::all_of(
        llvm::enumerate(llvm::zip(beginsAttr, endsAttr, stepsAttr,
                                  curInputType.getShape())),
        [&](auto pair) {
          auto [dimIdx, tuple] = pair;
          auto [dimBegin, dimEnd, dimStep, inputDimSize] = tuple;

          int32_t begin = mlir::cast<mlir::IntegerAttr>(dimBegin).getInt();
          int32_t end = mlir::cast<mlir::IntegerAttr>(dimEnd).getInt();
          int32_t step = mlir::cast<mlir::IntegerAttr>(dimStep).getInt();

          int32_t expectedBegin =
              (dimIdx == static_cast<size_t>(concatDim)) ? offset : 0;
          int32_t expectedEnd = (dimIdx == static_cast<size_t>(concatDim))
                                    ? offset + curInputSize
                                    : inputDimSize;

          return begin == expectedBegin && end == expectedEnd && step == 1;
        });

    if (matches) {
      return curInput;
    }
    offset += curInputSize;
  }

  return nullptr;
}

static mlir::OpFoldResult
constantFoldSliceStatic(mlir::tt::ttir::SliceStaticOp op,
                        Attribute constInput) {
  if (!constInput) {
    return nullptr;
  }

  auto inputShape = op.getInput().getType().getShape();
  llvm::SmallVector<int64_t> begins(inputShape.size());
  llvm::SmallVector<int64_t> step(inputShape.size());
  for (size_t i = 0; i < inputShape.size(); ++i) {
    int64_t begin = mlir::cast<mlir::IntegerAttr>(op.getBegins()[i]).getInt();
    step[i] = mlir::cast<mlir::IntegerAttr>(op.getStep()[i]).getInt();
    // Adjust negative begin.
    begins[i] = (begin < 0) ? (begin + inputShape[i]) : begin;
  }

  return constantFoldTM(
      op, constInput,
      [&begins, &step](const llvm::SmallVector<int64_t> &outputCoord) {
        llvm::SmallVector<int64_t> inputCoord(outputCoord.size());
        for (size_t i = 0; i != inputCoord.size(); ++i) {
          inputCoord[i] = begins[i] + step[i] * outputCoord[i];
        }
        return inputCoord;
      });
}

// SliceStaticOp Folder
mlir::OpFoldResult mlir::tt::ttir::SliceStaticOp::fold(FoldAdaptor adaptor) {

  if (auto foldResult = foldConsecutiveSliceStatic(*this)) {
    return foldResult;
  }

  if (auto foldResult = foldSliceOfConcat(*this)) {
    return foldResult;
  }

  if (auto foldResult = constantFoldSliceStatic(*this, adaptor.getInput())) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// SliceDynamicOp
//===----------------------------------------------------------------------===//

// SliceDynamicOp verification
::mlir::LogicalResult mlir::tt::ttir::SliceDynamicOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType beginsType = getBegins().getType();
  ::llvm::ArrayRef<int64_t> beginsShape = beginsType.getShape();
  ::mlir::RankedTensorType endsType = getEnds().getType();
  ::llvm::ArrayRef<int64_t> endsShape = endsType.getShape();
  ::mlir::ArrayAttr stepAttr = getStepAttr();
  ::mlir::RankedTensorType outputType = getType();

  // Verify that the input is at least 1D tensor.
  if (inputType.getRank() < 1) {
    return emitOpError("Input must be at least a 1D tensor");
  }

  // Verify that begins and ends are 1D tensors.
  size_t beginsRank = static_cast<size_t>(beginsType.getRank());
  size_t endsRank = static_cast<size_t>(endsType.getRank());
  if (beginsRank != 1 || endsRank != 1) {
    return emitOpError("Begins and ends must be 1D tensors");
  }

  // Verify that the input rank matches number of elements in begins, ends, and
  // step.
  auto inputRank = inputType.getRank();

  if (inputRank != beginsShape[0] || inputRank != endsShape[0] ||
      (stepAttr && static_cast<size_t>(inputRank) != stepAttr.size())) {
    return emitOpError("Begins, ends, and step must have the same "
                       "number of elements as the input tensor rank");
  }

  // Validate that the output tensor has the same element type as the input
  // tensor.
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError(
        "Output tensor must have the same element type as the input tensor");
  }

  // Verify the output tensor rank.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError(
        "Output tensor must have the same rank as the input tensor");
  }

  if (stepAttr) {
    // Verify that step isn't zero for any dimension.
    for (auto i = 0; i < inputRank; ++i) {
      int32_t step = ::mlir::cast<::mlir::IntegerAttr>(stepAttr[i]).getInt();
      if (step == 0) {
        return emitOpError("Step value for dimension " + std::to_string(i) +
                           " cannot be zero");
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// IndexOp
//===----------------------------------------------------------------------===//

// ANCHOR: decomposing_an_op_index_ttir_verify
// IndexOp verification
::mlir::LogicalResult mlir::tt::ttir::IndexOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::mlir::RankedTensorType outputType = getType();
  int32_t dim = getDim();
  int32_t begin = getBegin();
  int32_t end = getEnd();
  int32_t step = getStep();

  // Verify that the input is at least 1D tensor
  if (inputType.getRank() < 1) {
    return emitOpError("Input must be at least a 1D tensor");
  }

  // Validate that the output tensor has the same element type as the input
  // tensor
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError(
        "Output tensor must have the same element type as the input tensor");
  }

  // Verify the output tensor rank
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError(
        "Output tensor must have the same rank as the input tensor");
  }

  // Verify that the dim attribute is within the bounds of the input tensor
  if (dim < 0 || dim >= inputType.getRank()) {
    return emitOpError() << "Invalid dimension index " << dim
                         << ". Input tensor rank is " << inputType.getRank();
  }

  // Verify begin, end, step and the output tensor dimensions
  int64_t dimSize = inputShape[dim];

  // Adjust negative begin and end
  int32_t adjustedBegin = (begin < 0) ? (begin + dimSize) : begin;
  int32_t adjustedEnd = (end < 0) ? (end + dimSize) : end;

  std::ostringstream inputShapeStream;
  inputShapeStream << "(";
  for (size_t i = 0; i < inputShape.size(); ++i) {
    inputShapeStream << inputShape[i];
    if (i != inputShape.size() - 1) {
      inputShapeStream << ", ";
    }
  }
  inputShapeStream << ")";
  std::string inputShapeStr = inputShapeStream.str();

  if (adjustedBegin < 0 || adjustedBegin >= dimSize) {
    return emitOpError() << "Invalid begin index for dimension "
                         << std::to_string(dim) << ". Expected value in range ["
                         << std::to_string(-dimSize) << ", " << dimSize
                         << "), got " << begin
                         << ". Input shape: " << inputShapeStr;
  }
  if (adjustedEnd < 0 || adjustedEnd > dimSize) {
    return emitOpError() << "Invalid end index for dimension "
                         << std::to_string(dim) << ". Expected value in range ["
                         << std::to_string(-dimSize) << ", " << dimSize
                         << "], got " << end
                         << ". Input shape: " << inputShapeStr;
  }

  auto formatValueMessage = [](int value, int adjustedValue) {
    return value < 0 ? std::to_string(adjustedValue) + " (" +
                           std::to_string(value) + ")"
                     : std::to_string(value);
  };
  std::string beginValueMessage = formatValueMessage(begin, adjustedBegin);
  std::string endValueMessage = formatValueMessage(end, adjustedEnd);

  if (step == 0) {
    return emitOpError("Step value for dimension " + std::to_string(dim) +
                       " cannot be zero");
  }

  if (step > 0 && adjustedBegin > adjustedEnd) {
    return emitOpError() << "For positive step, begin index must be less "
                            "than or equal to end index for dimension "
                         << dim << ". Got begin: " << beginValueMessage
                         << ", end: " << endValueMessage << ", step: " << step
                         << ", input shape: " << inputShapeStr;
  }

  if (step < 0 && adjustedBegin < adjustedEnd) {
    return emitOpError() << "For negative step, begin index must be greater "
                            "than or equal to end index for dimension "
                         << dim << ". Got begin: " << beginValueMessage
                         << ", end: " << endValueMessage << ", step: " << step
                         << ", input shape: " << inputShapeStr;
  }

  // Calculate the expected size of the output dimension
  int32_t expectedDimSize =
      (std::abs(adjustedEnd - adjustedBegin) + std::abs(step) - 1) /
      std::abs(step);
  if (outputType.getDimSize(dim) != expectedDimSize) {
    return emitOpError() << "Mismatch in dimension " << std::to_string(dim)
                         << " of the output tensor: expected size "
                         << expectedDimSize << ", but got "
                         << outputType.getDimSize(dim);
  }

  return success();
}
// ANCHOR_END: decomposing_an_op_index_ttir_verify

//===----------------------------------------------------------------------===//
// IndexSelectOp
//===----------------------------------------------------------------------===//

// IndexSelectOp verification
::mlir::LogicalResult mlir::tt::ttir::IndexSelectOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();

  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError("Input and output tensors must have the same rank.");
  }

  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("Input and output tensors must have the same element "
                       "type.");
  }

  int32_t dim = getDim();
  int32_t origDim = dim;
  if (dim < 0) {
    dim += inputType.getRank();
  }

  if (dim < 0 || dim >= inputType.getRank()) {
    return emitOpError() << "Invalid dimension " << origDim
                         << " for select op with input tensor rank "
                         << inputType.getRank();
  }

  int32_t dimSize = inputType.getDimSize(dim);

  int32_t stride = getStride();
  if (stride == 0) {
    stride = dimSize;
  }

  if (stride < 0) {
    return emitOpError() << "Invalid stride " << stride << " for dimension "
                         << dim << ", stride must be non-negative";
  }

  if (stride > dimSize) {
    return emitOpError() << "Invalid stride " << stride << " for dimension "
                         << dim << " with size " << dimSize
                         << ". stride must be less than or equal to the "
                            "dimension size";
  }

  int32_t begin = getBegin();
  int32_t length = getLength();
  if (begin < 0 || begin >= dimSize) {
    return emitOpError() << "Invalid begin index " << begin << " for dimension "
                         << dim << " with size " << dimSize
                         << ". begin must be "
                            "in the range [0, dimSize)";
  }

  if (length < 1 || length > stride) {
    return emitOpError() << "Invalid length " << length << " for begin index "
                         << begin << " and stride " << stride
                         << " for dimension " << dim << " with size " << dimSize
                         << ". stride must be greater than or equal to length";
  }

  if (begin + length > dimSize) {
    return emitOpError() << "Invalid length " << length << " for begin index "
                         << begin << " and dimension " << dim << " with size "
                         << dimSize
                         << ". begin + length must be less than or "
                            "equal to the dimension size";
  }

  // Get the number of slices as the number of times the stride fits in the
  // dimension size starting from the begin index.
  int32_t numSlices = (dimSize - begin + stride - 1) / stride;
  int32_t totalLength = 0;
  for (int32_t i = 0; i < numSlices; i++) {
    int32_t newBegin = begin + i * stride;
    int32_t newEnd = std::min(newBegin + length, dimSize);
    totalLength += newEnd - newBegin;
  }

  if (totalLength != outputType.getDimSize(dim)) {
    return emitOpError() << "Sum of all slices must be equal to the output "
                            "dimension size for the given dimension. Expected "
                            "output dimension size: "
                         << outputType.getDimSize(dim) << ", but got "
                         << totalLength;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SqueezeOp
//===----------------------------------------------------------------------===//

// SqueezeOp verification
::mlir::LogicalResult mlir::tt::ttir::SqueezeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();
  int32_t dim = getDim();

  if (dim < 0) {
    dim += inputType.getRank();
  }

  // Check that the dimension `dim` is valid.
  if (dim < 0 || dim >= inputType.getRank()) {
    return emitOpError() << "Invalid dimension " << dim << " for squeezing.";
  }

  // Check that the dimension `dim` is 1 in the input tensor.
  if (inputType.getDimSize(dim) != 1) {
    return emitOpError() << "Dimension " << dim
                         << " in the input tensor must be 1.";
  }

  if (outputType.getRank() == 0) {
    return emitOpError() << "Output tensor must have at least one dimension.";
  }

  // Check that the rank of the output tensor is one less than the input tensor.
  if (outputType.getRank() != inputType.getRank() - 1) {
    return emitOpError()
           << "Output tensor rank must be one less than the input tensor rank.";
  }

  // Check that the dimensions of the output tensor are the same as the input
  // tensor except for dimension `dim`.
  for (int64_t i = 0, j = 0; i < inputType.getRank(); ++i) {
    if (i == dim) {
      continue;
    }
    if (inputType.getDimSize(i) != outputType.getDimSize(j)) {
      return emitOpError() << "Dimensions of the output tensor must be the "
                              "same as the input tensor except for dimension "
                           << dim << ".";
    }
    ++j;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// TransposeOp verification
::mlir::LogicalResult mlir::tt::ttir::TransposeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();
  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();
  int32_t dim0 = getDim0();
  int32_t dim1 = getDim1();
  if (inputType.getRank() < 2) {
    return emitOpError("Input must be at least a 2D tensor");
  }
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError("Input must have the same rank as output");
  }
  if (dim0 >= inputType.getRank() || dim0 < -inputType.getRank()) {
    return emitOpError(
        "Dimension 0 attribute must be within the bounds of the input tensor");
  }
  if (dim1 >= inputType.getRank() || dim1 < -inputType.getRank()) {
    return emitOpError(
        "Dimension 1 attribute must be within the bounds of the input tensor");
  }
  if (dim0 < 0) {
    dim0 += inputType.getRank();
  }
  if (dim1 < 0) {
    dim1 += inputType.getRank();
  }
  if (outputShape[dim0] != inputShape[dim1] ||
      outputShape[dim1] != inputShape[dim0]) {
    return emitOpError("Input-output transpose dimension mismatch.");
  }
  return success();
}

void mlir::tt::ttir::TransposeOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *) {
  patterns.add(+[](TransposeOp op, mlir::PatternRewriter &rewriter) {
    SmallVector<int64_t> permutation;
    for (int64_t i = 0; i < op.getInput().getType().getRank(); ++i) {
      permutation.push_back(i);
    }
    int64_t dim0 = op.getDim0() < 0
                       ? op.getDim0() + op.getInput().getType().getRank()
                       : op.getDim0();
    int64_t dim1 = op.getDim1() < 0
                       ? op.getDim1() + op.getInput().getType().getRank()
                       : op.getDim1();
    std::swap(permutation[dim0], permutation[dim1]);
    rewriter.replaceOpWithNewOp<PermuteOp>(op, op.getType(), op.getInput(),
                                           permutation);
    return success();
  });
}

//===----------------------------------------------------------------------===//
// BitcastConvertOp
//===----------------------------------------------------------------------===//

// BitcastConvertOp verification
::mlir::LogicalResult mlir::tt::ttir::BitcastConvertOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();

  auto inElementType = inputType.getElementType();
  auto outElementType = outputType.getElementType();

  if (!inElementType.isIntOrFloat()) {
    return emitOpError(
        "Input tensor type must be an integer or floating point type");
  }

  if (!outElementType.isIntOrFloat()) {
    return emitOpError(
        "Output tensor type must be an integer or floating point type");
  }

  if (inElementType.getIntOrFloatBitWidth() !=
      outElementType.getIntOrFloatBitWidth()) {
    return emitOpError(
        "Input and output tensor element types must have the same bit width");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TypecastOp
//===----------------------------------------------------------------------===//

static ::mlir::OpFoldResult
foldIdentityTypecast(mlir::tt::ttir::TypecastOp op) {
  if (op.getType() == op.getInput().getType()) {
    return op.getInput();
  }
  return nullptr;
}

// Helper to check if a float cast can be folded based on the conversion status.
static bool isFloatCastFoldSuccessful(llvm::APFloat::opStatus status) {
  // Don't fold if the conversion failed, as runtime semantics don't match IEEE
  // 754 in some special cases (e.g. overflow, NaN). Fold if the status is not
  // OK solely because the conversion is inexact.
  return status == llvm::APFloat::opOK || status == llvm::APFloat::opInexact;
}

// Helper to check if an integer constant cast can be safely folded.
static bool isIntCastFoldSafe(const llvm::APInt &value,
                              mlir::IntegerType sourceType,
                              mlir::IntegerType targetType) {
  unsigned targetWidth = targetType.getWidth();

  if (targetType.isUnsigned()) {
    if (!sourceType.isUnsigned() && value.isNegative()) {
      // Don't fold if the value is negative and the target type is unsigned.
      return false;
    }
    // Don't fold if the value overflows.
    return value.isIntN(targetWidth);
  }

  if (targetType.isSigned()) {
    if (!sourceType.isSigned()) {
      // Don't fold if the value overflows/underflows. This is checked
      // separately for unsigned source to avoid casting large positive
      // integers to negative.
      return value.isIntN(targetWidth - 1);
    }
    // Don't fold if the value overflows/underflows.
    return value.isSignedIntN(targetWidth);
  }

  // Fold conversion to signless integer type only if it would be correct for
  // both signed and unsigned target types.
  return value.isIntN(targetWidth - 1);
}

static ::mlir::OpFoldResult
constantFoldTypecast(mlir::tt::ttir::TypecastOp op,
                     mlir::tt::ttir::TypecastOp::FoldAdaptor adaptor) {
  auto input =
      mlir::dyn_cast_if_present<mlir::ElementsAttr>(adaptor.getInput());
  if (!input) {
    return nullptr;
  }
  if (!input.isSplat() && !shouldFold(op)) {
    return nullptr;
  }

  // We ignore `conservative_folding` attribute of typecast op here, because it
  // is meant for folding of consecutive ops without the known input. Constant
  // folding will not cause issues that this attribute is designed to prevent,
  // like the removal of narrowing conversions in conversion chains.

  auto outputType = op.getResult().getType();
  auto outputElementType = outputType.getElementType();
  auto inputElementType = input.getElementType();

  if (inputElementType.isFloat()) {
    if (auto targetType = mlir::dyn_cast<FloatType>(outputElementType)) {
      return foldCast<mlir::FloatAttr, mlir::FloatAttr>(
          adaptor.getOperands(), outputType,
          [targetType](llvm::APFloat value, bool &castStatus) -> llvm::APFloat {
            bool losesInfo{};
            llvm::APFloat::opStatus status =
                value.convert(targetType.getFloatSemantics(),
                              llvm::APFloat::rmNearestTiesToEven, &losesInfo);
            castStatus = isFloatCastFoldSuccessful(status);
            return value;
          });
    }

    if (auto targetType = mlir::dyn_cast<IntegerType>(outputElementType)) {
      return foldCast<mlir::FloatAttr, mlir::IntegerAttr>(
          adaptor.getOperands(), outputType,
          [targetType](const llvm::APFloat &value,
                       bool &castStatus) -> llvm::APInt {
            llvm::APSInt intValue(targetType.getWidth(),
                                  targetType.isUnsigned());
            bool isExact{};
            llvm::APFloat::opStatus status = value.convertToInteger(
                intValue, llvm::APFloat::rmTowardZero, &isExact);
            castStatus = isFloatCastFoldSuccessful(status);
            return intValue;
          });
    }
  }

  if (auto sourceType = mlir::dyn_cast<mlir::IntegerType>(inputElementType)) {
    if (auto targetType = mlir::dyn_cast<FloatType>(outputElementType)) {
      return foldCast<mlir::IntegerAttr, mlir::FloatAttr>(
          adaptor.getOperands(), outputType,
          [sourceType, targetType](const llvm::APInt &value,
                                   bool &castStatus) -> llvm::APFloat {
            llvm::APFloat floatValue(targetType.getFloatSemantics());
            // If target type is a signless integer, we treat it as signed to
            // keep negative floats negative after conversion.
            bool isSigned = !sourceType.isUnsigned();
            llvm::APFloat::opStatus status = floatValue.convertFromAPInt(
                value, isSigned, llvm::APFloat::rmNearestTiesToEven);
            castStatus = isFloatCastFoldSuccessful(status);
            return floatValue;
          });
    }

    if (auto targetType = mlir::dyn_cast<IntegerType>(outputElementType)) {
      return foldCast<mlir::IntegerAttr, mlir::IntegerAttr>(
          adaptor.getOperands(), outputType,
          [sourceType, targetType](const llvm::APInt &value,
                                   bool &castStatus) -> llvm::APInt {
            castStatus = isIntCastFoldSafe(value, sourceType, targetType);
            if (sourceType.isUnsigned()) {
              return value.zextOrTrunc(targetType.getWidth());
            }
            return value.sextOrTrunc(targetType.getWidth());
          });
    }
  }

  return nullptr;
}

// TypecastOp folder
mlir::OpFoldResult mlir::tt::ttir::TypecastOp::fold(FoldAdaptor adaptor) {
  if (auto foldResult = foldIdentityTypecast(*this)) {
    return foldResult;
  }

  if (auto foldResult = constantFoldTypecast(*this, adaptor)) {
    return foldResult;
  }

  return nullptr;
}

static bool isNarrowingConversion(const ::mlir::tt::ttcore::DataType srcDtype,
                                  const ::mlir::tt::ttcore::DataType dstDtype) {
  const bool srcIsFloat = isFloat(srcDtype);
  const bool dstIsFloat = isFloat(dstDtype);
  const auto srcNumberOfBits = getNumberOfBits(srcDtype);
  const auto dstNumberOfBits = getNumberOfBits(dstDtype);

  if (srcIsFloat && !dstIsFloat) {
    return true;
  }

  if (srcIsFloat && dstIsFloat) {
    const auto srcExponentSize = getExponentSize(srcDtype);
    const auto dstExponentSize = getExponentSize(dstDtype);
    const auto srcMantissaSize = getMantissaSize(srcDtype);
    const auto dstMantissaSize = getMantissaSize(dstDtype);
    return srcExponentSize > dstExponentSize ||
           srcMantissaSize > dstMantissaSize;
  }

  // For integer to FP, it is narrowing if the FP type has fewer bits in its
  // mantissa than the integer type's magnitude bits.
  if (!srcIsFloat && dstIsFloat) {
    if (isSignedInteger(srcDtype)) {
      return srcNumberOfBits - 1 > getMantissaSize(dstDtype);
    }
    return srcNumberOfBits > getMantissaSize(dstDtype);
  }

  assert(!srcIsFloat && !dstIsFloat);
  const auto srcIsSigned = isSignedInteger(srcDtype);
  const auto dstIsSigned = isSignedInteger(dstDtype);
  // When signedness are the same, reducing the number of bits is narrowing.
  if (srcIsSigned == dstIsSigned) {
    return srcNumberOfBits > dstNumberOfBits;
  }
  // Unsigned->Signed is narrowing when the signed type can't hold the largest.
  // value of the unsigned type
  if (!srcIsSigned && dstIsSigned) {
    return srcNumberOfBits >= dstNumberOfBits;
  }
  // Signed->Unsigned is always narrowing.
  assert(srcIsSigned && !dstIsSigned);
  return true;
}

// TypecastOp canonicalization method
::llvm::LogicalResult
mlir::tt::ttir::TypecastOp::canonicalize(mlir::tt::ttir::TypecastOp op,
                                         ::mlir::PatternRewriter &rewriter) {
  // Fold two consecutive typecast ops into a single one.
  ::mlir::tt::ttir::TypecastOp producerOp =
      op.getInput().getDefiningOp<mlir::tt::ttir::TypecastOp>();

  if (!producerOp) {
    return mlir::failure();
  }

  const bool conservativeFolding =
      op.getConservativeFolding() || producerOp.getConservativeFolding();

  if (conservativeFolding) {
    // Disable folding if it has the potential to cause too much numerical
    // differences.
    auto dtypeIn = ttcore::elementTypeToDataType(
        producerOp.getInput().getType().getElementType());
    auto dtypeMid =
        ttcore::elementTypeToDataType(op.getInput().getType().getElementType());
    auto dtypeOut =
        ttcore::elementTypeToDataType(op.getType().getElementType());

    assert(dtypeMid == ttcore::elementTypeToDataType(
                           producerOp.getType().getElementType()));

    // If the 1st Op is narrowing and the 2nd Op is widening, we shouldn't fold.
    // FP->Int->FP is special and should never fold, due to its truncation
    // semantics and application in QDQ models.
    const bool isNarrowingProducer = isNarrowingConversion(dtypeIn, dtypeMid);
    const bool isNarrowingConsumer = isNarrowingConversion(dtypeMid, dtypeOut);
    const bool isFpIntFp =
        isFloat(dtypeIn) && !isFloat(dtypeMid) && isFloat(dtypeOut);
    if (isFpIntFp || (isNarrowingProducer && !isNarrowingConsumer)) {
      return mlir::failure();
    }
  }

  // The resulting Op is conservative iff both typecast ops were conservative.
  rewriter.replaceOpWithNewOp<ttir::TypecastOp>(
      op, op.getType(), producerOp.getInput(),
      op.getConservativeFolding() && producerOp.getConservativeFolding());

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// UnsqueezeOp
//===----------------------------------------------------------------------===//

// UnsqueezeOp verification
::mlir::LogicalResult mlir::tt::ttir::UnsqueezeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();
  int32_t dim = getDim();

  // Convert negative dim to its positive equivalent
  if (dim < 0) {
    dim += inputType.getRank() + 1;
  }

  // Check that the dim is within the bounds of the input tensor
  if (dim > inputType.getRank() || dim < 0) {
    return emitOpError(
        "Dimension attribute must be within the bounds of the input tensor");
  }

  // Check that the output tensor has one more dimension than the input tensor
  if (outputType.getRank() != inputType.getRank() + 1) {
    return emitOpError(
        "Output tensor must have one more dimension than the input tensor");
  }

  // and that the dimension added is of size 1
  if (outputType.getDimSize(dim) != 1) {
    return emitOpError("Dimension added must be of size 1");
  }

  // All dimensions of the input tensor must be the same as the output tensor
  // except for the dimension added
  for (int64_t i = 0, j = 0; i < outputType.getRank(); ++i) {
    if (i == dim) {
      continue;
    }

    if (inputType.getDimSize(j) != outputType.getDimSize(i)) {
      return emitOpError("All dimensions of the input tensor must be the same "
                         "as the output tensor except for the dimension added");
    }

    j++;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EmbeddingOp
//===----------------------------------------------------------------------===//

// EmbeddingOp verification
::mlir::LogicalResult mlir::tt::ttir::EmbeddingOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType weightType = getWeight().getType();
  ::mlir::RankedTensorType outputType = getType();

  // Input tensor must be at most 2D tensor.
  if (inputType.getRank() > 2) {
    return emitOpError("input must be at most a 2D tensor, got ")
           << inputType.getRank() << "D ranked tensor";
  }

  // Weight tensor must be effectively 2D tensor. It means that it must have
  // shape of (1, 1,..., 1, N, M) where N is the dictionary size and M is the
  // embedding size.
  if (weightType.getRank() < 2) {
    return emitOpError("weight must be at least 2D tensor, got ")
           << weightType.getRank() << "D ranked tensor";
  }
  if (std::any_of(weightType.getShape().begin(),
                  weightType.getShape().end() - 2,
                  [](int64_t dim) { return dim != 1; })) {
    return emitOpError("weight must be effectively 2D tensor");
  }

  // Output tensor is expected to have the shape of (*inputTensorShape,
  // embeddingSize).
  int64_t embeddingSize = weightType.getDimSize(weightType.getRank() - 1);
  llvm::SmallVector<int64_t, 3> expectedOutputShape(inputType.getShape());
  expectedOutputShape.push_back(embeddingSize);

  if (!llvm::equal(expectedOutputShape, outputType.getShape())) {
    return emitOpError() << "expected output shape of (" << expectedOutputShape
                         << ") but got (" << outputType.getShape() << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EmbeddingBackwardOp
//===----------------------------------------------------------------------===//

// EmbeddingBackwardOp verification
::mlir::LogicalResult mlir::tt::ttir::EmbeddingBackwardOp::verify() {
  ::mlir::RankedTensorType weightType = getWeight().getType();
  ::mlir::RankedTensorType inputGradType = getInGradient().getType();
  ::mlir::RankedTensorType outputType = getType();

  // weightType must have rank of 2: (dictionary_size, embedding_size).
  if (weightType.getRank() != 2) {
    return emitOpError("Input must be a 2D tensor");
  }

  // inputGradType checks.
  if (inputGradType.getElementType() != outputType.getElementType()) {
    return emitOpError("Input gradient and output must have the same dtype");
  }

  // outputType should have the same shape as weightType.
  if (outputType.getShape() != weightType.getShape()) {
    return emitOpError("Output must have the same shape as weight");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//

struct ToLayoutFoldRedundantPattern : public OpRewritePattern<ToLayoutOp> {
  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  ToLayoutFoldRedundantPattern(MLIRContext *context)
      : OpRewritePattern<ToLayoutOp>(context) {
    setDebugName("ttir.ToLayoutFoldRedundantPattern");
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    ToLayoutOp producerLayoutOp = op.getInput().getDefiningOp<ToLayoutOp>();
    if (!producerLayoutOp) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<ToLayoutOp>(op, producerLayoutOp.getInput(),
                                            op.getOutput());
    return success();
  }
};

static ::mlir::LogicalResult
verifyLayoutOp(mlir::Operation *op, mlir::Type inputTensorOrMemrefTy,
               mlir::Type outputTensorOrMemrefTy, bool allowFormatChange,
               bool allowMemorySpaceChange, bool checkMemrefRank = false,
               bool checkMemrefGridShardForm = false,
               bool checkMemrefShardShape = false) {
  if (mlir::RankedTensorType inputTy =
          mlir::dyn_cast<mlir::RankedTensorType>(inputTensorOrMemrefTy)) {
    mlir::RankedTensorType outputTy =
        mlir::dyn_cast<mlir::RankedTensorType>(outputTensorOrMemrefTy);
    if (!outputTy) {
      return op->emitOpError("Input and output types must be the same");
    }

    auto inputLayout =
        mlir::dyn_cast_if_present<mlir::tt::ttcore::MetalLayoutAttr>(
            inputTy.getEncoding());
    auto outputLayout =
        mlir::dyn_cast_if_present<mlir::tt::ttcore::MetalLayoutAttr>(
            outputTy.getEncoding());

    if (!inputLayout || !outputLayout) {
      // If the input/output tensor does not have a layout, we can early exit.
      return mlir::success();
    }

    const bool isFormatChange =
        inputTy.getElementType() != outputTy.getElementType();
    if (isFormatChange && !allowFormatChange) {
      return op->emitOpError(
          "Input and output tensor element types must be the same");
    }

    const bool isMemorySpaceChange =
        inputLayout.getMemorySpace() != outputLayout.getMemorySpace();
    if (!allowMemorySpaceChange && isMemorySpaceChange) {
      return op->emitOpError(
          "Input and output layout memory spaces must be the same");
    }
    return mlir::success();
  }

  if (mlir::MemRefType inputTy =
          mlir::dyn_cast<mlir::MemRefType>(inputTensorOrMemrefTy)) {
    mlir::MemRefType outputTy =
        mlir::dyn_cast<mlir::MemRefType>(outputTensorOrMemrefTy);
    if (!outputTy) {
      return op->emitOpError("Input and output types must be the same");
    }

    const bool isFormatChange =
        inputTy.getElementType() != outputTy.getElementType();
    if (!allowFormatChange && isFormatChange) {
      return op->emitOpError(
          "Input and output layout element types must be the same");
    }

    const bool isMemorySpaceChange =
        inputTy.getMemorySpace() != outputTy.getMemorySpace();
    if (!allowMemorySpaceChange && isMemorySpaceChange) {
      return op->emitOpError(
          "Input and output memref memory spaces must be the same");
    }

    const bool sameRank = inputTy.getRank() == outputTy.getRank();
    if (checkMemrefRank && !sameRank) {
      return op->emitOpError("Input and output memref ranks must be the same");
    }

    auto inputDeviceLayout =
        mlir::dyn_cast<mlir::tt::ttcore::DeviceLayoutInterface>(
            inputTy.getLayout());
    if (checkMemrefGridShardForm && !inputDeviceLayout) {
      return op->emitOpError(
          "input memref must have device layout, i.e. have even rank, grid "
          "shape followed by shard shape of equal rank, e.g. GxGxSxS");
    }

    auto outputDeviceLayout =
        mlir::dyn_cast<mlir::tt::ttcore::DeviceLayoutInterface>(
            outputTy.getLayout());
    if (checkMemrefGridShardForm && !outputDeviceLayout) {
      return op->emitOpError(
          "output memref must have device layout, i.e. have even rank, grid "
          "shape followed by shard shape of equal rank, e.g. GxGxSxS");
    }

    return mlir::success();
  }

  return op->emitOpError("Unsupported input type for view");
}

// ToLayoutOp verification
::mlir::LogicalResult mlir::tt::ttir::ToLayoutOp::verify() {
  return verifyLayoutOp(*this, getInput().getType(), getOutput().getType(),
                        /*allowFormatChange*/ true,
                        /*allowMemorySpaceChange*/ true);
}

// ToLayoutOp utility methods
mlir::LogicalResult mlir::tt::ttir::ToLayoutOp::fold(
    FoldAdaptor, llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  mlir::RankedTensorType inputType =
      dyn_cast<mlir::RankedTensorType>(getInput().getType());
  mlir::RankedTensorType outputType =
      dyn_cast<mlir::RankedTensorType>(getOutput().getType());
  if (inputType && outputType && inputType == outputType) {
    results.push_back(getInput());
    return mlir::success();
  }
  return mlir::failure();
}

bool mlir::tt::ttir::ToLayoutOp::isHostToDevice() {
  const bool hostInput =
      mlir::cast<mlir::RankedTensorType>(getInput().getType()).getEncoding() ==
          nullptr ||
      mlir::isa<mlir::tt::ttcore::TensorMeshAttr>(
          mlir::cast<mlir::RankedTensorType>(getInput().getType())
              .getEncoding());
  const bool hostOutput =
      mlir::cast<mlir::RankedTensorType>(getOutput().getType()).getEncoding() ==
          nullptr ||
      mlir::isa<mlir::tt::ttcore::TensorMeshAttr>(
          mlir::cast<mlir::RankedTensorType>(getOutput().getType())
              .getEncoding());
  return hostInput && !hostOutput;
}

bool mlir::tt::ttir::ToLayoutOp::isDeviceToHost() {
  const bool hostInput =
      mlir::cast<mlir::RankedTensorType>(getInput().getType()).getEncoding() ==
          nullptr ||
      mlir::isa<mlir::tt::ttcore::TensorMeshAttr>(
          mlir::cast<mlir::RankedTensorType>(getInput().getType())
              .getEncoding());
  const bool hostOutput =
      mlir::cast<mlir::RankedTensorType>(getOutput().getType()).getEncoding() ==
          nullptr ||
      mlir::isa<mlir::tt::ttcore::TensorMeshAttr>(
          mlir::cast<mlir::RankedTensorType>(getOutput().getType())
              .getEncoding());
  return !hostInput && hostOutput;
}

void mlir::tt::ttir::ToLayoutOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // Fold into ttir.empty w/ desired layout
  patterns.add(+[](ToLayoutOp op, mlir::PatternRewriter &rewriter) {
    EmptyOp emptyOp = op.getInput().getDefiningOp<EmptyOp>();
    if (!emptyOp) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<EmptyOp>(op, op.getOutput().getType());
    return success();
  });

  patterns.add(std::make_unique<ToLayoutFoldRedundantPattern>(context));
}

//===----------------------------------------------------------------------===//
// TTNNMetalLayoutCastOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::tt::ttir::TTNNMetalLayoutCastOp::verify() {
  auto inputType = mlir::dyn_cast<mlir::ShapedType>(getInput().getType());
  auto outputType = mlir::dyn_cast<mlir::ShapedType>(getResult().getType());

  const bool inputIsMemref = mlir::isa<mlir::MemRefType>(inputType);
  const bool outputIsMemref = mlir::isa<mlir::MemRefType>(outputType);

  auto maybeInputTensor = mlir::dyn_cast<mlir::RankedTensorType>(inputType);
  auto maybeOutputTensor = mlir::dyn_cast<mlir::RankedTensorType>(outputType);

  auto maybeInputAttr =
      maybeInputTensor ? maybeInputTensor.getEncoding() : nullptr;
  auto maybeOutputAttr =
      maybeOutputTensor ? maybeOutputTensor.getEncoding() : nullptr;

  const bool inputIsTTNNTensor =
      maybeInputTensor &&
      (mlir::isa<mlir::tt::ttnn::TTNNLayoutAttr>(maybeInputAttr) ||
       mlir::isa<mlir::tt::ttnn::TTNNNDLayoutAttr>(maybeInputAttr));
  const bool outputIsTTNNTensor =
      maybeOutputTensor &&
      (mlir::isa<mlir::tt::ttnn::TTNNLayoutAttr>(maybeOutputAttr) ||
       mlir::isa<mlir::tt::ttnn::TTNNNDLayoutAttr>(maybeOutputAttr));

  const bool inputIsMetalTensor =
      maybeInputTensor &&
      mlir::isa<mlir::tt::ttcore::MetalLayoutAttr>(maybeInputAttr);
  const bool outputIsMetalTensor =
      maybeOutputTensor &&
      mlir::isa<mlir::tt::ttcore::MetalLayoutAttr>(maybeOutputAttr);

  if (inputIsTTNNTensor) {
    if (!outputIsMetalTensor && !outputIsMemref) {
      return emitOpError(
          "Input is ttnn tensor, output must be metal tensor or memref");
    }
  }

  if (inputIsMetalTensor || inputIsMemref) {
    if (!outputIsTTNNTensor) {
      return emitOpError(
          "Input is metal tensor or memref, output must be ttnn tensor");
    }
  }

  return success();
}

void mlir::tt::ttir::TTNNMetalLayoutCastOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "cast");
}

void mlir::tt::ttir::TTNNMetalLayoutCastOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // Eliminate back-to-back casts that form a no-op round-trip.
  //
  // Casts are always ttnn->metal or metal->ttnn (enforced by the verifier).
  // If the producer of the input is also a cast, we have one of:
  //   1. metal_0 -> ttnn -> metal_1
  //   2. ttnn_0  -> metal -> ttnn_1
  //
  // We can only fold when the outer types match (metal_0 == metal_1 or
  // ttnn_0 == ttnn_1).  Different TTNN shard strategies (e.g. height_sharded
  // vs block_sharded) can map to the same MetalLayoutAttr, so an intervening
  // d2m.to_layout that was folded away may leave back-to-back casts whose
  // outer types differ.

  patterns.add(+[](TTNNMetalLayoutCastOp op, mlir::PatternRewriter &rewriter) {
    TTNNMetalLayoutCastOp producerOp =
        op.getInput().getDefiningOp<TTNNMetalLayoutCastOp>();

    if (!producerOp) {
      return failure();
    }

    // Bail if we are post bufferization.
    auto producerInputTensor =
        mlir::dyn_cast<mlir::RankedTensorType>(producerOp.getInput().getType());
    auto producerOutputTensor = mlir::dyn_cast<mlir::RankedTensorType>(
        producerOp.getResult().getType());
    auto consumerInputTensor =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
    auto consumerOutputTensor =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());

    if (!producerInputTensor || !producerOutputTensor || !consumerInputTensor ||
        !consumerOutputTensor) {
      return failure();
    }

    // Don't fold when either cast carries a virtualGridInverseMapping.
    // Different TTNN shard strategies (e.g. height_sharded vs block_sharded)
    // can map to the same MetalLayoutAttr, so a cast with a VGM represents
    // a meaningful shard strategy that must be preserved.
    if (producerOp.getVirtualGridInverseMapping() ||
        op.getVirtualGridInverseMapping()) {
      return failure();
    }

    rewriter.replaceOp(op, producerOp.getInput());
    return success();
  });
}

mlir::LogicalResult mlir::tt::ttir::TTNNMetalLayoutCastOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {

  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (mlir::isa<mlir::MemRefType>(resultType) ||
      mlir::isa<mlir::MemRefType>(inputType)) {
    return success();
  }

  auto inputTensor = mlir::cast<mlir::RankedTensorType>(inputType);
  auto outputTensor = mlir::cast<mlir::RankedTensorType>(resultType);

  auto inputEncoding = inputTensor.getEncoding();
  auto outputEncoding = outputTensor.getEncoding();

  if (mlir::isa<mlir::tt::ttcore::MetalLayoutAttr>(inputEncoding)) {
    // metal_layout -> ttnn_layout becomes memref -> ttnn_layout
    bool isTTNNLayout =
        mlir::isa<mlir::tt::ttnn::TTNNLayoutAttr>(outputEncoding) ||
        mlir::isa<mlir::tt::ttnn::TTNNNDLayoutAttr>(outputEncoding);
    TT_assertv(isTTNNLayout,
               "Output tensor must have ttnn_layout or ttnn_nd_layout");
    auto maybeInputBuf =
        mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
    if (failed(maybeInputBuf)) {
      return maybeInputBuf;
    }
    rewriter.replaceOpWithNewOp<TTNNMetalLayoutCastOp>(
        *this, outputTensor, *maybeInputBuf, getVirtualGridInverseMappingAttr(),
        getVirtualGridForwardMappingAttr());
  } else if (mlir::isa<mlir::tt::ttcore::MetalLayoutAttr>(outputEncoding)) {
    // ttnn_layout -> metal_layout becomes ttnn_layout -> memref
    bool isTTNNLayout =
        mlir::isa<mlir::tt::ttnn::TTNNLayoutAttr>(inputEncoding) ||
        mlir::isa<mlir::tt::ttnn::TTNNNDLayoutAttr>(inputEncoding);
    TT_assertv(isTTNNLayout,
               "Input tensor must have ttnn_layout or ttnn_nd_layout");
    ::llvm::SmallVector<mlir::Value> dummy;
    auto bufferType = getBufferType(getResult(), options, state, dummy);
    if (failed(bufferType)) {
      return bufferType;
    }
    MemRefType outputMemrefType = mlir::cast<mlir::MemRefType>(*bufferType);
    mlir::bufferization::replaceOpWithNewBufferizedOp<TTNNMetalLayoutCastOp>(
        rewriter, *this, outputMemrefType, getInput(),
        getVirtualGridInverseMappingAttr(), getVirtualGridForwardMappingAttr());

  } else {
    return emitOpError("Neither input or output uses metal_layout");
  }
  return success();
}

mlir::bufferization::AliasingValueList
mlir::tt::ttir::TTNNMetalLayoutCastOp::getAliasingValues(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
mlir::tt::ttir::TTNNMetalLayoutCastOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  return mlir::tt::ttcore::getBufferType(value.getType(), /*isView=*/false);
}

bool mlir::tt::ttir::TTNNMetalLayoutCastOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // no-op
  return false;
}

bool mlir::tt::ttir::TTNNMetalLayoutCastOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // no-op
  return false;
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

// LinearOp verification
::mlir::LogicalResult mlir::tt::ttir::LinearOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  std::optional<::mlir::RankedTensorType> biasType =
      getBias() ? std::make_optional(getBias().getType()) : std::nullopt;
  ::mlir::RankedTensorType outputType = getType();

  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  llvm::SmallVector<int64_t> inputAShape(inputAType.getShape());
  llvm::SmallVector<int64_t> inputBShape(inputBType.getShape());

  // Verify that the input A is at least 1D tensor.
  if (inputAType.getRank() < 1) {
    return emitOpError("Input A must be at least a 1D tensor");
  }

  // Verify that the input B is at least 1D tensor.
  if (inputBType.getRank() < 1) {
    return emitOpError("Input B must be at least a 1D tensor");
  }

  // If input A is a vector (1D tensor), 1 is prepended to its dimensions for
  // the purpose of the matrix multiplication. After the matrix
  // multiplication, the prepended dimension is removed. Otherwise, check if
  // the LHS needs to be transposed.
  if (inputAType.getRank() == 1) {
    inputAShape.insert(inputAShape.begin(), 1);
  } else if (getTransposeA()) {
    std::swap(inputAShape[inputAShape.size() - 1],
              inputAShape[inputAShape.size() - 2]);
  }

  // If input B is a vector (1D tensor), a 1 is appended to its dimensions for
  // the purpose of the matrix-vector product and removed afterwards.
  // Otherwise, check if the RHS needs to be transposed.
  if (inputBType.getRank() == 1) {
    inputBShape.push_back(1);
  } else if (getTransposeB()) {
    std::swap(inputBShape[inputBShape.size() - 1],
              inputBShape[inputBShape.size() - 2]);
  }

  // Verify that the input A and input B has matching inner dimensions.
  if (inputAShape[inputAShape.size() - 1] !=
      inputBShape[inputBShape.size() - 2]) {
    return emitOpError("Input A[-1](")
           << inputAShape[inputAShape.size() - 1] << ") and B[-2]("
           << inputBShape[inputBShape.size() - 2]
           << ") must have matching inner dimensions";
  }

  llvm::SmallVector<int64_t> expectedOutputShape;
  // Verify that the batch dimensions are broadcast compatible and construct
  // the expected output shape. If either of input A or input B is at most 2D
  // tensors, the batch dimensions are trivially broadcast compatible.
  if (inputAShape.size() > 2 || inputBShape.size() > 2) {
    llvm::SmallVector<int64_t> inputABatchDims(inputAShape.begin(),
                                               inputAShape.end() - 2);
    llvm::SmallVector<int64_t> inputBBatchDims(inputBShape.begin(),
                                               inputBShape.end() - 2);

    // Verify that the batch dimensions of input A and B are broadcast
    // compatible.
    llvm::SmallVector<int64_t, 4> broadcastedShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(
            inputABatchDims, inputBBatchDims, broadcastedShape)) {

      return emitOpError("Batch dimensions of input A(" +
                         ttmlir::utils::join(inputABatchDims, ",") +
                         ") and B(" +
                         ttmlir::utils::join(inputBBatchDims, ",") +
                         ") are not broadcast compatible");
    }

    // Insert the broadcasted batch dimensions in the expected output shape.
    expectedOutputShape = std::move(broadcastedShape);
  }

  // Insert the input A and B inner dimensions in expected output shape
  // Consider the case where input A and B are vectors. In that case,
  // the dimension 1 is omitted from the output shape.
  if (inputAType.getRank() > 1) {
    expectedOutputShape.push_back(inputAShape[inputAShape.size() - 2]);
  }

  if (inputBType.getRank() > 1) {
    expectedOutputShape.push_back(inputBShape[inputBShape.size() - 1]);
  }

  if (biasType) {
    // Verify that the input bias is at least 1D tensor.
    if (biasType->getRank() < 1) {
      return emitOpError("Bias must be at least a 1D tensor");
    }

    llvm::SmallVector<int64_t> biasShape(biasType->getShape());

    // Verify that the dimensions of the matmul of A and B are broadcast
    // compatible with input bias.
    llvm::SmallVector<int64_t> broadcastShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(expectedOutputShape,
                                                  biasShape, broadcastShape)) {
      return emitOpError("Bias shape(")
             << ttmlir::utils::join(biasShape, ",")
             << ") is not broadcast compatible with the matmul output shape("
             << ttmlir::utils::join(expectedOutputShape, ",") << ")";
    }

    expectedOutputShape = broadcastShape;
  }

  // Check the case of a vector-vector product. At this moment we don't
  // support scalars in IR, hence check that the output is at least 1D tensor
  // of size 1.
  if (expectedOutputShape.size() == 0) {
    if (outputType.getRank() < 1) {
      return emitOpError("Scalar output is not supported, output must be at "
                         "least a 1D tensor");
    }

    if (outputType.getRank() > 1 || outputType.getShape()[0] != 1) {
      return emitOpError("Scalar output must be a 1D tensor of size 1");
    }

    return success();
  }

  // Verify that the output shape is correct.
  if (outputShape.size() != expectedOutputShape.size()) {
    return emitOpError("Output shape rank(")
           << outputShape.size()
           << ") must match the expected output shape rank("
           << expectedOutputShape.size() << ")";
  }

  // Verify each dim of the output shape.
  for (auto [index, outputDim, expectedDim] : llvm::zip(
           llvm::seq(outputShape.size()), outputShape, expectedOutputShape)) {
    if (outputDim != expectedDim) {
      return emitOpError("Output shape dimension[")
             << index << "](" << outputDim
             << ") doesn't match the expected output shape dimension[" << index
             << "](" << expectedDim << ")";
    }
  }

  return success();
}

// LinearOp canonicalization
void mlir::tt::ttir::LinearOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  // If bias is not provided, linear operation is equivalent to matmul.
  patterns.add(+[](ttir::LinearOp op, mlir::PatternRewriter &rewriter) {
    if (!op.getBias()) {
      rewriter.replaceOpWithNewOp<ttir::MatmulOp>(op, op.getType(), op.getA(),
                                                  op.getB(), op.getTransposeA(),
                                                  op.getTransposeB());
      return mlir::success();
    }
    return mlir::failure();
  });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

// ANCHOR: adding_an_op_matmul_ttir_verify
// MatmulOp verification
::mlir::LogicalResult mlir::tt::ttir::MatmulOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  ::mlir::RankedTensorType outputType = getType();

  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  llvm::SmallVector<int64_t> inputAShape(inputAType.getShape());
  llvm::SmallVector<int64_t> inputBShape(inputBType.getShape());

  // Verify that the input A is at least 1D tensor.
  if (inputAType.getRank() < 1) {
    return emitOpError("Input A must be at least a 1D tensor");
  }

  // Verify that the input B is at least 1D tensor.
  if (inputBType.getRank() < 1) {
    return emitOpError("Input B must be at least a 1D tensor");
  }

  // If input A is a vector (1D tensor), 1 is prepended to its dimensions for
  // the purpose of the matrix multiplication. After the matrix
  // multiplication, the prepended dimension is removed. Otherwise, check if
  // the LHS needs to be transposed.
  if (inputAType.getRank() == 1) {
    inputAShape.insert(inputAShape.begin(), 1);
  } else if (getTransposeA()) {
    std::swap(inputAShape[inputAShape.size() - 1],
              inputAShape[inputAShape.size() - 2]);
  }

  // If input B is a vector (1D tensor), a 1 is appended to its dimensions for
  // the purpose of the matrix-vector product and removed afterwards.
  // Otherwise, check if the RHS needs to be transposed.
  if (inputBType.getRank() == 1) {
    inputBShape.push_back(1);
  } else if (getTransposeB()) {
    std::swap(inputBShape[inputBShape.size() - 1],
              inputBShape[inputBShape.size() - 2]);
  }

  // Verify that the input A and input B has matching inner dimensions.
  if (inputAShape[inputAShape.size() - 1] !=
      inputBShape[inputBShape.size() - 2]) {
    return emitOpError("Input A[-1](")
           << inputAShape[inputAShape.size() - 1] << ") and B[-2]("
           << inputBShape[inputBShape.size() - 2]
           << ") must have matching inner dimensions";
  }

  llvm::SmallVector<int64_t> expectedOutputShape;
  // Verify that the batch dimensions are broadcast compatible and construct
  // the expected output shape. If either of input A or input B is at most 2D
  // tensors, the batch dimensions are trivially broadcast compatible.
  if (inputAShape.size() > 2 || inputBShape.size() > 2) {
    llvm::SmallVector<int64_t> inputABatchDims(inputAShape.begin(),
                                               inputAShape.end() - 2);
    llvm::SmallVector<int64_t> inputBBatchDims(inputBShape.begin(),
                                               inputBShape.end() - 2);

    // Verify that the batch dimensions of input A and B are broadcast
    // compatible.
    llvm::SmallVector<int64_t, 4> broadcastedShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(
            inputABatchDims, inputBBatchDims, broadcastedShape)) {

      return emitOpError("Batch dimensions of input A(" +
                         ttmlir::utils::join(inputABatchDims, ",") +
                         ") and B(" +
                         ttmlir::utils::join(inputBBatchDims, ",") +
                         ") are not broadcast compatible");
    }

    // Insert the broadcasted batch dimensions in the expected output shape.
    expectedOutputShape = std::move(broadcastedShape);
  }

  // Insert the input A and B inner dimensions in expected output shape
  // Consider the case where input A and B are vectors. In that case,
  // the dimension 1 is omitted from the output shape.
  if (inputAType.getRank() > 1) {
    expectedOutputShape.push_back(inputAShape[inputAShape.size() - 2]);
  }

  if (inputBType.getRank() > 1) {
    expectedOutputShape.push_back(inputBShape[inputBShape.size() - 1]);
  }

  // Check the case of a vector-vector product. At this moment we don't
  // support scalars in IR, hence check that the output is at least 1D tensor
  // of size 1.
  if (expectedOutputShape.size() == 0) {
    if (outputType.getRank() < 1) {
      return emitOpError("Scalar output is not supported, output must be at "
                         "least a 1D tensor");
    }

    if (outputType.getRank() > 1 || outputType.getShape()[0] != 1) {
      return emitOpError("Scalar output must be a 1D tensor of size 1");
    }

    return success();
  }

  // Verify that the output shape is correct.
  if (outputShape.size() != expectedOutputShape.size()) {
    return emitOpError("Output shape rank(")
           << outputShape.size()
           << ") must match the expected output shape rank("
           << expectedOutputShape.size() << ")";
  }

  // Verify each dim of the output shape.
  for (auto [index, outputDim, expectedDim] : llvm::zip(
           llvm::seq(outputShape.size()), outputShape, expectedOutputShape)) {
    if (outputDim != expectedDim) {
      return emitOpError("Output shape dimension[")
             << index << "](" << outputDim
             << ") doesn't match the expected output shape dimension[" << index
             << "](" << expectedDim << ")";
    }
  }

  return success();
}
// ANCHOR_END: adding_an_op_matmul_ttir_verify

// Returns the number of leading input dimensions that are merged into the
// first output dimension. Returns 0 if the reshape is not a leading dimension
// merge.
//
// A leading merge reshapes [d0, d1, ..., dk, t0, t1, ...] ->
//                          [d0*d1*...*dk, t0, t1, ...]
// where the trailing dimensions are preserved exactly.
static size_t getLeadingMergeCount(mlir::tt::ttir::ReshapeOp reshapeOp) {
  auto inShape = reshapeOp.getInput().getType().getShape();
  auto outShape = reshapeOp.getType().getShape();

  // Must reduce rank by at least 1.
  if (outShape.size() >= inShape.size()) {
    return 0;
  }

  size_t rankDiff = inShape.size() - outShape.size();
  size_t numMerged = rankDiff + 1;

  // Trailing dims must match exactly.
  if (inShape.drop_front(numMerged) != outShape.drop_front(1)) {
    return 0;
  }

  // Product of merged leading dims must equal the output's first dim.
  int64_t product = 1;
  for (size_t i = 0; i < numMerged; ++i) {
    if (inShape[i] <= 0) {
      return 0;
    }
    product *= inShape[i];
  }

  if (product != outShape[0]) {
    return 0;
  }

  return numMerged;
}

// Returns the leading dimensions that a single input dimension is split into,
// or an empty vector if the reshape is not a leading dimension split.
//
// A leading split reshapes [P, t0, t1, ...] -> [d0, d1, ..., dk, t0, t1, ...]
// where P == d0*d1*...*dk and the trailing dimensions are preserved.
static llvm::SmallVector<int64_t>
getLeadingSplitDims(mlir::tt::ttir::ReshapeOp reshapeOp) {
  auto inShape = reshapeOp.getInput().getType().getShape();
  auto outShape = reshapeOp.getType().getShape();

  // Must increase rank by at least 1.
  if (inShape.size() >= outShape.size()) {
    return {};
  }

  size_t rankDiff = outShape.size() - inShape.size();
  size_t numSplit = rankDiff + 1;

  // Trailing dims must match exactly.
  if (inShape.drop_front(1) != outShape.drop_front(numSplit)) {
    return {};
  }

  // Product of split dims must equal the input's first dim.
  int64_t product = 1;
  llvm::SmallVector<int64_t> splitDims;
  for (size_t i = 0; i < numSplit; ++i) {
    if (outShape[i] <= 0) {
      return {};
    }
    product *= outShape[i];
    splitDims.push_back(outShape[i]);
  }

  if (product != inShape[0]) {
    return {};
  }

  return splitDims;
}

// MatmulOp canonicalization: absorb leading dimension merge/split reshapes.
//
// Matches patterns where both matmul inputs have leading dimensions merged
// into a single batch dimension, and the output splits that dimension back.
// This covers both the leading-1 squeeze/unsqueeze case (e.g. [1,B,M,K] ->
// [B,M,K]) and the general batch-merge case (e.g. [B,H,M,K] -> [B*H,M,K]).
//
// Example:
//   %a = ttir.reshape [D0,D1,M,K] -> [D0*D1,M,K]
//   %b = ttir.reshape [D0,D1,K,N] -> [D0*D1,K,N]
//   %r = ttir.matmul (%a, %b) -> [D0*D1,M,N]
//   %o = ttir.reshape %r -> [D0,D1,M,N]
//
// Becomes:
//   %o = matmul %a_orig, %b_orig -> [D0,D1,M,N]
//
void mlir::tt::ttir::MatmulOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add(+[](ttir::MatmulOp op, mlir::PatternRewriter &rewriter) {
    if (!op.getResult().hasOneUse()) {
      return mlir::failure();
    }

    auto splitOp =
        mlir::dyn_cast<ttir::ReshapeOp>(*op.getResult().getUsers().begin());
    if (!splitOp) {
      return mlir::failure();
    }

    auto splitDims = getLeadingSplitDims(splitOp);
    if (splitDims.empty()) {
      return mlir::failure();
    }

    // Both inputs must be leading dimension merges.
    auto mergeA = op.getA().getDefiningOp<ttir::ReshapeOp>();
    auto mergeB = op.getB().getDefiningOp<ttir::ReshapeOp>();
    if (!mergeA || !mergeB) {
      return mlir::failure();
    }

    size_t mergeCountA = getLeadingMergeCount(mergeA);
    size_t mergeCountB = getLeadingMergeCount(mergeB);
    if (mergeCountA == 0 || mergeCountB == 0) {
      return mlir::failure();
    }

    // The merged leading dims must be identical in both inputs.
    auto aInShape = mergeA.getInput().getType().getShape();
    auto bInShape = mergeB.getInput().getType().getShape();
    auto aLeading = aInShape.take_front(mergeCountA);
    auto bLeading = bInShape.take_front(mergeCountB);
    if (aLeading != bLeading) {
      return mlir::failure();
    }

    // The split dims in the output must match the merged leading dims.
    if (llvm::SmallVector<int64_t>(aLeading.begin(), aLeading.end()) !=
        splitDims) {
      return mlir::failure();
    }

    mlir::Value newA = mergeA.getInput();
    mlir::Value newB = mergeB.getInput();

    auto newAType = mlir::cast<mlir::RankedTensorType>(newA.getType());
    auto newBType = mlir::cast<mlir::RankedTensorType>(newB.getType());

    // Bail out if the restored inputs would have different ranks.
    if (newAType.getRank() != newBType.getRank()) {
      return mlir::failure();
    }

    // Verify the new input rank matches the split output rank.
    if (newAType.getRank() != splitOp.getType().getRank()) {
      return mlir::failure();
    }

    auto newResultType = mlir::RankedTensorType::get(
        splitOp.getType().getShape(), splitOp.getType().getElementType(),
        splitOp.getType().getEncoding());

    rewriter.replaceOpWithNewOp<ttir::MatmulOp>(splitOp, newResultType, newA,
                                                newB, op.getTransposeA(),
                                                op.getTransposeB());

    return mlir::success();
  });
}

//===----------------------------------------------------------------------===//
// UpsampleOp
//===----------------------------------------------------------------------===//

// UpsampleOp verification
::mlir::LogicalResult mlir::tt::ttir::Upsample2dOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();

  // Input tensor is assumed to be 4D tensor.
  if (inputType.getRank() != 4) {
    return emitOpError("Expected rank of input tensor is 4, got rank " +
                       std::to_string(inputType.getRank()));
  }
  if (outputType.getRank() != 4) {
    return emitOpError("Expected rank of output tensor is 4, got rank " +
                       std::to_string(outputType.getRank()));
  }

  auto scaleFactor = ttmlir::utils::getPairOfInteger<int32_t>(getScaleFactor());
  if (auto error = scaleFactor.takeError()) {
    return emitOpError() << llvm::toString(std::move(error));
  }
  int32_t scaleH = scaleFactor->first;
  int32_t scaleW = scaleFactor->second;

  if (scaleH <= 0 || scaleW <= 0) {
    return emitOpError("Scale factors H = ")
           << scaleH << " and W = " << scaleW << " must be positive integers";
  }

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  // Input tensor is assumed to be in NHWC format.
  enum Dimensions { DIM_N = 0, DIM_H = 1, DIM_W = 2, DIM_C = 3 };
  if (inputShape[DIM_H] * scaleH != outputShape[DIM_H]) {
    return emitOpError("Expected output H dimension to be input H dimension * "
                       "scaleH = ")
           << (inputShape[DIM_H] * scaleH) << ", got " << outputShape[DIM_H];
  }
  if (inputShape[DIM_W] * scaleW != outputShape[DIM_W]) {
    return emitOpError("Expected output W dimension to be input W dimension * "
                       "scaleW = ")
           << (inputShape[DIM_W] * scaleW) << ", got " << outputShape[DIM_W];
  }
  if (inputShape[DIM_N] != outputShape[DIM_N]) {
    return emitOpError("Expected output N dimension to be ")
           << inputShape[DIM_N] << ", got " << outputShape[DIM_N];
  }
  if (inputShape[DIM_C] != outputShape[DIM_C]) {
    return emitOpError("Expected output C dimension to be ")
           << inputShape[DIM_C] << ", got " << outputShape[DIM_C];
  }

  // Verify that the mode attribute is one of the legal modes. These two modes
  // are currently only supported modes in TTNN.
  llvm::SmallVector<llvm::StringRef> legalModes = {"nearest", "bilinear"};
  if (std::find(legalModes.begin(), legalModes.end(), getMode()) ==
      legalModes.end()) {
    return emitOpError("Expected modes are (")
           << llvm::join(legalModes, ", ") << "), got \"" << getMode() << "\"";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

// AllocOp verification
::mlir::LogicalResult mlir::tt::ttir::AllocOp::verify() {
  auto layout = mlir::dyn_cast_if_present<mlir::tt::ttcore::MetalLayoutAttr>(
      getResult().getType().getEncoding());
  if (not layout) {
    return emitOpError("Result type missing layout attribute");
  }

  if (getSize() == 0) {
    return emitOpError("Alloc size must be non-zero");
  }

  auto memspace = layout.getMemorySpace();
  if (memspace != getMemorySpace()) {
    return emitOpError(
        "Input tensor layout memory space must match alloc memory space");
  }

  if (isSystemMemorySpace(getMemorySpace()) and getAddress() != 0) {
    return emitOpError("Allocating from system memory space must have address "
                       "set to 0, implicitly allocated by the runtime");
  }

  if (isDeviceMemorySpace(memspace) and getAddress() == 0) {
    return emitOpError(
        "Allocating from a device memory space must have address "
        "set to a non-zero value, device addresses are statically allocated");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//

// RepeatOp verification.
::mlir::LogicalResult mlir::tt::ttir::RepeatOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();
  llvm::ArrayRef<int64_t> repeatDimensions = getRepeatDimensions();

  // Input tensor and repeat dimension argument must have same rank.
  if (inputType.getRank() != static_cast<int64_t>(repeatDimensions.size())) {
    return emitOpError() << "Input tensor rank " << inputType.getRank()
                         << " doesn't match the number of repeat dimensions "
                         << repeatDimensions.size() << ".";
  }

  // Input and output tensors must have the same rank.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "Input tensor rank " << inputType.getRank()
                         << " doesn't match the output tensor rank "
                         << outputType.getRank() << ".";
  }

  // Verify output shape based on input shape and repeat dimension argument.
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  for (size_t i = 0; i < inputShape.size(); i++) {
    // Verify that the repeat dimension is greater than 0.
    if (repeatDimensions[i] <= 0) {
      return emitOpError() << "Repeat dimension at index " << i
                           << " must be greater than 0.";
    }

    int64_t expectedDimValue = inputShape[i] * repeatDimensions[i];
    if (expectedDimValue != outputShape[i]) {
      return emitOpError() << "Input tensor shape ("
                           << ttmlir::utils::join(inputShape, ",")
                           << ") at index " << i
                           << " does not repeat to output ("
                           << ttmlir::utils::join(outputShape, ",")
                           << ") using repeat value " << repeatDimensions[i]
                           << ".";
    }
  }

  return success();
}

// back to back producer-consumer RepeatOp can be folded
// into a single RepeatOp.
static mlir::OpFoldResult
foldConsecutiveRepeat(mlir::tt::ttir::RepeatOp consumerOp) {

  if (auto producerOp =
          consumerOp.getInput().getDefiningOp<mlir::tt::ttir::RepeatOp>()) {

    // If producerOp has multiple uses, do not fold
    if (!producerOp->hasOneUse()) {
      return nullptr;
    }

    mlir::RankedTensorType inputType = producerOp.getInput().getType();
    size_t inputRank = static_cast<size_t>(inputType.getRank());

    // Producer repeat dimensions
    llvm::ArrayRef<int64_t> producerRepeatDims =
        producerOp.getRepeatDimensions();

    // Consumer repeat dimensions
    llvm::ArrayRef<int64_t> consumerRepeatDims =
        consumerOp.getRepeatDimensions();

    llvm::SmallVector<int64_t> mergedRepeatDimensions(inputRank);
    for (size_t i = 0; i < inputRank; ++i) {
      mergedRepeatDimensions[i] = producerRepeatDims[i] * consumerRepeatDims[i];
    }
    llvm::ArrayRef<int64_t> mergedRepeatDimsRef(mergedRepeatDimensions);
    consumerOp.setRepeatDimensions(mergedRepeatDimsRef);
    consumerOp->setOperand(0, producerOp.getInput());
    return consumerOp.getResult();
  }

  return nullptr;
}

// Repeat op can be folded when repeat dimensions are all 1.
static mlir::OpFoldResult foldIdentityRepeat(mlir::tt::ttir::RepeatOp op) {
  if (llvm::all_of(op.getRepeatDimensions(),
                   [](int64_t dim) { return dim == 1; })) {
    return op.getInput();
  }
  return nullptr;
}

static mlir::OpFoldResult constantFoldRepeat(mlir::tt::ttir::RepeatOp op,
                                             mlir::Attribute input) {
  llvm::ArrayRef<int64_t> inputShape = op.getInput().getType().getShape();
  return constantFoldTM(
      op, input, [inputShape](const llvm::SmallVector<int64_t> &outputCoords) {
        llvm::SmallVector<int64_t> inputCoords(outputCoords.size());
        std::transform(outputCoords.begin(), outputCoords.end(),
                       inputShape.begin(), inputCoords.begin(),
                       std::modulus<int64_t>());
        return inputCoords;
      });
}

// RepeatOp Folder
mlir::OpFoldResult mlir::tt::ttir::RepeatOp::fold(FoldAdaptor fold) {

  if (auto foldResult = foldIdentityRepeat(*this)) {
    return foldResult;
  }
  if (auto foldResult = foldConsecutiveRepeat(*this)) {
    return foldResult;
  }
  if (auto foldResult = constantFoldRepeat(*this, fold.getInput())) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// RepeatInterleaveOp
//===----------------------------------------------------------------------===//

// RepeatInterleaveOp verification
::mlir::LogicalResult mlir::tt::ttir::RepeatInterleaveOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();
  uint32_t repeats = getRepeats();
  int32_t dim = getDim();

  // Verify that the input is at least a 1D tensor.
  if (inputType.getRank() < 1) {
    return emitOpError("Input must be at least a 1D tensor");
  }

  // Check that the repeats is not zero.
  if (repeats == 0) {
    return emitOpError("Repeats attribute must be non-zero");
  }

  // Check that the dim is within the bounds of the input tensor.
  if (dim >= inputType.getRank() || dim < -inputType.getRank()) {
    return emitOpError("Dimension attribute must be within the bounds")
           << "[" << -inputType.getRank() << ", " << inputType.getRank() << ")"
           << ", got " << inputType.getRank();
  }

  // Normalize dim to [0, n) range.
  if (dim < 0) {
    dim += inputType.getRank();
  }

  // Compute the expected output shape.
  llvm::SmallVector<int64_t> expectedOutputShape(inputType.getShape());
  expectedOutputShape[dim] *= repeats;

  // Verify that the output shape matches the expected shape.
  if (outputType.getShape() != ::llvm::ArrayRef(expectedOutputShape)) {
    return emitOpError("Output shape ")
           << "[" << ttmlir::utils::join(outputType.getShape(), ",") << "]"
           << " does not match the expected shape "
           << "[" << ttmlir::utils::join(expectedOutputShape, ",") << "]";
  }

  return success();
}

static mlir::OpFoldResult
foldIdentityRepeatInterleave(mlir::tt::ttir::RepeatInterleaveOp op) {
  if (op.getRepeats() == 1) {
    return op.getInput();
  }
  return nullptr;
}

static mlir::OpFoldResult
constantFoldRepeatInterleave(mlir::tt::ttir::RepeatInterleaveOp op,
                             mlir::Attribute input) {
  int32_t dim = op.getDim();
  uint32_t repeats = op.getRepeats();
  return constantFoldTM(
      op, input,
      [dim, repeats](const llvm::SmallVector<int64_t> &outputCoords) {
        llvm::SmallVector<int64_t> inputCoords(outputCoords.size());
        for (size_t i = 0; i != outputCoords.size(); ++i) {
          inputCoords[i] = outputCoords[i];
          if (i == static_cast<size_t>(dim)) {
            inputCoords[i] /= repeats;
          }
        }
        return inputCoords;
      });
}

// RepeatInterleaveOp Folder
mlir::OpFoldResult mlir::tt::ttir::RepeatInterleaveOp::fold(FoldAdaptor fold) {
  if (auto foldResult = foldIdentityRepeatInterleave(*this)) {
    return foldResult;
  }
  if (auto foldResult = constantFoldRepeatInterleave(*this, fold.getInput())) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

// SoftmaxOp verification
::mlir::LogicalResult mlir::tt::ttir::SoftmaxOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();

  // Shapes of input and output of a softmax operation must be the same
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("Input and output shapes must be the same");
  }

  int32_t dim = getDimension();

  // Check that the dim is within the bounds of the input tensor
  if (dim >= inputType.getRank() || dim < -inputType.getRank()) {
    return emitOpError(
        "Dimension attribute must be within the bounds of the input tensor");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

// SortOp verification
::mlir::LogicalResult mlir::tt::ttir::SortOp::verify() {
  auto dim = getDim();
  auto input = getInput();
  auto rank = input.getType().getRank();
  if (dim >= rank || dim < -rank) {
    return emitOpError("Dimension out of range (expected to be in range of [")
           << -rank << ", " << (rank - 1) << "], but got " << dim << ")";
  }

  auto indicesType =
      mlir::cast<RankedTensorType>(getResults().back().getType());
  auto values = getResults().front();
  if (input.getType() != values.getType()) {
    return emitOpError("Sorted tensor type does not match with input tensor.");
  }

  if (input.getType().getShape() != indicesType.getShape()) {
    return emitOpError("Indices shape does not match with input tensor shape.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//

// AllGatherOp verification
::mlir::LogicalResult mlir::tt::ttir::AllGatherOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  int32_t gatherDim = getAllGatherDim();

  if (gatherDim >= inputType.getRank() || gatherDim < -inputType.getRank()) {
    return emitOpError(
               "Invalid dimension for all gather op. Gather dimension must "
               "be "
               ">= to "
               "input tensor rank or < -input tensor rank, got gather_dim = ")
           << gatherDim;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllReduceOp
//===----------------------------------------------------------------------===//

// AllReduceOp verification
::mlir::LogicalResult mlir::tt::ttir::AllReduceOp::verify() {
  ::mlir::tt::ttcore::ReduceType reduceType = getReduceType();

  // Currently TTIR only supports the sum reduce types.
  if (reduceType != ::mlir::tt::ttcore::ReduceType::Sum) {
    return emitOpError("Invalid reduction op for all reduce op.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllReduceAsyncOp
//===----------------------------------------------------------------------===//

// AllReduceAsyncOp verification
::mlir::LogicalResult mlir::tt::ttir::AllReduceAsyncOp::verify() {
  ::mlir::tt::ttcore::ReduceType reduceType = getReduceType();

  // Currently TTIR only supports the sum reduce types.
  if (reduceType != ::mlir::tt::ttcore::ReduceType::Sum) {
    return emitOpError("Invalid reduction op for all reduce async op.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReduceScatterOp
//===----------------------------------------------------------------------===//

// ReduceScatterOp verification
::mlir::LogicalResult mlir::tt::ttir::ReduceScatterOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::tt::ttcore::ReduceType reduceType = getReduceType();
  int32_t scatterDim = getScatterDim();

  // Currently TTIR only supports the following reduce types.
  if (reduceType != ::mlir::tt::ttcore::ReduceType::Sum &&
      reduceType != ::mlir::tt::ttcore::ReduceType::Max &&
      reduceType != ::mlir::tt::ttcore::ReduceType::Min) {
    return emitOpError("Invalid reduction op for reduce scatter op.");
  }

  if (scatterDim >= inputType.getRank() || scatterDim < -inputType.getRank()) {
    return emitOpError(
               "Invalid dimension for reduce scatter op. Scatter dimension "
               "must be >= to input tensor rank or < -input tensor rank, got "
               "scatter_dim = ")
           << scatterDim;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

// CollectivePermuteOp verification
::mlir::LogicalResult mlir::tt::ttir::CollectivePermuteOp::verify() {
  auto sourceTargetPairs = getSourceTargetPairs().getValues<int64_t>();

  // Check that the rank of sourceTargetPairs is 2D.
  llvm::ArrayRef<int64_t> sourceTargetPairsShape =
      getSourceTargetPairs().getType().getShape();
  const size_t sourceTargetPairsRank = sourceTargetPairsShape.size();

  if (sourceTargetPairsRank != 2) {
    return emitOpError("The rank of source target pairs must be 2, got rank = ")
           << sourceTargetPairsRank;
  }

  /* Check that the 'src' values and 'dest' values in sourceTargetPairs is
  unique. Given a 2D rank tensor of source target pairs eg. [['src',
  'target'],
  ['src', 'target'] ...], we need to ensure that each 'src' is unique and each
  'target' is unique.
  */
  auto areElementsUnique = [](const auto &sourceTargetPairs) -> bool {
    for (size_t i = 0; i < sourceTargetPairs.size(); i++) {
      int target = sourceTargetPairs[i];
      for (size_t j = i + 2; j < sourceTargetPairs.size(); j += 2) {
        if (sourceTargetPairs[j] == target) {
          return false;
        }
      }
    }

    return true;
  };

  if (!areElementsUnique(sourceTargetPairs)) {
    return emitOpError(
        "There are duplicate 'src' or 'dest' devices in source target pairs");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MeshPartitionOp
//===----------------------------------------------------------------------===//

// MeshPartitionOp verification
::mlir::LogicalResult mlir::tt::ttir::MeshPartitionOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  int32_t dim = getDim();

  if (dim >= inputType.getRank() || dim < -inputType.getRank()) {
    return emitOpError(
               "Invalid tensor dimension for mesh partition op. Dimension "
               "must be >= to input tensor rank or < -input tensor rank, got "
               "dim = ")
           << dim;
  }

  std::optional<uint32_t> clusterAxis = getClusterAxis();
  if (clusterAxis.has_value() && clusterAxis.value() > 1) {
    return emitOpError("Cluster axis must be either None, 0 or 1, got " +
                       std::to_string(clusterAxis.value()));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MeshShardOp
//===----------------------------------------------------------------------===//

// MeshShardOp verification
mlir::LogicalResult mlir::tt::ttir::MeshShardOp::verify() {
  auto shardType = getShardType();

  // Currently, we are not supporting maximal from StableHLO.
  if (shardType == mlir::tt::ttcore::MeshShardType::Maximal) {
    return emitOpError("Invalid shard_type (maximal) for mesh_shard op.");
  }

  return success();
}

::mlir::OpFoldResult mlir::tt::ttir::MeshShardOp::fold(FoldAdaptor adaptor) {
  auto shardShapeArray = getShardShape();
  auto shardType = getShardType();
  if (shardType != mlir::tt::ttcore::MeshShardType::Replicate &&
      ttmlir::utils::volume(shardShapeArray) == 1) {
    return getInput();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

// ScatterOp folder: If the index or source tensor has a volume of 0, then the
// scatter operation is a no-op and can be folded to the input tensor.
::mlir::OpFoldResult mlir::tt::ttir::ScatterOp::fold(FoldAdaptor adaptor) {
  if (ttmlir::utils::volume(getIndex().getType().getShape()) == 0 ||
      ttmlir::utils::volume(getSource().getType().getShape()) == 0) {
    return getInput();
  }
  return nullptr;
}

::mlir::LogicalResult mlir::tt::ttir::ScatterOp::verify() {
  const ::mlir::RankedTensorType inputType = getInput().getType();
  const ::mlir::RankedTensorType indexType = getIndex().getType();
  const ::mlir::RankedTensorType sourceType = getSource().getType();

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> indexShape = indexType.getShape();
  llvm::ArrayRef<int64_t> sourceShape = sourceType.getShape();

  const size_t inputTypeRank = inputShape.size();
  const size_t indexTypeRank = indexShape.size();
  const size_t sourceTypeRank = sourceShape.size();

  if (inputTypeRank != indexTypeRank || inputTypeRank != sourceTypeRank ||
      indexTypeRank != sourceTypeRank) {
    return emitOpError() << "Input tensor, index tensor, and source tensor "
                            "must have the same rank. "
                         << "Got input rank = " << inputTypeRank
                         << ", index rank = " << indexTypeRank
                         << ", source rank = " << sourceTypeRank;
  }

  if (indexShape != sourceShape) {
    return emitOpError(
        "Index tensor must have the same shape as source tensor.");
  }

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::GatherOp::verify() {
  const ::mlir::RankedTensorType inputType = getInput().getType();
  const ::mlir::RankedTensorType indexType = getIndex().getType();
  const ::mlir::RankedTensorType resultType = getResult().getType();

  const int64_t inputRank = inputType.getRank();
  const int64_t indexRank = indexType.getRank();

  if (!indexType.getElementType().isInteger()) {
    return emitOpError() << "Index tensor must have an integer type, got "
                         << indexType.getElementType();
  }

  if (inputRank != indexRank) {
    return emitOpError()
           << "Input tensor and index tensor must have the same rank. "
           << "Got input rank = " << inputRank
           << ", index rank = " << indexRank;
  }

  int32_t dim = getDim();
  if (dim >= inputRank || dim < -inputRank) {
    return emitOpError() << "Dimension must be in the range [-" << inputRank
                         << ", " << inputRank << "), got dim = " << dim;
  }

  if (indexType.getShape() != resultType.getShape()) {
    return emitOpError(
        "Index tensor and result tensor must have the same shape.");
  }

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// UpdateCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::UpdateCacheOp::verify() {
  const ::mlir::RankedTensorType cacheType = getCache().getType();
  const ::mlir::RankedTensorType inputType = getInput().getType();

  const ttcore::DataType cacheDataType =
      ttcore::elementTypeToDataType(cacheType.getElementType());
  const ttcore::DataType inputDataType =
      ttcore::elementTypeToDataType(inputType.getElementType());

  if (cacheDataType != inputDataType) {
    return emitOpError(
        "Cache and input tensors must have the same dtype. "
        "Got cache dtype = " +
        DataTypeEnumToString(cacheDataType) +
        ", input dtype = " + DataTypeEnumToString(inputDataType));
  }

  if (cacheType.getRank() != 4) {
    return emitOpError("Cache tensor must be a 4D tensor");
  }

  if (inputType.getRank() != 4) {
    return emitOpError("Input tensor must be a 4D tensor");
  }

  if (inputType.getShape()[0] != 1) {
    return emitOpError("Input tensor requires that dim 0 have size 1, got "
                       "input dim 0 size = " +
                       std::to_string(inputType.getShape()[0]));
  }

  if (cacheType.getShape()[1] != inputType.getShape()[1] ||
      cacheType.getShape()[3] != inputType.getShape()[3]) {
    return emitOpError("Cache tensor shape must match input tensor shape on "
                       "dims 1 and 3. Got cache shape (" +
                       std::to_string(cacheType.getShape()[0]) + ", " +
                       std::to_string(cacheType.getShape()[1]) + ", " +
                       std::to_string(cacheType.getShape()[2]) + ", " +
                       std::to_string(cacheType.getShape()[3]) +
                       "), input shape ()" +
                       std::to_string(inputType.getShape()[0]) + "x" +
                       std::to_string(inputType.getShape()[1]) + "x" +
                       std::to_string(inputType.getShape()[2]) + "x" +
                       std::to_string(inputType.getShape()[3]) + ")");
  }

  return success();
}

void mlir::tt::ttir::UpdateCacheOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add(
      +[](mlir::tt::ttir::UpdateCacheOp op, mlir::PatternRewriter &rewriter) {
        auto cacheShape = op.getCache().getType().getShape();
        auto inputShape = op.getInput().getType().getShape();
        auto updateIndexShape = op.getUpdateIndex().getType().getShape();

        auto numUsers = cacheShape[0];
        auto numHeads = cacheShape[1];
        auto headDim = cacheShape[3];

        TypedValue<RankedTensorType> newInput = op.getInput();

        // Permute input if in the format [1, num_heads, num_users, head_dim]
        if (inputShape[2] == numUsers && inputShape[1] == numHeads) {
          llvm::SmallVector<int64_t> newInputShape = {1, numUsers, numHeads,
                                                      headDim};
          auto newInputType = RankedTensorType::get(
              newInputShape, newInput.getType().getElementType(),
              newInput.getType().getEncoding());
          newInput = rewriter.create<PermuteOp>(
              op.getLoc(), newInputType, newInput,
              rewriter.getDenseI64ArrayAttr({0, 2, 1, 3}));
        }

        // If the update index shape is [1] then repeat to num users
        TypedValue<RankedTensorType> newUpdateIndex = op.getUpdateIndex();
        if (updateIndexShape[0] == 1) {
          auto newUpdateIndexShape = {numUsers};
          auto newUpdateIndexType = RankedTensorType::get(
              newUpdateIndexShape, newUpdateIndex.getType().getElementType(),
              newUpdateIndex.getType().getEncoding());
          auto repeatDims = rewriter.getDenseI64ArrayAttr({numUsers});
          newUpdateIndex = rewriter.create<RepeatOp>(
              op.getLoc(), newUpdateIndexType, newUpdateIndex, repeatDims);
        }

        rewriter.replaceOpWithNewOp<ttir::PagedUpdateCacheOp>(
            op, op.getType(), op.getCache(), newInput, newUpdateIndex, false,
            nullptr);

        return mlir::success();
      });
}

//===----------------------------------------------------------------------===//
// PagedUpdateCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult PagedUpdateCacheOp::verify() {
  auto cacheType = getCache().getType();
  auto inputType = getInput().getType();
  auto updateIndexType = getUpdateIndex().getType();

  auto cacheShape = cacheType.getShape();
  auto inputShape = inputType.getShape();
  auto updateIndexShape = updateIndexType.getShape();

  bool usingStaticCache = getPageTable() == nullptr;

  if (cacheShape.size() != 4) {
    return emitOpError("Cache tensor must be a 4D tensor");
  }

  if (inputShape.size() != 4) {
    return emitOpError("Input tensor must be a 4D tensor");
  }

  if (updateIndexShape.size() != 1) {
    return emitOpError("Update index tensor must be a 1D tensor");
  }

  int64_t blockSize = cacheShape[2];
  int64_t headDim = cacheShape[3];
  int64_t numUsers = updateIndexShape[0];

  if (!usingStaticCache && blockSize % ttnn::TILE_HEIGHT != 0) {
    return emitOpError("Block size must be divisible by 32, got " +
                       std::to_string(blockSize));
  }

  if (inputShape[0] != 1) {
    return emitOpError("Input tensor must have dim 0 be equal to 1, got " +
                       std::to_string(inputShape[0]));
  }

  if (inputShape[1] != numUsers) {
    return emitOpError("Input tensor must have shape equal to the number of "
                       "users (determined by update index shape): " +
                       std::to_string(numUsers) + ", got " +
                       std::to_string(inputShape[1]));
  }

  if (inputShape[3] != headDim) {
    return emitOpError("Input tensor must have dim 3 be equal to the head "
                       "dimension (determined by cache shape): " +
                       std::to_string(headDim) + ", got " +
                       std::to_string(inputShape[3]));
  }

  if (!usingStaticCache) {
    auto pageTableType = getPageTable().getType();
    auto pageTableShape = pageTableType.getShape();
    if (pageTableShape.size() != 2) {
      return emitOpError("Page table tensor must be a 2D tensor");
    }

    if (pageTableShape[0] != numUsers) {
      return emitOpError(
          "Page table tensor must have dim 0 be equal to the "
          "number of users (determined by update index shape): " +
          std::to_string(numUsers) + ", got " +
          std::to_string(pageTableShape[0]));
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SamplingOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::SamplingOp::verify() {
  auto inputValuesType = getInputValues().getType();
  auto inputIndicesType = getInputIndices().getType();
  auto kType = getK().getType();
  auto pType = getP().getType();
  auto tempType = getTemp().getType();
  auto resultType = getResult().getType();

  if (inputValuesType.getRank() != 2) {
    return emitOpError("input_values must be 2D [batch, candidates]");
  }
  if (inputIndicesType.getRank() != 2) {
    return emitOpError("input_indices must be 2D [batch, candidates]");
  }
  if (inputValuesType.getShape() != inputIndicesType.getShape()) {
    return emitOpError(
        "input_values and input_indices must have the same shape");
  }

  int64_t batch = inputValuesType.getShape()[0];

  // k, p, temp must be 1D with the same batch dimension.
  for (auto [tensor, name] :
       llvm::zip(std::array<mlir::RankedTensorType, 3>{kType, pType, tempType},
                 std::array<llvm::StringRef, 3>{"k", "p", "temp"})) {
    if (tensor.getRank() != 1) {
      return emitOpError() << name << " must be 1D [batch]";
    }
    if (tensor.getShape()[0] != batch) {
      return emitOpError() << name << " batch dimension ("
                           << tensor.getShape()[0]
                           << ") must match input_values batch (" << batch
                           << ")";
    }
  }

  // Result must be 1D [batch].
  if (resultType.getRank() != 1 || resultType.getShape()[0] != batch) {
    return emitOpError("result must be 1D [batch]");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FillCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::FillCacheOp::verify() {
  const ::mlir::RankedTensorType cacheType = getCache().getType();
  const ::mlir::RankedTensorType inputType = getInput().getType();

  const ttcore::DataType cacheDataType =
      ttcore::elementTypeToDataType(cacheType.getElementType());
  const ttcore::DataType inputDataType =
      ttcore::elementTypeToDataType(inputType.getElementType());

  if (cacheDataType != inputDataType) {
    return emitOpError(
        "Cache and input tensors must have the same dtype. "
        "Got cache dtype = " +
        DataTypeEnumToString(cacheDataType) +
        ", input dtype = " + DataTypeEnumToString(inputDataType));
  }

  if (cacheType.getRank() != 4) {
    return emitOpError("Cache tensor must be a 4D tensor");
  }

  if (inputType.getRank() != 4) {
    return emitOpError("Input tensor must be a 4D tensor");
  }

  if (inputType.getShape()[2] > cacheType.getShape()[2]) {
    return emitOpError(
        "Input tensor requires that dim 2 have a size which is less than or "
        "equal to the size of dim 2 of the cache tensor. Got cache dim 2 "
        "size = " +
        std::to_string(cacheType.getShape()[2]) +
        ", input dim 2 size = " + std::to_string(inputType.getShape()[2]));
  }

  if (cacheType.getShape()[1] != inputType.getShape()[1] ||
      cacheType.getShape()[3] != inputType.getShape()[3]) {
    return emitOpError("Cache tensor shape must match input tensor shape on "
                       "dims 1 and 3. Got cache shape (" +
                       std::to_string(cacheType.getShape()[0]) + ", " +
                       std::to_string(cacheType.getShape()[1]) + ", " +
                       std::to_string(cacheType.getShape()[2]) + ", " +
                       std::to_string(cacheType.getShape()[3]) +
                       "), input shape (" +
                       std::to_string(inputType.getShape()[0]) + ", " +
                       std::to_string(inputType.getShape()[1]) + ", " +
                       std::to_string(inputType.getShape()[2]) + ", " +
                       std::to_string(inputType.getShape()[3]) + ")");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PagedFillCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::PagedFillCacheOp::verify() {
  auto cacheType = getCache().getType();
  auto inputType = getInput().getType();
  auto pageTableType = getPageTable().getType();

  auto cacheShape = cacheType.getShape();
  auto inputShape = inputType.getShape();
  auto pageTableShape = pageTableType.getShape();

  if (cacheShape.size() != 4) {
    return emitOpError("Cache tensor must be a 4D tensor");
  }

  if (inputShape.size() != 4) {
    return emitOpError("Input tensor must be a 4D tensor");
  }

  if (pageTableShape.size() != 2) {
    return emitOpError("Page table tensor must be a 2D tensor");
  }

  if (cacheType.getElementType() != inputType.getElementType()) {
    return emitOpError("Cache and input tensors must have the same dtype");
  }

  if (!cacheType.getElementType().isFloat()) {
    return emitOpError("Cache tensor must be a floating point type");
  }

  if (!inputType.getElementType().isFloat()) {
    return emitOpError("Input tensor must be a floating point type");
  }

  if (!pageTableType.getElementType().isInteger()) {
    return emitOpError("Page table tensor must be an integer type");
  }

  if (getBatchIdxTensor()) {
    auto batchIdxTensorType = getBatchIdxTensor().getType();
    if (batchIdxTensorType.getShape().size() != 1) {
      return emitOpError("Batch index tensor must be a 1D tensor");
    }
    if (batchIdxTensorType.getShape()[0] != 1) {
      return emitOpError(
          "Batch index tensor must have dim 0 be equal to 1, got " +
          std::to_string(batchIdxTensorType.getShape()[0]));
    }
    if (!batchIdxTensorType.getElementType().isInteger()) {
      return emitOpError("Batch index tensor must be an integer type");
    }
  }

  int64_t numCacheHeads = cacheShape[1];
  int64_t numInputHeads = inputShape[1];
  int64_t blockSize = cacheShape[2];
  int64_t headDim = cacheShape[3];

  if (blockSize % 32 != 0) {
    return emitOpError("Block size must be divisible by 32, got " +
                       std::to_string(blockSize));
  }

  if (numInputHeads % numCacheHeads != 0) {
    return emitOpError("Input must have a number of heads that is a multiple "
                       "of the number of heads in the cache.");
  }

  if (inputShape[3] != headDim) {
    return emitOpError("Input must have same head dimension as cache.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

// ReverseOp verification
::mlir::LogicalResult mlir::tt::ttir::ReverseOp::verify() {
  llvm::ArrayRef<int64_t> dimensions = getDimensions();

  // Check that all given dimensions are unique/not repeating.
  llvm::SmallDenseSet<int64_t> uniqueDims(dimensions.begin(), dimensions.end());

  if (uniqueDims.size() != dimensions.size()) {
    return emitOpError("dimensions should be unique. Got: ") << dimensions;
  }

  ::mlir::RankedTensorType operandTy = getInput().getType();

  // Check that each dimension is positive and within valid interval [0,
  // operandRank).
  for (int64_t dim : dimensions) {
    if (dim < 0) {
      return emitOpError(
                 "all dimensions should be non-negative. Got dimension: ")
             << dim;
    }

    if (dim >= operandTy.getRank()) {
      return emitOpError("all dimensions should be in interval [0, ")
             << operandTy.getRank() << "). Got dimension: " << dim;
    }
  }

  return success();
}

// ReverseOp canonicalization
void mlir::tt::ttir::ReverseOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  // Reverse dimensions of two consecutive ReverseOps can be folded into a
  // single ReverseOp where the dimensions are the symmetric difference of the
  // two sets of dimensions.
  patterns.add(+[](mlir::tt::ttir::ReverseOp op,
                   mlir::PatternRewriter &rewriter) {
    auto producerOp = op.getInput().getDefiningOp<ttir::ReverseOp>();
    if (!producerOp) {
      return mlir::failure();
    }

    llvm::SmallBitVector reverseDimensions(op.getInput().getType().getRank());
    llvm::for_each(op.getDimensions(), [&reverseDimensions](int64_t dim) {
      reverseDimensions.flip(dim);
    });
    llvm::for_each(
        producerOp.getDimensions(),
        [&reverseDimensions](int64_t dim) { reverseDimensions.flip(dim); });

    llvm::SmallVector<int64_t> setIndices;
    llvm::copy_if(llvm::seq<int64_t>(reverseDimensions.size()),
                  std::back_inserter(setIndices),
                  [&](int64_t i) { return reverseDimensions.test(i); });

    rewriter.replaceOpWithNewOp<ttir::ReverseOp>(
        op, op.getType(), producerOp.getInput(), setIndices);
    return success();
  });

  // ReverseOp with empty reverse dimensions is a no-op.
  patterns.add(
      +[](mlir::tt::ttir::ReverseOp op, mlir::PatternRewriter &rewriter) {
        if (!op.getDimensions().empty()) {
          return mlir::failure();
        }

        rewriter.replaceAllOpUsesWith(op, op.getInput());
        return mlir::success();
      });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// PermuteOp
//===----------------------------------------------------------------------===//

// PermuteOp verification
::mlir::LogicalResult mlir::tt::ttir::PermuteOp::verify() {
  llvm::ArrayRef<int64_t> inputShape = getInput().getType().getShape();
  const size_t inputRank = inputShape.size();
  llvm::ArrayRef<int64_t> resultShape = getResult().getType().getShape();

  // Check that given attribute `permutation` is a valid permutation of the
  // dimensions.
  llvm::ArrayRef<int64_t> permutation = getPermutation();
  llvm::SmallVector<int64_t> dimensions(inputRank);
  std::iota(dimensions.begin(), dimensions.end(), 0);
  if (inputRank != permutation.size() ||
      !std::is_permutation(permutation.begin(), permutation.end(),
                           dimensions.begin())) {
    return emitOpError("Expected a permutation of (")
           << ttmlir::utils::join(dimensions, ", ")
           << "), got (" + ttmlir::utils::join(permutation, ", ") << ")";
  }

  // Check that the result shape matches the shape of input tensor after
  // permutation is applied.
  llvm::SmallVector<int64_t> expectedResultShape =
      ttmlir::utils::applyPermutation(inputShape, permutation);
  if (!llvm::equal(expectedResultShape, resultShape)) {
    return emitOpError("Expected result shape (")
           << ttmlir::utils::join(expectedResultShape, ", ") << "), got ("
           << ttmlir::utils::join(resultShape, ", ") << ")";
  }

  return success();
}

// PermuteOp with identity permutation is a no-op.
// The input can be used directly as the output.
// This includes:
// 1. Sorted permutations like [0, 1, 2, 3]
// 2. Permutations where all swapped dimensions have size 1
//    (e.g., [2, 0, 1, 3] on shape 1x1x1x64 is a no-op)
static mlir::OpFoldResult foldIdentityPermute(mlir::tt::ttir::PermuteOp op) {
  // Case 1: True identity permutation (sorted)
  if (llvm::is_sorted(op.getPermutation())) {
    return op.getInput();
  }

  // Case 2: All non-identity dimension swaps are between dims of size 1
  auto inputShape = op.getInput().getType().getShape();
  for (auto [index, permuteIndex] : llvm::enumerate(op.getPermutation())) {
    if (permuteIndex != static_cast<int64_t>(index)) {
      // This dim position is changed - check if both dims involved are size 1
      if (inputShape[index] != 1 || inputShape[permuteIndex] != 1) {
        return nullptr; // Non-trivial swap
      }
    }
  }
  return op.getInput();
}

// If the producer is a PermuteOp we can compose the permutation attributes
// into `op`, and set the input to the producers input.
static mlir::OpFoldResult foldConsecutivePermute(mlir::tt::ttir::PermuteOp op) {
  // Don't fold decomposed permutes - they were intentionally split.
  if (op->hasAttr("decomposed")) {
    return nullptr;
  }
  if (auto producerOp =
          op.getInput().getDefiningOp<mlir::tt::ttir::PermuteOp>()) {
    llvm::SmallVector<int64_t> composedPermutation =
        ttmlir::utils::applyPermutation(producerOp.getPermutation(),
                                        op.getPermutation());
    op.setPermutation(composedPermutation);
    op->setOperand(0, producerOp.getInput());
    return op.getResult();
  }
  return nullptr;
}

static mlir::OpFoldResult constantFoldPermute(mlir::tt::ttir::PermuteOp op,
                                              Attribute input) {
  if (!input) {
    return nullptr;
  }

  // Invert the permutation to permute output to input coordinates
  SmallVector<int64_t> invPerm =
      mlir::invertPermutationVector(op.getPermutation());

  return constantFoldTM(op, input,
                        [&invPerm](llvm::SmallVector<int64_t> coord) {
                          mlir::applyPermutationToVector(coord, invPerm);
                          return coord;
                        });
}

// PermuteOp folder
mlir::OpFoldResult mlir::tt::ttir::PermuteOp::fold(FoldAdaptor adaptor) {

  if (auto foldResult = foldIdentityPermute(*this)) {
    return foldResult;
  }

  if (auto foldResult = foldConsecutivePermute(*this)) {
    return foldResult;
  }

  if (auto foldResult = constantFoldPermute(*this, adaptor.getInput())) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// FullOp
//===----------------------------------------------------------------------===//

// FullOp verification
mlir::LogicalResult mlir::tt::ttir::FullOp::verify() {
  // Verify that the shape is the shape of the output.
  if (!llvm::equal(getShape(), getType().getShape())) {
    return emitOpError() << "expected shape (" << getType().getShape()
                         << "), got (" << getShape() << ")";
  }

  return mlir::success();
}

static std::optional<std::string>
verifyReplicaGroups(mlir::DenseIntElementsAttr replicaGroups) {
  if (replicaGroups.getType().getRank() != 2) {
    return "replica_groups must be a 2D array";
  }

  auto replicaIds = replicaGroups.getValues<int64_t>();
  int64_t maxId = replicaIds.size() - 1;
  llvm::SmallDenseSet<int64_t> seen;
  for (auto id : replicaIds) {
    if (id < 0) {
      return "replica_groups values must be positive";
    }
    if (id > maxId) {
      return llvm::formatv(
                 "replica_groups values must be in the range [0, {0}], got {1}",
                 maxId, id)
          .str();
    }
    if (!seen.insert(id).second) {
      return "replica_groups must not contain duplicate values";
    }
  }
  return std::nullopt;
}

// Helper to convert type of fill_value attribute from i32/f32 to any
// integer/float type.
static mlir::Attribute convertFillValue(mlir::TypedAttr typedAttr,
                                        mlir::Type targetType) {
  assert((typedAttr.getType().isF32() || typedAttr.getType().isInteger(32)) &&
         "Expected fill_value attribute to be either f32 or i32");

  if (typedAttr.getType() == targetType) {
    return typedAttr;
  }

  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(typedAttr)) {
    llvm::APFloat floatVal = floatAttr.getValue();

    // Case A: Float -> Float (e.g., f32 -> f64)
    if (auto targetFloatType = mlir::dyn_cast<mlir::FloatType>(targetType)) {
      bool losesInfo;
      floatVal.convert(targetFloatType.getFloatSemantics(),
                       llvm::APFloat::rmNearestTiesToEven, &losesInfo);
      return mlir::FloatAttr::get(targetType, floatVal);
    }

    // Case B: Float -> Integer (e.g., f32 -> i32)
    if (auto targetIntType = mlir::dyn_cast<mlir::IntegerType>(targetType)) {
      llvm::APSInt intVal(targetIntType.getWidth(), targetIntType.isUnsigned());
      bool isExact;
      floatVal.convertToInteger(intVal, llvm::APFloat::rmTowardZero, &isExact);
      return mlir::IntegerAttr::get(targetType, intVal);
    }
  }

  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(typedAttr)) {
    llvm::APInt intVal = intAttr.getValue();

    // Case C: Integer -> Integer (e.g., i32 -> i64)
    if (auto targetIntType = mlir::dyn_cast<mlir::IntegerType>(targetType)) {
      intVal = intVal.sextOrTrunc(targetIntType.getWidth());
      return mlir::IntegerAttr::get(targetType, intVal);
    }

    // Case D: Integer -> Float (e.g., i32 -> f32)
    if (auto targetFloatType = mlir::dyn_cast<mlir::FloatType>(targetType)) {
      llvm::APFloat floatVal(targetFloatType.getFloatSemantics());
      // Source type is signless (i32) but we want to keep negative values
      // negative when converting to float.
      bool isSigned = true;
      floatVal.convertFromAPInt(intVal, isSigned,
                                llvm::APFloat::rmNearestTiesToEven);
      return mlir::FloatAttr::get(targetType, floatVal);
    }
  }

  llvm_unreachable("Expected floating point or integer types");
}

// FullOp folder
::mlir::OpFoldResult mlir::tt::ttir::FullOp::fold(FoldAdaptor adaptor) {
  auto fillValue = llvm::dyn_cast<TypedAttr>(getFillValueAttr());
  RankedTensorType resultType = getResult().getType();

  // Fill value is 32-bit float or 32-bit signless integer, but result type
  // might differ.
  auto convertedFillValue =
      convertFillValue(fillValue, resultType.getElementType());

  return SplatElementsAttr::get(resultType, convertedFillValue);
}

//===----------------------------------------------------------------------===//
// ZerosOp
//===----------------------------------------------------------------------===//

// ZerosOp folder
::mlir::OpFoldResult mlir::tt::ttir::ZerosOp::fold(FoldAdaptor adaptor) {
  RankedTensorType resultType = getResult().getType();
  mlir::Attribute value = makeScalarAttr(resultType.getElementType(), 0.0);
  return SplatElementsAttr::get(resultType, value);
}

//===----------------------------------------------------------------------===//
// OnesOp
//===----------------------------------------------------------------------===//

// OnesOp folder
::mlir::OpFoldResult mlir::tt::ttir::OnesOp::fold(FoldAdaptor adaptor) {
  RankedTensorType resultType = getResult().getType();
  mlir::Attribute value = makeScalarAttr(resultType.getElementType(), 1.0);
  return SplatElementsAttr::get(resultType, value);
}

//===----------------------------------------------------------------------===//
// AllToAllOp
//===----------------------------------------------------------------------===//

// AllToAllOp verification
::mlir::LogicalResult mlir::tt::ttir::AllToAllOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();
  auto inShape = inputType.getShape();
  int64_t splitDim = getSplitDim();
  int64_t splitCount = getSplitCount();
  if (splitDim < 0 || splitDim >= inputType.getRank()) {
    return emitOpError("splitDim must be in the range [0, ")
           << inputType.getRank() - 1 << "], got " << splitDim;
  }
  if (splitCount <= 0) {
    return emitOpError("splitCount must be a positive integer");
  }
  if (inShape[splitDim] % splitCount != 0) {
    return emitOpError("splitDim size must be divisible by splitCount");
  }
  int64_t concatDim = getConcatDim();
  if (concatDim < 0 || concatDim >= inputType.getRank()) {
    return emitOpError("concatDim must be in the range [0, ")
           << inputType.getRank() - 1 << "], got " << concatDim;
  }
  ::llvm::SmallVector<int64_t> expectedShape(inShape.begin(), inShape.end());
  expectedShape[splitDim] = expectedShape[splitDim] / splitCount;
  expectedShape[concatDim] = expectedShape[concatDim] * splitCount;
  if (expectedShape != outputType.getShape()) {
    return emitOpError("Output shape mismatch: expected = <")
           << expectedShape << "> output = <" << outputType.getShape() << ">";
  }
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("Input and output element types must match");
  }
  ::mlir::DenseIntElementsAttr replicaGroups = getReplicaGroups();

  if (auto errorMsg = verifyReplicaGroups(replicaGroups)) {
    return emitOpError() << *errorMsg;
  }
  auto replicaGroupsShape = replicaGroups.getType().getShape();
  if (replicaGroupsShape[1] != splitCount) {
    return emitOpError("replicaGroup count must match splitCount");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

// Common verifier for all Reduce ops.
static mlir::LogicalResult
verifyReduceOp(llvm::function_ref<mlir::InFlightDiagnostic()> emitOpError,
               mlir::RankedTensorType inputType,
               const std::optional<mlir::ArrayAttr> &reduceDims, bool keepDim,
               ::llvm::ArrayRef<int64_t> specifiedOutputShape) {

  int64_t inputTensorRank = inputType.getRank();

  llvm::BitVector reduceDimsMask(inputTensorRank, false);
  if (!reduceDims) {
    reduceDimsMask.set();
  } else {
    llvm::SmallSet<int64_t, 4> uniqueReduceDims;
    for (mlir::Attribute reduceDim : *reduceDims) {
      int64_t reduceDimInt = mlir::cast<mlir::IntegerAttr>(reduceDim).getInt();
      if (reduceDimInt < -inputTensorRank || reduceDimInt >= inputTensorRank) {
        return emitOpError() << "Reduce dimension " << reduceDimInt
                             << " is out of range for input tensor of rank "
                             << inputTensorRank;
      }
      uniqueReduceDims.insert(reduceDimInt);
      reduceDimsMask.set((reduceDimInt + inputTensorRank) % inputTensorRank);
    }

    if (uniqueReduceDims.size() != reduceDims->size()) {
      return emitOpError() << "Reduce dimensions are not unique";
    }
  }

  // Check that the output shape is valid.
  llvm::SmallVector<int64_t> expectedOutputShape;
  for (int64_t index = 0; index < inputTensorRank; ++index) {
    if (!reduceDimsMask[index]) {
      expectedOutputShape.push_back(inputType.getDimSize(index));
    } else if (keepDim) {
      expectedOutputShape.push_back(1);
    }
  }

  // Finally, compare shapes.
  if (!llvm::equal(specifiedOutputShape, expectedOutputShape)) {
    return emitOpError() << "Expected output shape ("
                         << ttmlir::utils::join(expectedOutputShape, ", ")
                         << "), got ("
                         << ttmlir::utils::join(specifiedOutputShape, ", ")
                         << ")";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

// MaxOp verification.
::mlir::LogicalResult mlir::tt::ttir::MaxOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

// MeanOp verification.
::mlir::LogicalResult mlir::tt::ttir::MeanOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

// SumOp verification.
::mlir::LogicalResult mlir::tt::ttir::SumOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce MinOp
//===----------------------------------------------------------------------===//

// MinOp verification.
::mlir::LogicalResult mlir::tt::ttir::MinOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce ProdOp
//===----------------------------------------------------------------------===//

// ProdOp verification.
::mlir::LogicalResult mlir::tt::ttir::ProdOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// ReduceAndOp
//===----------------------------------------------------------------------===//

// ReduceAndOp verification.
::mlir::LogicalResult mlir::tt::ttir::ReduceAndOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// ReduceOrOp
//===----------------------------------------------------------------------===//

// ReduceOrOp verification.
::mlir::LogicalResult mlir::tt::ttir::ReduceOrOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce ArgMaxOp
//===----------------------------------------------------------------------===//

// ArgMaxOp verification.
::mlir::LogicalResult mlir::tt::ttir::ArgMaxOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

// TopKOp verification
::mlir::LogicalResult mlir::tt::ttir::TopKOp::verify() {
  RankedTensorType inputType = getInputTensor().getType();
  int64_t inputRank = inputType.getRank();
  int32_t dim = getDim();
  int32_t K = getK();

  // Normalize dim to check if it's effectively the last dimension
  int normalizedDim = dim < 0 ? dim + inputRank : dim;
  if (normalizedDim < 0 || normalizedDim >= inputRank) {
    return emitOpError() << "specified dimension should be between "
                         << -inputRank << " and " << (inputRank - 1)
                         << ", but got: " << dim;
  }

  if (K <= 0 || K > inputType.getDimSize(normalizedDim)) {
    return emitOpError() << "K should be between 1 and the size of the "
                            "specified dimension ("
                         << normalizedDim << "), but got: " << K;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TopKRouterGptOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::TopKRouterGptOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType weightType = getWeight().getType();
  RankedTensorType biasType = getBias().getType();

  if (inputType.getRank() != 2) {
    return emitOpError() << "input must be a 2D tensor [B, hidden_dim], but "
                            "got rank "
                         << inputType.getRank();
  }
  if (weightType.getRank() != 2) {
    return emitOpError()
           << "weight must be a 2D tensor [hidden_dim, num_experts], but got "
              "rank "
           << weightType.getRank();
  }
  if (biasType.getRank() != 2) {
    return emitOpError()
           << "bias must be a 2D tensor [B, num_experts], but got rank "
           << biasType.getRank();
  }

  int64_t B = inputType.getDimSize(0);
  int64_t hiddenDim = inputType.getDimSize(1);
  if (weightType.getDimSize(0) != hiddenDim) {
    return emitOpError() << "weight dim 0 (" << weightType.getDimSize(0)
                         << ") must equal input hidden_dim (" << hiddenDim
                         << ")";
  }
  if (biasType.getDimSize(0) != B) {
    return emitOpError() << "bias dim 0 (" << biasType.getDimSize(0)
                         << ") must equal input batch size B (" << B << ")";
  }
  if (biasType.getDimSize(1) != weightType.getDimSize(1)) {
    return emitOpError() << "bias dim 1 (" << biasType.getDimSize(1)
                         << ") must equal weight num_experts ("
                         << weightType.getDimSize(1) << ")";
  }

  int32_t numExperts = getNumExperts();
  if (numExperts <= 0) {
    return emitOpError() << "num_experts must be positive, but got: "
                         << numExperts;
  }
  if (static_cast<int64_t>(numExperts) != weightType.getDimSize(1)) {
    return emitOpError() << "num_experts attribute (" << numExperts
                         << ") must equal weight dim 1 ("
                         << weightType.getDimSize(1) << ")";
  }

  int32_t k = getK();
  if (k <= 0) {
    return emitOpError() << "k must be positive, but got: " << k;
  }

  RankedTensorType indicesType = getExpertIndices().getType();
  RankedTensorType weightsType = getExpertWeights().getType();

  if (indicesType.getRank() != 2) {
    return emitOpError()
           << "expert_indices must be a 2D tensor [B, k], but got rank "
           << indicesType.getRank();
  }
  if (weightsType.getRank() != 2) {
    return emitOpError()
           << "expert_weights must be a 2D tensor [B, k], but got rank "
           << weightsType.getRank();
  }
  if (indicesType.getDimSize(0) != B) {
    return emitOpError() << "expert_indices dim 0 ("
                         << indicesType.getDimSize(0)
                         << ") must equal input batch size B (" << B << ")";
  }
  if (indicesType.getDimSize(1) != static_cast<int64_t>(k)) {
    return emitOpError() << "expert_indices dim 1 ("
                         << indicesType.getDimSize(1) << ") must equal k (" << k
                         << ")";
  }
  if (weightsType.getDimSize(0) != B) {
    return emitOpError() << "expert_weights dim 0 ("
                         << weightsType.getDimSize(0)
                         << ") must equal input batch size B (" << B << ")";
  }
  if (weightsType.getDimSize(1) != static_cast<int64_t>(k)) {
    return emitOpError() << "expert_weights dim 1 ("
                         << weightsType.getDimSize(1) << ") must equal k (" << k
                         << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CumSumOp
//===----------------------------------------------------------------------===//

// CumSumOp verification
::mlir::LogicalResult mlir::tt::ttir::CumSumOp::verify() {
  int64_t dim = getDim();
  int64_t inputRank = getInput().getType().getRank();
  if (dim < -inputRank || dim >= inputRank) {
    return emitOpError() << "specified dimension should be between "
                         << -inputRank << " and " << (inputRank - 1)
                         << ", but got: " << dim;
  }

  return success();
}

// CumSumOp folding
::mlir::OpFoldResult mlir::tt::ttir::CumSumOp::fold(FoldAdaptor adaptor) {
  // Normalize `dim` to be in range [0, rank).
  int64_t dim = getDim();
  int64_t rank = getInput().getType().getRank();
  if (dim < 0) {
    setDim(dim + rank);
    return getResult();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// BatchNorm verification helpers
//===----------------------------------------------------------------------===//
namespace {
// Shared verification logic for BatchNorm operations
static ::mlir::LogicalResult verifyBatchNormOp(mlir::Operation *op,
                                               mlir::RankedTensorType inputType,
                                               int64_t dimension) {
  int64_t inputRank = inputType.getRank();

  // Input must be 2D to 5D
  if (inputRank < 2 || inputRank > 5) {
    return op->emitOpError(
               "input tensor must have rank between 2 and 5, got rank ")
           << inputRank;
  }

  // Dimension attribute must be within bounds
  if (dimension < 0 || dimension >= inputRank) {
    return op->emitOpError(
               "dimension attribute must be within input rank bounds, "
               "got dimension ")
           << dimension << " for rank " << inputRank;
  }

  return success();
}
} // namespace

//===----------------------------------------------------------------------===//
// BatchNormInferenceOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::BatchNormInferenceOp::verify() {
  return verifyBatchNormOp(getOperation(), getOperand().getType(),
                           getDimension());
}

//===----------------------------------------------------------------------===//
// BatchNormTrainingOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::BatchNormTrainingOp::verify() {
  return verifyBatchNormOp(getOperation(), getOperand().getType(),
                           getDimension());
}

//===----------------------------------------------------------------------===//
// CollectiveBroadcastOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::CollectiveBroadcastOp::verify() {
  // Check input/output/result types are RankedTensorType
  auto inputType = getInput().getType();
  auto resultType = getType();

  // Check input == output type
  if (inputType != resultType) {
    return emitOpError("input and output must have the same type");
  }

  ::mlir::DenseIntElementsAttr replicaGroups = getReplicaGroups();
  if (auto errorMsg = verifyReplicaGroups(replicaGroups)) {
    return emitOpError() << *errorMsg;
  }

  return success();
}

mlir::OpFoldResult
mlir::tt::ttir::CollectiveBroadcastOp::fold(FoldAdaptor adaptor) {
  auto groupsType = getReplicaGroups().getType();
  // If there is no group, the broadcast is a no-op.
  if (groupsType.getShape()[0] < 1) {
    return getInput();
  }
  // If there is only one device in a group, the broadcast is a no-op.
  if (groupsType.getShape()[1] <= 1) {
    return getInput();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ConcatenateHeadsOp
//===----------------------------------------------------------------------===//

// ConcatenateHeadsOp verification
::mlir::LogicalResult mlir::tt::ttir::ConcatenateHeadsOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();

  // Input tensor must be 4D tensor
  if (inputType.getRank() != 4) {
    return emitOpError() << "expected rank of input tensor is 4, got rank "
                         << inputType.getRank();
  }

  // Output tensor must be 3D tensor.
  if (outputType.getRank() != 3) {
    return emitOpError() << "expected rank of output tensor is 3, got rank "
                         << outputType.getRank();
  }

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  // Input tensor dimensions [batch_size, num_heads, sequence_size, head_size].
  // Output tensor dimensions [batch_size, sequence_size, num_heads *
  // head_size].
  using namespace ttmlir::utils::transformer;

  // Verify batch_size dimension matches.
  if (inputShape[INPUT_BATCH] != outputShape[OUTPUT_BATCH]) {
    return emitOpError() << "expected output batch dimension to be "
                         << inputShape[INPUT_BATCH] << ", got "
                         << outputShape[OUTPUT_BATCH];
  }

  // Verify sequence_size dimension matches.
  if (inputShape[INPUT_SEQ] != outputShape[OUTPUT_SEQ]) {
    return emitOpError() << "expected output sequence dimension to be "
                         << inputShape[INPUT_SEQ] << ", got "
                         << outputShape[OUTPUT_SEQ];
  }

  // Verify that num_heads * head_size equals the output hidden dimension.
  int64_t expectedHiddenSize =
      inputShape[INPUT_NUM_HEADS] * inputShape[INPUT_HEAD_SIZE];
  if (expectedHiddenSize != outputShape[OUTPUT_HIDDEN]) {
    return emitOpError()
           << "expected output hidden dimension to be num_heads * "
              "head_size = "
           << expectedHiddenSize << ", got " << outputShape[OUTPUT_HIDDEN];
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SplitQueryKeyValueAndSplitHeadsOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult
mlir::tt::ttir::SplitQueryKeyValueAndSplitHeadsOp::verify() {
  ::mlir::RankedTensorType inputType = getInputTensor().getType();

  // Input tensor must be 3D tensor
  if (inputType.getRank() != 3) {
    return emitOpError() << "expected rank of input tensor is 3, got rank "
                         << inputType.getRank();
  }

  ::mlir::RankedTensorType queryOutputType = getQuery().getType();
  ::mlir::RankedTensorType keyOutputType = getKey().getType();
  ::mlir::RankedTensorType valueOutputType = getValue().getType();

  // Output tensors must be 4D tensors
  if (queryOutputType.getRank() != 4 || keyOutputType.getRank() != 4 ||
      valueOutputType.getRank() != 4) {
    return emitOpError() << "expected rank of query/key/value output tensor is "
                            "4, got query rank: "
                         << queryOutputType.getRank()
                         << ", key rank: " << keyOutputType.getRank()
                         << ", value rank: " << valueOutputType.getRank();
  }

  const uint32_t BATCH_DIM = 0;
  const uint32_t SEQUENCE_LENGTH_DIM = 1;
  const uint32_t HIDDEN_DIMENSION = 2;

  auto inputShape = inputType.getShape();
  auto kvInputShape = getKvInputTensor()
                          ? getKvInputTensor().getType().getShape()
                          : inputType.getShape();

  int64_t numHeads = getNumHeads();
  int64_t numKVHeads = getNumKvHeads() ? *getNumKvHeads() : numHeads;

  int64_t batchSizeQuery = inputShape[BATCH_DIM];
  int64_t sequenceLengthQuery = inputShape[SEQUENCE_LENGTH_DIM];
  int64_t headSizeQuery = 0;

  int64_t batchSizeKeyValue = kvInputShape[BATCH_DIM];
  int64_t sequenceLengthKeyValue = kvInputShape[SEQUENCE_LENGTH_DIM];
  int64_t headSizeKeyValue = 0;

  if (getKvInputTensor()) {
    headSizeQuery = inputShape[HIDDEN_DIMENSION] / numHeads;
    headSizeKeyValue = kvInputShape[HIDDEN_DIMENSION] / (2 * numKVHeads);
  } else {
    headSizeQuery = inputShape[HIDDEN_DIMENSION] / (numHeads + 2 * numKVHeads);
    headSizeKeyValue = headSizeQuery;
  }

  llvm::SmallVector<int64_t, 4> expectedQueryShape = {
      batchSizeQuery, numHeads, sequenceLengthQuery, headSizeQuery};

  if (!llvm::equal(expectedQueryShape, queryOutputType.getShape())) {
    return emitOpError() << "expected query output shape ("
                         << ttmlir::utils::join(expectedQueryShape, ", ")
                         << "), got ("
                         << ttmlir::utils::join(queryOutputType.getShape(),
                                                ", ")
                         << ")";
  }

  llvm::SmallVector<int64_t, 4> expectedKeyShape = {
      batchSizeKeyValue, numKVHeads, sequenceLengthKeyValue, headSizeKeyValue};

  if (getTransposeKey()) {
    std::swap(expectedKeyShape[2], expectedKeyShape[3]);
  }

  if (!llvm::equal(expectedKeyShape, keyOutputType.getShape())) {
    return emitOpError() << "expected key output shape ("
                         << ttmlir::utils::join(expectedKeyShape, ", ")
                         << "), got ("
                         << ttmlir::utils::join(keyOutputType.getShape(), ", ")
                         << ")";
  }

  llvm::SmallVector<int64_t, 4> expectedValueShape = {
      batchSizeKeyValue, numKVHeads, sequenceLengthKeyValue, headSizeKeyValue};

  if (!llvm::equal(expectedValueShape, valueOutputType.getShape())) {
    return emitOpError() << "expected value output shape ("
                         << ttmlir::utils::join(expectedValueShape, ", ")
                         << "), got ("
                         << ttmlir::utils::join(valueOutputType.getShape(),
                                                ", ")
                         << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RMSNormOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::RMSNormOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType outputType = getResult().getType();

  // Input and output must have the same shape.
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("input and output must have the same shape");
  }

  // Verify normalized_shape is valid for the input tensor.
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> normalizedShape = getNormalizedShape();

  // Check that normalized_shape is not empty.
  if (normalizedShape.empty()) {
    return emitOpError("normalized_shape cannot be empty");
  }

  // Check that normalized_shape is not larger than input tensor shape.
  if (normalizedShape.size() > inputShape.size()) {
    return emitOpError(
        "normalized_shape has more dimensions than input tensor");
  }

  // Check that the trailing dimensions of input match normalized_shape.
  // For example, if input shape is [2, 3, 4, 5] and normalized_shape is [4, 5],
  // then we check that input shape's last two dimensions (4, 5) match
  // normalized_shape.
  size_t offset = inputShape.size() - normalizedShape.size();
  for (size_t i = 0; i < normalizedShape.size(); ++i) {
    if (inputShape[offset + i] != normalizedShape[i]) {
      return emitOpError("normalized_shape dimensions must match trailing "
                         "dimensions of input tensor");
    }
  }

  // Verify weight tensor shape if present.
  if (getWeight()) {
    RankedTensorType weightType = getWeight().getType();
    if (weightType.getShape() != normalizedShape) {
      return emitOpError("weight tensor shape must match normalized_shape");
    }
  }

  // Verify bias tensor shape if present.
  if (getBias()) {
    RankedTensorType biasType = getBias().getType();
    if (biasType.getShape() != normalizedShape) {
      return emitOpError("bias tensor shape must match normalized_shape");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DistributedRMSNormOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::DistributedRMSNormOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType outputType = getResult().getType();

  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("output shape must match input shape");
  }

  // Verify cluster_axis is valid (must be 0 or 1 for 2D mesh).
  uint32_t clusterAxis = getClusterAxis();
  if (clusterAxis > 1) {
    return emitOpError("cluster_axis must be 0 or 1");
  }

  // Verify epsilon is positive.
  float epsilon = getEpsilon().convertToFloat();
  if (epsilon <= 0) {
    return emitOpError("epsilon must be positive");
  }

  // Verify residual tensor shape matches input if present.
  if (getResidual()) {
    RankedTensorType residualType = getResidual().getType();
    if (residualType.getShape() != inputType.getShape()) {
      return emitOpError("residual tensor shape must match input tensor shape");
    }
  }

  // Verify weight tensor's last dimension matches input's last dimension.
  if (getWeight()) {
    RankedTensorType weightType = getWeight().getType();
    int64_t inputLastDim = inputType.getShape().back();
    int64_t weightLastDim = weightType.getShape().back();

    if (weightLastDim != inputLastDim) {
      return emitOpError(
          "weight tensor's last dimension must match input's last dimension");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DistributedLayerNormOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::DistributedLayerNormOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType outputType = getResult().getType();

  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("output shape must match input shape");
  }

  // Verify cluster_axis is valid (must be 0 or 1 for 2D mesh).
  uint32_t clusterAxis = getClusterAxis();
  if (clusterAxis > 1) {
    return emitOpError("cluster_axis must be 0 or 1");
  }

  // Verify epsilon is positive.
  float epsilon = getEpsilon().convertToFloat();
  if (epsilon <= 0) {
    return emitOpError("epsilon must be positive");
  }

  // Verify residual tensor shape matches input if present.
  if (getResidual()) {
    RankedTensorType residualType = getResidual().getType();
    if (residualType.getShape() != inputType.getShape()) {
      return emitOpError("residual tensor shape must match input tensor shape");
    }
  }

  int64_t inputLastDim = inputType.getShape().back();

  // Verify weight tensor's last dimension matches input's last dimension.
  if (getWeight()) {
    RankedTensorType weightType = getWeight().getType();
    if (weightType.getShape().back() != inputLastDim) {
      return emitOpError(
          "weight tensor's last dimension must match input's last dimension");
    }
  }

  // Verify bias tensor's last dimension matches input's last dimension.
  if (getBias()) {
    RankedTensorType biasType = getBias().getType();
    if (biasType.getShape().back() != inputLastDim) {
      return emitOpError(
          "bias tensor's last dimension must match input's last dimension");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LayerNormOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::LayerNormOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType outputType = getResult().getType();

  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("input and output must have the same shape");
  }

  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> normalizedShape = getNormalizedShape();

  if (normalizedShape.empty()) {
    return emitOpError("normalized_shape cannot be empty");
  }

  if (normalizedShape.size() > inputShape.size()) {
    return emitOpError(
        "normalized_shape has more dimensions than input tensor");
  }

  // Check that the trailing dimensions of input match normalized_shape.
  // For example, if input shape is [2, 3, 4, 5] and normalized_shape is [4, 5],
  // then we check that input shape's last two dimensions (4, 5) match
  // normalized_shape.
  size_t offset = inputShape.size() - normalizedShape.size();
  for (size_t i = 0; i < normalizedShape.size(); ++i) {
    if (inputShape[offset + i] != normalizedShape[i]) {
      return emitOpError("normalized_shape dimensions must match trailing "
                         "dimensions of input tensor");
    }
  }

  if (getWeight()) {
    RankedTensorType weightType = getWeight().getType();
    if (weightType.getShape() != normalizedShape) {
      return emitOpError("weight tensor shape must match normalized_shape");
    }
  }

  if (getBias()) {
    RankedTensorType biasType = getBias().getType();
    if (biasType.getShape() != normalizedShape) {
      return emitOpError("bias tensor shape must match normalized_shape");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GroupNormOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::GroupNormOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType outputType = getResult().getType();

  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("input and output must have the same shape");
  }

  // Input must be 4D.
  if (inputType.getRank() != 4) {
    return emitOpError("input must be a 4D tensor, got rank ")
           << inputType.getRank();
  }

  int64_t numGroups = getNumGroups();
  if (numGroups <= 0) {
    return emitOpError("num_groups must be positive, got ") << numGroups;
  }

  // Validate channel_dim is within bounds.
  int64_t channelDimIdx = getChannelDim();
  if (channelDimIdx < 0 || channelDimIdx >= inputType.getRank()) {
    return emitOpError("channel_dim must be in range [0, rank), got ")
           << channelDimIdx << " for rank " << inputType.getRank();
  }

  // Channel dimension must be divisible by num_groups.
  int64_t c = inputType.getShape()[channelDimIdx];
  if (c % numGroups != 0) {
    return emitOpError("channel dimension (dim ")
           << channelDimIdx << ") must be divisible by num_groups; got C=" << c
           << ", num_groups=" << numGroups;
  }

  // Weight must be 1D with size matching channel dimension.
  if (getWeight()) {
    RankedTensorType weightType = getWeight().getType();
    if (weightType.getRank() != 1 || weightType.getShape()[0] != c) {
      return emitOpError(
                 "weight must be 1D with size matching channel dimension C=")
             << c;
    }
  }

  // Bias must be 1D with size matching channel dimension.
  if (getBias()) {
    RankedTensorType biasType = getBias().getType();
    if (biasType.getRank() != 1 || biasType.getShape()[0] != c) {
      return emitOpError(
                 "bias must be 1D with size matching channel dimension C=")
             << c;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionDecodeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult
mlir::tt::ttir::ScaledDotProductAttentionDecodeOp::verify() {

  RankedTensorType queryType = getQuery().getType();
  RankedTensorType keyType = getKey().getType();
  RankedTensorType valueType = getValue().getType();
  RankedTensorType curPosTensorType = getCurPosTensor().getType();
  RankedTensorType resultType = getResult().getType();

  if (queryType != resultType) {
    return emitOpError("Query and result must have the same type");
  }

  if (!curPosTensorType.getElementType().isInteger()) {
    return emitOpError("Cur pos tensor must be a tensor of integers");
  }

  if (curPosTensorType.getShape().size() != 1) {
    return emitOpError("Cur pos tensor must be a 1D tensor");
  }

  if (keyType != valueType) {
    return emitOpError("Key and value must have the same type");
  }
  if (queryType.getShape().size() != 4) {
    return emitOpError("Query must be a 4D tensor");
  }
  if (keyType.getShape().size() != 4) {
    return emitOpError("Key/Value must be a 4D tensor");
  }
  if (resultType.getShape().size() != 4) {
    return emitOpError("Output must be a 4D tensor");
  }

  if (queryType.getShape()[0] != 1) {
    return emitOpError("Query dim 0 must be 1");
  }

  int64_t batchSize = queryType.getShape()[1];
  int64_t nQueryHeads = queryType.getShape()[2];
  int64_t nKVHeads = keyType.getShape()[1];
  int64_t headSize = queryType.getShape()[3];
  int64_t maxSeqLen = keyType.getShape()[2];

  if (curPosTensorType.getShape()[0] != batchSize) {
    return emitOpError("Cur pos tensor batch size must match query batch size");
  }

  if (keyType.getShape()[0] != batchSize) {
    return emitOpError("Key/Value batch size must match query batch size");
  }

  if (keyType.getShape()[3] != headSize) {
    return emitOpError("Key/Value head size must match query head size");
  }

  if (nQueryHeads % nKVHeads != 0) {
    return emitOpError(
        "Query num heads must be divisible by key/value num heads");
  }

  if (getAttentionMask()) {
    if (getIsCausal()) {
      return emitOpError(
          "Attention mask is not allowed when is_causal is true");
    }
    RankedTensorType attentionMaskType = getAttentionMask().getType();
    if (attentionMaskType.getShape().size() != 4) {
      return emitOpError("Attention mask must be a 4D tensor");
    }
    if (attentionMaskType.getShape()[0] != 1) {
      return emitOpError("Attention mask dim 0 must be 1");
    }
    if (attentionMaskType.getShape()[1] != 1 &&
        attentionMaskType.getShape()[1] != batchSize) {
      return emitOpError("Attention mask batch size must be 1 (broadcast) or "
                         "match query batch size");
    }
    if (attentionMaskType.getShape()[2] != 1 &&
        attentionMaskType.getShape()[2] != nQueryHeads) {
      return emitOpError("Attention mask num heads must be 1 (broadcast) or "
                         "match query num heads");
    }
    if (attentionMaskType.getShape()[3] != maxSeqLen) {
      return emitOpError("Attention mask sequence length must match key/value "
                         "sequence length");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PagedScaledDotProductAttentionDecodeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult
mlir::tt::ttir::PagedScaledDotProductAttentionDecodeOp::verify() {

  RankedTensorType queryType = getQuery().getType();
  RankedTensorType keyType = getKey().getType();
  RankedTensorType valueType = getValue().getType();
  RankedTensorType pageTableType = getPageTable().getType();

  auto queryShape = queryType.getShape();
  auto keyShape = keyType.getShape();
  auto pageTableShape = pageTableType.getShape();

  auto numUsers = queryShape[1];
  auto numQueryHeads = queryShape[2];
  auto headSize = queryShape[3];
  auto numKVHeads = keyShape[1];
  auto blockSize = keyShape[2];

  // Verify element types.
  if (queryType.getElementType() != keyType.getElementType() ||
      queryType.getElementType() != valueType.getElementType()) {
    return emitOpError(
        "Query, key, and value must have the same element type.");
  }

  if (!queryType.getElementType().isFloat()) {
    return emitOpError("Query, key, and value must be float tensors.");
  }

  if (!pageTableType.getElementType().isInteger()) {
    return emitOpError("Page table must be an integer tensor.");
  }

  // Verify key and value are identical shapes/dtypes
  if (keyType != valueType) {
    return emitOpError("Key and value must have the same shape and data type.");
  }

  // Verify ranks.
  if (queryType.getShape().size() != 4) {
    return emitOpError("Query must be a 4D tensor.");
  }
  if (keyType.getShape().size() != 4) {
    return emitOpError("Key/Value tensor must be a 4D tensor.");
  }
  if (pageTableType.getShape().size() != 2) {
    return emitOpError("Page table tensor must be a 2D tensor.");
  }

  // Verify shapes.
  if (headSize != keyShape[3]) {
    return emitOpError("Query head size must match key/value head size.");
  }
  if (numQueryHeads % numKVHeads != 0) {
    return emitOpError(
        "Query num heads must be divisible by key/value num heads.");
  }
  if (blockSize % 32 != 0) {
    return emitOpError("Block size must be divisible by 32.");
  }

  if (pageTableShape[0] != numUsers) {
    return emitOpError(
        "Page table number of users must match query number of users.");
  }

  bool isCausal = getIsCausal();
  if (isCausal) {
    if (getAttentionMask()) {
      return emitOpError(
          "Attention mask is not allowed when is_causal is true.");
    }
  } else {
    if (!getAttentionMask()) {
      return emitOpError("Attention mask is required when is_causal is false.");
    }
  }

  if (getCurPosTensor()) {
    if (!getCurPosTensor().getType().getElementType().isInteger()) {
      return emitOpError("Cur pos tensor must be an integer tensor.");
    }
    if (getCurPosTensor().getType().getShape().size() != 1) {
      return emitOpError("Cur pos tensor must be a 1D tensor.");
    }
    if (getCurPosTensor().getType().getShape()[0] != numUsers) {
      return emitOpError(
          "Cur pos tensor number of users must match query number of users.");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PagedFlashMultiLatentAttentionDecodeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult
mlir::tt::ttir::PagedFlashMultiLatentAttentionDecodeOp::verify() {

  RankedTensorType queryType = getQuery().getType();
  RankedTensorType keyType = getKey().getType();
  RankedTensorType pageTableType = getPageTable().getType();

  // Verify ranks.
  if (queryType.getShape().size() != 4) {
    return emitOpError("Query must be a 4D tensor.");
  }
  if (keyType.getShape().size() != 4) {
    return emitOpError("Key tensor must be a 4D tensor.");
  }
  if (pageTableType.getShape().size() != 2) {
    return emitOpError("Page table tensor must be a 2D tensor.");
  }

  // Verify element types.
  if (!queryType.getElementType().isFloat()) {
    return emitOpError("Query must be a float tensor.");
  }
  if (queryType.getElementType() != keyType.getElementType()) {
    return emitOpError("Query and key must have the same element type.");
  }
  if (!pageTableType.getElementType().isInteger()) {
    return emitOpError("Page table must be an integer tensor.");
  }

  // Verify value if present.
  if (getValue()) {
    RankedTensorType valueType = getValue().getType();
    if (valueType.getShape().size() != 4) {
      return emitOpError("Value tensor must be a 4D tensor.");
    }
    if (queryType.getElementType() != valueType.getElementType()) {
      return emitOpError("Query and value must have the same element type.");
    }
  }

  // head_dim_v must be > 0.
  if (getHeadDimV() == 0) {
    return emitOpError("head_dim_v must be greater than 0.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::ScaledDotProductAttentionOp::verify() {

  RankedTensorType queryType = getQuery().getType();
  RankedTensorType keyType = getKey().getType();
  RankedTensorType valueType = getValue().getType();
  RankedTensorType resultType = getResult().getType();

  if (queryType != resultType) {
    return emitOpError("Query and result must have the same type");
  }

  if (keyType != valueType) {
    return emitOpError("Key and value must have the same type");
  }
  if (queryType.getShape().size() != 4) {
    return emitOpError("Query must be a 4D tensor");
  }
  if (keyType.getShape().size() != 4) {
    return emitOpError("Key/Value must be a 4D tensor");
  }
  if (resultType.getShape().size() != 4) {
    return emitOpError("Output must be a 4D tensor");
  }

  int64_t batchSize = queryType.getShape()[0];
  int64_t nQueryHeads = queryType.getShape()[1];
  int64_t nKVHeads = keyType.getShape()[1];
  int64_t headSize = queryType.getShape()[3];
  int64_t seqLen = queryType.getShape()[2];
  int64_t maxSeqLen = keyType.getShape()[2];

  if (keyType.getShape()[0] != batchSize) {
    return emitOpError("Key/Value batch size must match query batch size");
  }

  if (keyType.getShape()[3] != headSize) {
    return emitOpError("Key/Value head size must match query head size");
  }

  if (nQueryHeads % nKVHeads != 0) {
    return emitOpError(
        "Query num heads must be divisible by key/value num heads");
  }

  if (getAttentionMask()) {
    if (getIsCausal()) {
      return emitOpError(
          "Attention mask is not allowed when is_causal is true");
    }
    RankedTensorType attentionMaskType = getAttentionMask().getType();
    if (attentionMaskType.getShape().size() != 4) {
      return emitOpError("Attention mask must be a 4D tensor");
    }
    if (attentionMaskType.getShape()[0] != 1 &&
        attentionMaskType.getShape()[0] != batchSize) {
      return emitOpError("Attention mask batch size must be 1 (broadcast) or "
                         "match query batch size");
    }
    if (attentionMaskType.getShape()[1] != 1 &&
        attentionMaskType.getShape()[1] != nQueryHeads) {
      return emitOpError("Attention mask dim 1 must be 1 (broadcast) or match "
                         "query num heads");
    }
    if (attentionMaskType.getShape()[2] != seqLen) {
      return emitOpError(
          "Attention mask at dim 2 must match query sequence length");
    }
    if (attentionMaskType.getShape()[3] != maxSeqLen) {
      return emitOpError("Attention mask at dim 3 must match key/value "
                         "sequence length (max sequence length)");
    }
  }

  if (getIsCausal()) {
    if (seqLen != maxSeqLen) {
      return emitOpError("Sequence length must match key/value sequence length "
                         "when is_causal is true");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalAvgPool2dOp
//===----------------------------------------------------------------------===//

// GlobalAvgPool2dOp verification
::mlir::LogicalResult mlir::tt::ttir::GlobalAvgPool2dOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType outputType = getResult().getType();

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  int64_t rank = inputType.getRank();
  if (rank < 2) {
    return emitOpError("input tensor must have at least 2 dimensions for "
                       "global average pooling over 2 spatial dimensions");
  }

  if (outputType.getRank() != rank) {
    return emitOpError("output tensor must have the same rank as input tensor");
  }

  if (inputShape[0] != outputShape[0]) {
    return emitOpError(
        "batch dimension must remain the same between input and output");
  }

  if (outputShape[rank - 2] != 1 || outputShape[rank - 3] != 1) {
    return emitOpError("spatial dimensions must be reduced to 1");
  }

  if (inputShape[rank - 1] != outputShape[rank - 1]) {
    return emitOpError(
        "channel dimension must remain the same between input and output");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GeluBackwardOp
//===----------------------------------------------------------------------===//

// GeluBackwardOp verification
::mlir::LogicalResult mlir::tt::ttir::GeluBackwardOp::verify() {
  llvm::StringRef approximate = getApproximate();

  if (approximate != "none" && approximate != "tanh") {
    return emitOpError("approximate attribute must be either 'none' or 'tanh', "
                       "but got '")
           << approximate << "'";
  }

  RankedTensorType lhsType = getLhs().getType();
  RankedTensorType rhsType = getRhs().getType();

  int64_t lhsRank = lhsType.getRank();
  int64_t rhsRank = rhsType.getRank();

  if (lhsRank < 2 || lhsRank > 4) {
    return emitOpError(
               "gradient tensor (lhs) must have rank 2, 3, or 4, but got rank ")
           << lhsRank;
  }

  if (rhsRank < 2 || rhsRank > 4) {
    return emitOpError(
               "input tensor (rhs) must have rank 2, 3, or 4, but got rank ")
           << rhsRank;
  }

  if (lhsRank != rhsRank) {
    return emitOpError("gradient tensor (lhs) and input tensor (rhs) must have "
                       "the same rank, "
                       "but got lhs rank ")
           << lhsRank << " and rhs rank " << rhsRank;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SparseMatmulOp
//===----------------------------------------------------------------------===//

// SparseMatmulOp verification
::mlir::LogicalResult mlir::tt::ttir::SparseMatmulOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  ::mlir::RankedTensorType sparsityType = getSparsity().getType();
  ::mlir::RankedTensorType outputType = getType();

  // Verify that input A is at least 4D tensor
  if (inputAType.getRank() < 4) {
    return emitOpError("Input A must be at least a 4D tensor");
  }

  // Verify that input B is exactly 4D tensor [1, E, K, N]
  if (inputBType.getRank() != 4) {
    return emitOpError("Input B must be a 4D tensor [1, E, K, N]");
  }

  // Verify that sparsity is a 4D tensor
  if (sparsityType.getRank() != 4) {
    return emitOpError("Sparsity tensor must be a 4D tensor");
  }

  bool isInputASparse = getIsInputASparse();
  bool isInputBSparse = getIsInputBSparse();

  // Verify that at least one input is sparse
  if (!isInputASparse && !isInputBSparse) {
    return emitOpError("At least one of is_input_a_sparse or is_input_b_sparse "
                       "must be true");
  }

  // Get dimensions
  llvm::ArrayRef<int64_t> aShape = inputAType.getShape();
  llvm::ArrayRef<int64_t> bShape = inputBType.getShape();
  int64_t E = bShape[1];
  int64_t K = bShape[2];
  int64_t N = bShape[3];
  int64_t M = aShape[aShape.size() - 2];

  // Verify input B first dimension is 1
  if (bShape[0] != 1) {
    return emitOpError("Input B first dimension must be 1");
  }

  // Verify inner dimensions match
  if (aShape[aShape.size() - 1] != K) {
    return emitOpError(
        "Input A inner dimension must match Input B K dimension");
  }

  // Verify sparsity tensor has correct number of experts
  if (sparsityType.getShape()[sparsityType.getRank() - 1] != E) {
    return emitOpError("Sparsity tensor last dimension must match number of "
                       "experts in Input B");
  }

  // Verify output shape based on sparse mode (including batch dimensions)
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  if (isInputASparse && isInputBSparse) {
    // [1, E, M, K] @ [1, E, K, N] -> [1, E, M, N]
    if (outputShape.size() != 4) {
      return emitOpError("Output must be 4D for sparse-sparse mode");
    }
    if (outputShape[0] != 1 || outputShape[1] != E || outputShape[2] != M ||
        outputShape[3] != N) {
      return emitOpError(
          "Output shape must be [1, E, M, N] for sparse-sparse mode");
    }
  } else if (!isInputASparse && isInputBSparse) {
    // [A, B, M, K] @ [1, E, K, N] -> [A, B, 1, E, M, N]
    if (outputShape.size() != 6) {
      return emitOpError("Output must be 6D for dense-sparse mode");
    }
    int64_t A = aShape[0];
    int64_t B = aShape[1];
    if (outputShape[0] != A || outputShape[1] != B || outputShape[2] != 1 ||
        outputShape[3] != E || outputShape[4] != M || outputShape[5] != N) {
      return emitOpError(
          "Output shape must be [A, B, 1, E, M, N] for dense-sparse mode");
    }
  } else if (isInputASparse && !isInputBSparse) {
    // [A, E, M, K] @ [1, E, K, N] -> [A, E, M, N]
    if (outputShape.size() != 4) {
      return emitOpError("Output must be 4D for sparse-dense mode");
    }
    int64_t A = aShape[0];
    if (outputShape[0] != A || outputShape[1] != E || outputShape[2] != M ||
        outputShape[3] != N) {
      return emitOpError(
          "Output shape must be [A, E, M, N] for sparse-dense mode");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllToAllDispatchOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::AllToAllDispatchOp::verify() {
  ::mlir::RankedTensorType inputType = getInputTensor().getType();
  ::mlir::RankedTensorType indicesType = getExpertIndices().getType();
  ::mlir::RankedTensorType mappingType = getExpertMapping().getType();
  ::mlir::RankedTensorType dispatchedType = getDispatched().getType();
  ::mlir::RankedTensorType metadataType = getMetadata().getType();

  // All tensors must be 4D
  if (inputType.getRank() != 4) {
    return emitOpError("input_tensor must be a 4D tensor [B, S, 1, H]");
  }
  if (indicesType.getRank() != 4) {
    return emitOpError("expert_indices must be a 4D tensor [B, S, 1, K]");
  }
  if (mappingType.getRank() != 4) {
    return emitOpError("expert_mapping must be a 4D tensor [1, 1, E, D]");
  }
  if (dispatchedType.getRank() != 4) {
    return emitOpError("dispatched output must be a 4D tensor [1, B*D, S, H]");
  }
  if (metadataType.getRank() != 4) {
    return emitOpError("metadata output must be a 4D tensor [1, B*D, S, K]");
  }

  // Verify num_devices > 0
  if (getNumDevices() <= 0) {
    return emitOpError("num_devices must be positive");
  }

  // Verify cluster_axis is 0 or 1.
  int64_t clusterAxis = static_cast<int64_t>(getClusterAxis());
  if (clusterAxis < 0 || clusterAxis > 1) {
    return emitOpError("cluster_axis must be 0 or 1");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllToAllDispatchMetadataOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::AllToAllDispatchMetadataOp::verify() {
  ::mlir::RankedTensorType inputType = getInputTensor().getType();
  ::mlir::RankedTensorType indicesType = getExpertIndices().getType();
  ::mlir::RankedTensorType scoresType = getExpertScores().getType();
  ::mlir::RankedTensorType mappingType = getExpertMapping().getType();
  ::mlir::RankedTensorType dispatchedType = getDispatched().getType();
  ::mlir::RankedTensorType indicesOutType = getIndices().getType();
  ::mlir::RankedTensorType scoresOutType = getScores().getType();

  // Inputs must be 4D
  if (inputType.getRank() != 4) {
    return emitOpError("input_tensor must be a 4D tensor [1, 1, M, H]");
  }
  if (indicesType.getRank() != 4) {
    return emitOpError("expert_indices must be a 4D tensor [1, 1, M, K]");
  }
  if (scoresType.getRank() != 4) {
    return emitOpError("expert_scores must be a 4D tensor [1, 1, M, K]");
  }
  if (mappingType.getRank() != 4) {
    return emitOpError("expert_mapping must be a 4D tensor [1, 1, D, E]");
  }
  // Outputs are 3D matching the metal kernel output shapes
  if (dispatchedType.getRank() != 3) {
    return emitOpError(
        "dispatched output must be a 3D tensor [1, tokens_global, H]");
  }
  if (indicesOutType.getRank() != 3) {
    return emitOpError(
        "indices output must be a 3D tensor [1, tokens_global, K]");
  }
  if (scoresOutType.getRank() != 3) {
    return emitOpError(
        "scores output must be a 3D tensor [1, tokens_global, K]");
  }

  // Verify num_devices > 0
  if (getNumDevices() <= 0) {
    return emitOpError("num_devices must be positive");
  }

  // Verify cluster_axis is 0 or 1.
  int64_t clusterAxis = static_cast<int64_t>(getClusterAxis());
  if (clusterAxis < 0 || clusterAxis > 1) {
    return emitOpError("cluster_axis must be 0 or 1");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllToAllCombineOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::AllToAllCombineOp::verify() {
  ::mlir::RankedTensorType inputType = getInputTensor().getType();
  ::mlir::RankedTensorType metadataType = getExpertMetadata().getType();
  ::mlir::RankedTensorType mappingType = getExpertMapping().getType();
  ::mlir::RankedTensorType resultType = getResult().getType();

  // All tensors must be 4D
  if (inputType.getRank() != 4) {
    return emitOpError("input_tensor must be a 4D tensor [E_local, B*D, S, H]");
  }
  if (metadataType.getRank() != 4) {
    return emitOpError("expert_metadata must be a 4D tensor [1, B*D, S, K]");
  }
  if (mappingType.getRank() != 4) {
    return emitOpError("expert_mapping must be a 4D tensor [1, 1, E, D]");
  }
  if (resultType.getRank() != 4) {
    return emitOpError("result must be a 4D tensor [K, B, S, H]");
  }

  // Verify num_devices > 0
  if (getNumDevices() <= 0) {
    return emitOpError("num_devices must be positive");
  }

  // Verify cluster_axis is 0 or 1.
  int64_t clusterAxis = static_cast<int64_t>(getClusterAxis());
  if (clusterAxis < 0 || clusterAxis > 1) {
    return emitOpError("cluster_axis must be 0 or 1");
  }

  // Verify num_experts_per_tok > 0
  if (getNumExpertsPerTok() <= 0) {
    return emitOpError("num_experts_per_tok must be positive");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SelectiveReduceCombineOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::SelectiveReduceCombineOp::verify() {
  // Verify select_experts_k > 0
  if (getSelectExpertsK() == 0) {
    return emitOpError("select_experts_k must be positive");
  }

  // Verify experts > 0
  if (getExperts() == 0) {
    return emitOpError("experts must be positive");
  }

  // Verify hidden_size > 0
  if (getHiddenSize() == 0) {
    return emitOpError("hidden_size must be positive");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MoeExpertTokenRemapOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::MoeExpertTokenRemapOp::verify() {
  ::mlir::RankedTensorType topkType = getTopkTensor().getType();
  ::mlir::RankedTensorType mappingInputType = getExpertMapping().getType();
  ::mlir::RankedTensorType metadataType = getExpertMetadata().getType();
  ::mlir::RankedTensorType mappingOutputType = getMapping().getType();
  ::mlir::RankedTensorType reducedType = getReduced().getType();

  if (topkType.getRank() != 4) {
    return emitOpError("topk_tensor must be a 4D tensor [D, B, S, E]");
  }
  if (mappingInputType.getRank() != 4) {
    return emitOpError("expert_mapping must be a 4D tensor [1, 1, E, D]");
  }
  if (metadataType.getRank() != 4) {
    return emitOpError("expert_metadata must be a 4D tensor [D, B, S, K]");
  }
  if (mappingOutputType.getRank() != 4) {
    return emitOpError("mapping output must be a 4D tensor [1, B, S, E_local]");
  }
  if (reducedType.getRank() != 4) {
    return emitOpError(
        "reduced output must be a 4D tensor [1, 1, ceil(B*S/R), E_local]");
  }

  if (getReductionSize() <= 0) {
    return emitOpError("reduction_size must be positive");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::AbsOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnary(
      *this, adaptor.getInput(),
      [](const llvm::APFloat &x) { return llvm::abs(x); },
      [](const llvm::APInt &x) { return x.abs(); });
}

//===----------------------------------------------------------------------===//
// AtanOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::AtanOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                       ApplyToAPFloat(::atanf));
}

//===----------------------------------------------------------------------===//
// BitwiseNotOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::BitwiseNotOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryInt(*this, adaptor.getInput(),
                                     [](const llvm::APInt &x) { return ~x; });
}

//===----------------------------------------------------------------------===//
// CbrtOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::CbrtOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                       ApplyToAPFloat(::cbrtf));
}

//===----------------------------------------------------------------------===//
// CeilOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::CeilOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                       ApplyToAPFloat(::ceilf));
}

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::CosOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                       ApplyToAPFloat(::cosf));
}

//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::ExpOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                       ApplyToAPFloat(::expf));
}

//===----------------------------------------------------------------------===//
// Expm1Op
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::Expm1Op::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                       ApplyToAPFloat(::expm1f));
}

//===----------------------------------------------------------------------===//
// FloorOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::FloorOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                       ApplyToAPFloat(::floorf));
}

//===----------------------------------------------------------------------===//
// IsFiniteOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::IsFiniteOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(
      *this, adaptor.getInput(), [](const llvm::APFloat &x) {
        return x.isFinite() ? llvm::APFloat::getOne(x.getSemantics())
                            : llvm::APFloat::getZero(x.getSemantics());
      });
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::LogOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                       ApplyToAPFloat(::logf));
}

//===----------------------------------------------------------------------===//
// Log1pOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::Log1pOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                       ApplyToAPFloat(::log1pf));
}

//===----------------------------------------------------------------------===//
// LogicalNotOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::LogicalNotOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnary(
      *this, adaptor.getInput(),
      [](const llvm::APFloat &x) {
        return llvm::APFloat(x.getSemantics(), x.isZero() ? 1 : 0);
      },
      [](const llvm::APInt &x) {
        return llvm::APInt(x.getBitWidth(), x.isZero() ? 1 : 0,
                           /*isSigned=*/false);
      });
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

// NegOp folder
::mlir::OpFoldResult mlir::tt::ttir::NegOp::fold(FoldAdaptor adaptor) {
  auto neg = std::negate<>();
  return constantFoldEltwiseUnary(*this, adaptor.getInput(), neg, neg);
}

//===----------------------------------------------------------------------===//
// ReciprocalOp
//===----------------------------------------------------------------------===//

static bool noneZero(mlir::Attribute attr) {
  if (auto elementsAttr = mlir::dyn_cast_if_present<mlir::ElementsAttr>(attr)) {
    if (elementsAttr.getElementType().isFloat()) {
      if (elementsAttr.isSplat()) {
        return !elementsAttr.getSplatValue<llvm::APFloat>().isZero();
      }
      return llvm::none_of(elementsAttr.getValues<llvm::APFloat>(),
                           std::mem_fn(&llvm::APFloat::isZero));
    }
    if (elementsAttr.getElementType().isInteger()) {
      if (elementsAttr.isSplat()) {
        return !elementsAttr.getSplatValue<llvm::APInt>().isZero();
      }
      return llvm::none_of(elementsAttr.getValues<llvm::APInt>(),
                           std::mem_fn(&llvm::APInt::isZero));
    }
  }
  return false;
}

::mlir::OpFoldResult mlir::tt::ttir::ReciprocalOp::fold(FoldAdaptor adaptor) {
  if (!noneZero(adaptor.getInput())) {
    // Don't fold if the result is inf because runtime doesn't support it fully.
    return nullptr;
  }
  return constantFoldEltwiseUnaryFloat(
      *this, adaptor.getInput(), [](const llvm::APFloat &x) {
        return llvm::APFloat::getOne(x.getSemantics()) / x;
      });
}

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::RsqrtOp::fold(FoldAdaptor adaptor) {
  if (!noneZero(adaptor.getInput())) {
    // Don't fold if the result is inf because runtime doesn't support it fully.
    return nullptr;
  }
  auto rsqrt = ApplyToAPFloat([](float x) { return 1.0f / ::sqrtf(x); });
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(), rsqrt);
}

//===----------------------------------------------------------------------===//
// SignOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::SignOp::fold(FoldAdaptor adaptor) {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(getType().getElementType());
  bool isUnsigned = intType && intType.isUnsignedInteger();
  return constantFoldEltwiseUnary(
      *this, adaptor.getInput(),
      [](const llvm::APFloat &x) {
        if (x.isZero()) {
          return llvm::APFloat::getZero(x.getSemantics());
        }
        if (x.isNegative()) {
          return llvm::APFloat::getOne(x.getSemantics(), /*Negative=*/true);
        }
        if (x.isNaN()) {
          return x;
        }
        // x is positive.
        return llvm::APFloat::getOne(x.getSemantics());
      },
      [isUnsigned](const llvm::APInt &x) {
        if (x.isZero()) {
          return llvm::APInt::getZero(x.getBitWidth());
        }
        if (isUnsigned || x.isStrictlyPositive()) {
          return llvm::APInt(x.getBitWidth(), 1);
        }
        return llvm::APInt(x.getBitWidth(), -1, true);
      });
}

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::SinOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                       ApplyToAPFloat(::sinf));
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::SqrtOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                       ApplyToAPFloat(::sqrtf));
}

//===----------------------------------------------------------------------===//
// TanOp
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::tt::ttir::TanOp::fold(FoldAdaptor adaptor) {
  return constantFoldEltwiseUnaryFloat(*this, adaptor.getInput(),
                                       ApplyToAPFloat(::tanf));
}

} // namespace mlir::tt::ttir
