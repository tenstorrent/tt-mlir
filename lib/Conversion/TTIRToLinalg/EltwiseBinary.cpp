// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/EltwiseBinary.h"
#include "ttmlir/Conversion/TTIRToLinalg/Utils.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>
#include <cstdint>

namespace mlir::tt::ttir_to_linalg {

// Conversion patterns for TTIR binary elementwise ops are organized by
// implementation strategy, in order of preference:
//
// 1. TOSA 1:1        — Direct mapping to a single TOSA op.
//                      Preferred when a TOSA equivalent exists. TOSA ops
//                      handle broadcasting natively.
// 2. linalg.generic + arith/math — A linalg.generic body containing scalar
//                      arith or math ops. Broadcasting is handled implicitly
//                      through affine indexing maps, avoiding materialization
//                      of intermediate broadcast buffers.
// 3. Custom          — Multi-op sequences in TOSA, linalg, or arith dialects.
//                      Used for compound operations (e.g. gelu_bw, remainder).

//===----------------------------------------------------------------------===//
// TOSA Binary Conversion Patterns
//===----------------------------------------------------------------------===//

// TOSA binary ops support implicit broadcasting natively, so operands are
// passed directly without explicit broadcast materialization.
namespace {
template <typename TTIROpTy, typename TosaOpTy>
class ElementwiseBinaryOpToTosaPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    rewriter.replaceOpWithNewOp<TosaOpTy>(
        op, resultType, ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    return success();
  }
};
} // namespace

// Direct comparison operations (where TTIR and TOSA ops match directly).
namespace {
template <typename TTIROpTy, typename TosaOpTy>
class DirectComparisonOpToTosaPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    Location loc = op.getLoc();
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));
    auto boolResult = TosaOpTy::create(rewriter, loc, boolType,
                                       adaptor.getLhs(), adaptor.getRhs());

    rewriter.replaceOpWithNewOp<tosa::CastOp>(op, resultType, boolResult);
    return success();
  }
};

// Swapped comparison operations (where TTIR and TOSA ops have swapped operands
// e.g. ttir.lt must use inverted tosa.greater).
template <typename TTIROpTy, typename TosaOpTy>
class SwappedComparisonOpToTosaPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    Location loc = op.getLoc();
    // Swapped operands: rhs, lhs.
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));
    auto boolResult = TosaOpTy::create(rewriter, loc, boolType,
                                       adaptor.getRhs(), adaptor.getLhs());

    rewriter.replaceOpWithNewOp<tosa::CastOp>(op, resultType, boolResult);
    return success();
  }
};

// Negated comparison operations (where TTIR op is the negation of a TOSA op,
// e.g. ttir.not_equal).
template <typename TTIROpTy, typename TosaOpTy>
class NegatedComparisonOpToTosaPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    Location loc = op.getLoc();
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));
    auto boolResult = TosaOpTy::create(rewriter, loc, boolType,
                                       adaptor.getLhs(), adaptor.getRhs());
    auto notResult =
        tosa::LogicalNotOp::create(rewriter, loc, boolType, boolResult);

    rewriter.replaceOpWithNewOp<tosa::CastOp>(op, resultType, notResult);
    return success();
  }
};
} // namespace

// Logical binary operations pattern (LogicalAnd, LogicalOr, LogicalXor).
// These operations:
// 1. Convert inputs to boolean (non-zero = true)
// 2. Apply the TOSA logical operation (implicit broadcasting)
// 3. Convert boolean result back to original type
namespace {
template <typename TTIROpTy, typename TosaOpTy>
class LogicalBinaryOpToTosaPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    Location loc = op.getLoc();

    // Convert both inputs to boolean tensors.
    Value boolLhs = convertToBooleanTensor(adaptor.getLhs(), loc, rewriter);
    Value boolRhs = convertToBooleanTensor(adaptor.getRhs(), loc, rewriter);

    // Get the boolean type for the result.
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));

    // Apply the logical operation (TOSA handles broadcasting).
    auto logicalResult =
        TosaOpTy::create(rewriter, loc, boolType, boolLhs, boolRhs);

    // Convert boolean result back to original type.
    rewriter.replaceOpWithNewOp<tosa::CastOp>(op, resultType, logicalResult);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Linalg Generic + Math/Arith Dialect Patterns
//===----------------------------------------------------------------------===//

// Build a broadcast-aware affine map for an operand. The map has as many
// result expressions as the operand rank. Handles rank extension (fewer dims
// than result — leading dims are dropped) and size-1 broadcasting (maps to 0).
static AffineMap buildBroadcastAffineMap(ArrayRef<int64_t> operandShape,
                                         ArrayRef<int64_t> resultShape,
                                         MLIRContext *ctx) {
  int64_t resultRank = resultShape.size();
  int64_t operandRank = operandShape.size();
  int64_t rankDiff = resultRank - operandRank;

  // Only emit expressions for the operand's own dimensions,
  // aligned to the trailing dimensions of the result.
  SmallVector<AffineExpr> exprs;
  for (int64_t i = 0; i < operandRank; ++i) {
    int64_t resultDim = i + rankDiff;
    if (operandShape[i] == 1 && resultShape[resultDim] != 1) {
      exprs.push_back(getAffineConstantExpr(0, ctx));
    } else {
      exprs.push_back(getAffineDimExpr(resultDim, ctx));
    }
  }
  return AffineMap::get(resultRank, 0, exprs, ctx);
}

// Base class for TTIR binary ops lowered via linalg.generic. Subclasses only
// need to implement buildBody() to emit the scalar computation.
// Broadcasting is handled implicitly through affine indexing maps, avoiding
// materialization of intermediate broadcast buffers.
namespace {
template <typename TTIROpTy>
class ElementwiseBinaryOpToLinalgGenericPatternBase
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());

    int64_t rank = resultType.getRank();
    MLIRContext *ctx = rewriter.getContext();

    AffineMap lhsMap =
        buildBroadcastAffineMap(lhsType.getShape(), resultType.getShape(), ctx);
    AffineMap rhsMap =
        buildBroadcastAffineMap(rhsType.getShape(), resultType.getShape(), ctx);
    AffineMap resultMap = AffineMap::getMultiDimIdentityMap(rank, ctx);

    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType());

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, resultType, ValueRange{lhs, rhs},
        ValueRange{emptyTensor},
        SmallVector<AffineMap>{lhsMap, rhsMap, resultMap}, iteratorTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result = buildBody(b, nestedLoc, args, resultType);
          linalg::YieldOp::create(b, nestedLoc, result);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }

protected:
  virtual Value buildBody(OpBuilder &b, Location loc, ValueRange args,
                          RankedTensorType resultType) const = 0;
};
} // namespace

// Template for TTIR ops that map 1:1 to a math dialect op via linalg.generic.
namespace {
template <typename TTIROpTy, typename MathOpTy>
class ElementwiseBinaryOpToMathPattern
    : public ElementwiseBinaryOpToLinalgGenericPatternBase<TTIROpTy> {
public:
  using ElementwiseBinaryOpToLinalgGenericPatternBase<
      TTIROpTy>::ElementwiseBinaryOpToLinalgGenericPatternBase;

protected:
  Value buildBody(OpBuilder &b, Location loc, ValueRange args,
                  RankedTensorType /*resultType*/) const override {
    return MathOpTy::create(b, loc, args[0], args[1]);
  }
};
} // namespace

// Template for TTIR ops that map to arith dialect ops (float/int variants).
namespace {
template <typename TTIROpTy, typename ArithFOpTy, typename ArithIOpTy>
class ElementwiseBinaryOpToArithPattern
    : public ElementwiseBinaryOpToLinalgGenericPatternBase<TTIROpTy> {
public:
  using ElementwiseBinaryOpToLinalgGenericPatternBase<
      TTIROpTy>::ElementwiseBinaryOpToLinalgGenericPatternBase;

protected:
  Value buildBody(OpBuilder &b, Location loc, ValueRange args,
                  RankedTensorType /*resultType*/) const override {
    if (isa<FloatType>(args[0].getType())) {
      return ArithFOpTy::create(b, loc, args[0], args[1]);
    }
    return ArithIOpTy::create(b, loc, args[0], args[1]);
  }
};
} // namespace

// Remainder: Python-style modulo (sign follows divisor).
// arith.remf/remsi produce C-style remainder (sign follows dividend).
// To get Python semantics: rem = a % b; result = (rem != 0 && sign differs) ?
// rem + b : rem.
namespace {
class RemainderOpToLinalgGenericPattern
    : public ElementwiseBinaryOpToLinalgGenericPatternBase<ttir::RemainderOp> {
public:
  using ElementwiseBinaryOpToLinalgGenericPatternBase::
      ElementwiseBinaryOpToLinalgGenericPatternBase;

protected:
  Value buildBody(OpBuilder &b, Location loc, ValueRange args,
                  RankedTensorType resultType) const override {
    Value lhs = args[0];
    Value rhs = args[1];

    if (isa<FloatType>(resultType.getElementType())) {
      // Python-style float modulo: a - floor(a / b) * b
      Value div = arith::DivFOp::create(b, loc, lhs, rhs);
      Value floored = math::FloorOp::create(b, loc, div);
      Value prod = arith::MulFOp::create(b, loc, floored, rhs);
      return arith::SubFOp::create(b, loc, lhs, prod);
    }

    // Python-style integer modulo: adjust C remainder when signs differ.
    Value rem = arith::RemSIOp::create(b, loc, lhs, rhs);
    Value zero = arith::ConstantOp::create(
        b, loc, b.getIntegerAttr(resultType.getElementType(), 0));
    Value sum = arith::AddIOp::create(b, loc, rem, rhs);
    Value remNeZero =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, rem, zero);
    Value xorVal = arith::XOrIOp::create(b, loc, rem, rhs);
    Value signsDiffer =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, xorVal, zero);
    Value needAdjust = arith::AndIOp::create(b, loc, remNeZero, signsDiffer);
    return arith::SelectOp::create(b, loc, needAdjust, sum, rem);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Compound Binary Patterns (multi-op TOSA sequences)
//===----------------------------------------------------------------------===//

// GeluBackward: gelu_bw(grad, x) for both exact and tanh approximations.
namespace {
class GeluBackwardOpToTosaPattern
    : public OpConversionPattern<ttir::GeluBackwardOp> {
public:
  using OpConversionPattern<ttir::GeluBackwardOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GeluBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value grad = adaptor.getLhs();
    Value x = adaptor.getRhs();
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    Location loc = op.getLoc();

    // Compound patterns use resultType for all intermediate ops, so operands
    // must be pre-broadcast to the result shape.
    grad = broadcastToShape(grad, resultType.getShape(), loc, rewriter);
    x = broadcastToShape(x, resultType.getShape(), loc, rewriter);

    auto approximate = op.getApproximate();

    if (approximate == "tanh") {
      return matchAndRewriteTanh(grad, x, resultType, loc, op, rewriter);
    }
    return matchAndRewriteExact(grad, x, resultType, loc, op, rewriter);
  }

private:
  // gelu_bw(grad, x) = grad * (cdf(x) + x * pdf(x))
  // where cdf(x) = 0.5 * (1 + erf(x / sqrt(2)))
  //       pdf(x) = exp(-x^2/2) / sqrt(2*pi)
  LogicalResult
  matchAndRewriteExact(Value grad, Value x, RankedTensorType resultType,
                       Location loc, ttir::GeluBackwardOp op,
                       ConversionPatternRewriter &rewriter) const {
    auto elemTy = resultType.getElementType();
    int64_t rank = resultType.getRank();
    Value shift = createTosaMulShift(rewriter, loc);

    Value half = createTosaConst(rewriter, loc, elemTy, rank, 0.5);
    Value one = createTosaConst(rewriter, loc, elemTy, rank, 1.0);
    Value invSqrt2 = createTosaConst(rewriter, loc, elemTy, rank, M_SQRT1_2);
    Value negHalf = createTosaConst(rewriter, loc, elemTy, rank, -0.5);
    Value invSqrt2Pi = createTosaConst(rewriter, loc, elemTy, rank,
                                       1.0 / std::sqrt(2.0 * M_PI));

    // cdf = 0.5 * (1 + erf(x * invSqrt2))
    Value xScaled =
        tosa::MulOp::create(rewriter, loc, resultType, x, invSqrt2, shift);
    Value erfVal = tosa::ErfOp::create(rewriter, loc, resultType, xScaled);
    Value onePlusErf =
        tosa::AddOp::create(rewriter, loc, resultType, one, erfVal);
    Value cdf =
        tosa::MulOp::create(rewriter, loc, resultType, half, onePlusErf, shift);

    // pdf = exp(-x^2/2) / sqrt(2*pi)
    Value xSq = tosa::MulOp::create(rewriter, loc, resultType, x, x, shift);
    Value negHalfXSq =
        tosa::MulOp::create(rewriter, loc, resultType, negHalf, xSq, shift);
    Value expVal = tosa::ExpOp::create(rewriter, loc, resultType, negHalfXSq);
    Value pdf = tosa::MulOp::create(rewriter, loc, resultType, invSqrt2Pi,
                                    expVal, shift);

    // result = grad * (cdf + x * pdf)
    Value xTimesPdf =
        tosa::MulOp::create(rewriter, loc, resultType, x, pdf, shift);
    Value cdfPlusXPdf =
        tosa::AddOp::create(rewriter, loc, resultType, cdf, xTimesPdf);
    rewriter.replaceOpWithNewOp<tosa::MulOp>(op, resultType, grad, cdfPlusXPdf,
                                             shift);
    return success();
  }

  // gelu_bw with tanh approximation:
  // k = sqrt(2/pi), a = 0.044715
  // inner = k * (x + a * x^3)
  // tanh_val = tanh(inner)
  // left = 0.5 * (1 + tanh_val)
  // right = 0.5 * x * (1 - tanh_val^2) * k * (1 + 3*a*x^2)
  // gelu_bw = grad * (left + right)
  LogicalResult matchAndRewriteTanh(Value grad, Value x,
                                    RankedTensorType resultType, Location loc,
                                    ttir::GeluBackwardOp op,
                                    ConversionPatternRewriter &rewriter) const {
    auto elemTy = resultType.getElementType();
    int64_t rank = resultType.getRank();
    Value shift = createTosaMulShift(rewriter, loc);

    Value k =
        createTosaConst(rewriter, loc, elemTy, rank, std::sqrt(2.0 / M_PI));
    Value a = createTosaConst(rewriter, loc, elemTy, rank, 0.044715);
    Value threeA = createTosaConst(rewriter, loc, elemTy, rank, 3.0 * 0.044715);
    Value half = createTosaConst(rewriter, loc, elemTy, rank, 0.5);
    Value one = createTosaConst(rewriter, loc, elemTy, rank, 1.0);

    // x^2
    Value xSq = tosa::MulOp::create(rewriter, loc, resultType, x, x, shift);
    // a * x^2 * x = a * x^3
    Value aXSq = tosa::MulOp::create(rewriter, loc, resultType, a, xSq, shift);
    Value aXCub =
        tosa::MulOp::create(rewriter, loc, resultType, aXSq, x, shift);
    // inner = k * (x + a*x^3)
    Value xPlusAXCub = tosa::AddOp::create(rewriter, loc, resultType, x, aXCub);
    Value inner =
        tosa::MulOp::create(rewriter, loc, resultType, k, xPlusAXCub, shift);

    // tanh_val = tanh(inner)
    Value tanhVal = tosa::TanhOp::create(rewriter, loc, resultType, inner);

    // sech^2 = 1 - tanh^2
    Value tanhSq =
        tosa::MulOp::create(rewriter, loc, resultType, tanhVal, tanhVal, shift);
    Value negTanhSq = tosa::NegateOp::create(rewriter, loc, resultType, tanhSq);
    Value sechSq =
        tosa::AddOp::create(rewriter, loc, resultType, one, negTanhSq);

    // left = 0.5 * (1 + tanh_val)
    Value onePlusTanh =
        tosa::AddOp::create(rewriter, loc, resultType, one, tanhVal);
    Value left = tosa::MulOp::create(rewriter, loc, resultType, half,
                                     onePlusTanh, shift);

    // right = 0.5 * x * sech^2 * k * (1 + 3*a*x^2)
    Value threeAXSq =
        tosa::MulOp::create(rewriter, loc, resultType, threeA, xSq, shift);
    Value onePlus3AXSq =
        tosa::AddOp::create(rewriter, loc, resultType, one, threeAXSq);
    Value sechK =
        tosa::MulOp::create(rewriter, loc, resultType, sechSq, k, shift);
    Value sechKTerm = tosa::MulOp::create(rewriter, loc, resultType, sechK,
                                          onePlus3AXSq, shift);
    Value xTerm =
        tosa::MulOp::create(rewriter, loc, resultType, x, sechKTerm, shift);
    Value right =
        tosa::MulOp::create(rewriter, loc, resultType, half, xTerm, shift);

    // gelu_bw = grad * (left + right)
    Value leftPlusRight =
        tosa::AddOp::create(rewriter, loc, resultType, left, right);
    rewriter.replaceOpWithNewOp<tosa::MulOp>(op, resultType, grad,
                                             leftPlusRight, shift);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateTTIRToLinalgEltwiseBinaryPatterns(MLIRContext *ctx,
                                               RewritePatternSet &patterns,
                                               TypeConverter &typeConverter) {
  // linalg.generic + arith ops
  patterns.add<ElementwiseBinaryOpToArithPattern<ttir::MultiplyOp,
                                                 arith::MulFOp, arith::MulIOp>,
               ElementwiseBinaryOpToArithPattern<ttir::DivOp, arith::DivFOp,
                                                 arith::DivSIOp>>(typeConverter,
                                                                  ctx);

  // linalg.generic + math dialect ops
  patterns.add<ElementwiseBinaryOpToMathPattern<ttir::Atan2Op, math::Atan2Op>>(
      typeConverter, ctx);

  // Custom linalg-based patterns
  patterns.add<RemainderOpToLinalgGenericPattern>(typeConverter, ctx);
}

void populateTTIRToTosaEltwiseBinaryPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter) {
  // Elementwise binary operations (1:1 TOSA mappings)
  patterns.add<
      ElementwiseBinaryOpToTosaPattern<ttir::AddOp, tosa::AddOp>,
      ElementwiseBinaryOpToTosaPattern<ttir::SubtractOp, tosa::SubOp>,
      ElementwiseBinaryOpToTosaPattern<ttir::PowOp, tosa::PowOp>,
      ElementwiseBinaryOpToTosaPattern<ttir::MinimumOp, tosa::MinimumOp>,
      ElementwiseBinaryOpToTosaPattern<ttir::MaximumOp, tosa::MaximumOp>,
      ElementwiseBinaryOpToTosaPattern<ttir::BitwiseAndOp, tosa::BitwiseAndOp>,
      ElementwiseBinaryOpToTosaPattern<ttir::BitwiseOrOp, tosa::BitwiseOrOp>,
      ElementwiseBinaryOpToTosaPattern<ttir::BitwiseXorOp, tosa::BitwiseXorOp>,
      ElementwiseBinaryOpToTosaPattern<ttir::LogicalLeftShiftOp,
                                       tosa::LogicalLeftShiftOp>,
      ElementwiseBinaryOpToTosaPattern<ttir::LogicalRightShiftOp,
                                       tosa::LogicalRightShiftOp>>(
      typeConverter, ctx);

  // Comparison operations
  patterns.add<
      DirectComparisonOpToTosaPattern<ttir::EqualOp, tosa::EqualOp>,
      DirectComparisonOpToTosaPattern<ttir::GreaterThanOp, tosa::GreaterOp>,
      DirectComparisonOpToTosaPattern<ttir::GreaterEqualOp,
                                      tosa::GreaterEqualOp>,
      SwappedComparisonOpToTosaPattern<ttir::LessThanOp, tosa::GreaterOp>,
      SwappedComparisonOpToTosaPattern<ttir::LessEqualOp, tosa::GreaterEqualOp>,
      NegatedComparisonOpToTosaPattern<ttir::NotEqualOp, tosa::EqualOp>>(
      typeConverter, ctx);

  // Logical binary operations
  patterns.add<
      LogicalBinaryOpToTosaPattern<ttir::LogicalAndOp, tosa::LogicalAndOp>,
      LogicalBinaryOpToTosaPattern<ttir::LogicalOrOp, tosa::LogicalOrOp>,
      LogicalBinaryOpToTosaPattern<ttir::LogicalXorOp, tosa::LogicalXorOp>>(
      typeConverter, ctx);

  // Compound binary operations (multi-op TOSA sequences)
  patterns.add<GeluBackwardOpToTosaPattern>(typeConverter, ctx);
}

} // namespace mlir::tt::ttir_to_linalg
