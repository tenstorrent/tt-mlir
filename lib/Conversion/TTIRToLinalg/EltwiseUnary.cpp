// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/EltwiseUnary.h"
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

#include <cstdint>

namespace mlir::tt::ttir_to_linalg {

// Conversion patterns for TTIR unary elementwise ops are organized by
// implementation strategy, in order of preference:
//
// 1. TOSA 1:1        — Direct mapping to a single TOSA op.
//                      Preferred when a TOSA equivalent exists.
// 2. Named linalg    — Direct mapping to a named linalg op (e.g. linalg.sqrt).
//                      Used when no TOSA equivalent exists but a named linalg
//                      op does.
// 3. linalg.generic + math — A linalg.generic body containing a single math
//                      dialect op. Used for ops with no TOSA or named linalg
//                      equivalent. Math ops eventually lower to libm calls.
// 4. Custom          — Multi-op sequences in TOSA, linalg, or arith dialects.
//                      Used for compound operations (e.g. gelu, sign, mish).

//===----------------------------------------------------------------------===//
// TOSA Unary Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
template <typename TTIROpTy, typename TosaOpTy>
class ElementwiseUnaryOpToTosaConversionPattern
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

    Value input = broadcastToShape(adaptor.getInput(), resultType.getShape(),
                                   op.getLoc(), rewriter);
    auto result = rewriter.create<TosaOpTy>(op.getLoc(), resultType, input);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Linalg Named Op Conversion Pattern
//===----------------------------------------------------------------------===//

namespace {
// Conversion pattern for TTIR unary ops lowered to named linalg ops.
// Supports implicit broadcasting by broadcasting input to result shape.
template <typename TTIROpTy, typename LinAlgOpTy>
class ElementwiseUnaryOpToNamedLinalgConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    Value input = broadcastToShape(adaptor.getInput(), resultType.getShape(),
                                   loc, rewriter);
    auto output = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                   resultType.getElementType());
    rewriter.replaceOpWithNewOp<LinAlgOpTy>(op, resultType, ValueRange{input},
                                            output.getResult());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Linalg Generic + Math Dialect Patterns
//===----------------------------------------------------------------------===//

// Base class for TTIR unary ops lowered via linalg.generic. Subclasses only
// need to implement buildBody() to emit the scalar computation. Supports
// implicit broadcasting by broadcasting input to result shape.
namespace {
template <typename TTIROpTy>
class ElementwiseUnaryOpToLinalgGenericConversionPatternBase
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
    if (!isa<FloatType>(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(op,
                                         "element type must be a float type");
    }

    Location loc = op.getLoc();
    Value input = broadcastToShape(adaptor.getInput(), resultType.getShape(),
                                   loc, rewriter);

    int64_t rank = resultType.getRank();
    auto indexingMap =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{input}, ValueRange{emptyTensor},
        SmallVector<AffineMap>{indexingMap, indexingMap}, iteratorTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result = buildBody(b, nestedLoc, args, resultType);
          b.create<linalg::YieldOp>(nestedLoc, result);
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
class ElementwiseUnaryOpToMathConversionPattern
    : public ElementwiseUnaryOpToLinalgGenericConversionPatternBase<TTIROpTy> {
public:
  using ElementwiseUnaryOpToLinalgGenericConversionPatternBase<
      TTIROpTy>::ElementwiseUnaryOpToLinalgGenericConversionPatternBase;

protected:
  Value buildBody(OpBuilder &b, Location loc, ValueRange args,
                  RankedTensorType /*resultType*/) const override {
    return b.create<MathOpTy>(loc, args[0]);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Compound Unary Patterns (multi-op TOSA sequences)
//===----------------------------------------------------------------------===//

// Sign: -1 if x<0, 0 if x==0, 1 if x>0.
namespace {
class SignOpConversionPattern : public OpConversionPattern<ttir::SignOp> {
public:
  using OpConversionPattern<ttir::SignOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }
    Location loc = op.getLoc();
    input = broadcastToShape(input, resultType.getShape(), loc, rewriter);
    int64_t rank = resultType.getRank();
    auto elemTy = resultType.getElementType();
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));

    Value zero = createTosaConst(rewriter, loc, elemTy, rank, 0.0);
    Value one = createTosaConst(rewriter, loc, elemTy, rank, 1.0);
    Value negOne = createTosaConst(rewriter, loc, elemTy, rank, -1.0);

    Value gtZero = rewriter.create<tosa::GreaterOp>(loc, boolType, input, zero);
    Value eqZero = rewriter.create<tosa::EqualOp>(loc, boolType, input, zero);
    Value posOrNeg =
        rewriter.create<tosa::SelectOp>(loc, resultType, gtZero, one, negOne);
    rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, resultType, eqZero, zero,
                                                posOrNeg);
    return success();
  }
};
} // namespace

// IsFinite: true if x is neither NaN nor Inf.
namespace {
class IsFiniteOpConversionPattern
    : public ElementwiseUnaryOpToLinalgGenericConversionPatternBase<
          ttir::IsFiniteOp> {
public:
  using ElementwiseUnaryOpToLinalgGenericConversionPatternBase::
      ElementwiseUnaryOpToLinalgGenericConversionPatternBase;

protected:
  Value buildBody(OpBuilder &b, Location loc, ValueRange args,
                  RankedTensorType resultType) const override {
    Value elem = args[0];
    // isfinite(x) = (x - x) == 0.
    // If x is NaN, x - x is NaN and OEQ returns false.
    // If x is Inf, x - x is NaN and OEQ returns false.
    Value diff = b.create<arith::SubFOp>(loc, elem, elem);
    Value zero =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(elem.getType(), 0.0));
    Value isFinite =
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, diff, zero);
    return b.create<arith::UIToFPOp>(loc, resultType.getElementType(),
                                     isFinite);
  }
};
} // namespace

// Gelu: 0.5 * x * (1 + erf(x / sqrt(2))).
namespace {
class GeluOpConversionPattern : public OpConversionPattern<ttir::GeluOp> {
public:
  using OpConversionPattern<ttir::GeluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GeluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }
    Location loc = op.getLoc();
    auto elemTy = resultType.getElementType();
    if (!isa<FloatType>(elemTy)) {
      return rewriter.notifyMatchFailure(op,
                                         "element type must be a float type");
    }
    input = broadcastToShape(input, resultType.getShape(), loc, rewriter);
    int64_t rank = resultType.getRank();
    Value shift = createTosaMulShift(rewriter, loc);

    Value half = createTosaConst(rewriter, loc, elemTy, rank, 0.5);
    Value one = createTosaConst(rewriter, loc, elemTy, rank, 1.0);
    // 1/sqrt(2), used in the standard Gelu formula.
    constexpr double kInvSqrt2 = 0.7071067811865475;
    Value invSqrt2 = createTosaConst(rewriter, loc, elemTy, rank, kInvSqrt2);

    Value xScaled =
        rewriter.create<tosa::MulOp>(loc, resultType, input, invSqrt2, shift);
    Value erfVal = rewriter.create<tosa::ErfOp>(loc, resultType, xScaled);
    Value onePlusErf =
        rewriter.create<tosa::AddOp>(loc, resultType, one, erfVal);
    Value halfX =
        rewriter.create<tosa::MulOp>(loc, resultType, half, input, shift);
    rewriter.replaceOpWithNewOp<tosa::MulOp>(op, resultType, halfX, onePlusErf,
                                             shift);
    return success();
  }
};
} // namespace

// Silu: x * sigmoid(x).
namespace {
class SiluOpConversionPattern : public OpConversionPattern<ttir::SiluOp> {
public:
  using OpConversionPattern<ttir::SiluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SiluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }
    Location loc = op.getLoc();
    if (!isa<FloatType>(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(op,
                                         "element type must be a float type");
    }
    input = broadcastToShape(input, resultType.getShape(), loc, rewriter);
    Value shift = createTosaMulShift(rewriter, loc);

    Value sigm = rewriter.create<tosa::SigmoidOp>(loc, resultType, input);
    rewriter.replaceOpWithNewOp<tosa::MulOp>(op, resultType, input, sigm,
                                             shift);
    return success();
  }
};
} // namespace

// LeakyRelu: x if x>0, alpha*x otherwise.
namespace {
class LeakyReluOpConversionPattern
    : public OpConversionPattern<ttir::LeakyReluOp> {
public:
  using OpConversionPattern<ttir::LeakyReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LeakyReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }
    Location loc = op.getLoc();
    auto elemTy = resultType.getElementType();
    if (!isa<FloatType>(elemTy)) {
      return rewriter.notifyMatchFailure(op,
                                         "element type must be a float type");
    }
    input = broadcastToShape(input, resultType.getShape(), loc, rewriter);
    int64_t rank = resultType.getRank();
    Value shift = createTosaMulShift(rewriter, loc);
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));

    double alpha = op.getParameter().convertToDouble();
    Value zero = createTosaConst(rewriter, loc, elemTy, rank, 0.0);
    Value alphaC = createTosaConst(rewriter, loc, elemTy, rank, alpha);
    Value scaled =
        rewriter.create<tosa::MulOp>(loc, resultType, alphaC, input, shift);
    Value positive =
        rewriter.create<tosa::GreaterOp>(loc, boolType, input, zero);
    rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, resultType, positive, input,
                                                scaled);
    return success();
  }
};
} // namespace

// Hardsigmoid: clamp((x + 3) / 6, 0, 1).
namespace {
class HardsigmoidOpConversionPattern
    : public OpConversionPattern<ttir::HardsigmoidOp> {
public:
  using OpConversionPattern<ttir::HardsigmoidOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::HardsigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }
    Location loc = op.getLoc();
    input = broadcastToShape(input, resultType.getShape(), loc, rewriter);
    auto elemTy = resultType.getElementType();
    if (!isa<FloatType>(elemTy)) {
      return rewriter.notifyMatchFailure(op,
                                         "element type must be a float type");
    }
    int64_t rank = resultType.getRank();
    Value shift = createTosaMulShift(rewriter, loc);

    Value three = createTosaConst(rewriter, loc, elemTy, rank, 3.0);
    Value sixth = createTosaConst(rewriter, loc, elemTy, rank, 1.0 / 6.0);

    Value xPlus3 = rewriter.create<tosa::AddOp>(loc, resultType, input, three);
    Value scaled =
        rewriter.create<tosa::MulOp>(loc, resultType, xPlus3, sixth, shift);
    rewriter.replaceOpWithNewOp<tosa::ClampOp>(
        op, resultType, scaled, rewriter.getFloatAttr(elemTy, 0.0),
        rewriter.getFloatAttr(elemTy, 1.0));
    return success();
  }
};
} // namespace

// Mish: x * tanh(softplus(x)).
// Uses numerically stable softplus: max(x,0) + log(1 + exp(-|x|)).
namespace {
class MishOpConversionPattern : public OpConversionPattern<ttir::MishOp> {
public:
  using OpConversionPattern<ttir::MishOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MishOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }
    Location loc = op.getLoc();
    auto elemTy = resultType.getElementType();
    if (!isa<FloatType>(elemTy)) {
      return rewriter.notifyMatchFailure(op,
                                         "element type must be a float type");
    }
    input = broadcastToShape(input, resultType.getShape(), loc, rewriter);
    int64_t rank = resultType.getRank();
    Value shift = createTosaMulShift(rewriter, loc);

    Value zero = createTosaConst(rewriter, loc, elemTy, rank, 0.0);
    Value one = createTosaConst(rewriter, loc, elemTy, rank, 1.0);

    // softplus(x) = max(x, 0) + log(1 + exp(-|x|)).
    Value absX = rewriter.create<tosa::AbsOp>(loc, resultType, input);
    Value negAbsX = rewriter.create<tosa::NegateOp>(loc, resultType, absX);
    Value expNegAbsX = rewriter.create<tosa::ExpOp>(loc, resultType, negAbsX);
    Value onePlusExp =
        rewriter.create<tosa::AddOp>(loc, resultType, one, expNegAbsX);
    Value logPart = rewriter.create<tosa::LogOp>(loc, resultType, onePlusExp);
    Value maxXZero =
        rewriter.create<tosa::MaximumOp>(loc, resultType, input, zero);
    Value softplus =
        rewriter.create<tosa::AddOp>(loc, resultType, maxXZero, logPart);

    Value tanhSP = rewriter.create<tosa::TanhOp>(loc, resultType, softplus);
    rewriter.replaceOpWithNewOp<tosa::MulOp>(op, resultType, input, tanhSP,
                                             shift);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Special Unary Patterns
//===----------------------------------------------------------------------===//

namespace {
class LogicalNotOpConversionPattern
    : public OpConversionPattern<ttir::LogicalNotOp> {
public:
  using OpConversionPattern<ttir::LogicalNotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LogicalNotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    Location loc = op.getLoc();
    Value input = broadcastToShape(adaptor.getInput(), resultType.getShape(),
                                   loc, rewriter);

    // First convert the input to a boolean tensor.
    Value boolInput = convertToBooleanTensor(input, loc, rewriter);

    // Get the boolean type for the intermediate result.
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));

    // Apply logical not to the boolean tensor.
    auto notResult =
        rewriter.create<tosa::LogicalNotOp>(loc, boolType, boolInput);

    // Convert boolean result back to original type using cast.
    auto result = rewriter.create<tosa::CastOp>(loc, resultType, notResult);

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
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    Location loc = op.getLoc();
    Value input = broadcastToShape(adaptor.getInput(), resultType.getShape(),
                                   loc, rewriter);

    DenseElementsAttr zeroAttr = createDenseElementsAttr(resultType, 0);
    if (!zeroAttr) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported element type for ReLU zero constant");
    }

    auto zeroes = rewriter.create<arith::ConstantOp>(loc, resultType, zeroAttr);

    auto output = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                   resultType.getElementType());
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
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    Location loc = op.getLoc();
    Value input = broadcastToShape(adaptor.getInput(), resultType.getShape(),
                                   loc, rewriter);
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

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateTTIRToLinalgEltwiseUnaryPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {
  // Named linalg ops (no TOSA equivalent)
  patterns.add<ElementwiseUnaryOpToNamedLinalgConversionPattern<
      ttir::SqrtOp, linalg::SqrtOp>>(typeConverter, ctx);

  // linalg.generic + math dialect ops
  patterns.add<
      ElementwiseUnaryOpToMathConversionPattern<ttir::AcosOp, math::AcosOp>,
      ElementwiseUnaryOpToMathConversionPattern<ttir::AsinOp, math::AsinOp>,
      ElementwiseUnaryOpToMathConversionPattern<ttir::AsinhOp, math::AsinhOp>,
      ElementwiseUnaryOpToMathConversionPattern<ttir::AtanOp, math::AtanOp>,
      ElementwiseUnaryOpToMathConversionPattern<ttir::CbrtOp, math::CbrtOp>,
      ElementwiseUnaryOpToMathConversionPattern<ttir::ErfcOp, math::ErfcOp>,
      ElementwiseUnaryOpToMathConversionPattern<ttir::Expm1Op, math::ExpM1Op>,
      ElementwiseUnaryOpToMathConversionPattern<ttir::Log1pOp, math::Log1pOp>,
      ElementwiseUnaryOpToMathConversionPattern<ttir::RsqrtOp, math::RsqrtOp>,
      ElementwiseUnaryOpToMathConversionPattern<ttir::TanOp, math::TanOp>>(
      typeConverter, ctx);

  // Custom linalg-based patterns
  patterns.add<ReluOpConversionPattern, IsFiniteOpConversionPattern>(
      typeConverter, ctx);
}

void populateTTIRToTosaEltwiseUnaryPatterns(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {
  // Elementwise unary operations (1:1 TOSA mappings)
  patterns.add<
      ElementwiseUnaryOpToTosaConversionPattern<ttir::AbsOp, tosa::AbsOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::BitwiseNotOp,
                                                tosa::BitwiseNotOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::CeilOp, tosa::CeilOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::CosOp, tosa::CosOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::ErfOp, tosa::ErfOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::ExpOp, tosa::ExpOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::FloorOp, tosa::FloorOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::LogOp, tosa::LogOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::NegOp, tosa::NegateOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::ReciprocalOp,
                                                tosa::ReciprocalOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::SigmoidOp,
                                                tosa::SigmoidOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::SinOp, tosa::SinOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::TanhOp, tosa::TanhOp>,
      ElementwiseUnaryOpToTosaConversionPattern<ttir::TypecastOp,
                                                tosa::CastOp>>(typeConverter,
                                                               ctx);

  // Compound unary operations (multi-op TOSA sequences)
  patterns.add<SignOpConversionPattern, GeluOpConversionPattern,
               SiluOpConversionPattern, LeakyReluOpConversionPattern,
               HardsigmoidOpConversionPattern, MishOpConversionPattern>(
      typeConverter, ctx);

  // Unary operations using other dialects
  patterns.add<LogicalNotOpConversionPattern, Relu6OpConversionPattern>(
      typeConverter, ctx);
}

} // namespace mlir::tt::ttir_to_linalg
