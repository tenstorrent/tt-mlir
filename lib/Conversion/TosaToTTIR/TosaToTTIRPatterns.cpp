// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ttmlir/Conversion/TosaToTTIR/TosaToTTIR.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

using namespace mlir;
using namespace mlir::tt;

namespace {

// TODO(sdjukic): extract this pattern into separate file and use it for both
// TOSA and StableHLO

template <typename SrcOp, typename DestOp,
          typename Adaptor = typename SrcOp::Adaptor>
class TosaToTTIRDefaultDPSOpConversionPattern
    : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    LogicalResult legalityResult =
        checkConversionLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    RankedTensorType outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult().getType());
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
    rewriter.replaceOpWithNewOp<DestOp>(
        srcOp, TypeRange(outputTensor.getType()), adaptor.getOperands(),
        ValueRange(outputTensor),
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }

private:
  virtual LogicalResult
  checkConversionLegality(SrcOp srcOp, Adaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    return success();
  }
};

class TosaToTTIRMultiplyOpConversionPattern
    : public TosaToTTIRDefaultDPSOpConversionPattern<
          tosa::MulOp, mlir::tt::ttir::MultiplyOp> {
  using TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::MulOp,
      mlir::tt::ttir::MultiplyOp>::TosaToTTIRDefaultDPSOpConversionPattern;

private:
  LogicalResult
  checkConversionLegality(tosa::MulOp srcOp, tosa::MulOp::Adaptor adaptor,
                          ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getShift() != 0) {
      return rewriter.notifyMatchFailure(
          srcOp, "TTIR MultiplyOp doesn't support shifted multiply.");
    }
    return success();
  }
};

class TosaToTTIRMatmulOpConversionPattern
    : public OpConversionPattern<tosa::MatMulOp> {
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;
  using Adaptor = tosa::MatMulOp::Adaptor;

public:
  LogicalResult
  matchAndRewrite(tosa::MatMulOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult =
        checkConversionLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }
    auto outputType = mlir::cast<RankedTensorType>(srcOp.getResult().getType());
    auto outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
    auto operands = adaptor.getOperands();

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::MatmulOp>(
        srcOp, TypeRange(outputTensor.getType()), operands[0], operands[1],
        outputTensor,
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }

private:
  LogicalResult
  checkConversionLegality(tosa::MatMulOp srcOp, Adaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    if (srcOp.getQuantizationInfo().has_value()) {
      return rewriter.notifyMatchFailure(
          srcOp, "TTIR MatmulOp currently doesn't support quantization.");
    }
    return success();
  }
};

template <typename SrcOp, typename DestOp,
          typename Adaptor = typename SrcOp::Adaptor>
class TosaToTTIRReduceOpConversionPattern : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult().getType());
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    rewriter.replaceOpWithNewOp<DestOp>(
        srcOp, outputTensor.getType(), adaptor.getInput(), outputTensor,
        true /*keepdim*/,
        rewriter.getArrayAttr(SmallVector<Attribute>(1, adaptor.getAxisAttr())),
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }
};

void addElementwiseUnaryOpsConversionPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {

  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<tosa::AbsOp,
                                                       mlir::tt::ttir::AbsOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::CastOp, mlir::tt::ttir::TypecastOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<tosa::CeilOp,
                                                       mlir::tt::ttir::CeilOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<tosa::CosOp,
                                                       mlir::tt::ttir::CosOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<tosa::ExpOp,
                                                       mlir::tt::ttir::ExpOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::FloorOp, mlir::tt::ttir::FloorOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<tosa::NegateOp,
                                                       mlir::tt::ttir::NegOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::ReciprocalOp, mlir::tt::ttir::ReciprocalOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::RsqrtOp, mlir::tt::ttir::RsqrtOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::SigmoidOp, mlir::tt::ttir::SigmoidOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<tosa::SinOp,
                                                       mlir::tt::ttir::SinOp>>(
      typeConverter, ctx);
}

void addElementwiseBinaryOpsConversionPatterns(MLIRContext *ctx,
                                               RewritePatternSet &patterns,
                                               TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<tosa::AddOp,
                                                       mlir::tt::ttir::AddOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::MaximumOp, mlir::tt::ttir::MaximumOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::MinimumOp, mlir::tt::ttir::MinimumOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRMultiplyOpConversionPattern>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::SubOp, mlir::tt::ttir::SubtractOp>>(typeConverter, ctx);
}

void addElementwiseTernaryOpsConversionPatterns(MLIRContext *ctx,
                                                RewritePatternSet &patterns,
                                                TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::SelectOp, mlir::tt::ttir::WhereOp>>(typeConverter, ctx);
}

void addLogicalOpsConversionPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::LogicalAndOp, mlir::tt::ttir::LogicalAndOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::LogicalNotOp, mlir::tt::ttir::LogicalNotOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::LogicalOrOp, mlir::tt::ttir::LogicalOrOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::LogicalXorOp, mlir::tt::ttir::LogicalXorOp>>(typeConverter, ctx);
}

void addCompareOpsConversionPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::EqualOp, mlir::tt::ttir::EqualOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::GreaterEqualOp, mlir::tt::ttir::GreaterEqualOp>>(typeConverter,
                                                             ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::GreaterOp, mlir::tt::ttir::GreaterThanOp>>(typeConverter, ctx);
}

void addMatmulOpsConversionPatterns(MLIRContext *ctx,
                                    RewritePatternSet &patterns,
                                    TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRMatmulOpConversionPattern>(typeConverter, ctx);
}

void addReductionOpsConversionPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRReduceOpConversionPattern<tosa::ReduceMaxOp,
                                                   mlir::tt::ttir::MaxOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRReduceOpConversionPattern<tosa::ReduceSumOp,
                                                   mlir::tt::ttir::SumOp>>(
      typeConverter, ctx);
}
} // namespace

namespace mlir::tt {

void populateTosaToTTIRPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  addElementwiseUnaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addElementwiseBinaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addElementwiseTernaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addLogicalOpsConversionPatterns(ctx, patterns, typeConverter);
  addCompareOpsConversionPatterns(ctx, patterns, typeConverter);
  addMatmulOpsConversionPatterns(ctx, patterns, typeConverter);
  addReductionOpsConversionPatterns(ctx, patterns, typeConverter);
}

} // namespace mlir::tt
