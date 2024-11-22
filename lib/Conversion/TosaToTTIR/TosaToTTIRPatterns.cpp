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

void addElementwiseUnaryOpsConversionPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {

  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<tosa::AbsOp,
                                                       mlir::tt::ttir::AbsOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<tosa::NegateOp,
                                                       mlir::tt::ttir::NegOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<tosa::SinOp,
                                                       mlir::tt::ttir::SinOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::SigmoidOp, mlir::tt::ttir::SigmoidOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::ReciprocalOp, mlir::tt::ttir::ReciprocalOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::RsqrtOp, mlir::tt::ttir::RsqrtOp>>(typeConverter, ctx);
}

void addElementwiseBinaryOpsConversionPatterns(MLIRContext *ctx,
                                               RewritePatternSet &patterns,
                                               TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<tosa::AddOp,
                                                       mlir::tt::ttir::AddOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRMultiplyOpConversionPattern>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::SubOp, mlir::tt::ttir::SubtractOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::MaximumOp, mlir::tt::ttir::MaximumOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::MinimumOp, mlir::tt::ttir::MinimumOp>>(typeConverter, ctx);
}

void addCompareOpsConversionPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::GreaterEqualOp, mlir::tt::ttir::GreaterEqualOp>>(typeConverter,
                                                             ctx);
}

void addElementwiseTernaryOpsConversionPatterns(MLIRContext *ctx,
                                                RewritePatternSet &patterns,
                                                TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::SelectOp, mlir::tt::ttir::WhereOp>>(typeConverter, ctx);
}
} //  namespace

namespace mlir::tt {

void populateTosaToTTIRPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  addElementwiseUnaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addElementwiseBinaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addElementwiseTernaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addCompareOpsConversionPatterns(ctx, patterns, typeConverter);
}

} // namespace mlir::tt
