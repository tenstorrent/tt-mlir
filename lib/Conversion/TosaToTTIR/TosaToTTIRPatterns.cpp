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
#include "ttmlir/Utils.h"

using namespace mlir;
using namespace mlir::tt;

// TODO(sdjukic): extract this pattern into separate file and use it for both
// TOSA and StableHLO

namespace {
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

    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    ttmlir::utils::replaceOpWithNewDPSOp<DestOp>(rewriter, srcOp, outputType,
                                                 adaptor.getOperands());

    return success();
  }

private:
  virtual LogicalResult
  checkConversionLegality(SrcOp srcOp, Adaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    return success();
  }
};
} // namespace

namespace {
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
} // namespace

namespace {
class TosaToTTIRClampOpConversionPattern
    : public OpConversionPattern<tosa::ClampOp> {
  using OpConversionPattern<tosa::ClampOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(tosa::ClampOp srcOp, tosa::ClampOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    ttmlir::utils::replaceOpWithNewDPSOp<ttir::ClampOp>(
        rewriter, srcOp, outputType, adaptor.getInput(), adaptor.getMinFp(),
        adaptor.getMaxFp());

    return success();
  }
};
} // namespace

namespace {
class TosaToTTIRConcatOpConversionPattern
    : public OpConversionPattern<tosa::ConcatOp> {
  using OpConversionPattern<tosa::ConcatOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(tosa::ConcatOp srcOp, tosa::ConcatOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    ttmlir::utils::replaceOpWithNewDPSOp<ttir::ConcatOp>(
        rewriter, srcOp, outputType, adaptor.getOperands(), adaptor.getAxis());

    return success();
  }
};
} // namespace

namespace {
class TosaToTTIRMatmulOpConversionPattern
    : public OpConversionPattern<tosa::MatMulOp> {
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;
  using Adaptor = tosa::MatMulOp::Adaptor;

public:
  LogicalResult
  matchAndRewrite(tosa::MatMulOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    ttmlir::utils::replaceOpWithNewDPSOp<ttir::MatmulOp>(
        rewriter, srcOp, outputType, adaptor.getA(), adaptor.getB());

    return success();
  }
};
} // namespace

namespace {
template <typename SrcOp, typename DestOp,
          typename Adaptor = typename SrcOp::Adaptor>
class TosaToTTIRReduceOpConversionPattern : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    ttmlir::utils::replaceOpWithNewDPSOp<DestOp>(
        rewriter, srcOp, outputType, adaptor.getInput(), /*keep_dim=*/true,
        rewriter.getI32ArrayAttr(adaptor.getAxis()));

    return success();
  }
};
} // namespace

namespace {
class TosaToTTIRMaxPool2DOpConversionPattern
    : public OpConversionPattern<tosa::MaxPool2dOp> {
  using OpConversionPattern<tosa::MaxPool2dOp>::OpConversionPattern;
  using Adaptor = tosa::MaxPool2dOp::Adaptor;

public:
  LogicalResult
  matchAndRewrite(tosa::MaxPool2dOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    auto dims = srcOp.getKernel();
    auto strides = srcOp.getStride();
    auto pad = srcOp.getPad();

    // TODO (azecevic) Add comment about the parameters.
    ttmlir::utils::replaceOpWithNewDPSOp<ttir::MaxPool2dOp>(
        rewriter, srcOp, outputType, adaptor.getInput(), dims[0], dims[1],
        strides[0], strides[1], 1, 1, false, pad[2], pad[3], pad[0], pad[1]);

    return success();
  }
};
} // namespace

static void
addElementwiseUnaryOpsConversionPatterns(MLIRContext *ctx,
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

static void
addElementwiseBinaryOpsConversionPatterns(MLIRContext *ctx,
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

static void
addElementwiseTernaryOpsConversionPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::SelectOp, mlir::tt::ttir::WhereOp>>(typeConverter, ctx);
}

static void addLogicalOpsConversionPatterns(MLIRContext *ctx,
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

static void addCompareOpsConversionPatterns(MLIRContext *ctx,
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

static void addMatmulOpsConversionPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRMatmulOpConversionPattern>(typeConverter, ctx);
}

static void addReductionOpsConversionPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRReduceOpConversionPattern<tosa::ReduceMaxOp,
                                                   mlir::tt::ttir::MaxOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRReduceOpConversionPattern<tosa::ReduceSumOp,
                                                   mlir::tt::ttir::SumOp>>(
      typeConverter, ctx);
}

static void addPoolingOpsConversionPatterns(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRMaxPool2DOpConversionPattern>(typeConverter, ctx);
}

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
  addPoolingOpsConversionPatterns(ctx, patterns, typeConverter);

  patterns.add<TosaToTTIRClampOpConversionPattern,
               TosaToTTIRConcatOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
