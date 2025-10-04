// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToTTIR/TTNNToTTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

template <typename SrcOp, typename DestOp>
class TTNNToTTIROpConversionPattern : public mlir::OpConversionPattern<SrcOp> {
  using mlir::OpConversionPattern<SrcOp>::OpConversionPattern;
  using Adaptor = typename SrcOp::Adaptor;

public:
  mlir::LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    mlir::tt::ttir::utils::replaceOpWithNewDPSOp<DestOp>(
        rewriter, srcOp, outputType, adaptor.getOperands());

    return mlir::success();
  }
};

template <typename SrcOp, typename DestOp>
class TTNNToTTIRElementwiseConversionPattern
    : public TTNNToTTIROpConversionPattern<SrcOp, DestOp> {
public:
  using TTNNToTTIROpConversionPattern<SrcOp,
                                      DestOp>::TTNNToTTIROpConversionPattern;
  // Reuse base; this exists simply for clarity and potential future extensions
};

template <typename SrcOp, typename DestOp>
class TTNNToTTIRReductionConversionPattern
    : public TTNNToTTIROpConversionPattern<SrcOp, DestOp> {
public:
  using TTNNToTTIROpConversionPattern<SrcOp,
                                      DestOp>::TTNNToTTIROpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SrcOp srcOp, typename SrcOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    // Handle attribute differences between TTNN and TTIR
    auto dimArgAttr =
        srcOp.getDimArg()
            ? rewriter.getI32ArrayAttr(llvm::to_vector(llvm::map_range(
                  srcOp.getDimArg().value(),
                  [](mlir::Attribute attr) {
                    return static_cast<int32_t>(
                        mlir::cast<mlir::IntegerAttr>(attr).getInt());
                  })))
            : nullptr;

    mlir::tt::ttir::utils::replaceOpWithNewDPSOp<DestOp>(
        rewriter, srcOp, outputType, adaptor.getInput(), srcOp.getKeepDim(),
        dimArgAttr);

    return mlir::success();
  }
};

class TTNNArgMaxToTTIRArgMaxConversionPattern
    : public mlir::OpConversionPattern<mlir::tt::ttnn::ArgMaxOp> {
public:
  using mlir::OpConversionPattern<
      mlir::tt::ttnn::ArgMaxOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ArgMaxOp srcOp,
                  mlir::tt::ttnn::ArgMaxOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    // ArgMax: TTNN has dim (I32Attr), TTIR has dim_arg (I32ArrayAttr)
    auto keepDimAttr = rewriter.getBoolAttr(srcOp.getKeepDim());
    auto dimArgAttr =
        srcOp.getDim()
            ? rewriter.getI32ArrayAttr({static_cast<int32_t>(*srcOp.getDim())})
            : nullptr;

    mlir::tt::ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::ArgMaxOp>(
        rewriter, srcOp, outputType, adaptor.getInput(), keepDimAttr,
        dimArgAttr);

    return mlir::success();
  }
};

class TTNNProdToTTIRProdConversionPattern
    : public mlir::OpConversionPattern<mlir::tt::ttnn::ProdOp> {
public:
  using mlir::OpConversionPattern<mlir::tt::ttnn::ProdOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ProdOp srcOp,
                  mlir::tt::ttnn::ProdOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    auto keepDimAttr = rewriter.getBoolAttr(srcOp.getKeepDim());

    // TTNN_ProdOp uses I64Attr for TTNN dimArg (other reduction ops use
    // I32ArrayAttr). Convert to a single-element I32ArrayAttr for TTIR.
    // In both TTNN and TTIR, if dimArg is not provided, the reduction is
    // performed over all dimensions.
    auto dimArgAttr = srcOp.getDimArg()
                          ? rewriter.getI32ArrayAttr(
                                {static_cast<int32_t>(*srcOp.getDimArg())})
                          : nullptr;

    mlir::tt::ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::ProdOp>(
        rewriter, srcOp, outputType, adaptor.getInput(), keepDimAttr,
        dimArgAttr);

    return mlir::success();
  }
};

class TTNNMorehCumSumToTTIRCumSumConversionPattern
    : public mlir::OpConversionPattern<mlir::tt::ttnn::MorehCumSumOp> {
public:
  using mlir::OpConversionPattern<
      mlir::tt::ttnn::MorehCumSumOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tt::ttnn::MorehCumSumOp srcOp,
                  mlir::tt::ttnn::MorehCumSumOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    mlir::tt::ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::CumSumOp>(
        rewriter, srcOp, outputType, adaptor.getInput(), srcOp.getDim());

    return mlir::success();
  }
};

class TTNNMatmulToTTIRMatmulConversionPattern
    : public mlir::OpConversionPattern<mlir::tt::ttnn::MatmulOp> {

  using mlir::OpConversionPattern<
      mlir::tt::ttnn::MatmulOp>::OpConversionPattern;

public:
  mlir::LogicalResult
  matchAndRewrite(mlir::tt::ttnn::MatmulOp srcOp,
                  mlir::tt::ttnn::MatmulOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto outputType = mlir::cast<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    mlir::tt::ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::MatmulOp>(
        rewriter, srcOp, outputType, adaptor.getA(), adaptor.getB(),
        adaptor.getTransposeA(), adaptor.getTransposeB());

    // Note that TTNN attributes that have no TTIR equivalents are simply
    // dropped

    return mlir::success();
  }
};
} // namespace

static void
addElementwiseUnaryOpsConversionPatterns(mlir::MLIRContext *ctx,
                                         mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &typeConverter) {

  patterns
      .add<TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::AbsOp,
                                                  mlir::tt::ttir::AbsOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::CbrtOp,
                                                  mlir::tt::ttir::CbrtOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::TypecastOp,
                                                  mlir::tt::ttir::TypecastOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::CeilOp,
                                                  mlir::tt::ttir::CeilOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::CosOp,
                                                  mlir::tt::ttir::CosOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::ExpOp,
                                                  mlir::tt::ttir::ExpOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::FloorOp,
                                                  mlir::tt::ttir::FloorOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::IsFiniteOp,
                                                  mlir::tt::ttir::IsFiniteOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::NegOp,
                                                  mlir::tt::ttir::NegOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::RsqrtOp,
                                                  mlir::tt::ttir::RsqrtOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::SinOp,
                                                  mlir::tt::ttir::SinOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::SqrtOp,
                                                  mlir::tt::ttir::SqrtOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::Log1pOp,
                                                  mlir::tt::ttir::Log1pOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::Expm1Op,
                                                  mlir::tt::ttir::Expm1Op>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::SignOp,
                                                  mlir::tt::ttir::SignOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::SigmoidOp,
                                                  mlir::tt::ttir::SigmoidOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::TanOp,
                                                  mlir::tt::ttir::TanOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::TanhOp,
                                                  mlir::tt::ttir::TanhOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::LogOp,
                                                  mlir::tt::ttir::LogOp>>(
          typeConverter, ctx);
}

static void
addElementwiseBinaryOpsConversionPatterns(mlir::MLIRContext *ctx,
                                          mlir::RewritePatternSet &patterns,
                                          mlir::TypeConverter &typeConverter) {

  patterns
      .add<TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::AddOp,
                                                  mlir::tt::ttir::AddOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::DivideOp,
                                                  mlir::tt::ttir::DivOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::MaxOp,
                                                  mlir::tt::ttir::MaximumOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::MinOp,
                                                  mlir::tt::ttir::MinimumOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::MultiplyOp,
                                                  mlir::tt::ttir::MultiplyOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::SubtractOp,
                                                  mlir::tt::ttir::SubtractOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::RemainderOp,
                                                  mlir::tt::ttir::RemainderOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::WhereOp,
                                                  mlir::tt::ttir::WhereOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::PowTensorOp,
                                                  mlir::tt::ttir::PowOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::Atan2Op,
                                                  mlir::tt::ttir::Atan2Op>,
           TTNNToTTIRElementwiseConversionPattern<
               mlir::tt::ttnn::LogicalRightShiftOp,
               mlir::tt::ttir::LogicalRightShiftOp>,
           TTNNToTTIRElementwiseConversionPattern<
               mlir::tt::ttnn::LogicalLeftShiftOp,
               mlir::tt::ttir::LogicalLeftShiftOp>>(typeConverter, ctx);
}

static void
addReductionOpsConversionPatterns(mlir::MLIRContext *ctx,
                                  mlir::RewritePatternSet &patterns,
                                  mlir::TypeConverter &typeConverter) {

  patterns.add<TTNNToTTIRReductionConversionPattern<mlir::tt::ttnn::SumOp,
                                                    mlir::tt::ttir::SumOp>,
               TTNNToTTIRReductionConversionPattern<mlir::tt::ttnn::MeanOp,
                                                    mlir::tt::ttir::MeanOp>,
               TTNNToTTIRReductionConversionPattern<mlir::tt::ttnn::MaxOp,
                                                    mlir::tt::ttir::MaxOp>,
               TTNNToTTIRReductionConversionPattern<mlir::tt::ttnn::MinOp,
                                                    mlir::tt::ttir::MinOp>,
               TTNNArgMaxToTTIRArgMaxConversionPattern,
               TTNNProdToTTIRProdConversionPattern,
               TTNNMorehCumSumToTTIRCumSumConversionPattern>(typeConverter,
                                                             ctx);
}

namespace mlir::tt {

void populateTTNNToTTIRPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  addElementwiseUnaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addElementwiseBinaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addReductionOpsConversionPatterns(ctx, patterns, typeConverter);
  patterns.add<TTNNMatmulToTTIRMatmulConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
