// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TosaToTTIR/TosaToTTIR.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

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
    doRewrite(srcOp, adaptor, rewriter, outputType);

    return success();
  }

private:
  virtual LogicalResult
  checkConversionLegality(SrcOp srcOp, Adaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    return success();
  }

  virtual void doRewrite(SrcOp srcOp, Adaptor adaptor,
                         ConversionPatternRewriter &rewriter,
                         RankedTensorType outputType) const {
    ttir::utils::replaceOpWithNewDPSOp<DestOp>(rewriter, srcOp, outputType,
                                               adaptor.getOperands());
  }
};
} // namespace

namespace {
template <typename SrcOp, typename DestOp>
class TosaToTTIRDefaultUnaryDPSOpConversionPattern
    : public TosaToTTIRDefaultDPSOpConversionPattern<SrcOp, DestOp> {
  using TosaToTTIRDefaultDPSOpConversionPattern<
      SrcOp, DestOp>::TosaToTTIRDefaultDPSOpConversionPattern;
  using Adaptor = typename SrcOp::Adaptor;

private:
  void doRewrite(SrcOp srcOp, Adaptor adaptor,
                 ConversionPatternRewriter &rewriter,
                 RankedTensorType outputType) const override {
    ttir::utils::replaceOpWithNewDPSOp<DestOp>(rewriter, srcOp, outputType,
                                               adaptor.getInput1());
  }
};
} // namespace

namespace {
template <typename SrcOp, typename DestOp>
class TosaToTTIRDefaultBinaryDPSOpConversionPattern
    : public TosaToTTIRDefaultDPSOpConversionPattern<SrcOp, DestOp> {
  using TosaToTTIRDefaultDPSOpConversionPattern<
      SrcOp, DestOp>::TosaToTTIRDefaultDPSOpConversionPattern;
  using Adaptor = typename SrcOp::Adaptor;

private:
  void doRewrite(SrcOp srcOp, Adaptor adaptor,
                 ConversionPatternRewriter &rewriter,
                 RankedTensorType outputType) const override {
    ttir::utils::replaceOpWithNewDPSOp<DestOp>(
        rewriter, srcOp, outputType, adaptor.getInput1(), adaptor.getInput2());
  }
};
} // namespace

namespace {
class TosaToTTIRMultiplyOpConversionPattern
    : public TosaToTTIRDefaultBinaryDPSOpConversionPattern<tosa::MulOp,
                                                           ttir::MultiplyOp> {
  using TosaToTTIRDefaultBinaryDPSOpConversionPattern<
      tosa::MulOp,
      ttir::MultiplyOp>::TosaToTTIRDefaultBinaryDPSOpConversionPattern;
  using Adaptor = tosa::MulOp::Adaptor;

private:
  LogicalResult
  checkConversionLegality(tosa::MulOp srcOp, Adaptor adaptor,
                          ConversionPatternRewriter &rewriter) const override {
    TypedValue<RankedTensorType> shift = srcOp.getShift();
    if (!shift) {
      return success();
    }

    auto constOp = shift.getDefiningOp<tosa::ConstOp>();
    if (!constOp) {
      return srcOp.emitOpError(
          "conversion expects shift value to be defined by a "
          "tosa.const op");
    }

    auto denseIntAttr =
        mlir::dyn_cast<DenseIntElementsAttr>(constOp.getValues());
    if (!denseIntAttr) {
      return srcOp.emitOpError("conversion expects shift value to come from a "
                               "DenseIntElementsAttr");
    }

    if (denseIntAttr.getSplatValue<APInt>().getSExtValue() != 0) {
      return srcOp.emitOpError("conversion does not support shifted multiply");
    }

    return success();
  }
};
} // namespace

namespace {
class TosaToTTIRReshapeOpConversionPattern
    : public OpConversionPattern<tosa::ReshapeOp> {
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;
  using Adaptor = tosa::ReshapeOp::Adaptor;

public:
  LogicalResult
  matchAndRewrite(tosa::ReshapeOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    llvm::SmallVector<int32_t> newShape(outputType.getShape());
    ArrayAttr newShapeAttr = rewriter.getI32ArrayAttr(newShape);
    ttir::utils::replaceOpWithNewDPSOp<ttir::ReshapeOp>(
        rewriter, srcOp, outputType, adaptor.getInput(), newShapeAttr);
    return success();
  }
};
} // namespace

namespace {
class TosaToTTIRClampOpConversionPattern
    : public OpConversionPattern<tosa::ClampOp> {
  using OpConversionPattern<tosa::ClampOp>::OpConversionPattern;
  using Adaptor = tosa::ClampOp::Adaptor;

public:
  LogicalResult
  matchAndRewrite(tosa::ClampOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    FloatAttr minValAttr;
    FloatAttr maxValueAttr;

    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(srcOp.getMinVal())) {
      minValAttr = rewriter.getF32FloatAttr(intAttr.getSInt());
    } else {
      minValAttr = rewriter.getF32FloatAttr(
          cast<mlir::FloatAttr>(srcOp.getMinVal()).getValue().convertToFloat());
    }

    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(srcOp.getMaxVal())) {
      maxValueAttr = rewriter.getF32FloatAttr(intAttr.getSInt());
    } else {
      maxValueAttr = rewriter.getF32FloatAttr(
          cast<mlir::FloatAttr>(srcOp.getMaxVal()).getValue().convertToFloat());
    }

    ttir::utils::replaceOpWithNewDPSOp<ttir::ClampScalarOp>(
        rewriter, srcOp, outputType, adaptor.getInput(), minValAttr,
        maxValueAttr);

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

    ttir::utils::replaceOpWithNewDPSOp<ttir::ConcatOp>(
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

    ttir::utils::replaceOpWithNewDPSOp<ttir::MatmulOp>(
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

    ttir::utils::replaceOpWithNewDPSOp<DestOp>(
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

    auto kernelAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(srcOp.getKernel()[0]),
        static_cast<int32_t>(srcOp.getKernel()[1]),
    });

    auto strideAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(srcOp.getStride()[0]),
        static_cast<int32_t>(srcOp.getStride()[1]),
    });

    auto dilationAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(1),
        static_cast<int32_t>(1),
    });

    // Tosa max pool 2D op has padding in the order of
    // [top, bottom, left, right], while TTIR expects it in the order
    // of [top, left, bottom, right].
    // Thus, we need to rearrange the padding values.
    auto paddingAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(srcOp.getPad()[0]),
        static_cast<int32_t>(srcOp.getPad()[2]),
        static_cast<int32_t>(srcOp.getPad()[1]),
        static_cast<int32_t>(srcOp.getPad()[3]),
    });

    // TODO (azecevic) Add comment about the parameters.
    ttir::utils::replaceOpWithNewDPSOp<ttir::MaxPool2dOp>(
        rewriter, srcOp, outputType, adaptor.getInput(), kernelAttr, strideAttr,
        dilationAttr, paddingAttr, /*ceil_mode=*/false,
        /*flattened_compat_info=*/nullptr);

    return success();
  }
};
} // namespace

namespace {
class TosaToTTIRConstantOpConversionPattern
    : public OpConversionPattern<tosa::ConstOp> {
  using OpConversionPattern<tosa::ConstOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(tosa::ConstOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttir::ConstantOp>(
        srcOp,
        this->getTypeConverter()->convertType(srcOp.getResult().getType()),
        srcOp.getValues());
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
  patterns.add<TosaToTTIRDefaultUnaryDPSOpConversionPattern<
      tosa::NegateOp, mlir::tt::ttir::NegOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::FloorOp, mlir::tt::ttir::FloorOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::ReciprocalOp, mlir::tt::ttir::ReciprocalOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::RsqrtOp, mlir::tt::ttir::RsqrtOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::SigmoidOp, mlir::tt::ttir::SigmoidOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<tosa::SinOp,
                                                       mlir::tt::ttir::SinOp>>(
      typeConverter, ctx);
  patterns.add<TosaToTTIRClampOpConversionPattern>(typeConverter, ctx);
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

static void addBitwiseOpsConversionPatterns(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::BitwiseAndOp, mlir::tt::ttir::BitwiseAndOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::BitwiseNotOp, mlir::tt::ttir::BitwiseNotOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::BitwiseOrOp, mlir::tt::ttir::BitwiseOrOp>>(typeConverter, ctx);
  patterns.add<TosaToTTIRDefaultDPSOpConversionPattern<
      tosa::BitwiseXorOp, mlir::tt::ttir::BitwiseXorOp>>(typeConverter, ctx);
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

void addShapeOpsConversionPatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   TypeConverter &typeConverter) {
  patterns.add<TosaToTTIRReshapeOpConversionPattern>(typeConverter, ctx);
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
  addBitwiseOpsConversionPatterns(ctx, patterns, typeConverter);
  addCompareOpsConversionPatterns(ctx, patterns, typeConverter);
  addMatmulOpsConversionPatterns(ctx, patterns, typeConverter);
  addReductionOpsConversionPatterns(ctx, patterns, typeConverter);
  addPoolingOpsConversionPatterns(ctx, patterns, typeConverter);
  addShapeOpsConversionPatterns(ctx, patterns, typeConverter);

  patterns.add<TosaToTTIRConcatOpConversionPattern,
               TosaToTTIRConstantOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
