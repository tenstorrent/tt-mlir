// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"

#include "mlir/Dialect/Traits.h"
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <stablehlo/dialect/StablehloOps.h>

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

using namespace mlir;
using namespace mlir::tt;

namespace {

template <typename SrcOp, typename DestOp,
          typename Adaptor = typename SrcOp::Adaptor>
class StableHLOToTTIROpDefaultConversionPattern
    : public OpConversionPattern<SrcOp> {

  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
    rewriter.replaceOpWithNewOp<DestOp>(
        srcOp,
        TypeRange(
            this->getTypeConverter()->convertType(outputTensor.getType())),
        adaptor.getOperands(), ValueRange(outputTensor),
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }
};

class StableHLOToTTIRReduceOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReduceOp> {

  using OpConversionPattern<mlir::stablehlo::ReduceOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceOp srcOp,
                  mlir::stablehlo::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult err = checkBasicLegality(srcOp, adaptor, rewriter);
    if (not err.succeeded()) {
      return err;
    }

    const mlir::Operation &innerOp = srcOp.getBody().front().front();

    if (mlir::isa<mlir::stablehlo::AddOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::SumOp>(srcOp, adaptor,
                                                            rewriter);
    }
    if (mlir::isa<mlir::stablehlo::MaxOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::MaxOp>(srcOp, adaptor,
                                                            rewriter);
    }

    return failure();
  }

private:
  LogicalResult checkBasicLegality(mlir::stablehlo::ReduceOp &srcOp,
                                   mlir::stablehlo::ReduceOp::Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    if (!srcOp.getBody().hasOneBlock()) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "Expecting StableHLO Reduce OP to have one block inside its body.");
    }

    if (srcOp.getBody().front().empty()) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "Expecting StableHLO Reduce OP to have a body operation defined.");
    }

    return success();
  }

  template <typename DestOp>
  LogicalResult
  matchAndRewriteInternal(mlir::stablehlo::ReduceOp &srcOp,
                          mlir::stablehlo::ReduceOp::Adaptor &adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResultTypes().front()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    mlir::ArrayAttr dimArg = rewriter.getArrayAttr(SmallVector<Attribute>(
        1, rewriter.getI32IntegerAttr(adaptor.getDimensionsAttr()[0])));

    // If someone changes definition of TTIR_ReductionOp this constant will
    // become outdated, but I currently see no way to get this info (without
    // manually constructing the adaptor for dest OP).
    const std::size_t ttirReduceOpOperandsCount = 2;
    mlir::ArrayAttr operandConstraints =
        rewriter.getArrayAttr(SmallVector<Attribute>(
            ttirReduceOpOperandsCount, rewriter.getAttr<OperandConstraintAttr>(
                                           OperandConstraint::AnyDeviceTile)));

    rewriter.replaceOpWithNewOp<DestOp>(
        srcOp, outputType, adaptor.getInputs().front(), outputTensor,
        false /* keep_dim */, dimArg, operandConstraints);

    return success();
  }
};

class StableHLOToTTIRTransposeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::TransposeOp> {
  using OpConversionPattern<mlir::stablehlo::TransposeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::TransposeOp srcOp,
                  mlir::stablehlo::TransposeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    LogicalResult err = checkBasicLegality(srcOp, adaptor, rewriter);
    if (not err.succeeded()) {
      return err;
    }

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::TransposeOp>(
        srcOp, getTypeConverter()->convertType(outputTensor.getType()),
        Value(adaptor.getOperand()), Value(outputTensor),
        adaptor.getPermutation()[0], adaptor.getPermutation()[1],
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }

  LogicalResult
  checkBasicLegality(mlir::stablehlo::TransposeOp &srcOp,
                     mlir::stablehlo::TransposeOp::Adaptor &adaptor,
                     ConversionPatternRewriter &rewriter) const {

    if (adaptor.getPermutation().size() != 2) {
      return rewriter.notifyMatchFailure(
          srcOp, "TTIR supports only two dimensional transposeOp.");
    }

    return success();
  }
};

class StableHLOToTTIRDotGeneralOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::DotGeneralOp> {
  using OpConversionPattern<mlir::stablehlo::DotGeneralOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::DotGeneralOp srcOp,
                  mlir::stablehlo::DotGeneralOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    // This is a basic version that can only work for cases that can be directly
    // converted to matmul. The op should be extended as other ops such as
    // ttir.permute and ttir.broadcast_in_dim become available.

    LogicalResult err = checkBasicLegality(srcOp, adaptor, rewriter);
    if (not err.succeeded()) {
      return err;
    }

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::MatmulOp>(
        srcOp, getTypeConverter()->convertType(outputTensor.getType()),
        adaptor.getLhs(), adaptor.getRhs(), Value(outputTensor),
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }

private:
  LogicalResult
  checkBasicLegality(mlir::stablehlo::DotGeneralOp &srcOp,
                     mlir::stablehlo::DotGeneralOp::Adaptor &adaptor,
                     ConversionPatternRewriter &rewriter) const {

    ::mlir::stablehlo::DotDimensionNumbersAttr dimensions =
        adaptor.getDotDimensionNumbers();

    if (dimensions.getLhsContractingDimensions().empty() ||
        dimensions.getRhsContractingDimensions().empty()) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Contracting dimension is missing.");
    }

    if (dimensions.getLhsContractingDimensions()[0] != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "Only non-transposed matmul is currently supported in TTIR.");
    }

    if (dimensions.getRhsContractingDimensions()[0] != 0) {
      return rewriter.notifyMatchFailure(
          srcOp, "Only non-transposed matmul is currently supported in TTIR.");
    }

    if (not dimensions.getLhsBatchingDimensions().empty()) {
      return rewriter.notifyMatchFailure(
          srcOp, "Only non-transposed matmul is currently supported in TTIR.");
    }

    if (not dimensions.getRhsBatchingDimensions().empty()) {
      return rewriter.notifyMatchFailure(
          srcOp, "Only non-transposed matmul is currently supported in TTIR.");
    }

    return success();
  }
};

class StableHLOToTTIRConstantOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ConstantOp> {

  using OpConversionPattern<mlir::stablehlo::ConstantOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConstantOp srcOp,
                  mlir::stablehlo::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ConstantOp>(srcOp, outputType,
                                                            srcOp.getValue());
    return success();
  }
};

class StableHLOToTTIRBroadcastInDimOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpConversionPattern<
      mlir::stablehlo::BroadcastInDimOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::BroadcastInDimOp srcOp,
                  mlir::stablehlo::BroadcastInDimOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    LogicalResult err = checkBasicLegality(srcOp, adaptor, rewriter);
    if (not err.succeeded()) {
      return err;
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    mlir::ArrayAttr dimArg =
        rewriter.getI64ArrayAttr(adaptor.getBroadcastDimensions());

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::BroadcastOp>(
        srcOp, getTypeConverter()->convertType(outputTensor.getType()),
        Value(adaptor.getOperand()), Value(outputTensor), dimArg,
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));

    return success();
  }

private:
  LogicalResult
  checkBasicLegality(mlir::stablehlo::BroadcastInDimOp &srcOp,
                     mlir::stablehlo::BroadcastInDimOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {

    llvm::SmallVector<int64_t, 4> broadcastedShape;
    auto srcType =
        getTypeConverter()->convertType(srcOp.getOperand().getType());
    auto inputShape = mlir::cast<mlir::RankedTensorType>(srcType).getShape();
    auto outputShape = mlir::cast<mlir::RankedTensorType>(srcType).getShape();

    if (!OpTrait::util::getBroadcastedShape(inputShape, outputShape,
                                            broadcastedShape)) {
      return rewriter.notifyMatchFailure(
          srcOp, "Input cannot be broadcasted to provided dimensions.");
    }

    return success();
  }
};

class StableHLOToTTIRCompareOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompareOp> {
  using OpConversionPattern<mlir::stablehlo::CompareOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompareOp srcOp,
                  mlir::stablehlo::CompareOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // StableHLO has one 'compare' op to do all type of comparison (EQ, NE, GE,
    // GT, LE, and LT) and use direction to determine the type of comparison.
    mlir::stablehlo::ComparisonDirection direction =
        srcOp.getComparisonDirection();

    switch (direction) {
    case mlir::stablehlo::ComparisonDirection::EQ: {
      return matchAndRewriteInternal<mlir::tt::ttir::EqualOp>(srcOp, adaptor,
                                                              rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::NE: {
      return matchAndRewriteInternal<mlir::tt::ttir::NotEqualOp>(srcOp, adaptor,
                                                                 rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::GE: {
      return matchAndRewriteInternal<mlir::tt::ttir::GreaterEqualOp>(
          srcOp, adaptor, rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::GT: {
      return matchAndRewriteInternal<mlir::tt::ttir::GreaterThanOp>(
          srcOp, adaptor, rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::LE: {
      return matchAndRewriteInternal<mlir::tt::ttir::LessEqualOp>(
          srcOp, adaptor, rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::LT: {
      return matchAndRewriteInternal<mlir::tt::ttir::LessThanOp>(srcOp, adaptor,
                                                                 rewriter);
      break;
    }
    }
    return success();
  }

private:
  template <typename DestOp>
  LogicalResult
  matchAndRewriteInternal(mlir::stablehlo::CompareOp srcOp,
                          mlir::stablehlo::CompareOp::Adaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    // TTNN doesn't support boolean data type. TTNN compare operations have same
    // output data type as of input data type (e.g. comparing float32 will
    // produce float32 result). So input operand is used to create output type
    // and output tensor and the generated output tensor will be different type
    // (instead of boolean) depending on input operands.
    mlir::RankedTensorType outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp->getOperand(0).getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    DestOp TTIROp = rewriter.replaceOpWithNewOp<DestOp>(
        srcOp,
        TypeRange(
            this->getTypeConverter()->convertType(outputTensor.getType())),
        adaptor.getOperands(), ValueRange(outputTensor),
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));

    // rewriter may miss replacing all uses due to different output tensor type.
    // Replacing all uses of srcOp explicitly.
    rewriter.replaceAllOpUsesWith(srcOp, TTIROp);

    return success();
  }
};

void addElementwiseUnaryOpsConversionPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::AbsOp, mlir::tt::ttir::AbsOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::ConvertOp, mlir::tt::ttir::TypecastOp>>(typeConverter,
                                                               ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::ExpOp, mlir::tt::ttir::ExpOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::NegOp, mlir::tt::ttir::NegOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::RsqrtOp, mlir::tt::ttir::RsqrtOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SqrtOp, mlir::tt::ttir::SqrtOp>>(typeConverter, ctx);
}

void addElementwiseBinaryOpsConversionPatterns(MLIRContext *ctx,
                                               RewritePatternSet &patterns,
                                               TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::AddOp, mlir::tt::ttir::AddOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SubtractOp, mlir::tt::ttir::SubtractOp>>(typeConverter,
                                                                ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::MulOp, mlir::tt::ttir::MultiplyOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::DivOp, mlir::tt::ttir::DivOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::MaxOp, mlir::tt::ttir::MaximumOp>>(typeConverter, ctx);
}

void addReduceOpsConversionPatterns(MLIRContext *ctx,
                                    RewritePatternSet &patterns,
                                    TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReduceOpConversionPattern>(typeConverter, ctx);
}

void addTransposeOpsConversionPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIRTransposeOpConversionPattern>(typeConverter, ctx);
}

void addMatmulOpsConversionPatterns(MLIRContext *ctx,
                                    RewritePatternSet &patterns,
                                    TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRDotGeneralOpConversionPattern>(typeConverter,
                                                             ctx);
}

void addTensorCreationOpsConversionPatterns(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRConstantOpConversionPattern>(typeConverter, ctx);
}

void addBroadcastOpConversionPattern(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIRBroadcastInDimOpConversionPattern>(typeConverter,
                                                                 ctx);
}

void addCompareOpsConversionPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRCompareOpConversionPattern>(typeConverter, ctx);
}

} // namespace

namespace mlir::tt {

void populateStableHLOToTTIRPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  addElementwiseUnaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addElementwiseBinaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addReduceOpsConversionPatterns(ctx, patterns, typeConverter);
  addTransposeOpsConversionPatterns(ctx, patterns, typeConverter);
  addMatmulOpsConversionPatterns(ctx, patterns, typeConverter);
  addTensorCreationOpsConversionPatterns(ctx, patterns, typeConverter);
  addBroadcastOpConversionPattern(ctx, patterns, typeConverter);
  addCompareOpsConversionPatterns(ctx, patterns, typeConverter);
}

} // namespace mlir::tt
