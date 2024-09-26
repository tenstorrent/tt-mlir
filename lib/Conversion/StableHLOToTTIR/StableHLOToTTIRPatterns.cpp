// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"

#include <vector>

#include "mlir/Dialect/Traits.h"
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
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
    RankedTensorType outputType = mlir::cast<RankedTensorType>(
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

    if (srcOp.getDimensions().size() > 2) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "Reduce on more than two dimensions is not currently supported.");
    }

    RankedTensorType inputTensorType =
        mlir::cast<RankedTensorType>(srcOp.getInputs().getTypes().front());
    int64_t inputTensorRank =
        static_cast<int64_t>(inputTensorType.getShape().size());
    for (int64_t dim : srcOp.getDimensions()) {
      if (dim < -inputTensorRank || dim >= inputTensorRank) {
        return rewriter.notifyMatchFailure(
            srcOp, "Reduce dimensions are out of range.");
      }
    }

    return success();
  }

  template <typename DestOp>
  LogicalResult
  matchAndRewriteInternal(mlir::stablehlo::ReduceOp &srcOp,
                          mlir::stablehlo::ReduceOp::Adaptor &adaptor,
                          ConversionPatternRewriter &rewriter) const {
    RankedTensorType inputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getInputs().getTypes().front()));
    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResultTypes().front()));

    // If someone changes definition of TTIR_ReductionOp or TTIR_ReshapeOp this
    // constant will become outdated, but I currently see no way to get this
    // info (without manually constructing the adaptor for dest OP).
    const std::size_t operandsCount = 2;
    mlir::ArrayAttr operandConstraints =
        rewriter.getArrayAttr(SmallVector<Attribute>(
            operandsCount, rewriter.getAttr<OperandConstraintAttr>(
                               OperandConstraint::AnyDeviceTile)));

    // StableHLO.ReduceOp always removes reduce dimensions from the result shape
    // so ideally we would just convert it to TTIR.ReduceOp with keepDim=False.
    // Unfortunately we cannot do this because Metal TTNN implementation of
    // Reduce doesn't yet support keepDim=False. As a workaround, we convert it
    // to combination of TTIR.ReduceOp with keepDim=True + TTIR.ReshapeOp to
    // remove the reduce dims so that the rest of the graph is not affected.
    // In case when this is not needed (because type converter already promoted
    // rank of the op result) then we avoid adding unnecessary Reshape op.
    DestOp reduceOp = createReduceOp<DestOp>(
        srcOp, adaptor, rewriter, inputType, outputType, operandConstraints);
    if (outputType.getShape().size() < inputType.getShape().size()) {
      createReshapeOp<DestOp>(srcOp, reduceOp, rewriter, outputType,
                              operandConstraints);
    } else {
      rewriter.replaceOp(srcOp, reduceOp);
    }

    return success();
  }

  template <typename DestOp>
  DestOp createReduceOp(mlir::stablehlo::ReduceOp &srcOp,
                        mlir::stablehlo::ReduceOp::Adaptor &adaptor,
                        ConversionPatternRewriter &rewriter,
                        RankedTensorType &inputType,
                        RankedTensorType &outputType,
                        mlir::ArrayAttr &operandConstraints) const {
    std::vector<int64_t> reduceOpShapeVec = inputType.getShape().vec();
    for (int64_t dim : srcOp.getDimensions()) {
      reduceOpShapeVec[dim < 0 ? srcOp.getDimensions().size() -
                                     static_cast<size_t>(-dim)
                               : static_cast<size_t>(dim)] = 1;
    }
    llvm::ArrayRef<int64_t> reduceOpShape(reduceOpShapeVec);

    RankedTensorType reduceOpOutputType = RankedTensorType::get(
        reduceOpShape, outputType.getElementType(), outputType.getEncoding());

    tensor::EmptyOp reduceOpOutputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), reduceOpOutputType.getShape(),
        reduceOpOutputType.getElementType(), reduceOpOutputType.getEncoding());

    mlir::ArrayAttr reduceOpDimArg =
        rewriter.getI32ArrayAttr(llvm::ArrayRef<int32_t>(std::vector<int32_t>(
            srcOp.getDimensions().begin(), srcOp.getDimensions().end())));

    DestOp reduceOp = rewriter.create<DestOp>(
        srcOp.getLoc(), reduceOpOutputType, adaptor.getInputs().front(),
        reduceOpOutputTensor, true /* keep_dim */, reduceOpDimArg,
        operandConstraints);

    return reduceOp;
  }

  template <typename DestOp>
  void createReshapeOp(mlir::stablehlo::ReduceOp &srcOp, DestOp &reduceOp,
                       ConversionPatternRewriter &rewriter,
                       RankedTensorType &outputType,
                       mlir::ArrayAttr &operandConstraints) const {
    tensor::EmptyOp reshapeOpOutputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType(),
        outputType.getEncoding());

    mlir::ArrayAttr reshapeOpShape =
        rewriter.getI32ArrayAttr(llvm::ArrayRef<int32_t>(std::vector<int32_t>(
            outputType.getShape().begin(), outputType.getShape().end())));

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ReshapeOp>(
        srcOp, outputType, reduceOp, reshapeOpOutputTensor, reshapeOpShape,
        operandConstraints);
  }

  // TODO: When Metal adds support for keep_dim=False we should switch to this
  // version of matchAndRewrite (issue #).
  template <typename DestOp>
  LogicalResult matchAndRewriteInternalWithKeepDimFalse(
      mlir::stablehlo::ReduceOp &srcOp,
      mlir::stablehlo::ReduceOp::Adaptor &adaptor,
      ConversionPatternRewriter &rewriter) const {
    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResultTypes().front()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    mlir::ArrayAttr dimArg =
        rewriter.getI32ArrayAttr(llvm::ArrayRef<int32_t>(std::vector<int32_t>(
            srcOp.getDimensions().begin(), srcOp.getDimensions().end())));

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
    RankedTensorType outputType = mlir::cast<RankedTensorType>(
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
    RankedTensorType outputType = mlir::cast<RankedTensorType>(
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

// class StableHLOToTTIRConstantOpConversionPattern
//     : public OpConversionPattern<mlir::stablehlo::ConstantOp> {

//   using
//   OpConversionPattern<mlir::stablehlo::ConstantOp>::OpConversionPattern;

// public:
//   LogicalResult
//   matchAndRewrite(mlir::stablehlo::ConstantOp srcOp,
//                   mlir::stablehlo::ConstantOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     RankedTensorType outputType = mlir::cast<RankedTensorType>(
//         getTypeConverter()->convertType(srcOp.getResult().getType()));

//     rewriter.replaceOpWithNewOp<mlir::tt::ttir::ConstantOp>(srcOp,
//     outputType,
//                                                             srcOp.getValue());
//     return success();
//   }
// };

class StableHLOToTTIRConstantOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ConstantOp> {

  using OpConversionPattern<mlir::stablehlo::ConstantOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConstantOp srcOp,
                  mlir::stablehlo::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult = checkBasicLegality(srcOp, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    mlir::ElementsAttr valueAttr = srcOp.getValue();
    if (valueAttr.getShapedType().getShape().empty()) {
      // Scalar tensors are not supported by TTIR so we have to convert them to
      // 1-D tensors.
      mlir::ShapedType valueType = mlir::cast<mlir::ShapedType>(
          getTypeConverter()->convertType(valueAttr.getShapedType()));
      // It is important to separate these two get calls and invoke them with
      // explicit type because template type inference fails in this case,
      // integer values are being detected as float element type.
      if (valueAttr.getElementType().isInteger()) {
        valueAttr = mlir::DenseElementsAttr::get<int>(
            valueType, valueAttr.getSplatValue<int>());
      } else {
        valueAttr = mlir::DenseElementsAttr::get<float>(
            valueType, valueAttr.getSplatValue<float>());
      }
    }

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ConstantOp>(srcOp, outputType,
                                                            valueAttr);
    return success();
  }

private:
  LogicalResult checkBasicLegality(mlir::stablehlo::ConstantOp &srcOp,
                                   ConversionPatternRewriter &rewriter) const {
    if (srcOp.getValue().getShapedType().getShape().empty() &&
        !srcOp.getValue().getElementType().isIntOrFloat()) {
      return rewriter.notifyMatchFailure(srcOp, "Unsupported element type.");
    }

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

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
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
    mlir::Type srcType =
        getTypeConverter()->convertType(srcOp.getOperand().getType());
    llvm::ArrayRef<int64_t> inputShape =
        mlir::cast<mlir::RankedTensorType>(srcType).getShape();
    llvm::ArrayRef<int64_t> outputShape =
        mlir::cast<mlir::RankedTensorType>(srcType).getShape();

    if (!OpTrait::util::getBroadcastedShape(inputShape, outputShape,
                                            broadcastedShape)) {
      return rewriter.notifyMatchFailure(
          srcOp, "Input cannot be broadcasted to provided dimensions.");
    }

    return success();
  }
};

void addElementwiseUnaryOpsConversionPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::AbsOp, mlir::tt::ttir::AbsOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SqrtOp, mlir::tt::ttir::SqrtOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::RsqrtOp, mlir::tt::ttir::RsqrtOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::ExpOp, mlir::tt::ttir::ExpOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::NegOp, mlir::tt::ttir::NegOp>>(typeConverter, ctx);
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
}

} // namespace mlir::tt
