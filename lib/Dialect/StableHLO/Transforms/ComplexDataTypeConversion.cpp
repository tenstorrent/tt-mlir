// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_STABLEHLOCOMPLEXDATATYPECONVERSIONPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

// stablehlo.complex(lhs, rhs) → unsqueeze both + concat on trailing dim.
//   tensor<...xfN>, tensor<...xfN> → tensor<...x2xfN>
struct StablehloComplexToDecomposedPattern
    : public OpConversionPattern<mlir::stablehlo::ComplexOp> {
  using OpConversionPattern<mlir::stablehlo::ComplexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    auto lhsType = mlir::cast<RankedTensorType>(adaptor.getLhs().getType());
    SmallVector<int64_t> unsqueezedShape(lhsType.getShape());
    unsqueezedShape.push_back(1);
    auto unsqueezedType =
        RankedTensorType::get(unsqueezedShape, lhsType.getElementType());
    auto reshapedLhs = rewriter.create<mlir::stablehlo::ReshapeOp>(
        loc, unsqueezedType, adaptor.getLhs());
    auto reshapedRhs = rewriter.create<mlir::stablehlo::ReshapeOp>(
        loc, unsqueezedType, adaptor.getRhs());

    auto concatDim = static_cast<int64_t>(unsqueezedShape.size() - 1);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConcatenateOp>(
        op, resultType,
        ValueRange{reshapedLhs.getResult(), reshapedRhs.getResult()},
        concatDim);
    return success();
  }
};

// stablehlo.real(operand: tensor<...x2xfN>) → slice index 0 + squeeze.
//   tensor<...x2xfN> → tensor<...xfN>
struct StablehloRealToDecomposedPattern
    : public OpConversionPattern<mlir::stablehlo::RealOp> {
  using OpConversionPattern<mlir::stablehlo::RealOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::RealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType =
        mlir::cast<RankedTensorType>(adaptor.getOperand().getType());
    auto inputShape = inputType.getShape();
    int64_t rank = inputType.getRank();

    SmallVector<int64_t> begins(rank, 0);
    SmallVector<int64_t> ends(inputShape.begin(), inputShape.end());
    SmallVector<int64_t> steps(rank, 1);
    begins[rank - 1] = 0;
    ends[rank - 1] = 1;

    SmallVector<int64_t> sliceShape(inputShape.begin(), inputShape.end());
    sliceShape[rank - 1] = 1;
    auto sliceType =
        RankedTensorType::get(sliceShape, inputType.getElementType());

    auto sliceOp = rewriter.create<mlir::stablehlo::SliceOp>(
        loc, sliceType, adaptor.getOperand(),
        rewriter.getDenseI64ArrayAttr(begins),
        rewriter.getDenseI64ArrayAttr(ends),
        rewriter.getDenseI64ArrayAttr(steps));

    auto resultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(
        op, resultType, sliceOp.getResult());
    return success();
  }
};

// stablehlo.imag(operand: tensor<...x2xfN>) → slice index 1 + squeeze.
//   tensor<...x2xfN> → tensor<...xfN>
struct StablehloImagToDecomposedPattern
    : public OpConversionPattern<mlir::stablehlo::ImagOp> {
  using OpConversionPattern<mlir::stablehlo::ImagOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ImagOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType =
        mlir::cast<RankedTensorType>(adaptor.getOperand().getType());
    auto inputShape = inputType.getShape();
    int64_t rank = inputType.getRank();

    SmallVector<int64_t> begins(rank, 0);
    SmallVector<int64_t> ends(inputShape.begin(), inputShape.end());
    SmallVector<int64_t> steps(rank, 1);
    begins[rank - 1] = 1;
    ends[rank - 1] = 2;

    SmallVector<int64_t> sliceShape(inputShape.begin(), inputShape.end());
    sliceShape[rank - 1] = 1;
    auto sliceType =
        RankedTensorType::get(sliceShape, inputType.getElementType());

    auto sliceOp = rewriter.create<mlir::stablehlo::SliceOp>(
        loc, sliceType, adaptor.getOperand(),
        rewriter.getDenseI64ArrayAttr(begins),
        rewriter.getDenseI64ArrayAttr(ends),
        rewriter.getDenseI64ArrayAttr(steps));

    auto resultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(
        op, resultType, sliceOp.getResult());
    return success();
  }
};

struct ConstantComplexConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ConstantOp> {
  using OpConversionPattern<mlir::stablehlo::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    if (!mlir::isa<mlir::ComplexType>(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex element type");
    }

    auto newType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(resultType));
    auto denseAttr = mlir::cast<DenseElementsAttr>(op.getValue());

    SmallVector<APFloat> floatValues;
    for (auto complexVal : denseAttr.getValues<std::complex<APFloat>>()) {
      floatValues.push_back(complexVal.real());
      floatValues.push_back(complexVal.imag());
    }

    auto newAttr = DenseElementsAttr::get(newType, floatValues);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, newType,
                                                             newAttr);
    return success();
  }
};

struct ReshapeComplexConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReshapeOp> {
  using OpConversionPattern<mlir::stablehlo::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    if (!mlir::isa<mlir::ComplexType>(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex element type");
    }

    auto newResultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(resultType));

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(
        op, newResultType, adaptor.getOperand());
    return success();
  }
};

// stablehlo.broadcast_in_dim with complex result type:
//   tensor<...xcomplex<fN>> → tensor<...x2xfN>
// The converted operand already has a trailing x2 dim; append the
// corresponding output dim index to broadcast_dimensions.
struct BroadcastInDimComplexConversionPattern
    : public OpConversionPattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpConversionPattern<
      mlir::stablehlo::BroadcastInDimOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    if (!mlir::isa<mlir::ComplexType>(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex element type");
    }

    auto newResultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(resultType));

    // The operand is already converted to have a trailing x2 dimension.
    // Map that new trailing input dim to the new trailing output dim.
    auto dims = op.getBroadcastDimensions();
    SmallVector<int64_t> newDims(dims.begin(), dims.end());
    newDims.push_back(newResultType.getRank() - 1);

    rewriter.replaceOpWithNewOp<mlir::stablehlo::BroadcastInDimOp>(
        op, newResultType, adaptor.getOperand(),
        rewriter.getDenseI64ArrayAttr(newDims));
    return success();
  }
};

struct StableHLOComplexDataTypeConversionPass
    : public impl::StableHLOComplexDataTypeConversionPassBase<
          StableHLOComplexDataTypeConversionPass> {
  using impl::StableHLOComplexDataTypeConversionPassBase<
      StableHLOComplexDataTypeConversionPass>::
      StableHLOComplexDataTypeConversionPassBase;

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::stablehlo::StablehloDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<BuiltinDialect>();

    // Complex constants must be converted to float-pair constants.
    target.addDynamicallyLegalOp<mlir::stablehlo::ConstantOp>(
        [](mlir::stablehlo::ConstantOp op) {
          auto resultType =
              mlir::cast<RankedTensorType>(op.getResult().getType());
          return !mlir::isa<mlir::ComplexType>(resultType.getElementType());
        });

    // StableHLO complex ops are illegal.
    target.addIllegalOp<mlir::stablehlo::ComplexOp>();
    target.addIllegalOp<mlir::stablehlo::RealOp>();
    target.addIllegalOp<mlir::stablehlo::ImagOp>();

    // Reshape ops that produce complex tensors must be converted to operate
    // on the float-pair representation instead.
    target.addDynamicallyLegalOp<mlir::stablehlo::ReshapeOp>(
        [](mlir::stablehlo::ReshapeOp op) {
          auto resultType =
              mlir::cast<RankedTensorType>(op.getResult().getType());
          return !mlir::isa<mlir::ComplexType>(resultType.getElementType());
        });

    // broadcast_in_dim ops that produce complex tensors must be converted.
    target.addDynamicallyLegalOp<mlir::stablehlo::BroadcastInDimOp>(
        [](mlir::stablehlo::BroadcastInDimOp op) {
          auto resultType =
              mlir::cast<RankedTensorType>(op.getResult().getType());
          return !mlir::isa<mlir::ComplexType>(resultType.getElementType());
        });

    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });
    // tensor<...xcomplex<fN>> → tensor<...x2xfN>: append a trailing dimension
    // of 2 for [real, imag] components, keeping the element type as fN.
    typeConverter.addConversion(
        [](RankedTensorType type) -> std::optional<Type> {
          auto complexTy =
              mlir::dyn_cast<mlir::ComplexType>(type.getElementType());
          if (!complexTy) {
            return std::nullopt;
          }
          auto floatTy =
              mlir::dyn_cast<mlir::FloatType>(complexTy.getElementType());
          if (!floatTy) {
            return std::nullopt;
          }
          SmallVector<int64_t> newShape(type.getShape());
          newShape.push_back(2);
          return RankedTensorType::get(newShape, floatTy);
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantComplexConversionPattern>(typeConverter,
                                                   &getContext());
    patterns.add<StablehloComplexToDecomposedPattern>(typeConverter,
                                                      &getContext());
    patterns.add<StablehloRealToDecomposedPattern>(typeConverter,
                                                   &getContext());
    patterns.add<StablehloImagToDecomposedPattern>(typeConverter,
                                                   &getContext());
    patterns.add<ReshapeComplexConversionPattern>(typeConverter, &getContext());
    patterns.add<BroadcastInDimComplexConversionPattern>(typeConverter,
                                                         &getContext());

    // Function type conversions: update func signatures and return ops so that
    // complex<fN> argument/result types become float-pair types.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::stablehlo
