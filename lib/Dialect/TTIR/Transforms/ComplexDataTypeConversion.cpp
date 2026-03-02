// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRCOMPLEXDATATYPECONVERSIONPASS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

struct StablehloComplexToComplexPattern
    : public OpConversionPattern<ttir::StablehloComplexOp> {
  using OpConversionPattern<ttir::StablehloComplexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::StablehloComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto newOp = rewriter.create<ttir::ComplexOp>(
        op.getLoc(), resultType, adaptor.getReal(), adaptor.getImag());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct StablehloRealToRealPattern
    : public OpConversionPattern<ttir::StablehloRealOp> {
  using OpConversionPattern<ttir::StablehloRealOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::StablehloRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto newOp = rewriter.create<ttir::RealOp>(op.getLoc(), resultType,
                                               adaptor.getInput());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct StablehloImagToImagPattern
    : public OpConversionPattern<ttir::StablehloImagOp> {
  using OpConversionPattern<ttir::StablehloImagOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::StablehloImagOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto newOp = rewriter.create<ttir::ImagOp>(op.getLoc(), resultType,
                                               adaptor.getInput());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConstantComplexConversionPattern
    : public OpConversionPattern<ttir::ConstantOp> {
  using OpConversionPattern<ttir::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConstantOp op, OpAdaptor adaptor,
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
    rewriter.replaceOpWithNewOp<ttir::ConstantOp>(op, newType, newAttr);
    return success();
  }
};

struct TTIRComplexDataTypeConversionPass
    : public impl::TTIRComplexDataTypeConversionPassBase<
          TTIRComplexDataTypeConversionPass> {
  using impl::TTIRComplexDataTypeConversionPassBase<
      TTIRComplexDataTypeConversionPass>::TTIRComplexDataTypeConversionPassBase;

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<BuiltinDialect>();

    // Complex constants must be converted to float-pair constants.
    target.addDynamicallyLegalOp<ttir::ConstantOp>([](ttir::ConstantOp op) {
      auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
      return !mlir::isa<mlir::ComplexType>(resultType.getElementType());
    });

    // Stablehlo complex ops and intermediate complex ops are illegal.
    target.addIllegalOp<ttir::StablehloComplexOp>();
    target.addIllegalOp<ttir::StablehloRealOp>();
    target.addIllegalOp<ttir::StablehloImagOp>();

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
    patterns.add<StablehloComplexToComplexPattern>(typeConverter,
                                                   &getContext());
    patterns.add<StablehloRealToRealPattern>(typeConverter, &getContext());
    patterns.add<StablehloImagToImagPattern>(typeConverter, &getContext());

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

} // namespace mlir::tt::ttir
