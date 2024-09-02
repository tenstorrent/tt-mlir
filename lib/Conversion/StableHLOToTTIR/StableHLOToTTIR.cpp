// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <stablehlo/dialect/StablehloOps.h>

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

using namespace mlir;
using namespace tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

template <typename SrcOp, typename DestOp,
          typename Adaptor = typename SrcOp::Adaptor>
class StableHLOToTTIROpDefaultConversionPattern :
  public OpConversionPattern<SrcOp> {

using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType =
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
};

template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
class StableHLOToTTIRReduceOpConversionPattern :
  public OpConversionPattern<SrcOp> {

using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!checkBasicLegality(srcOp)) {
      return failure();
    }

    mlir::TypeID srcOpTypeID =
      srcOp.getBody().front().front().getName().getTypeID();

    // Convert based on type.
    if (srcOpTypeID ==
        mlir::detail::TypeIDResolver<
          mlir::stablehlo::AddOp>::resolveTypeID()) {
      return matchAndRewriteInternal<mlir::tt::ttir::SumOp>(
        srcOp, adaptor, rewriter);
    }
    else if (srcOpTypeID ==
             mlir::detail::TypeIDResolver<
               mlir::stablehlo::MaxOp>::resolveTypeID()) {
      return matchAndRewriteInternal<mlir::tt::ttir::MaxOp>(
        srcOp, adaptor, rewriter);
    }

    return failure();
  }

private:
  bool checkBasicLegality(SrcOp &srcOp) const {
    if (!srcOp.getBody().hasOneBlock()) {
      // Expecting StableHLO Reduce OP to have one block inside its body.
      return false;
    }

    if (srcOp.getBody().front().empty()) {
      // Expecting StableHLO Reduce OP to have a body operation defined.
      return false;
    }

    return true;
  }

  template <typename DestOp>
  LogicalResult
  matchAndRewriteInternal(SrcOp &srcOp, Adaptor &adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto outputType =
      mlir::cast<RankedTensorType>(srcOp.getResultTypes().front());
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
      srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    mlir::ArrayAttr dimArg =
      rewriter.getArrayAttr(
        SmallVector<Attribute>(1,
                               rewriter.getI32IntegerAttr(
                                 adaptor.getDimensionsAttr()[0])));

    // If someone changes definition of TTIR_ReductionOp this constant will
    // become outdated, but I currently see no way to get this info (without
    // manually constructing the adaptor for dest OP).
    const std::size_t ttirReduceOpOperandsCount = 2;
    mlir::ArrayAttr operandConstraints =
      rewriter.getArrayAttr(
        SmallVector<Attribute>(ttirReduceOpOperandsCount,
                               rewriter.getAttr<OperandConstraintAttr>(
                                 OperandConstraint::AnyDeviceTile)));

    rewriter.replaceOpWithNewOp<DestOp>(
      srcOp, outputType, adaptor.getInputs().front(), outputTensor,
      false /* keep_dim */, dimArg, operandConstraints);

    return success();
  }
};

struct ConvertStableHLOToTTIRPass
  : public ttir::impl::ConvertStableHLOToTTIRBase<
      ConvertStableHLOToTTIRPass> {

  void addElementwiseUnaryOpsConversionPatterns(
    RewritePatternSet &patterns,
    const TypeConverter &typeConverter) {

    patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::ExpOp, mlir::tt::ttir::ExpOp>>(
        typeConverter, &getContext());
  }

  void addElementwiseBinaryOpsConversionPatterns(
    RewritePatternSet &patterns,
    const TypeConverter &typeConverter) {

    patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::AddOp, mlir::tt::ttir::AddOp>>(
        typeConverter, &getContext());
    patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SubtractOp, mlir::tt::ttir::SubtractOp>>(
        typeConverter, &getContext());
    patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::MulOp, mlir::tt::ttir::MultiplyOp>>(
        typeConverter, &getContext());
    patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::DivOp, mlir::tt::ttir::DivOp>>(
        typeConverter, &getContext());
  }

  void addReduceOpsConversionPatterns(RewritePatternSet &patterns,
                                      const TypeConverter &typeConverter) {
    patterns.add<StableHLOToTTIRReduceOpConversionPattern<
        mlir::stablehlo::ReduceOp>>(typeConverter, &getContext());
  }

  void addConversionPatterns(RewritePatternSet &patterns,
                             const TypeConverter &typeConverter) {
    addElementwiseUnaryOpsConversionPatterns(patterns, typeConverter);
    addElementwiseBinaryOpsConversionPatterns(patterns, typeConverter);
    addReduceOpsConversionPatterns(patterns, typeConverter);
  }

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();

    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalOp<mlir::tensor::EmptyOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();

    // For now keep the same type assuming StableHLO ops operate on builtin
    // tensor.
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) {
      assert(isa<RankedTensorType>(type) &&
             "only ranked tensor type supported");
      return type;
    });
    RewritePatternSet patterns(&getContext());

    addConversionPatterns(patterns, typeConverter);

    // Apply conversion.
    if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertStableHLOToTTIRPass() {
  return std::make_unique<ConvertStableHLOToTTIRPass>();
}

} // namespace mlir::tt
