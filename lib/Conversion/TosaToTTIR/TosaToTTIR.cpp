// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TosaToTTIR/TosaToTTIR.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTOSATOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

template <typename SrcOp, typename DestOp,
          typename Adaptor = typename SrcOp::Adaptor>
class TosaToTTIROpConversionPattern : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if constexpr (std::is_same<SrcOp, tosa::MulOp>::value) {
      assert(srcOp.getShift() == 0);
    }

    auto outputType = mlir::cast<RankedTensorType>(srcOp.getResult().getType());
    auto outputTensor = rewriter.create<tensor::EmptyOp>(
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

struct ConvertTosaToTTIRPass
    : public ttir::impl::ConvertTosaToTTIRBase<ConvertTosaToTTIRPass> {
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addIllegalDialect<tosa::TosaDialect>();

    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalOp<mlir::tensor::EmptyOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();

    // For now keep the same type assuming tosa ops operate on builtin tensor.
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) {
      assert(isa<RankedTensorType>(type) &&
             "only ranked tensor type supported");
      return type;
    });
    RewritePatternSet patterns(&getContext());

    // Add conversion patterns.
    patterns
        .add<TosaToTTIROpConversionPattern<tosa::AddOp, mlir::tt::ttir::AddOp>>(
            typeConverter, &getContext());
    patterns.add<
        TosaToTTIROpConversionPattern<tosa::MulOp, mlir::tt::ttir::MultiplyOp>>(
        typeConverter, &getContext());
    patterns.add<
        TosaToTTIROpConversionPattern<tosa::SubOp, mlir::tt::ttir::SubtractOp>>(
        typeConverter, &getContext());
    patterns.add<TosaToTTIROpConversionPattern<tosa::GreaterEqualOp,
                                               mlir::tt::ttir::GreaterEqualOp>>(
        typeConverter, &getContext());

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

std::unique_ptr<OperationPass<ModuleOp>> createConvertTosaToTTIRPass() {
  return std::make_unique<ConvertTosaToTTIRPass>();
}

} // namespace mlir::tt
