// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"

#include "ttmlir/Conversion/StableHLOToTTIR/EmptyOpTypeConversion.h"
#include "ttmlir/Conversion/StableHLOToTTIR/ShardyToTTIR.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOTTIR
#define GEN_PASS_DEF_CONVERTSHARDYTOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

namespace {
struct ConvertStableHLOToTTIRPass
    : public ttir::impl::ConvertStableHLOToTTIRBase<
          ConvertStableHLOToTTIRPass> {

  ConvertStableHLOToTTIRPass() = default;
  ConvertStableHLOToTTIRPass(const ConvertStableHLOToTTIROptions &options)
      : Base(options) {}

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());

    // Common legal/illegal ops/dialects, regardless of partial conversion.
    target.addLegalDialect<mlir::quant::QuantDialect>();
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalOp<mlir::tt::ttir::EmptyOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();
    target.addLegalOp<mlir::func::CallOp>();
    target.addIllegalOp<mlir::tensor::EmptyOp>();

    if (enablePartialConversion) {
      // In partial conversion mode, explicitly mark only the ops we can
      // convert. Based on the patterns in
      // addElementwiseUnaryOpsConversionPatterns.
      target.addIllegalOp<mlir::stablehlo::AbsOp, mlir::stablehlo::CbrtOp,
                          mlir::stablehlo::ConvertOp, mlir::stablehlo::CeilOp,
                          mlir::stablehlo::CosineOp, mlir::stablehlo::ExpOp,
                          mlir::stablehlo::FloorOp, mlir::stablehlo::IsFiniteOp,
                          mlir::stablehlo::NegOp, mlir::stablehlo::RsqrtOp,
                          mlir::stablehlo::SineOp, mlir::stablehlo::SqrtOp,
                          mlir::stablehlo::Log1pOp, mlir::stablehlo::Expm1Op,
                          mlir::stablehlo::SignOp, mlir::stablehlo::LogisticOp,
                          mlir::stablehlo::TanOp, mlir::stablehlo::TanhOp,
                          mlir::stablehlo::LogOp>();

      // Based on addElementwiseBinaryOpsConversionPatterns.
      target.addIllegalOp<mlir::stablehlo::AddOp, mlir::stablehlo::DivOp,
                          mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp,
                          mlir::stablehlo::MulOp, mlir::stablehlo::SubtractOp,
                          mlir::stablehlo::RemOp, mlir::stablehlo::SelectOp,
                          mlir::stablehlo::PowOp, mlir::stablehlo::Atan2Op>();

      // Based on other conversion patterns.
      target.addIllegalOp<
          mlir::stablehlo::ReduceOp, mlir::stablehlo::DotGeneralOp,
          mlir::stablehlo::GetDimensionSizeOp, mlir::stablehlo::ConstantOp,
          mlir::stablehlo::BroadcastInDimOp, mlir::stablehlo::ConvolutionOp,
          mlir::stablehlo::UniformQuantizeOp,
          mlir::stablehlo::UniformDequantizeOp, mlir::stablehlo::ReduceWindowOp,
          mlir::stablehlo::CompareOp, mlir::stablehlo::ConcatenateOp,
          mlir::stablehlo::TransposeOp, mlir::stablehlo::ReshapeOp,
          mlir::stablehlo::AllReduceOp, mlir::stablehlo::AllGatherOp,
          mlir::stablehlo::ReduceScatterOp,
          mlir::stablehlo::CollectivePermuteOp, mlir::stablehlo::CustomCallOp,
          mlir::stablehlo::AllToAllOp, mlir::stablehlo::AndOp,
          mlir::stablehlo::OrOp, mlir::stablehlo::XorOp, mlir::stablehlo::NotOp,
          mlir::stablehlo::SliceOp, mlir::stablehlo::ClampOp,
          mlir::stablehlo::GatherOp, mlir::stablehlo::IotaOp,
          mlir::stablehlo::DynamicIotaOp, mlir::stablehlo::ScatterOp,
          mlir::stablehlo::ReverseOp, mlir::stablehlo::PadOp,
          mlir::stablehlo::BatchNormInferenceOp>();

    } else {
      // Full conversion - any SHLO or Sdy op is illegal.
      target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();
      target.addIllegalDialect<mlir::sdy::SdyDialect>();
    }

    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });
    RewritePatternSet patterns(&getContext());

    // Func type conversions.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    addEmptyOpTypeConversionPattern(&getContext(), patterns, typeConverter);

    populateStableHLOToTTIRPatterns(&getContext(), patterns, typeConverter);
    populateShardyToTTIRPatterns(&getContext(), patterns, typeConverter);

    // Apply full or partial conversion, based on option flag.
    if (enablePartialConversion) {
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    } else {
      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace
} // namespace mlir::tt::ttir

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertStableHLOToTTIRPass() {
  return std::make_unique<ttir::ConvertStableHLOToTTIRPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertStableHLOToTTIRPass(
    const ttir::ConvertStableHLOToTTIROptions &options) {
  return std::make_unique<ttir::ConvertStableHLOToTTIRPass>(options);
}

} // namespace mlir::tt
