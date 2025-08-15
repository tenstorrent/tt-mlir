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
#include "mlir/Transforms/DialectConversion.h"
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

    // Common legal/illegal ops/dialects
    target.addLegalDialect<mlir::quant::QuantDialect>();
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalOp<mlir::tt::ttir::EmptyOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();
    target.addLegalOp<mlir::func::CallOp>();
    target.addIllegalOp<mlir::tensor::EmptyOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    RewritePatternSet patterns(&getContext());

    // Func type conversions
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
    ::mlir::tt::populateStableHLOToTTIRPatterns(&getContext(), patterns,
                                                typeConverter);
    populateShardyToTTIRPatterns(&getContext(), patterns, typeConverter);

    if (enablePartialConversion) {
      // For partial conversion: Don't explicitly mark StableHLO/Sdy as illegal
      // Instead, mark unknown ops as dynamically legal based on whether we have
      // patterns This should make the framework only try to convert ops we have
      // patterns for

      // Mark unknown operations as legal by default
      // This means ops without patterns won't be touched
      target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

      // Track which ops couldn't be converted
      DenseSet<Operation *> unlegalizedOps;
      ConversionConfig config;
      config.unlegalizedOps = &unlegalizedOps;

      // Don't mark StableHLO/Sdy as illegal - let them be "unknown"
      // The patterns will still apply to ops they match

      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns), config))) {
        signalPassFailure();
        return;
      }

      // Report what couldn't be converted
      if (!unlegalizedOps.empty()) {
        emitRemark(getOperation()->getLoc())
            << "Partial conversion: " << unlegalizedOps.size()
            << " ops could not be converted";
      }
    } else {
      // Full conversion - any StableHLO or Sdy op is illegal
      target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();
      target.addIllegalDialect<mlir::sdy::SdyDialect>();

      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
    // Full conversion - any StableHLO or Sdy op is illegal
    target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();
    target.addIllegalDialect<mlir::sdy::SdyDialect>();

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
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
