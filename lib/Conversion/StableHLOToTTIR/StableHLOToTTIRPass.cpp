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
    MLIRContext *context = &getContext();

    // Setup common type converter (all types map 1:1).
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(context);
    populateStableHLOToTTIRPatterns(context, patterns, typeConverter);
    populateShardyToTTIRPatterns(context, patterns, typeConverter);
    addEmptyOpTypeConversionPattern(context, patterns, typeConverter);

    if (enablePartialConversion) {
      // Partial conversion: use greedy rewriting.
      // This naturally only converts ops with matching patterns.
      // Note that this is not as resilient to type changes as
      // applyFull/PartialConversion; however, we don't expect types to change
      // anyway.
      GreedyRewriteConfig config;
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                       config))) {
        // In partial conversion, continue even if some patterns fail
        emitWarning(getOperation()->getLoc())
            << "Some patterns failed during partial conversion.";
      }
    } else {
      // Full conversion: use dialect conversion framework.
      mlir::ConversionTarget target(*context);
      setupConversionTarget(target, typeConverter);

      // Add function conversion patterns for full conversion.
      addFunctionConversionPatterns(patterns, typeConverter, target);

      // Apply full conversion - will fail if any StableHLO/Sdy ops remain.
      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns)))) {
        signalPassFailure();
      }
    }
  }

private:
  // Dialect and op-level conversion rules.
  void setupConversionTarget(ConversionTarget &target,
                             TypeConverter &typeConverter) {
    target.addLegalDialect<mlir::quant::QuantDialect>();
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalOp<mlir::tt::ttir::EmptyOp>();
    target.addLegalOp<mlir::ModuleOp>();

    target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();
    target.addIllegalDialect<mlir::sdy::SdyDialect>();
    target.addIllegalOp<mlir::tensor::EmptyOp>();
  }

  // Func dialect type conversions.
  void addFunctionConversionPatterns(RewritePatternSet &patterns,
                                     TypeConverter &typeConverter,
                                     ConversionTarget &target) {
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
