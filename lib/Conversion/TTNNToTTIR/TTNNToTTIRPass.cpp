// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToTTIR/TTNNToTTIR.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

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

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTNNTOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

namespace {
struct ConvertTTNNToTTIRPass
    : public ttir::impl::ConvertTTNNToTTIRBase<ConvertTTNNToTTIRPass> {

  ConvertTTNNToTTIRPass() = default;

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());

    // Common legal/illegal ops/dialects for both partial and full conversion.
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<ttcore::TTCoreDialect>();
    target.addLegalOp<mlir::tt::ttir::EmptyOp>();
    target.addLegalOp<mlir::ModuleOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());

    ::mlir::tt::populateTTNNToTTIRPatterns(&getContext(), patterns,
                                           typeConverter);

    // Function type conversions.
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

    // For partial conversion, we can leave TTNN dialect as neither
    // explicitly legal so the patterns run, nor explicitly illegal so
    // leftover ops won't throw an error.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::tt::ttir

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToTTIRPass() {
  return std::make_unique<ttir::ConvertTTNNToTTIRPass>();
}

} // namespace mlir::tt
