// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToTTIR/TTNNToTTIR.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

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

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_CONVERTTTNNTOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

#define GET_OP_LIST
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h.inc"

namespace {
struct ConvertTTNNToTTIRPass
    : public ttnn::impl::ConvertTTNNToTTIRBase<ConvertTTNNToTTIRPass> {

  ConvertTTNNToTTIRPass() = default;

  void runOnOperation() final {
    ::mlir::ConversionTarget target(getContext());

    // Common legal/illegal ops/dialects for both partial and full conversion.
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<ttcore::TTCoreDialect>();
    target.addLegalOp<::mlir::ModuleOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());

    // Mark TTNN dialect as dynamically legal for all ops that do NOT have the
    // `hoist_generic_via_d2m` attribute
    target.addDynamicallyLegalDialect<ttnn::TTNNDialect>(
        [&](Operation *op) { return !utils::isTTNNHoistGenericViaD2MOp(op); });

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

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::tt::ttnn

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToTTIRPass() {
  return std::make_unique<ttnn::ConvertTTNNToTTIRPass>();
}

} // namespace mlir::tt
