// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "TypeConverter.h"

#include "ttmlir/Dialect/TT/IR/TTOpsDialect.h.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace {

void populateTTNNToEmitCPatterns(mlir::MLIRContext *ctx,
                                 mlir::RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  // Device ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::OpenDeviceOp>>(typeConverter,
                                                               ctx, true);
  patterns.add<DefaultOpConversionPattern<ttnn::CloseDeviceOp>>(typeConverter,
                                                                ctx);

  // Memory ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::ToMemoryConfigOp>>(
      typeConverter, ctx);

  // Tensor ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::FullOp>>(typeConverter, ctx);

  // Math ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::AddOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::SubtractOp>>(typeConverter,
                                                             ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::GreaterEqualOp>>(typeConverter,
                                                                 ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::SumOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::SoftmaxOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::MultiplyOp>>(typeConverter,
                                                             ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::MatmulOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::ReluOp>>(typeConverter, ctx);
}

struct ConvertTTNNToEmitCPass
    : public ttnn::ConvertTTNNToEmitCBase<ConvertTTNNToEmitCPass> {
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<ttnn::TTNNDialect>();

    // Add header imports to front of module
    //
    {
      auto module = getOperation();
      OpBuilder builder(module);

      builder.create<emitc::IncludeOp>(module.getLoc(), "ttnn/device.h",
                                       /*isStandard=*/false);
      builder.create<emitc::IncludeOp>(
          module.getLoc(), "ttnn/operations/eltwise/binary/binary.hpp",
          /*isStandard=*/false);
      builder.create<emitc::IncludeOp>(
          module.getLoc(), "ttnn/operations/core.hpp", /*isStandard=*/false);
      builder.create<emitc::IncludeOp>(module.getLoc(),
                                       "ttnn/operations/creation.hpp",
                                       /*isStandard=*/false);
      builder.create<emitc::IncludeOp>(
          module.getLoc(),
          "ttnn/operations/reduction/generic/generic_reductions.hpp",
          /*isStandard=*/false);
      builder.create<emitc::IncludeOp>(module.getLoc(),
                                       "ttnn/operations/normalization.hpp",
                                       /*isStandard=*/false);
    }

    // TTNN -> EmitC
    //
    {
      TTNNToEmitCTypeConverter typeConverter(&getContext());
      RewritePatternSet patterns(&getContext());

      // Func dialect handling
      //
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

      // TTNN -> EmitC patterns
      //
      populateTTNNToEmitCPatterns(&getContext(), patterns, typeConverter);

      // Apply conversion
      //
      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  };
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTTNNToEmitCPass() {
  return std::make_unique<ConvertTTNNToEmitCPass>();
}

} // namespace mlir::tt
