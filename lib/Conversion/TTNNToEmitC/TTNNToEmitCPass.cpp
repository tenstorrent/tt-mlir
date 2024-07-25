// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"

#include "../PassDetail.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

using namespace mlir;
using namespace mlir::tt;

namespace {

class TTNNToEmitCTypeConverter : public TypeConverter {
public:
  TTNNToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](mlir::tt::DeviceType type) -> Type {
      return emitc::OpaqueType::get(ctx, "ttnn::Device");
    });
    addConversion([ctx](TensorType type) -> Type {
      return emitc::OpaqueType::get(ctx, "ttnn::Tensor");
    });
  }
};

struct ConvertTTNNToEmitCPass
    : public ttnn::impl::ConvertTTNNToEmitCBase<ConvertTTNNToEmitCPass> {
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<ttnn::TTNNDialect>();
    target.addLegalOp<mlir::ModuleOp>();

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

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToEmitCPass() {
  return std::make_unique<ConvertTTNNToEmitCPass>();
}

} // namespace mlir::tt
