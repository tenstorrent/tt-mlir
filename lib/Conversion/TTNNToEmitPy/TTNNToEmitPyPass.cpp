// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_CONVERTTTNNTOEMITPY
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttnn

namespace {

class TTNNToEmitPyTypeConverter : public TypeConverter {
public:
  TTNNToEmitPyTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](tt::ttnn::DeviceType type) -> emitpy::OpaqueType {
      return emitpy::OpaqueType::get(ctx, "ttnn.Device");
    });
    addConversion([ctx](mlir::TensorType type) -> emitpy::OpaqueType {
      return emitpy::OpaqueType::get(ctx,
                                     ttnn_to_emitpy::TypeNameV<::ttnn::Tensor>);
    });
    addConversion([ctx](mlir::TupleType type) -> emitpy::OpaqueType {
      return emitpy::OpaqueType::get(
          ctx, ttnn_to_emitpy::TypeNameV<std::vector<::ttnn::Tensor>>);
    });
  }
};

struct ConvertTTNNToEmitPyPass
    : public tt::ttnn::impl::ConvertTTNNToEmitPyBase<ConvertTTNNToEmitPyPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    // Only run conversion on top-level moduleOp.
    if (module->getParentOp() != nullptr) {
      return;
    }

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<emitpy::EmitPyDialect>();
    target.addIllegalDialect<tt::ttnn::TTNNDialect>();
    // mlir::ModuleOp is legal only if no attributes are present on it
    //
    target.addDynamicallyLegalOp<mlir::ModuleOp>(
        [&](mlir::ModuleOp op) { return op->getAttrs().empty(); });

    OpBuilder builder(module);

    if (module.getBodyRegion().empty()) {
      // Parent module is empty, nothing to do here
      //
      signalPassFailure();
    }

    // Set insertion point to start of first module child
    //
    builder.setInsertionPointToStart(module.getBody(0));

    // Include headers
    //
    builder.create<emitpy::ImportOp>(module.getLoc(), "ttnn", nullptr, nullptr,
                                     nullptr, nullptr);

    builder.create<emitpy::ImportOp>(module->getLoc(), "my_get_device", nullptr,
                                     nullptr, nullptr, nullptr);

    // Unwrap device_module into top-level ModuleOp (if present)
    {
      OpPassManager pm(ModuleOp::getOperationName());
      pm.addPass(tt::ttcore::createTTCoreUnwrapDeviceModulePass());

      if (failed(runPipeline(pm, module))) {
        signalPassFailure();
        return;
      }
    }

    // TTNN -> EmitPy
    //
    {
      TTNNToEmitPyTypeConverter typeConverter(&getContext());
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

      // TTNN -> EmitPy patterns
      //
      populateTTNNToEmitPyPatterns(&getContext(), patterns, typeConverter);

      // Apply full conversion
      //
      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToEmitPyPass() {
  return std::make_unique<ConvertTTNNToEmitPyPass>();
}

} // namespace mlir::tt
