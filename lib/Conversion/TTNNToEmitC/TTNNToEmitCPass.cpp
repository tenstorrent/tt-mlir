// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"

#include "ttmlir/Conversion/TTNNToEmitC/EmitCConversion.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_CONVERTTTNNTOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttnn

namespace {

class TTNNToEmitCTypeConverter : public TypeConverter {
public:
  TTNNToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](mlir::tt::ttnn::DeviceType type) -> emitc::PointerType {
      return emitc::PointerType::get(
          emitc::OpaqueType::get(ctx, "ttnn::distributed::MeshDevice"));
    });
    addConversion([ctx](mlir::RankedTensorType type) -> emitc::OpaqueType {
      if (mlir::isa_and_present<mlir::tt::ttnn::TraceIdAttr>(
              type.getEncoding())) {
        return emitc::OpaqueType::get(ctx, "ttnn::MeshTraceId");
      }
      return emitc::OpaqueType::get(ctx,
                                    ttnn_to_emitc::TypeNameV<::ttnn::Tensor>);
    });
    addConversion([ctx](mlir::TupleType type) -> emitc::OpaqueType {
      return emitc::OpaqueType::get(
          ctx, ttnn_to_emitc::TypeNameV<std::vector<::ttnn::Tensor>>);
    });
  }
};

struct ConvertTTNNToEmitCPass
    : public mlir::tt::ttnn::impl::ConvertTTNNToEmitCBase<
          ConvertTTNNToEmitCPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    // Only run conversion on top-level moduleOp.
    if (module->getParentOp() != nullptr) {
      return;
    }

    mlir::ConversionTarget target(getContext());

    // EmitC is legal, TTNN is illegal
    //
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<mlir::tt::ttnn::TTNNDialect>();

    // mlir::ModuleOp is legal only if no attributes are present on it
    //
    target.addDynamicallyLegalOp<mlir::ModuleOp>(
        [&](mlir::ModuleOp op) { return op->getAttrs().empty(); });

    // Add header imports to front of module
    //
    {
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
      builder.create<emitc::IncludeOp>(module.getLoc(), "ttnn-precompiled.hpp",
                                       /*isStandard=*/false);
    }

    // Unwrap device_module into top-level ModuleOp (if present)
    {
      OpPassManager pm(ModuleOp::getOperationName());
      pm.addPass(mlir::tt::ttcore::createTTCoreUnwrapDeviceModulePass());

      if (failed(runPipeline(pm, module))) {
        signalPassFailure();
        return;
      }
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
      // Disallow arg attrs on func op
      //
      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody()) &&
               (!op.getArgAttrs().has_value() ||
                op.getArgAttrs().value().empty());
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
      if (failed(applyFullConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToEmitCPass() {
  return std::make_unique<ConvertTTNNToEmitCPass>();
}

} // namespace mlir::tt
