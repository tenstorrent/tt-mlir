// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPy.h"

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

// Helper function to enable torch conversion for CPU-hoisted functions.
//
// Inserts ttnn.to_torch calls for function arguments at the beginning of the
// function, and ttnn.from_torch calls for return values before return ops.
//
void enableTorchConversion(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());

  // Insert to_torch calls for tensor arguments at the beginning of the
  // function.
  //
  Block &entryBlock = funcOp.getBody().front();
  builder.setInsertionPointToStart(&entryBlock);

  for (BlockArgument arg : funcOp.getArguments()) {
    // Create ttnn.to_torch call.
    //
    auto toTorchOp = builder.create<emitpy::CallOpaqueOp>(
        funcOp.getLoc(), arg.getType(), "ttnn.to_torch", ValueRange{arg},
        nullptr, nullptr);

    // Replace all uses of the original argument with the to_torch result,
    // except for the to_torch op itself.
    //
    arg.replaceAllUsesExcept(toTorchOp.getResult(0), toTorchOp);
  }

  // Insert from_torch calls for tensor return values.
  //
  funcOp.walk([&](func::ReturnOp returnOp) {
    builder.setInsertionPoint(returnOp);

    SmallVector<Value> newReturnOperands;
    for (Value returnValue : returnOp.getOperands()) {
      // Create ttnn.from_torch call.
      //
      auto fromTorchOp = builder.create<emitpy::CallOpaqueOp>(
          returnOp.getLoc(), returnValue.getType(), "ttnn.from_torch",
          ValueRange{returnValue}, nullptr, nullptr);

      newReturnOperands.push_back(fromTorchOp.getResult(0));
    }

    // Update the return op with the new operands.
    //
    returnOp->setOperands(newReturnOperands);
  });
}

struct ConvertTTNNToEmitPyPass
    : public tt::ttnn::impl::ConvertTTNNToEmitPyBase<ConvertTTNNToEmitPyPass> {

  using tt::ttnn::impl::ConvertTTNNToEmitPyBase<
      ConvertTTNNToEmitPyPass>::ConvertTTNNToEmitPyBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
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
    builder.create<emitpy::ImportOp>(module->getLoc(), "ttnn", nullptr, nullptr,
                                     nullptr, nullptr);
    builder.create<emitpy::ImportOp>(module->getLoc(), "utils", nullptr,
                                     nullptr, nullptr, nullptr);

    // Create a global cache dictionary
    //
    auto opaqueAttr = emitpy::OpaqueAttr::get(&getContext(), "{}");
    auto globalCacheDict = builder.create<emitpy::GlobalOp>(
        module->getLoc(), "_CONST_EVAL_CACHE", opaqueAttr);
    globalCacheDict->setAttr("emitpy.consteval", builder.getUnitAttr());

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
      populateTTNNToEmitPyPatterns(&getContext(), patterns, typeConverter,
                                   enableGoldenMode);

      // Apply full conversion
      //
      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns)))) {
        signalPassFailure();
        return;
      }

      // Enable torch tensor conversions if the golden mode is enabled.
      //
      if (enableGoldenMode) {
        module.walk(
            [&](func::FuncOp funcOp) { enableTorchConversion(funcOp); });
      }
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToEmitPyPass() {
  return std::make_unique<ConvertTTNNToEmitPyPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTTNNToEmitPyPass(const ConvertTTNNToEmitPyOptions &options) {
  return std::make_unique<ConvertTTNNToEmitPyPass>(options);
}

} // namespace mlir::tt
