// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"

#include "ttmlir/Conversion/TTNNToEmitC/EmitCConversion.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
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
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

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
    addConversion(
        [ctx](mlir::tt::ttnn::GlobalSemaphoreType type) -> emitc::OpaqueType {
          return emitc::OpaqueType::get(ctx, "::ttnn::GlobalSemaphore");
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
    mlir::ConversionTarget target(getContext());

    // Inlining a composite can move a cached call before its helper's
    // definition. Remember the helpers before load_cached is lowered to an
    // opaque C++ function pointer, losing the symbolic reference.
    llvm::SetVector<StringAttr> cachedCallees;
    module.walk([&](ttcore::LoadCachedOp op) {
      cachedCallees.insert(op.getCalleeAttr().getAttr());
    });

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

    // This pipeline retains func.func, which emitc.declare_func cannot
    // reference. Let the C++ emitter render a bodyless EmitC function with the
    // converted signature instead, then place the declaration after includes
    // and before any definitions. Each cached helper is declared only once.
    OpBuilder builder(module.getContext());
    auto include = *module.getOps<emitc::IncludeOp>().begin();
    builder.setInsertionPointAfter(include);
    for (StringAttr callee : cachedCallees) {
      auto func = module.lookupSymbol<func::FuncOp>(callee.getValue());
      if (!func) {
        module.emitError("cached helper not found after EmitC conversion: ")
            << callee.getValue();
        signalPassFailure();
        return;
      }
      OpBuilder declarationBuilder(module.getContext());
      OwningOpRef<emitc::FuncOp> declaration =
          declarationBuilder.create<emitc::FuncOp>(
              func.getLoc(), func.getSymName(), func.getFunctionType());
      std::string cpp;
      llvm::raw_string_ostream stream(cpp);
      if (failed(emitc::translateToCpp(declaration.get(), stream))) {
        signalPassFailure();
        return;
      }
      builder.create<emitc::VerbatimOp>(func.getLoc(), StringRef(cpp).trim());
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToEmitCPass() {
  return std::make_unique<ConvertTTNNToEmitCPass>();
}

} // namespace mlir::tt
