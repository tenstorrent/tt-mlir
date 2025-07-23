// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTMetalToEmitC/TTMetalToEmitC.h"

#include "ttmlir/Conversion/TTMetalToEmitC/EmitCConversion.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_CONVERTTTMETALTOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttmetal

namespace {

class TTMetalToEmitCTypeConverter : public TypeConverter {
public:
  TTMetalToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    
    // Convert MemRef types to Buffer pointers
    addConversion([ctx](MemRefType type) -> emitc::PointerType {
      return emitc::PointerType::get(
          emitc::OpaqueType::get(ctx, ttmetal_to_emitc::TypeNameV<::tt::tt_metal::Buffer>));
    });
    
    // Convert function types
    addConversion([this](FunctionType type) -> FunctionType {
      auto inputs = llvm::map_to_vector(
          type.getInputs(), [this](Type t) { return convertType(t); });
      auto results = llvm::map_to_vector(
          type.getResults(), [this](Type t) { return convertType(t); });
      return FunctionType::get(type.getContext(), inputs, results);
    });
  }
};

struct ConvertTTMetalToEmitCPass
    : public ttmetal::impl::ConvertTTMetalToEmitCBase<ConvertTTMetalToEmitCPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    TTMetalToEmitCTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    // Configure conversion target
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalOp<ModuleOp>();
    
    // Mark TTMetal operations as illegal
    target.addIllegalDialect<ttmetal::TTMetalDialect>();

    // Add conversion patterns
    populateTTMetalToEmitCPatterns(context, patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tt::createConvertTTMetalToEmitCPass() {
  return std::make_unique<ConvertTTMetalToEmitCPass>();
}