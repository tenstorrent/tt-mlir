// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToTTIR/TTNNToTTIR.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/ScopeExit.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNMATERIALIZED2M
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

/// Clones a function into a standalone temporary ModuleOp for isolated
/// compilation. The caller is responsible for erasing the returned module.
static FailureOr<ModuleOp>
cloneFuncIntoStandaloneModule(func::FuncOp origFunc) {
  MLIRContext *ctx = origFunc.getContext();
  Location loc = origFunc.getLoc();

  OpBuilder builder(ctx);

  // Create a standalone module wrapper for the D2M pipeline.
  ModuleOp moduleWrapper = builder.create<ModuleOp>(loc, "d2m_subgraph");
  builder.setInsertionPointToStart(moduleWrapper.getBody());

  // Clone the original function into the new module.
  IRMapping irMapper;
  builder.clone(*origFunc.getOperation(), irMapper);

  return moduleWrapper;
}

class TTNNMaterializeD2M
    : public impl::TTNNMaterializeD2MBase<TTNNMaterializeD2M> {

public:
  using impl::TTNNMaterializeD2MBase<
      TTNNMaterializeD2M>::TTNNMaterializeD2MBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Collect functions marked for D2M compilation.
    SmallVector<func::FuncOp> funcsToCompile;
    SymbolTable symbolTable(moduleOp);
    moduleOp.walk([&](ttir::DispatchD2MOp dispatchD2MOp) {
      // Look up the func symbol in the module's symbol table
      auto funcOp = symbolTable.lookup<func::FuncOp>(dispatchD2MOp.getFunc());
      if (funcOp) {
        funcsToCompile.push_back(funcOp);
      }
    });

    if (funcsToCompile.empty()) {
      return;
    }

    // Process each function that needs D2M compilation.
    for (func::FuncOp origFunc : funcsToCompile) {
      if (failed(compileViaD2M(origFunc))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult compileViaD2M(func::FuncOp origFunc) {
    FailureOr<ModuleOp> tempModuleResult =
        cloneFuncIntoStandaloneModule(origFunc);
    if (failed(tempModuleResult)) {
      return origFunc.emitError("Failed to clone function for D2M compilation");
    }
    ModuleOp tempModule = *tempModuleResult;

    auto moduleCleanup = llvm::make_scope_exit([&]() { tempModule->erase(); });

    auto D2MPm = PassManager::on<ModuleOp>(tempModule.getContext());
    D2MPm.enableIRPrinting(
        /*shouldPrintBeforePass=*/[](Pass *, Operation *) { return false; },
        /*shouldPrintAfterPass=*/[](Pass *, Operation *) { return true; },
        /*printModuleScope=*/false, // true requires single-threading
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false,
        /*out=*/llvm::errs());

    ttmetal::TTIRToTTMetalPipelineOptions d2mOptions;
    // d2mOptions.systemDescPath = systemDescPath;
    // d2mOptions.mockSystemDescArch = mockSystemDescArch;
    // d2mOptions.meshShape = llvm::to_vector(meshShape);
    d2mOptions.ttnnMode = true;

    // D2MPm.addPass(createConvertTTNNToTTIRPass());
    ttmetal::createTTIRToTTMetalFrontendPipeline(D2MPm, d2mOptions);
    ttmetal::createTTIRToTTMetalMiddleendPipeline(D2MPm, d2mOptions);
    ttmetal::createTTIRToTTMetalBackendPipeline(D2MPm, d2mOptions);

    if (failed(D2MPm.run(tempModule))) {
      return origFunc.emitError("D2M pipeline failed for function: ")
             << origFunc.getName();
    }

    // Extract the compiled function and merge back into original module
    return success();
  }
};

} // namespace
} // namespace mlir::tt::ttnn
