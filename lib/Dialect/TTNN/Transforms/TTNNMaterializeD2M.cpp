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
#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNMATERIALIZED2M
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNMaterializeD2M
    : public impl::TTNNMaterializeD2MBase<TTNNMaterializeD2M> {

public:
  using impl::TTNNMaterializeD2MBase<
      TTNNMaterializeD2M>::TTNNMaterializeD2MBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Collect dispatch ops and their referenced subgraph modules
    SmallVector<std::pair<ttir::DispatchD2MOp, ModuleOp>> toCompile;
    SymbolTable symbolTable(moduleOp);
    moduleOp.walk([&](ttir::DispatchD2MOp dispatchD2MOp) {
      auto subgraphModule =
          symbolTable.lookup<ModuleOp>(dispatchD2MOp.getSubgraph());
      if (subgraphModule) {
        toCompile.push_back({dispatchD2MOp, subgraphModule});
      }
    });

    if (toCompile.empty()) {
      return;
    }

    // Process each subgraph module that needs D2M compilation.
    // Use a set to avoid compiling the same module multiple times.
    llvm::DenseSet<ModuleOp> compiledModules;
    for (auto &[dispatchOp, subgraphModule] : toCompile) {
      if (compiledModules.contains(subgraphModule)) {
        continue;
      }
      if (failed(compileViaD2M(subgraphModule))) {
        signalPassFailure();
        return;
      }
      compiledModules.insert(subgraphModule);
    }
  }

private:
  LogicalResult compileViaD2M(ModuleOp subgraphModule) {
    // Debug: Print the subgraph module before running D2M pipeline
    llvm::errs() << "\n=== D2M Input Module ("
                 << subgraphModule.getSymName().value_or("unnamed")
                 << ") ===\n";
    subgraphModule.print(llvm::errs());
    llvm::errs() << "\n=== End D2M Input Module ===\n\n";

    auto D2MPm = PassManager::on<ModuleOp>(subgraphModule.getContext());
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

    ttmetal::createTTIRToTTMetalFrontendPipeline(D2MPm, d2mOptions);
    ttmetal::createTTIRToTTMetalMiddleendPipeline(D2MPm, d2mOptions);
    ttmetal::createTTIRToTTMetalBackendPipeline(D2MPm, d2mOptions);

    if (failed(D2MPm.run(subgraphModule))) {
      return subgraphModule.emitError(
                 "D2M pipeline failed for subgraph module: ")
             << subgraphModule.getSymName().value_or("unnamed");
    }

    // Debug: Print the compiled module
    llvm::errs() << "\n=== D2M Output Module ("
                 << subgraphModule.getSymName().value_or("unnamed")
                 << ") ===\n";
    subgraphModule.print(llvm::errs());
    llvm::errs() << "\n=== End D2M Output Module ===\n\n";

    return success();
  }
};

} // namespace
} // namespace mlir::tt::ttnn
