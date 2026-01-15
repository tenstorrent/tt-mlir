// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNCOLLASPED2M
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNCollaspeD2M : public impl::TTNNCollaspeD2MBase<TTNNCollaspeD2M> {

public:
  using impl::TTNNCollaspeD2MBase<TTNNCollaspeD2M>::TTNNCollaspeD2MBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // llvm::errs() << "\n=== Module before D2M collapse ===\n";
    // moduleOp.print(llvm::errs());
    // llvm::errs() << "\n=== End Module before D2M collapse ===\n\n";

    SymbolTable symbolTable(moduleOp);

    // Find dispatch_d2m ops and their subgraph modules
    SmallVector<std::pair<ttir::DispatchD2MOp, ModuleOp>> dispatchOps;
    moduleOp.walk([&](ttir::DispatchD2MOp dispatchOp) {
      auto subgraphModule =
          symbolTable.lookup<ModuleOp>(dispatchOp.getSubgraph());
      if (subgraphModule) {
        dispatchOps.push_back({dispatchOp, subgraphModule});
      }
    });

    if (dispatchOps.empty()) {
      llvm::errs() << "No D2M subgraph modules found to collapse.\n";
      return;
    }

    llvm::errs() << "Found " << dispatchOps.size()
                 << " dispatch_d2m op(s) with subgraph modules to inline.\n";

    // Track modules to erase after all dispatch ops are inlined
    llvm::DenseSet<ModuleOp> modulesToErase;

    // For each dispatch op, inline the compiled function body
    for (auto &[dispatchOp, subgraphModule] : dispatchOps) {
      if (failed(inlineDispatchOp(moduleOp, dispatchOp, subgraphModule))) {
        signalPassFailure();
        return;
      }
      modulesToErase.insert(subgraphModule);
    }

    // Erase all the subgraph modules after inlining
    for (ModuleOp mod : modulesToErase) {
      mod.erase();
    }

    // llvm::errs() << "\n=== Module after D2M collapse ===\n";
    // moduleOp.print(llvm::errs());
    // llvm::errs() << "\n=== End Module after D2M collapse ===\n\n";
  }

private:
  // Inline the compiled function body at the dispatch_d2m call site,
  // and copy all kernel functions to the parent module.
  LogicalResult inlineDispatchOp(ModuleOp parentModule,
                                 ttir::DispatchD2MOp dispatchOp,
                                 ModuleOp subgraphModule) {
    // Find the entry function in the subgraph module (same name as module).
    // The function may be nested inside device_module/builtin.module, so walk.
    StringRef moduleName = subgraphModule.getSymName().value_or("");
    func::FuncOp entryFunc = nullptr;

    // Collect all functions from the nested module
    SmallVector<func::FuncOp> allFunctions;
    subgraphModule.walk([&](func::FuncOp funcOp) {
      allFunctions.push_back(funcOp);
      if (funcOp.getSymName() == moduleName) {
        entryFunc = funcOp;
      }
    });

    // If no function with module name found, take the first public function
    if (!entryFunc) {
      for (auto funcOp : allFunctions) {
        if (!funcOp.isPrivate()) {
          entryFunc = funcOp;
          break;
        }
      }
    }

    if (!entryFunc) {
      return dispatchOp.emitOpError("could not find entry function in "
                                    "subgraph module: ")
             << moduleName;
    }

    // Copy all private kernel functions to the parent module with unique names
    SymbolTable parentSymbolTable(parentModule);
    OpBuilder moduleBuilder(parentModule.getContext());
    moduleBuilder.setInsertionPointToEnd(parentModule.getBody());

    std::string prefix = (moduleName + "_").str();

    for (func::FuncOp funcOp : allFunctions) {
      if (funcOp == entryFunc) {
        continue;
      }

      StringRef oldName = funcOp.getSymName();
      std::string newName = prefix + oldName.str();
      StringAttr newNameAttr =
          StringAttr::get(parentModule.getContext(), newName);

      // Update all symbol references in the entry function then clone+rename
      // the function
      (void)SymbolTable::replaceAllSymbolUses(funcOp, newNameAttr, entryFunc);
      IRMapping funcMapping;
      Operation *clonedOp =
          moduleBuilder.clone(*funcOp.getOperation(), funcMapping);
      auto clonedFunc = cast<func::FuncOp>(clonedOp);
      clonedFunc.setSymName(newName);
      parentSymbolTable.insert(clonedOp);
    }

    // Copy all the ttnn.generic ops to the dispatch_d2m op call site
    OpBuilder builder(dispatchOp);
    IRMapping mapping;
    for (auto [funcArg, dispatchInput] :
         llvm::zip(entryFunc.getArguments(), dispatchOp.getInputs())) {
      mapping.map(funcArg, dispatchInput);
    }

    // Clone all ops from the function body except the return
    Value resultValue;
    for (Operation &op : entryFunc.getBody().front()) {
      if (auto returnOp = dyn_cast<func::ReturnOp>(&op)) {
        resultValue = mapping.lookup(returnOp.getOperand(0));
      } else {
        builder.clone(op, mapping);
      }
    }

    if (!resultValue) {
      return dispatchOp.emitOpError("could not find return value in "
                                    "entry function");
    }

    dispatchOp.getResult().replaceAllUsesWith(resultValue);
    dispatchOp.erase();

    return success();
  }
};

} // namespace
} // namespace mlir::tt::ttnn
