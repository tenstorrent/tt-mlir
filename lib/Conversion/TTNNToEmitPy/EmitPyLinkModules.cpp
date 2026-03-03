// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt {

#define GEN_PASS_DEF_EMITPYLINKMODULES
#include "ttmlir/Conversion/Passes.h.inc"

namespace {

// Pass which links the CPU and Device modules by moving all operations from
// both modules into the root module. CPU-hoisted function declarations in the
// Device module are skipped (not moved), and call sites are updated to call
// the actual function definitions from the CPU module.
//
// Operations are moved in the following order:
// 1. Imports.
// 2. Ops from the CPU module (CPU-hoisted function definitions).
// 3. Ops from the Device module (excluding CPU-hoisted function declarations).
//
class EmitPyLinkModulesPass
    : public impl::EmitPyLinkModulesBase<EmitPyLinkModulesPass> {
public:
  using impl::EmitPyLinkModulesBase<
      EmitPyLinkModulesPass>::EmitPyLinkModulesBase;

  void runOnOperation() override {
    mlir::ModuleOp rootModule = getOperation();

    // Ensure we only run this on top-level ModuleOp.
    //
    if (rootModule->getParentOp() != nullptr) {
      rootModule.emitError("EmitPyLinkModules pass must run on root module!");
      return signalPassFailure();
    }

    // Find DeviceModuleOp.
    //
    auto deviceModuleOps = rootModule.getOps<ttcore::DeviceModuleOp>();
    if (deviceModuleOps.empty()) {
      rootModule.emitError(
          "No ttcore.device_module found in the top-level module!");
      return signalPassFailure();
    }

    ttcore::DeviceModuleOp deviceModuleOp = *deviceModuleOps.begin();
    auto deviceModule =
        mlir::cast<mlir::ModuleOp>(deviceModuleOp.getBody()->front());

    // Transfer attributes from device module to the root module.
    //
    for (const auto &attr : deviceModule->getAttrs()) {
      if (!rootModule->hasAttr(attr.getName())) {
        rootModule->setAttr(attr.getName(), attr.getValue());
      }
    }

    // Find CPUModuleOp (optional).
    //
    auto cpuModuleOps = rootModule.getOps<ttcore::CPUModuleOp>();
    ttcore::CPUModuleOp cpuModuleOp =
        cpuModuleOps.empty() ? nullptr : *cpuModuleOps.begin();

    auto &rootBody = rootModule.getBodyRegion().front();
    auto &deviceBody = deviceModule.getBodyRegion().front();

    // Collect imports from Device module and CPU module (if it exists).
    //
    llvm::SmallVector<emitpy::ImportOp, 8> imports;

    // Helper to collect imports from a module, erasing duplicates from the
    // module.
    //
    auto collectImports = [&imports](mlir::ModuleOp module) {
      llvm::SmallVector<emitpy::ImportOp> moduleImports(
          module.getOps<emitpy::ImportOp>());

      for (auto importOp : moduleImports) {
        if (!llvm::any_of(imports, [&](emitpy::ImportOp existingImportOp) {
              return existingImportOp.getModuleName() ==
                     importOp.getModuleName();
            })) {
          // Add unique import to the collection.
          // The import will be moved to the root module later.
          //
          imports.push_back(importOp);
        } else {
          // Erase duplicate import from the module.
          //
          importOp->erase();
        }
      }
    };

    collectImports(deviceModule);

    // Helper to collect globals from a module, erasing duplicates from the
    // module.
    //
    llvm::SmallVector<emitpy::GlobalOp> globals;
    auto collectGlobals = [&globals](mlir::ModuleOp module) {
      llvm::SmallVector<emitpy::GlobalOp> moduleGlobals(
          module.getOps<emitpy::GlobalOp>());
      for (auto &moduleGlobal : moduleGlobals) {
        if (llvm::all_of(globals, [&](emitpy::GlobalOp globalOp) {
              return moduleGlobal.getName() != globalOp.getName();
            })) {
          globals.push_back(moduleGlobal);
        } else {
          moduleGlobal->erase();
        }
      }
    };

    collectGlobals(deviceModule);

    if (cpuModuleOp) {
      auto cpuModule =
          mlir::cast<mlir::ModuleOp>(cpuModuleOp.getBody()->front());
      collectImports(cpuModule);
      collectGlobals(cpuModule);
    }

    // Move all collected globals to the root module.
    //
    for (auto globalOp : llvm::reverse(globals)) {
      globalOp->moveBefore(&rootBody, rootBody.begin());
    }

    // Move all collected imports to the root module.
    //
    for (auto importOp : llvm::reverse(imports)) {
      importOp->moveBefore(&rootBody, rootBody.begin());
    }

    // Move all operations from CPU module (CPU-hoisted function
    // definitions).
    //
    if (cpuModuleOp) {
      auto cpuModule =
          mlir::cast<mlir::ModuleOp>(cpuModuleOp.getBody()->front());

      llvm::SmallVector<Operation *> cpuOps;

      for (auto &op : *cpuModule.getBody()) {
        cpuOps.push_back(&op);
      }
      for (auto *op : cpuOps) {
        op->moveBefore(&rootBody, rootBody.end());
      }

      cpuModuleOp->erase();
    }

    // Collect the CPU-hoisted declarations from Device module and update their
    // call sites.
    //
    OpBuilder builder(&getContext());
    llvm::SmallVector<func::FuncOp> declarations;
    for (auto funcOp : deviceModule.getOps<func::FuncOp>()) {
      if (!ttmlir::utils::isForwardCPUDeclarationFunc(funcOp)) {
        continue;
      }

      llvm::StringRef declarationName = funcOp.getSymName();
      llvm::StringRef definitionName = declarationName;

      (void)funcOp.replaceAllSymbolUses(builder.getStringAttr(definitionName),
                                        deviceModule);

      declarations.push_back(funcOp);
    }

    // Erase the collected CPU-hoisted declarations.
    //
    for (auto funcOp : declarations) {
      funcOp->erase();
    }

    // Move all remaining operations from the Device module.
    //
    rootBody.getOperations().splice(rootBody.end(), deviceBody.getOperations());

    // Erase the Device module.
    //
    deviceModuleOp->erase();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyLinkModulesPass() {
  return std::make_unique<EmitPyLinkModulesPass>();
}

} // namespace mlir::tt
