// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::tt {

#define GEN_PASS_DEF_EMITPYLINKMODULES
#include "ttmlir/Conversion/Passes.h.inc"

namespace {

// Pass which links the CPU and Device modules by:
//
// - Replacing the CPU-hoisted declarations in the Device module with the
//   corresponding CPU-hoisted definitions from the CPU module.
//
// - Moving the imports from the CPU module to the corresponding scope in the
//   Device module, which could be either the module itself, or a file inside
//   the Device module (if TTNNFileSplit is performed);
//
// - Erasing the CPU module.
//
// - Finally, moving all ops from the Device module to the Root module and
//   erasing the Device module.
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

    // Find CPUModuleOp (optional).
    //
    auto cpuModuleOps = rootModule.getOps<ttcore::CPUModuleOp>();
    ttcore::CPUModuleOp cpuModuleOp =
        cpuModuleOps.empty() ? nullptr : *cpuModuleOps.begin();

    if (cpuModuleOp) {
      auto cpuModule =
          mlir::cast<mlir::ModuleOp>(cpuModuleOp.getBody()->front());

      // Perform linking by replacing CPU-hoisted declarations with their
      // corresponding definitions.
      //
      if (failed(linkCPUModule(deviceModule, cpuModule))) {
        return signalPassFailure();
      }

      // Erase the CPU module.
      //
      cpuModuleOp->erase();
    }

    // Move all operations from the Device module to the Root module.
    //
    auto &rootBody = rootModule.getBodyRegion().front();
    auto &deviceBody = deviceModule.getBodyRegion().front();
    rootBody.getOperations().splice(rootBody.end(), deviceBody.getOperations());

    // Transfer attributes from device module to the root module.
    //
    for (const auto &attr : deviceModule->getAttrs()) {
      if (!rootModule->hasAttr(attr.getName())) {
        rootModule->setAttr(attr.getName(), attr.getValue());
      }
    }

    // Erase the Device module.
    //
    deviceModuleOp->erase();
  }

private:
  // Link CPU module into the device module by replacing CPU-hoisted
  // declarations with definitions and moving CPU imports to the enclosing
  // scope of each declaration.
  //
  LogicalResult linkCPUModule(mlir::ModuleOp deviceModule,
                              mlir::ModuleOp cpuModule) {
    // Build a map of CPU-hoisted function definitions by symbol name.
    //
    llvm::StringMap<func::FuncOp> cpuDefs;
    for (auto funcOp : cpuModule.getOps<func::FuncOp>()) {
      if (!funcOp.isDeclaration()) {
        cpuDefs[funcOp.getSymName()] = funcOp;
      }
    }

    // Collect CPU module imports.
    //
    llvm::SmallVector<emitpy::ImportOp> cpuImports(
        cpuModule.getOps<emitpy::ImportOp>());

    // Walk all CPU-hoisted declarations in the device module and replace
    // each with the corresponding definition from the CPU module.
    //
    // The definition is inserted before the first operation in the scope
    // that calls the symbol, so it appears right before its first usage.
    // CPU imports are cloned into the scope once per scope.
    //
    llvm::SmallVector<func::FuncOp> decls;
    deviceModule.walk([&](func::FuncOp funcOp) {
      if (ttmlir::utils::isForwardCPUDeclarationFunc(funcOp)) {
        decls.push_back(funcOp);
      }
    });

    llvm::SmallPtrSet<Block *, 4> blocksWithCPUImportsCloned;
    for (auto decl : decls) {
      Block *scope = decl->getBlock();

      auto it = cpuDefs.find(decl.getSymName());
      if (it == cpuDefs.end()) {
        return decl.emitError(
            "CPU-hoisted declaration has no matching definition "
            "in the CPU module");
      }

      // Find the first operation in the scope that calls this symbol
      // and insert the definition before it.
      //
      Operation *insertionPoint = findFirstCaller(decl.getSymName(), *scope);
      if (insertionPoint) {
        it->second->moveBefore(insertionPoint);
      } else {
        it->second->moveBefore(decl);
      }
      decl->erase();

      // Clone CPU imports into this scope (once per scope).
      //
      if (blocksWithCPUImportsCloned.insert(scope).second) {
        cloneCPUImports(cpuImports, *scope);
      }
    }

    return success();
  }

  // Find the first top-level operation in the scope that contains a call
  // to the given symbol. Returns nullptr if no caller is found.
  //
  Operation *findFirstCaller(StringRef symbolName, Block &scope) {
    for (auto &op : scope) {
      bool found = false;
      op.walk([&](func::CallOp callOp) {
        if (callOp.getCallee() == symbolName) {
          found = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (found) {
        return &op;
      }
    }
    return nullptr;
  }

  // Clone CPU module imports into the given scope, skipping duplicates.
  //
  void cloneCPUImports(ArrayRef<emitpy::ImportOp> cpuImports, Block &scope) {
    llvm::StringSet<> existingImports;
    for (auto &op : scope) {
      if (auto importOp = dyn_cast<emitpy::ImportOp>(op)) {
        existingImports.insert(importOp.getModuleName());
      }
    }

    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(&scope);

    for (auto importOp : cpuImports) {
      if (existingImports.contains(importOp.getModuleName())) {
        continue;
      }
      builder.clone(*importOp);
      existingImports.insert(importOp.getModuleName());
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyLinkModulesPass() {
  return std::make_unique<EmitPyLinkModulesPass>();
}

} // namespace mlir::tt
