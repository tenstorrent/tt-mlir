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
// When TTNNFileSplit is performed, Device module has two file ops: main and
// consteval. CPU definitions are placed in the consteval file and
// declarations from the main file are replaced with an proper import statement.
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
  // Import all functions from consteval file that have their declaration in the
  // main file. Do not erase the declarations from the main file so that
  // func.call ops can resolve the symbol.
  void createImportForDecls(emitpy::FileOp mainFile,
                            ArrayRef<func::FuncOp> decls) {
    OpBuilder builder(&getContext());
    llvm::SmallVector<Attribute> memberNames;
    llvm::SmallVector<Attribute> emptyAliases;
    for (auto funcDecl : decls) {
      memberNames.push_back(builder.getStringAttr(funcDecl.getSymName()));
      emptyAliases.push_back(builder.getStringAttr(""));
    }

    builder.setInsertionPointToStart(&mainFile.getBodyRegion().front());
    builder.create<emitpy::ImportOp>(
        mainFile.getLoc(), builder.getStringAttr("consteval"),
        /*module_alias=*/nullptr,
        /*members_to_import=*/builder.getArrayAttr(memberNames),
        /*member_aliases=*/builder.getArrayAttr(emptyAliases),
        /*import_all=*/nullptr);
  }

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

    // Find main and consteval files in the device module.
    //
    emitpy::FileOp mainFileOp, constevalFileOp;
    for (auto fileOp : deviceModule.getOps<emitpy::FileOp>()) {
      if (fileOp.getId() == "main") {
        mainFileOp = fileOp;
      }
      if (fileOp.getId() == "consteval") {
        constevalFileOp = fileOp;
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

    if (cpuModuleOp) {
      auto cpuModule =
          mlir::cast<mlir::ModuleOp>(cpuModuleOp.getBody()->front());
      collectImports(cpuModule);
    }

    // Move all collected imports to the root module.
    //
    for (auto importOp : llvm::reverse(imports)) {
      importOp->moveBefore(&rootBody, rootBody.begin());
    }

    Block *blockToMoveCPUOpsTo = nullptr;
    if (constevalFileOp) {
      blockToMoveCPUOpsTo = &constevalFileOp.getBodyRegion().front();
    } else {
      blockToMoveCPUOpsTo = &rootBody;
    }

    // Move all operations from CPU module (CPU-hoisted function
    // definitions). If TTNNFileSplit was performed, move them to the consteval
    // file. Otherwise, move them to the root module.
    //
    if (cpuModuleOp) {
      auto cpuModule =
          mlir::cast<mlir::ModuleOp>(cpuModuleOp.getBody()->front());

      llvm::SmallVector<Operation *> cpuOps;

      for (auto &op : *cpuModule.getBody()) {
        cpuOps.push_back(&op);
      }
      for (auto *op : llvm::reverse(cpuOps)) {
        op->moveBefore(blockToMoveCPUOpsTo, blockToMoveCPUOpsTo->begin());
      }

      cpuModuleOp->erase();
    }

    // Erase CPU-hoisted declarations. If TTNNFileSplit was performed, erase
    // declarations from the consteval file and replace all declarations in the
    // main file with a proper import statement. Otherwise, erase declarations
    // from the device module.
    //
    auto eraseCPUDecls = [](auto container) {
      llvm::SmallVector<func::FuncOp> decls;
      for (auto funcOp : container.template getOps<func::FuncOp>()) {
        if (ttmlir::utils::isForwardCPUDeclarationFunc(funcOp)) {
          decls.push_back(funcOp);
        }
      }
      for (auto funcOp : decls) {
        funcOp->erase();
      }
    };

    if (constevalFileOp) {
      llvm::SmallVector<func::FuncOp> mainDecls;
      for (auto funcOp : mainFileOp.getOps<func::FuncOp>()) {
        if (funcOp.isDeclaration()) {
          mainDecls.push_back(funcOp);
        }
      }
      if (!mainDecls.empty()) {
        createImportForDecls(mainFileOp, mainDecls);
      }
      eraseCPUDecls(constevalFileOp);
    } else {
      eraseCPUDecls(deviceModule);
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
