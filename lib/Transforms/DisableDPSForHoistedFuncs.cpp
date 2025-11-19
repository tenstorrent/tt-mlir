// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir::tt::transforms {
#define GEN_PASS_DEF_DISABLEDPSFORHOISTEDFUNCS
#include "ttmlir/Transforms/Passes.h.inc"

class DisableDPSForHoistedFuncs
    : public impl::DisableDPSForHoistedFuncsBase<DisableDPSForHoistedFuncs> {
public:
  using impl::DisableDPSForHoistedFuncsBase<
      DisableDPSForHoistedFuncs>::DisableDPSForHoistedFuncsBase;
  void runOnOperation() final {
    auto rootModule = getOperation();

    if (rootModule->getParentOp() != nullptr) {
      return;
    }

    ttcore::DeviceModuleOp deviceModuleOp;
    for (auto &op : rootModule.getBodyRegion().front()) {
      if (auto maybeDeviceModule = dyn_cast<ttcore::DeviceModuleOp>(op)) {
        deviceModuleOp = maybeDeviceModule;
        break;
      }
    }

    assert(deviceModuleOp && "DeviceModuleOp not found!");

    mlir::ModuleOp deviceModule = dyn_cast<mlir::ModuleOp>(
        deviceModuleOp.getBodyRegion().front().front());

    assert(deviceModule && "Inner ModuleOp not found!");

    ttcore::CPUModuleOp cpuModuleOp;
    for (auto &op : rootModule.getBodyRegion().front()) {
      if (auto maybeCPUModule = dyn_cast<ttcore::CPUModuleOp>(op)) {
        cpuModuleOp = maybeCPUModule;
        break;
      }
    }

    assert(cpuModuleOp && "CPUModuleOp not found!");

    mlir::ModuleOp cpuModule =
        dyn_cast<mlir::ModuleOp>(cpuModuleOp.getBodyRegion().front().front());

    assert(cpuModule && "Inner CPUModuleOp ModuleOp not found!");

    // Gathering hoisted functions from CPU module
    llvm::SmallVector<func::FuncOp, 4> hoistedFuncs;
    cpuModule.walk([&](func::FuncOp funcOp) {
      if (funcOp->hasAttr(ttir::ReturnToOutputMappingAttr::name)) {
        hoistedFuncs.push_back(funcOp);
      }
    });

    for (auto funcOp : hoistedFuncs) {
      // Removing the DPS argument from hoisted functions in device module
      auto funcType = funcOp.getFunctionType();

      SmallVector<Type, 4> newInputTypes(funcType.getInputs().begin(),
                                         funcType.getInputs().end() - 1);

      funcOp.setType(mlir::FunctionType::get(&getContext(), newInputTypes,
                                             funcType.getResults()));

      auto &entryBlock = funcOp.front();
      entryBlock.getArgument(entryBlock.getNumArguments() - 1).dropAllUses();
      entryBlock.eraseArgument(entryBlock.getNumArguments() - 1);

      // Removing the ReturnToOutputMappingAttr attribute
      funcOp->removeAttr(ttir::ReturnToOutputMappingAttr::name);
    }

    // Gathering hoisted function prototypes in the device module
    auto isHoistedFunctionDecl = [&hoistedFuncs](func::FuncOp funcOp) {
      return funcOp.isDeclaration() &&
             llvm::any_of(hoistedFuncs, [&](func::FuncOp hoistedFunc) {
               return funcOp.getSymName() ==
                      (hoistedFunc.getSymName().str() + "_decl");
             });
    };

    llvm::SmallVector<func::FuncOp, 4> hoistedFuncStubs;
    deviceModule.walk([&](func::FuncOp funcOp) {
      if (isHoistedFunctionDecl(funcOp)) {
        hoistedFuncStubs.push_back(funcOp);
      }
    });

    // Removing the DPS argument from hoisted function prototypes in device
    // module
    for (auto funcOp : hoistedFuncStubs) {
      auto funcType = funcOp.getFunctionType();
      SmallVector<Type, 4> newInputTypes(funcType.getInputs().begin(),
                                         funcType.getInputs().end() - 1);
      funcOp.setType(mlir::FunctionType::get(&getContext(), newInputTypes,
                                             funcType.getResults()));
    }

    // Finally, removing the DPS argument from the call ops
    deviceModule.walk([&](func::CallOp callOp) {
      auto calleeFuncOp =
          deviceModule.lookupSymbol<func::FuncOp>(callOp.getCallee());
      if (isHoistedFunctionDecl(calleeFuncOp)) {
        auto dpsOperand = callOp->getOperand(callOp.getNumOperands() - 1);

        // Remove all ops which use this operand (except the callOp itself)
        for (auto &use : dpsOperand.getUses()) {
          if (use.getOwner() != callOp) {
            use.getOwner()->erase();
          }
        }

        // Remove the DPS operand from the call op
        callOp->eraseOperand(callOp.getNumOperands() - 1);
      }
    });
  }
};
} // namespace mlir::tt::transforms
