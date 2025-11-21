// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir::tt::transforms {
#define GEN_PASS_DEF_ENABLEDPSFORHOISTEDFUNCS
#include "ttmlir/Transforms/Passes.h.inc"

class EnableDPSForHoistedFuncs
    : public impl::EnableDPSForHoistedFuncsBase<EnableDPSForHoistedFuncs> {
public:
  using impl::EnableDPSForHoistedFuncsBase<
      EnableDPSForHoistedFuncs>::EnableDPSForHoistedFuncsBase;
  void runOnOperation() final {
    auto module = getOperation();

    // This pass should run either on inner ModuleOp within DeviceModuleOp or
    // CPUModuleOp.
    if (auto deviceModule = module->getParentOfType<ttcore::DeviceModuleOp>()) {
      enableDPSInDeviceModule(module);
      return;
    }

    if (auto cpuModule = module->getParentOfType<ttcore::CPUModuleOp>()) {
      enableDPSInCPUModule(module);
      return;
    }

    signalPassFailure();
  }

private:
  void enableDPSInCPUModule(mlir::ModuleOp cpuModule) {
    // Gathering hoisted functions from CPU module
    llvm::SmallVector<func::FuncOp, 4> hoistedFuncs;
    cpuModule.walk([&](func::FuncOp funcOp) {
      if (funcOp->hasAttr(ttir::HoistedFuncAttr::name)) {
        hoistedFuncs.push_back(funcOp);
      }
    });

    // Adding DPS semantics by adding an extra output argument
    for (auto funcOp : hoistedFuncs) {
      auto returnValue = funcOp.getBody().back().getTerminator()->getOperand(0);
      auto returnType =
          llvm::dyn_cast_or_null<RankedTensorType>(returnValue.getType());

      assert(returnType && "Return type is not RankedTensorType!");

      OpBuilder builder(funcOp);

      // Create new function type with extra output argument and void return
      // type
      llvm::SmallVector<Type, 4> inputTypes(funcOp.getArgumentTypes().begin(),
                                            funcOp.getArgumentTypes().end());
      inputTypes.push_back(returnType);

      auto newFuncType =
          builder.getFunctionType(inputTypes, llvm::ArrayRef<Type>{});
      funcOp.setType(newFuncType);

      // Adjust block arguments
      funcOp.getBody().front().addArgument(returnType, funcOp.getLoc());

      // Add bufferization access attribute to new output argument
      funcOp.setArgAttr(funcOp.getNumArguments() - 1, "bufferization.access",
                        builder.getStringAttr("write"));

      // Adjust arg_ranks attribute
      auto argRanksAttr = funcOp->getAttrOfType<ArrayAttr>("arg_ranks");
      assert(argRanksAttr &&
             "arg_ranks attribute not found on hoisted function");
      llvm::SmallVector<Attribute, 4> newArgRanksAttrValues(
          argRanksAttr.getValue().begin(), argRanksAttr.getValue().end());
      newArgRanksAttrValues.push_back(
          builder.getIntegerAttr(builder.getI64Type(), returnType.getRank()));
      funcOp->setAttr("arg_ranks", builder.getArrayAttr(newArgRanksAttrValues));

      auto returnOp = llvm::dyn_cast<func::ReturnOp>(
          funcOp.getBody().back().getTerminator());
      builder.setInsertionPoint(returnOp);

      // Add linalg.copy from return value to output argument
      builder.create<linalg::CopyOp>(
          returnOp.getLoc(), returnValue,
          funcOp.getArgument(funcOp.getNumArguments() - 1));

      // Replace return operation to not return any values
      builder.create<func::ReturnOp>(returnOp.getLoc());
      returnOp.erase();
    }
  }

  void enableDPSInDeviceModule(mlir::ModuleOp deviceModule) {
    // Gathering hoisted function prototypes in the device module
    llvm::SmallVector<func::FuncOp, 4> hoistedFuncStubs;
    deviceModule.walk([&](func::FuncOp funcOp) {
      if (funcOp->hasAttr(ttir::HoistedFuncAttr::name)) {
        hoistedFuncStubs.push_back(funcOp);
      }
    });

    // Adding DPS semantics by adding an extra output argument and updating the
    // call sites
    for (auto funcOp : hoistedFuncStubs) {
      OpBuilder builder(funcOp);

      auto returnType = funcOp.getFunctionType().getResult(0);
      // Create new function type with extra output argument and void return
      // type
      llvm::SmallVector<Type, 4> inputTypes(funcOp.getArgumentTypes().begin(),
                                            funcOp.getArgumentTypes().end());
      inputTypes.push_back(returnType);

      auto newFuncType = builder.getFunctionType(inputTypes, {returnType});
      funcOp.setType(newFuncType);

      // Update all call sites
      deviceModule.walk([&](func::CallOp callOp) {
        if (callOp.getCallee() == funcOp.getName()) {
          builder.setInsertionPoint(callOp);

          auto outputTensor =
              builder.create<ttir::EmptyOp>(callOp.getLoc(), returnType);

          // Create new call op with extra output argument
          llvm::SmallVector<Value, 4> callOperands(callOp.getOperands().begin(),
                                                   callOp.getOperands().end());
          callOperands.push_back(outputTensor);

          auto newCallOp = builder.create<func::CallOp>(callOp.getLoc(), funcOp,
                                                        callOperands);

          // Copy attributes from old call op to new call op
          newCallOp->setAttrs(callOp->getAttrs());

          callOp.getResult(0).replaceAllUsesWith(newCallOp->getResult(0));

          // Erase the original call op
          callOp.erase();
        }
      });
    }
  }
};

} // namespace mlir::tt::transforms
