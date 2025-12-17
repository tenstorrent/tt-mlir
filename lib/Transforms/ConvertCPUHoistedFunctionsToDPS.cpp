// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"

namespace mlir::tt::transforms {
#define GEN_PASS_DEF_CONVERTCPUHOISTEDFUNCTIONSTODPS
#include "ttmlir/Transforms/Passes.h.inc"

/// Pass to add Destination Passing Style semantics to CPU-hoisted functions.
/// This pass can be run on either DeviceModuleOp or CPUModuleOp.
/// - If run on DeviceModuleOp, it updates the hoisted function prototypes
///   to add extra output arguments for each of the op results and updates all
///   call sites accordingly.
/// - If run on CPUModuleOp, it updates the hoisted function definitions to
///   add extra output arguments, modifies the function body to write the
///   return values to these arguments, and changes the return type to void.
class ConvertCPUHoistedFunctionsToDPS
    : public impl::ConvertCPUHoistedFunctionsToDPSBase<
          ConvertCPUHoistedFunctionsToDPS> {
public:
  using impl::ConvertCPUHoistedFunctionsToDPSBase<
      ConvertCPUHoistedFunctionsToDPS>::ConvertCPUHoistedFunctionsToDPSBase;
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
    // Gathering hoisted functions from CPU module.
    llvm::SmallVector<func::FuncOp> hoistedFuncs;
    cpuModule.walk([&](func::FuncOp funcOp) {
      if (funcOp->hasAttr(ttir::CPUHoistedFuncAttr::name)) {
        hoistedFuncs.push_back(funcOp);
      }
    });

    // Adding DPS semantics by adding extra output arguments.
    for (auto funcOp : hoistedFuncs) {
      OpBuilder builder(funcOp);

      auto returnOp = llvm::dyn_cast<func::ReturnOp>(
          funcOp.getBody().back().getTerminator());

      assert(returnOp &&
             "Hoisted function does not have a valid return operation!");

      auto returnedValues = returnOp.getOperands();

      const auto returnTypes =
          llvm::map_to_vector(returnedValues, [](mlir::Value val) {
            auto tensorType =
                llvm::dyn_cast_or_null<RankedTensorType>(val.getType());
            assert(tensorType &&
                   "Return type can't be casted to RankedTensorType!");
            return tensorType;
          });

      // Create new function type with extra output arguments and void return
      // type.
      auto inputTypes =
          ttmlir::utils::flatten(funcOp.getArgumentTypes(), returnTypes);

      auto newFuncType =
          builder.getFunctionType(inputTypes, llvm::ArrayRef<Type>{});
      funcOp.setType(newFuncType);

      // Adjust block arguments.
      for (auto returnType : returnTypes) {
        funcOp.getBody().front().addArgument(returnType, funcOp.getLoc());
      }

      // Add bufferization access attribute to new output arguments.
      for (auto [i, _] : llvm::enumerate(returnTypes)) {
        funcOp.setArgAttr(funcOp.getNumArguments() - returnTypes.size() + i,
                          "bufferization.access",
                          builder.getStringAttr("write"));
      }

      // Adjust arg_ranks attribute.
      auto argRanksAttr = funcOp->getAttrOfType<ArrayAttr>("arg_ranks");
      assert(argRanksAttr &&
             "arg_ranks attribute not found on hoisted function");

      llvm::SmallVector<Attribute> newArgRanksAttrValues(
          argRanksAttr.getValue().begin(), argRanksAttr.getValue().end());

      for (auto returnType : returnTypes) {
        newArgRanksAttrValues.push_back(
            builder.getIntegerAttr(builder.getI64Type(), returnType.getRank()));
      }

      funcOp->setAttr("arg_ranks", builder.getArrayAttr(newArgRanksAttrValues));

      builder.setInsertionPoint(returnOp);

      // Insert linalg.copy ops to write return values to output arguments.
      // TODO(dmilinkovic): Optimize copies away where possible by relying on
      // the DPS nature of linalg ops - issue #6096.
      for (auto [index, returnValue] : llvm::enumerate(returnedValues)) {
        auto outputArgument = funcOp.getArgument(funcOp.getNumArguments() -
                                                 returnTypes.size() + index);
        linalg::CopyOp::create(builder, returnOp.getLoc(), returnValue,
                               outputArgument);
      }

      // Insert void return op.
      func::ReturnOp::create(builder, returnOp.getLoc());
      returnOp.erase();
    }
  }

  void enableDPSInDeviceModule(mlir::ModuleOp deviceModule) {
    // Gathering hoisted function prototypes in the device module.
    llvm::SmallVector<func::FuncOp> hoistedFuncStubs;
    deviceModule.walk([&](func::FuncOp funcOp) {
      if (funcOp->hasAttr(ttir::CPUHoistedFuncAttr::name)) {
        hoistedFuncStubs.push_back(funcOp);
      }
    });

    // Adding DPS semantics by adding extra output arguments and updating the
    // call sites.
    for (auto funcOp : hoistedFuncStubs) {
      OpBuilder builder(funcOp);

      auto returnTypes = funcOp.getFunctionType().getResults();

      // Create new function type with extra output arguments.
      llvm::SmallVector<Type> inputTypes(funcOp.getArgumentTypes().begin(),
                                         funcOp.getArgumentTypes().end());
      for (auto returnType : returnTypes) {
        inputTypes.push_back(returnType);
      }

      auto newFuncType = builder.getFunctionType(inputTypes, returnTypes);
      funcOp.setType(newFuncType);

      llvm::SmallVector<func::CallOp> callsToErase;

      // Update all call sites.
      deviceModule.walk([&](func::CallOp callOp) {
        if (callOp.getCallee() != funcOp.getName()) {
          return;
        }

        builder.setInsertionPoint(callOp);

        // Create new call op with extra output arguments.
        llvm::SmallVector<Value> callOperands(callOp.getOperands().begin(),
                                              callOp.getOperands().end());

        for (auto returnType : returnTypes) {
          auto outputTensor =
              ttir::EmptyOp::create(builder, callOp.getLoc(), returnType);

          callOperands.push_back(outputTensor);
        }

        auto newCallOp = func::CallOp::create(builder, callOp.getLoc(), funcOp,
                                              callOperands);

        // Copy attributes from old call op to new call op.
        newCallOp->setAttrs(callOp->getAttrs());

        // Replace uses of old call op results with new call op results.
        for (auto [index, result] : llvm::enumerate(callOp.getResults())) {
          result.replaceAllUsesWith(newCallOp.getResult(index));
        }

        // Schedule original call op for deletion.
        callsToErase.push_back(callOp);
      });

      for (auto callOp : callsToErase) {
        callOp.erase();
      }
    }
  }
};

} // namespace mlir::tt::transforms
