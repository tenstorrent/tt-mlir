// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/FunctionTypes.h"
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

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNCONVERTCPUHOISTEDFUNCTIONCALLSTODPS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

/// Pass to add Destination Passing Style semantics to CPU-hoisted functions.
/// This pass runs on DeviceModuleOp. It updates the hoisted function prototypes
/// to add extra output arguments for each of the op results and updates all
/// call sites accordingly.
class TTNNConvertCPUHoistedFunctionCallsToDPS
    : public impl::TTNNConvertCPUHoistedFunctionCallsToDPSBase<
          TTNNConvertCPUHoistedFunctionCallsToDPS> {
public:
  using impl::TTNNConvertCPUHoistedFunctionCallsToDPSBase<
      TTNNConvertCPUHoistedFunctionCallsToDPS>::
      TTNNConvertCPUHoistedFunctionCallsToDPSBase;
  void runOnOperation() final {
    auto module = getOperation()->getParentOfType<ttcore::DeviceModuleOp>();

    // This pass should run on inner ModuleOp within DeviceModuleOp.
    if (!module) {
      signalPassFailure();
      return;
    }

    // Gathering hoisted function prototypes in the device module.
    llvm::SmallVector<func::FuncOp> hoistedFuncStubs;
    module.walk([&](func::FuncOp funcOp) {
      if (ttmlir::utils::isForwardX280CPUDeclarationFunc(funcOp)) {
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
      module.walk([&](func::CallOp callOp) {
        if (callOp.getCallee() != funcOp.getName()) {
          return;
        }

        builder.setInsertionPoint(callOp);

        // Create new call op with extra output arguments.
        llvm::SmallVector<Value> callOperands(callOp.getOperands().begin(),
                                              callOp.getOperands().end());

        for (auto returnType : returnTypes) {
          auto outputTensor =
              builder.create<ttir::EmptyOp>(callOp.getLoc(), returnType);

          callOperands.push_back(outputTensor);
        }

        auto newCallOp =
            builder.create<func::CallOp>(callOp.getLoc(), funcOp, callOperands);

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

} // namespace

} // namespace mlir::tt::ttnn
