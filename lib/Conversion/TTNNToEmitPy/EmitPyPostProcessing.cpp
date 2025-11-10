// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "llvm/ADT/DenseMap.h"

#include <cstdint>

namespace mlir::tt {

#define GEN_PASS_DEF_EMITPYPOSTPROCESSING
#include "ttmlir/Conversion/Passes.h.inc"

namespace {

class EmitPyPostProcessingPass
    : public impl::EmitPyPostProcessingBase<EmitPyPostProcessingPass> {
public:
  using impl::EmitPyPostProcessingBase<
      EmitPyPostProcessingPass>::EmitPyPostProcessingBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    llvm::StringMap<uint64_t> inputCounter;
    llvm::DenseMap<Value, std::string> valueNames;

    // Handle emitpy::CallOpaqueOp operations:
    module.walk([&](emitpy::CallOpaqueOp callOp) {
      if (auto calleeAttr = callOp.getCalleeAttr()) {
        std::string argName =
            calleeAttr.getValue().str() + "_" +
            std::to_string(inputCounter[calleeAttr.getValue()]++);
        // Only insert into valueNames if the operation has results
        if (callOp.getNumResults() > 0) {
          valueNames.insert({callOp.getResult(0), argName});
        }
        StringAttr suggestName = StringAttr::get(ctx, argName);
        callOp->setAttr("suggest_name", suggestName);
      }
    });

    // Handle func::CallOp operations:
    module.walk([&](func::CallOp callOp) {
      if (auto calleeAttr = callOp.getCalleeAttr()) {
        std::string argName =
            calleeAttr.getValue().str() + "_" +
            std::to_string(inputCounter[calleeAttr.getValue()]++);
        // Only insert into valueNames if the operation has results
        if (callOp.getNumResults() > 0) {
          valueNames.insert({callOp.getResult(0), argName});
        }
        StringAttr suggestName = StringAttr::get(ctx, argName);
        callOp->setAttr("suggest_name", suggestName);
      }
    });

    // Handle func::FuncOp operations:
    module.walk([&](func::FuncOp funcOp) {
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        std::string argName = funcOp.getNumArguments() > 1
                                  ? "input_" + std::to_string(i)
                                  : "input";
        valueNames.insert({funcOp.getArgument(i), argName});
        StringAttr suggestName = StringAttr::get(ctx, argName);
        funcOp.setArgAttr(i, "emitpy.suggest_name", suggestName);
      }
    });

    // Handle subscript operations:
    module.walk([&](emitpy::SubscriptOp subscriptOp) {
      Value operand = subscriptOp.getOperand(0);
      Value indexValue = subscriptOp.getIndex();

      // Extract the index string from the LiteralOp
      if (auto literalOp = indexValue.getDefiningOp<emitpy::LiteralOp>()) {
        std::string indexStr = literalOp.getValue().str();
        std::string argName = valueNames[operand] + "_" + indexStr;
        StringAttr suggestName = StringAttr::get(ctx, argName);
        subscriptOp->setAttr("suggest_name", suggestName);
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyPostProcessingPass() {
  return std::make_unique<EmitPyPostProcessingPass>();
}

} // namespace mlir::tt
