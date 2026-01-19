// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"

#include <cstdint>

namespace mlir::tt {

#define GEN_PASS_DEF_EMITPYNAMEVARS
#include "ttmlir/Conversion/Passes.h.inc"

namespace {

class EmitPyNameVarsPass : public impl::EmitPyNameVarsBase<EmitPyNameVarsPass> {
public:
  using impl::EmitPyNameVarsBase<EmitPyNameVarsPass>::EmitPyNameVarsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    llvm::StringMap<uint64_t> inputCounter;
    llvm::DenseMap<Value, std::string> valueNames;
    llvm::StringSet<> reservedNames;

    module.walk(
        [&](func::FuncOp funcOp) { reservedNames.insert(funcOp.getName()); });

    // Handle emitpy::CallOpaqueOp operations:
    module.walk([&](emitpy::CallOpaqueOp callOp) {
      if (auto calleeAttr = callOp.getCalleeAttr()) {
        std::string argName =
            calleeAttr.getValue().str() + "_" +
            std::to_string(inputCounter[calleeAttr.getValue()]++);

        // Check if the generated name collides with a function name
        while (reservedNames.contains(argName)) {
          argName = argName + "_" + std::to_string(inputCounter[argName]++);
        }

        // Insert all results into valueNames for multi-result operations
        for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
          valueNames.insert({callOp.getResult(i), argName});
        }
        StringAttr suggestName = StringAttr::get(ctx, argName);
        callOp->setAttr("emitpy.name", suggestName);
      }
    });

    // Handle func::CallOp operations:
    module.walk([&](func::CallOp callOp) {
      if (auto calleeAttr = callOp.getCalleeAttr()) {
        std::string argName =
            calleeAttr.getValue().str() + "_" +
            std::to_string(inputCounter[calleeAttr.getValue()]++);

        // Check if the generated name collides with a function name
        while (reservedNames.contains(argName)) {
          argName = argName + "_" + std::to_string(inputCounter[argName]++);
        }

        // Insert all results into valueNames for multi-result operations
        for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
          valueNames.insert({callOp.getResult(i), argName});
        }
        StringAttr suggestName = StringAttr::get(ctx, argName);
        callOp->setAttr("emitpy.name", suggestName);
      }
    });

    // Handle func::FuncOp operations:
    module.walk([&](func::FuncOp funcOp) {
      if (funcOp.isDeclaration()) {
        return;
      }

      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        std::string argName;

        // Check if an emitpy.name attribute already exists for this argument.
        // If so, use it instead of generating a new name.
        if (auto existingNameAttr =
                funcOp.getArgAttrOfType<StringAttr>(i, "emitpy.name")) {
          argName = existingNameAttr.getValue().str();
        } else {
          argName = funcOp.getNumArguments() > 1 ? "input_" + std::to_string(i)
                                                 : "input";
        }

        // Check if parameter name collides with the function name
        if (reservedNames.contains(argName)) {
          argName = argName + "_arg";
        }

        valueNames.insert({funcOp.getArgument(i), argName});
        StringAttr suggestName = StringAttr::get(ctx, argName);
        funcOp.setArgAttr(i, "emitpy.name", suggestName);
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

        // Check if the generated name collides with a function name
        // If so, add a "var" suffix to disambiguate
        if (reservedNames.contains(argName)) {
          argName = argName + "_var";
        }

        valueNames.insert({subscriptOp.getResult(), argName});
        StringAttr suggestName = StringAttr::get(ctx, argName);
        subscriptOp->setAttr("emitpy.name", suggestName);
      }
    });

    // Handle constant operations:
    module.walk([&](emitpy::ConstantOp constantOp) {
      std::string argName = "const";
      if (IntegerAttr value =
              llvm::dyn_cast<IntegerAttr>(constantOp.getValue())) {
        argName += std::to_string(value.getValue().getSExtValue());
      }
      argName += "_" + std::to_string(inputCounter[argName]++);

      // Check if the generated name collides with a function name
      while (reservedNames.contains(argName)) {
        argName = argName + "_" + std::to_string(inputCounter[argName]++);
      }

      valueNames.insert({constantOp.getResult(), argName});
      StringAttr suggestName = StringAttr::get(ctx, argName);
      constantOp->setAttr("emitpy.name", suggestName);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyNameVarsPass() {
  return std::make_unique<EmitPyNameVarsPass>();
}

} // namespace mlir::tt
