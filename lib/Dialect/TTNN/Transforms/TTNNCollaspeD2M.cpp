// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNCOLLASPED2M
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNCollaspeD2M : public impl::TTNNCollaspeD2MBase<TTNNCollaspeD2M> {

public:
  using impl::TTNNCollaspeD2MBase<TTNNCollaspeD2M>::TTNNCollaspeD2MBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Inline the main func and move kernel funcs to parent module scope.
    moduleOp.walk([&](DispatchD2MOp dispatchOp) -> WalkResult {
      if (failed(inlineDispatchOp(moduleOp, dispatchOp))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }

private:
  LogicalResult inlineDispatchOp(ModuleOp parentModule,
                                 DispatchD2MOp dispatchOp) {
    func::FuncOp mainFunc = dispatchOp.getD2MMainFunc();
    if (!mainFunc) {
      return dispatchOp.emitOpError("could not find D2M function '")
             << dispatchOp.getD2mFunc() << "' in nested module";
    }

    // Clone kernel funcs to parent module with unique names.
    SmallVector<func::FuncOp> kernelFuncs = dispatchOp.getKernelFuncs();
    SymbolTable parentSymbolTable(parentModule);
    OpBuilder moduleBuilder(parentModule.getContext());
    moduleBuilder.setInsertionPointToEnd(parentModule.getBody());

    std::string prefix = mainFunc.getSymName().str() + "_";

    // Clone kernel funcs and update symbol references in mainFunc.
    for (func::FuncOp kernelFunc : kernelFuncs) {
      std::string newName = prefix + kernelFunc.getSymName().str();
      StringAttr newNameAttr =
          StringAttr::get(parentModule.getContext(), newName);

      Operation *clonedOp = moduleBuilder.clone(*kernelFunc.getOperation());
      auto clonedFunc = cast<func::FuncOp>(clonedOp);
      clonedFunc.setSymName(newName);
      parentSymbolTable.insert(clonedOp);

      (void)SymbolTable::replaceAllSymbolUses(kernelFunc, newNameAttr,
                                              mainFunc);
    }

    // Map func args to dispatch op operands.
    OpBuilder builder(dispatchOp);
    IRMapping mapping;
    auto allOperands =
        llvm::concat<Value>(dispatchOp.getInputs(), dispatchOp.getOutputs());
    for (auto [arg, operand] :
         llvm::zip(mainFunc.getArguments(), allOperands)) {
      mapping.map(arg, operand);
    }

    // Clone function body operations and map return value.
    Block &funcBody = mainFunc.getBody().front();
    for (Operation &op : funcBody.without_terminator()) {
      builder.clone(op, mapping);
    }

    auto returnOp = dyn_cast<func::ReturnOp>(funcBody.getTerminator());
    if (!returnOp) {
      return dispatchOp.emitOpError(
          "Function must have func.return terminator");
    }
    SmallVector<Value> resultValues;
    for (Value operand : returnOp.getOperands()) {
      resultValues.push_back(mapping.lookup(operand));
    }

    if (resultValues.size() != dispatchOp.getNumResults()) {
      return dispatchOp.emitOpError("Return value count mismatch: expected ")
             << dispatchOp.getNumResults() << " but got "
             << resultValues.size();
    }

    for (auto [result, replacement] :
         llvm::zip(dispatchOp.getResults(), resultValues)) {
      result.replaceAllUsesWith(replacement);
    }

    dispatchOp.erase();
    return success();
  }
};

} // namespace
} // namespace mlir::tt::ttnn
