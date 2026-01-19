// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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

    // Find ttnn.dispatch_d2m ops and inline main func, move kernel funcs to
    // module scope.
    SmallVector<DispatchD2MOp> dispatchOps;
    moduleOp.walk(
        [&](DispatchD2MOp dispatchOp) { dispatchOps.push_back(dispatchOp); });

    if (dispatchOps.empty()) {
      return;
    }

    llvm::errs() << "Found " << dispatchOps.size()
                 << " ttnn.dispatch_d2m op(s) to inline.\n";

    for (DispatchD2MOp dispatchOp : dispatchOps) {
      if (failed(inlineDispatchOp(moduleOp, dispatchOp))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult inlineDispatchOp(ModuleOp parentModule,
                                 DispatchD2MOp dispatchOp) {
    func::FuncOp mainFunc = dispatchOp.lookupD2MMainFunc();
    if (!mainFunc) {
      return dispatchOp.emitOpError("could not find D2M function '")
             << dispatchOp.getD2mFunc() << "' in body region";
    }

    // Copy all kernel funcs to the parent module with unique names.
    SmallVector<func::FuncOp> kernelFuncs = dispatchOp.getKernelFuncs();
    SymbolTable parentSymbolTable(parentModule);
    OpBuilder moduleBuilder(parentModule.getContext());
    moduleBuilder.setInsertionPointToEnd(parentModule.getBody());

    std::string prefix = mainFunc.getSymName().str() + "_";

    // Update symbol references in mainFunc and build rename map for later.
    llvm::DenseMap<StringAttr, StringAttr> symbolRenameMap;
    for (func::FuncOp kernelFunc : kernelFuncs) {
      StringRef oldName = kernelFunc.getSymName();
      std::string newName = prefix + oldName.str();
      StringAttr oldNameAttr =
          StringAttr::get(parentModule.getContext(), oldName);
      StringAttr newNameAttr =
          StringAttr::get(parentModule.getContext(), newName);
      symbolRenameMap[oldNameAttr] = newNameAttr;

      // Update all references to this kernel within mainFunc.
      (void)SymbolTable::replaceAllSymbolUses(kernelFunc, newNameAttr,
                                              mainFunc);
    }

    // Clone and rename kernel funcs to parent module.
    for (func::FuncOp kernelFunc : kernelFuncs) {
      StringRef oldName = kernelFunc.getSymName();
      std::string newName = prefix + oldName.str();

      IRMapping funcMapping;
      Operation *clonedOp =
          moduleBuilder.clone(*kernelFunc.getOperation(), funcMapping);
      auto clonedFunc = cast<func::FuncOp>(clonedOp);
      clonedFunc.setSymName(newName);
      parentSymbolTable.insert(clonedOp);
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

    // Clone all ops except return and collect return values for replacement.
    SmallVector<Value> resultValues;
    for (Operation &op : mainFunc.getBody().front()) {
      if (auto returnOp = mlir::dyn_cast<func::ReturnOp>(&op)) {
        for (Value operand : returnOp.getOperands()) {
          resultValues.push_back(mapping.lookup(operand));
        }
      } else {
        builder.clone(op, mapping);
      }
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
