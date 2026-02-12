// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNCOLLASPED2M
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNCollaspeD2M : public impl::TTNNCollaspeD2MBase<TTNNCollaspeD2M> {

public:
  using impl::TTNNCollaspeD2MBase<TTNNCollaspeD2M>::TTNNCollaspeD2MBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SmallVector<func::FuncOp> funcsToDelete;

    moduleOp.walk([&](D2MSubgraphOp dispatchOp) {
      func::FuncOp mainFunc = dispatchOp.getD2MMainFunc();
      if (failed(inlineDispatchOp(moduleOp, dispatchOp))) {
        signalPassFailure();
        return;
      }
      funcsToDelete.push_back(mainFunc);
    });

    for (func::FuncOp func : funcsToDelete) {
      func.erase();
    }
  }

private:
  LogicalResult inlineDispatchOp(ModuleOp parentModule,
                                 D2MSubgraphOp dispatchOp) {
    func::FuncOp mainFunc = dispatchOp.getD2MMainFunc();
    if (!mainFunc) {
      return dispatchOp.emitOpError("could not find D2M function '")
             << dispatchOp.getD2mFunc() << "' in parent module";
    }

    // Map func args to dispatch op input operands
    OpBuilder builder(dispatchOp);
    IRMapping mapping;
    for (auto [arg, input] :
         llvm::zip(mainFunc.getArguments(), dispatchOp.getInputs())) {
      mapping.map(arg, input);
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
