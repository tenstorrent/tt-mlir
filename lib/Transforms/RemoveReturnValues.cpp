// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::transforms {

#define GEN_PASS_DEF_REMOVERETURNVALUES
#include "ttmlir/Transforms/Passes.h.inc"

namespace {
// Pass to remove return values from functions that have been cleaned up; we do
// not want our hoisted funcs to return, since this would involve allocating new
// tensors etc.
class RemoveReturnValuesPass
    : public impl::RemoveReturnValuesBase<RemoveReturnValuesPass> {
public:
  using impl::RemoveReturnValuesBase<
      RemoveReturnValuesPass>::RemoveReturnValuesBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Find all functions with return values
    moduleOp.walk([&](func::FuncOp funcOp) {
      auto funcType = funcOp.getFunctionType();
      if (funcType.getResults().empty()) {
        return;
      }

      // Find the return operation.
      func::ReturnOp returnOp;
      for (Block &block : funcOp.getBlocks()) {
        if (auto retOp = dyn_cast<func::ReturnOp>(block.getTerminator())) {
          returnOp = retOp;
          break;
        }
      }

      if (!returnOp) {
        funcOp->emitError() << "Function does not have a return operation";
        signalPassFailure();
        return;
      }

      if (returnOp.getNumOperands() == 0) {
        // Function already has no return values, nothing to transform.
        return;
      }

      // Replace the return operation with an empty return.
      rewriter.setInsertionPoint(returnOp);
      rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp);

      // Update the function type to remove the return values
      auto newFuncType =
          FunctionType::get(funcOp.getContext(), funcType.getInputs(), {});
      funcOp.setType(newFuncType);
    });
  }
};
} // namespace
} // namespace mlir::tt::transforms
