// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"

#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt {

#define GEN_PASS_DEF_EMITPYWRAPFUNCARGSFORGOLDEN
#include "ttmlir/Conversion/Passes.h.inc"

namespace {

class EmitPyWrapFuncArgsForGoldenPass
    : public impl::EmitPyWrapFuncArgsForGoldenBase<
          EmitPyWrapFuncArgsForGoldenPass> {
public:
  using impl::EmitPyWrapFuncArgsForGoldenBase<
      EmitPyWrapFuncArgsForGoldenPass>::EmitPyWrapFuncArgsForGoldenBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) { processFunction(funcOp); });
  }

private:
  void processFunction(func::FuncOp funcOp) {
    // Only process functions marked as CPU hoisted.
    // TODO(dmilinkovic): a bit hacky, reconsider.
    if (!funcOp->hasAttr(ttir::CPUHoistedFuncAttr::name)) {
      return;
    }

    OpBuilder builder(funcOp.getContext());

    // Insert to_torch calls for tensor arguments at the beginning of the
    // function.
    Block &entryBlock = funcOp.getBody().front();
    builder.setInsertionPointToStart(&entryBlock);

    for (BlockArgument arg : funcOp.getArguments()) {
      // Create ttnn.to_torch call.
      auto toTorchOp = builder.create<emitpy::CallOpaqueOp>(
          funcOp.getLoc(), arg.getType(), "ttnn.to_torch", ValueRange{arg},
          /*args=*/nullptr, /*keywordArgs=*/nullptr);

      // Replace all uses of the original argument with the to_torch result,
      // except for the to_torch op itself.
      arg.replaceAllUsesExcept(toTorchOp.getResult(0), toTorchOp);
    }

    // Insert from_torch calls for tensor return values.
    funcOp.walk([&](func::ReturnOp returnOp) {
      builder.setInsertionPoint(returnOp);

      SmallVector<Value> newReturnOperands;
      for (Value returnValue : returnOp.getOperands()) {
        // Create ttnn.from_torch call.
        auto fromTorchOp = builder.create<emitpy::CallOpaqueOp>(
            returnOp.getLoc(), returnValue.getType(), "ttnn.from_torch",
            ValueRange{returnValue}, /*args=*/nullptr, /*keywordArgs=*/nullptr);

        newReturnOperands.push_back(fromTorchOp.getResult(0));
      }

      // Update the return op with the new operands.
      returnOp.getOperandsMutable().assign(newReturnOperands);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createEmitPyWrapFuncArgsForGoldenPass() {
  return std::make_unique<EmitPyWrapFuncArgsForGoldenPass>();
}

} // namespace mlir::tt
