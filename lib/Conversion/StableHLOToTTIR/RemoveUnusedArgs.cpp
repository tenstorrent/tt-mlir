// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/RemoveUnusedArgs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_REMOVEUNUSEDARGS
#include "ttmlir/Conversion/Passes.h.inc"
} // namespace mlir::tt::ttir

namespace {
struct RemoveUnusedArgsPass
    : public ttir::impl::RemoveUnusedArgsBase<RemoveUnusedArgsPass> {

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<unsigned> argsToRemove;

    module.walk([&](func::FuncOp func) {
      Block &entry = func.front();
      argsToRemove.clear();

      // Step 1: detect unused args
      for (auto arg : llvm::enumerate(entry.getArguments())) {
        if (arg.value().use_empty()) {
          argsToRemove.push_back(arg.index());
        }
      }
      if (argsToRemove.empty()) {
        return;
      }

      // Step 2: construct new function type
      FunctionType oldType = func.getFunctionType();
      SmallVector<Type> newInputs;
      for (auto [i, t] : llvm::enumerate(oldType.getInputs())) {
        if (!llvm::is_contained(argsToRemove, i)) {
          newInputs.push_back(t);
        }
      }
      auto newType =
          FunctionType::get(func.getContext(), newInputs, oldType.getResults());

      // Step 3: update signature
      func.setType(newType);

      // Step 4: update block arguments
      for (unsigned i : llvm::reverse(argsToRemove)) {
        entry.eraseArgument(i);
      }

      // Step 5: fix all call sites
      for (Operation *user : llvm::make_early_inc_range(func->getUsers())) {
        if (auto call = dyn_cast<func::CallOp>(user)) {
          SmallVector<Value> newOperands;
          for (auto [i, operand] : llvm::enumerate(call.getArgOperands())) {
            if (!llvm::is_contained(argsToRemove, i)) {
              newOperands.push_back(operand);
            }
          }
          OpBuilder builder(call);
          auto newCall = builder.create<func::CallOp>(
              call.getLoc(), func.getName(), newType.getResults(), newOperands);
          call.replaceAllUsesWith(newCall.getOperation());
          call.erase();
        }
      }
    });
  }
};
} // namespace

namespace mlir::tt {
std::unique_ptr<mlir::Pass> createRemoveUnusedArgsPass() {
  return std::make_unique<RemoveUnusedArgsPass>();
}
} // namespace mlir::tt
