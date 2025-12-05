// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

using namespace mlir;
using namespace mlir::tt::ttnn;

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNISOLATEOPS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"
} // namespace mlir::tt::ttnn

namespace {

/// Helper to determine if an operation should be isolated
bool isTargetOp(Operation *op, ArrayRef<std::string> filterOps) {
  // Only process TTNN dialect operations
  StringRef opName = op->getName().getStringRef();
  if (!opName.starts_with("ttnn.")) {
    return false;
  }

  // If no filter ops specified, skip only utility/management ops
  if (filterOps.empty()) {
    return !isa<DeallocateOp, GetDeviceOp, FromDeviceOp, ToDeviceOp,
               ToLayoutOp>(op);
  }

  // Check if op name matches any filter
  for (const auto &filter : filterOps) {
    if (opName.contains(filter)) {
      return true;
    }
  }
  return false;
}

/// Create a minimal isolated function with only the target operation
func::FuncOp createIsolatedFunction(Operation *targetOp,
                                     StringRef baseName,
                                     int index,
                                     OpBuilder &builder) {

  // Build function name
  std::string funcName = (baseName + "_" + Twine(index)).str();

  // Function arguments are the direct operands of the target op
  // with their final types (after preprocessing)
  SmallVector<Type> inputTypes;
  for (Value operand : targetOp->getOperands()) {
    inputTypes.push_back(operand.getType());
  }

  // Result types from the target op
  SmallVector<Type> resultTypes;
  for (Value result : targetOp->getResults()) {
    resultTypes.push_back(result.getType());
  }

  auto funcType = builder.getFunctionType(inputTypes, resultTypes);

  // Create the function
  auto isolatedFunc = builder.create<func::FuncOp>(
      targetOp->getLoc(), funcName, funcType);
  isolatedFunc.setPrivate();

  // Create function body
  Block *entryBlock = isolatedFunc.addEntryBlock();
  OpBuilder funcBuilder = OpBuilder::atBlockBegin(entryBlock);

  // Map target op operands directly to function arguments
  IRMapping mapping;
  for (auto [operand, arg] : llvm::zip(targetOp->getOperands(), entryBlock->getArguments())) {
    mapping.map(operand, arg);
  }

  // Clone only the target operation
  Operation *clonedTarget = funcBuilder.clone(*targetOp, mapping);

  // Create return operation
  SmallVector<Value> returnValues;
  for (Value result : clonedTarget->getResults()) {
    returnValues.push_back(result);
  }
  funcBuilder.create<func::ReturnOp>(targetOp->getLoc(), returnValues);

  return isolatedFunc;
}

struct TTNNIsolateOpsPass
    : public ::mlir::tt::ttnn::impl::TTNNIsolateOpsBase<TTNNIsolateOpsPass> {
  using ::mlir::tt::ttnn::impl::TTNNIsolateOpsBase<TTNNIsolateOpsPass>::TTNNIsolateOpsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    SmallVector<func::FuncOp> funcsToErase;

    // Walk through all nested modules to find functions
    module.walk([&](func::FuncOp funcOp) {
      // Skip already isolated functions or trace functions
      if (funcOp.isPrivate() || funcOp->hasAttr("ttnn.trace")) {
        return;
      }

      // Get the parent module where we should insert isolated functions
      auto parentModule = funcOp->getParentOfType<ModuleOp>();
      if (!parentModule) {
        return;
      }

      // Find all target operations in this function
      SmallVector<Operation *> targetOpsFound;
      funcOp.walk([&](Operation *op) {
        if (isTargetOp(op, this->filterOps)) {
          targetOpsFound.push_back(op);
        }
      });

      // Skip functions with no target ops
      if (targetOpsFound.empty()) {
        return;
      }

      // Create isolated function for each target op
      int index = 0;
      for (Operation *targetOp : targetOpsFound) {
        // Create base name from op type (remove "ttnn." prefix if present)
        std::string baseName = targetOp->getName().getStringRef().str();
        size_t dotPos = baseName.find('.');
        if (dotPos != std::string::npos) {
          baseName = baseName.substr(dotPos + 1);
        }

        // Create isolated function in the same parent module
        builder.setInsertionPointToEnd(parentModule.getBody());
        createIsolatedFunction(targetOp, baseName, index++, builder);
      }

      // Mark original function for removal if preserveOriginal is false
      if (!this->preserveOriginal) {
        funcsToErase.push_back(funcOp);
      }
    });

    // Erase original functions after walking (to avoid iterator invalidation)
    for (auto funcOp : funcsToErase) {
      if (auto *parentOp = funcOp->getParentOp(); isa<ModuleOp>(parentOp)) {
        SymbolTable symbolTable(cast<ModuleOp>(parentOp));
        symbolTable.erase(funcOp);
      } else {
        funcOp.erase();
      }
    }
  }
};

} // namespace
