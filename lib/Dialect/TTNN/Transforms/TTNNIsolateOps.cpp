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
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Utils.h"

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
  // Skip device operands - they will be created with get_device inside
  SmallVector<Type> inputTypes;
  SmallVector<Value> nonDeviceOperands;
  SmallVector<bool> isDeviceOperand;

  for (Value operand : targetOp->getOperands()) {
    Type opType = operand.getType();
    if (isa<DeviceType>(opType)) {
      // Mark this position as device operand (will be replaced with get_device)
      isDeviceOperand.push_back(true);
    } else {
      inputTypes.push_back(opType);
      nonDeviceOperands.push_back(operand);
      isDeviceOperand.push_back(false);
    }
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

  // Mark function as isolated
  isolatedFunc->setAttr("isolated", builder.getUnitAttr());

  // Create function body
  Block *entryBlock = isolatedFunc.addEntryBlock();
  OpBuilder funcBuilder = OpBuilder::atBlockBegin(entryBlock);

  // Check if we need to create a device op (only if target op has device operands)
  bool hasDeviceOperand = llvm::any_of(isDeviceOperand, [](bool b) { return b; });
  Value deviceValue;
  if (hasDeviceOperand) {
    // Create get_device op for device operands using the utility function
    // We need an IRRewriter for the utility, so create one from the OpBuilder
    IRRewriter rewriter(funcBuilder);
    GetDeviceOp deviceOp = utils::getOrInsertDevice(rewriter, entryBlock);
    deviceValue = deviceOp.getResult();
  }

  // Map target op operands to function arguments or device value
  IRMapping mapping;
  size_t argIdx = 0;
  for (size_t i = 0; i < targetOp->getNumOperands(); ++i) {
    if (isDeviceOperand[i]) {
      // Map device operand to the get_device result
      mapping.map(targetOp->getOperand(i), deviceValue);
    } else {
      // Map non-device operand to function argument
      mapping.map(targetOp->getOperand(i), entryBlock->getArgument(argIdx++));
    }
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

    // Global counter map for each operation type to ensure unique names
    llvm::StringMap<int> opTypeCounters;

    // Walk through all nested modules to find functions
    module.walk([&](func::FuncOp funcOp) {
      // Skip already isolated functions or trace functions
      if (ttmlir::utils::isIsolatedFunc(funcOp)) {
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

      // Skip functions with no target ops - don't mark for erasure
      if (targetOpsFound.empty()) {
        return;
      }

      // Mark original function for removal if preserveOriginal is false
      // Only mark functions that actually have operations being isolated
      if (!this->preserveOriginal) {
        funcsToErase.push_back(funcOp);
      }

      // Create isolated function for each target op
      for (Operation *targetOp : targetOpsFound) {
        // Create base name from op type (remove "ttnn." prefix if present)
        std::string baseName = targetOp->getName().getStringRef().str();
        size_t dotPos = baseName.find('.');
        if (dotPos != std::string::npos) {
          baseName = baseName.substr(dotPos + 1);
        }

        // Get and increment counter for this op type
        int index = opTypeCounters[baseName]++;

        // Create isolated function in the same parent module
        builder.setInsertionPointToEnd(parentModule.getBody());
        createIsolatedFunction(targetOp, baseName, index, builder);
      }
    });

    if (this->preserveOriginal) {
      return;
    }

    // Erase original functions after walking (to avoid iterator invalidation)
    for (auto funcOp : funcsToErase) {
      if (auto *parentOp = funcOp->getParentOp(); isa<ModuleOp>(parentOp)) {
        SymbolTable symbolTable(cast<ModuleOp>(parentOp));
        symbolTable.erase(funcOp);
      } else {
        funcOp.erase();
      }
    }

    // Clean up unreferenced functions (e.g., const_eval functions)
    // This removes functions that are no longer called after removing originals
    module.walk([&](ModuleOp moduleOp) {
      SymbolTable symbolTable(moduleOp);
      SmallVector<func::FuncOp> deadFuncs;

      // Collect potentially dead functions
      for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
        // Skip isolated functions we just created
        if (ttmlir::utils::isIsolatedFunc(funcOp)) {
          continue;
        }

        // Check if this function is referenced by any other function
        auto uses = symbolTable.getSymbolUses(funcOp, moduleOp);
        if (!uses || uses->empty()) {
          deadFuncs.push_back(funcOp);
        }
      }

      // Remove dead functions
      for (auto funcOp : deadFuncs) {
        symbolTable.erase(funcOp);
      }
    });
  }
};

} // namespace
