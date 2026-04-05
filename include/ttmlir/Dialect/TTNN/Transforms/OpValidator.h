// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_OPVALIDATOR_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_OPVALIDATOR_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
//
// This file defines an op validation wrapper that:
// 1. Creates an op in an isolated module
// 2. Applies workaround and validation/fallback passes
// 3. Decides whether the op is valid based on whether 2. succeeded or not.
//
//===----------------------------------------------------------------------===//

/// Configuration for op validation.
struct OpValidationConfig {
  /// Whether to apply decomposition workarounds.
  bool applyDecompositionWorkarounds = true;

  /// Whether to apply layout workarounds.
  bool applyLayoutWorkarounds = true;

  /// Maximum number of fallback configurations to try (0 = unlimited).
  uint32_t maxFallbackAttempts = 10000;
};

/// Result of op validation.
struct OpValidationResult {
  enum Status {
    Success,           // Op validated successfully
    WorkaroundFailed,  // Workaround passes failed
    ValidationFailed,  // Op validation/fallback passes failed
    PreconditionFailed // Missing required context (e.g. system_desc)
  };

  Status status;
  std::string errorMessage;

  bool isSuccess() const { return status == Success; }

  static OpValidationResult success() { return {Success, ""}; }

  static OpValidationResult failure(Status status, const std::string &msg) {
    return {status, msg};
  }
};

/// Validates a single op by creating it in an isolated module,
/// applying workaround passes, and checking op model constraints.
class IsolatedIRValidationWrapper {
public:
  IsolatedIRValidationWrapper(MLIRContext *context,
                              const OpValidationConfig &config = {})
      : context(context), config(config) {}

  /// Validate an op by creating it in an isolated module.
  /// @param srcOp The source operation to get parent module context from.
  /// @param loc Location for the op.
  /// @param resultTypes Result types of the op.
  /// @param args Arguments forwarded to the op constructor.
  template <typename OpType, typename... Args>
  OpValidationResult validateOp(Operation *srcOp, Location loc,
                                llvm::ArrayRef<Type> resultTypes,
                                Args &&...args);

private:
  /// Create a validation function in the module containing the op.
  /// Handles block argument substitution and result pinning.
  template <typename OpType, typename... Args>
  void createValidationFunc(ModuleOp module, Location loc,
                            llvm::ArrayRef<Type> resultTypes, Args &&...args);

  /// Run workaround, validation, and fallback passes on the module.
  /// Returns a typed result distinguishing workaround vs validation failure.
  OpValidationResult runValidationPipeline(ModuleOp module);

  MLIRContext *context;
  OpValidationConfig config;
};

// Template implementations
template <typename OpType, typename... Args>
void IsolatedIRValidationWrapper::createValidationFunc(
    ModuleOp module, Location loc, llvm::ArrayRef<Type> resultTypes,
    Args &&...args) {
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(module.getBody());

  // Create an empty function with a return terminator.
  auto funcType = builder.getFunctionType({}, {});
  auto func = builder.create<mlir::func::FuncOp>(module->getLoc(),
                                                 "validation_func", funcType);
  func.addEntryBlock();
  auto *block = &func.getBody().front();
  builder.setInsertionPointToEnd(block);
  builder.create<mlir::func::ReturnOp>(func->getLoc());

  // Capture Value args and create corresponding block arguments.
  builder.setInsertionPointToStart(block);

  llvm::SmallVector<Value> capturedValues;
  (
      [&] {
        if constexpr (!std::is_same_v<std::decay_t<Args>, std::nullptr_t> &&
                      std::is_convertible_v<std::decay_t<Args>, Value>) {
          // Skip empty Values (optional operands that are not present).
          if (Value(args)) {
            capturedValues.push_back(args);
          }
        }
      }(),
      ...);

  llvm::SmallVector<Type> inputTypes;
  llvm::DenseMap<Value, Value> subs;
  for (auto val : capturedValues) {
    subs[val] = block->addArgument(val.getType(), loc);
    inputTypes.push_back(val.getType());
  }

  // Substitution helper: replace original Values with block args.
  // Empty Values (optional operands) are passed through unchanged.
  auto sub = [&](auto &&a) -> decltype(auto) {
    if constexpr (!std::is_same_v<std::decay_t<decltype(a)>, std::nullptr_t> &&
                  std::is_convertible_v<std::decay_t<decltype(a)>, Value>) {
      if (!Value(a)) {
        return Value(a);
      }
      return subs.lookup(Value(a));
    } else {
      return std::forward<decltype(a)>(a);
    }
  };

  // Create the op.
  auto op = builder.create<OpType>(loc, resultTypes, sub(args)...);

  // Pin results: update return and function type so passes don't DCE the op.
  auto returnOp = cast<mlir::func::ReturnOp>(block->getTerminator());
  llvm::SmallVector<Value> opResults(op->getResults());
  OpBuilder retBuilder(returnOp);
  retBuilder.create<mlir::func::ReturnOp>(returnOp.getLoc(), opResults);
  returnOp.erase();

  llvm::SmallVector<Type> outTypes;
  for (auto v : opResults) {
    outTypes.push_back(v.getType());
  }
  func.setFunctionType(builder.getFunctionType(inputTypes, outTypes));
}

template <typename OpType, typename... Args>
OpValidationResult
IsolatedIRValidationWrapper::validateOp(Operation *srcOp, Location loc,
                                        llvm::ArrayRef<Type> resultTypes,
                                        Args &&...args) {
  // Find the parent module carrying system_desc.
  auto parentModule = srcOp->getParentOfType<ModuleOp>();
  ModuleOp moduleWithSystemDesc = parentModule;
  while (moduleWithSystemDesc &&
         !moduleWithSystemDesc->hasAttr("ttcore.system_desc")) {
    moduleWithSystemDesc = moduleWithSystemDesc->getParentOfType<ModuleOp>();
  }

  if (!moduleWithSystemDesc) {
    return OpValidationResult::failure(
        OpValidationResult::PreconditionFailed,
        "No parent module with ttcore.system_desc found");
  }

  // Create temporary validation module.
  auto module = ModuleOp::create(UnknownLoc::get(context));

  // Copy attributes from the module carrying system_desc.
  for (const auto &attr : moduleWithSystemDesc->getAttrs()) {
    if (attr.getName() != "sym_name" && attr.getName() != "sym_visibility") {
      module->setAttr(attr.getName(), attr.getValue());
    }
  }
  // Clone DeviceOp so lookupDevice works during validation.
  if (auto deviceOp = ttcore::lookupDeviceOp(moduleWithSystemDesc)) {
    OpBuilder deviceBuilder(context);
    deviceBuilder.setInsertionPointToEnd(module.getBody());
    deviceBuilder.clone(*deviceOp.getOperation());
  }

  // Create validation function containing the op.
  createValidationFunc<OpType>(module, loc, resultTypes,
                               std::forward<Args>(args)...);

  // Run workaround and validation passes.
  auto result = runValidationPipeline(module);
  module->erase();
  return result;
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_OPVALIDATOR_H
