// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_FUSIONVALIDATOR_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_FUSIONVALIDATOR_H

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
// This file defines a validation wrapper for fusion patterns that:
// 1. Creates a fused op in an isolated module
// 2. Applies workaround and validation/fallback passes
// 3. Decides whether fusion should proceed wether 2. succeeded or not.
//
//===----------------------------------------------------------------------===//

/// Configuration for fusion validation.
struct FusionValidationConfig {
  /// Whether to apply decomposition workarounds.
  bool applyDecompositionWorkarounds = true;

  /// Whether to apply layout workarounds.
  bool applyLayoutWorkarounds = true;

  /// Maximum number of fallback configurations to try (0 = unlimited).
  uint32_t maxFallbackAttempts = 10000;
};

/// Result of fusion validation.
struct FusionValidationResult {
  enum Status {
    Success,           // Fused op validated successfully
    WorkaroundFailed,  // Workaround passes failed
    ValidationFailed,  // Op validation/fallback passes failed
    PreconditionFailed // Missing required context (e.g. system_desc)
  };

  Status status;
  std::string errorMessage;

  bool isSuccess() const { return status == Success; }

  static FusionValidationResult success() { return {Success, ""}; }

  static FusionValidationResult failure(Status status, const std::string &msg) {
    return {status, msg};
  }
};

/// Validates a single fused op by creating it in an isolated module,
/// applying workaround passes, and checking op model constraints.
class FusionValidator {
public:
  FusionValidator(MLIRContext *context,
                  const FusionValidationConfig &config = {})
      : context(context), config(config) {}

  /// Validate a fusion by creating the fused op in an isolated module.
  /// @param srcOp The source operation to get parent module context from.
  /// @param loc Location for the fused op.
  /// @param resultTypes Result types of the fused op.
  /// @param args Arguments forwarded to the fused op constructor.
  template <typename FusedOpType, typename... Args>
  FusionValidationResult validateFusion(Operation *srcOp, Location loc,
                                        llvm::ArrayRef<Type> resultTypes,
                                        Args &&...args);

private:
  /// Create a validation function in the module containing the fused op.
  /// Handles block argument substitution and result pinning.
  template <typename FusedOpType, typename... Args>
  void createValidationFunc(ModuleOp module, Location loc,
                            llvm::ArrayRef<Type> resultTypes, Args &&...args);

  /// Run workaround, validation, and fallback passes on the module.
  /// Returns a typed result distinguishing workaround vs validation failure.
  FusionValidationResult runValidationPipeline(ModuleOp module);

  MLIRContext *context;
  FusionValidationConfig config;
};

// Template implementations
template <typename FusedOpType, typename... Args>
void FusionValidator::createValidationFunc(ModuleOp module, Location loc,
                                           llvm::ArrayRef<Type> resultTypes,
                                           Args &&...args) {
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(module.getBody());

  // Create an empty function with a return terminator.
  auto funcType = builder.getFunctionType({}, {});
  auto func = mlir::func::FuncOp::create(builder, module->getLoc(),
                                         "validation_func", funcType);
  func.addEntryBlock();
  auto *block = &func.getBody().front();
  builder.setInsertionPointToEnd(block);
  mlir::func::ReturnOp::create(builder, func->getLoc());

  // Capture Value args and create corresponding block arguments.
  builder.setInsertionPointToStart(block);

  llvm::SmallVector<Value> capturedValues;
  (
      [&] {
        if constexpr (!std::is_same_v<std::decay_t<Args>, std::nullptr_t> &&
                      std::is_convertible_v<std::decay_t<Args>, Value>) {
          capturedValues.push_back(args);
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
  auto sub = [&](auto &&a) -> decltype(auto) {
    if constexpr (!std::is_same_v<std::decay_t<decltype(a)>, std::nullptr_t> &&
                  std::is_convertible_v<std::decay_t<decltype(a)>, Value>) {
      return subs.lookup(Value(a));
    } else {
      return std::forward<decltype(a)>(a);
    }
  };

  // Create the fused op.
  auto op = FusedOpType::create(builder, loc, resultTypes, sub(args)...);

  // Pin results: update return and function type so passes don't DCE the op.
  auto returnOp = cast<mlir::func::ReturnOp>(block->getTerminator());
  llvm::SmallVector<Value> opResults(op->getResults());
  OpBuilder retBuilder(returnOp);
  mlir::func::ReturnOp::create(retBuilder, returnOp.getLoc(), opResults);
  returnOp.erase();

  llvm::SmallVector<Type> outTypes;
  for (auto v : opResults) {
    outTypes.push_back(v.getType());
  }
  func.setFunctionType(builder.getFunctionType(inputTypes, outTypes));
}

template <typename FusedOpType, typename... Args>
FusionValidationResult
FusionValidator::validateFusion(Operation *srcOp, Location loc,
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
    return FusionValidationResult::failure(
        FusionValidationResult::PreconditionFailed,
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

  // Create validation function containing the fused op.
  createValidationFunc<FusedOpType>(module, loc, resultTypes,
                                    std::forward<Args>(args)...);

  // Run workaround and validation passes.
  auto result = runValidationPipeline(module);
  module->erase();
  return result;
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_FUSIONVALIDATOR_H
