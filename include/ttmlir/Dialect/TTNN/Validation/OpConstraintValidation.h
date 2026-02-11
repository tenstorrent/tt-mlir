// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATION_H
#define TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATION_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::ttnn {

// Context-agnostic utility functions for systematic validation of TTNN
// operation configurations.
namespace op_constraint_validation {

enum class ValidationStatus {
  Success,
  NotImplemented,
  MetalBackendError,
  UnmatchedReferenceConfig,
  OutOfMemoryError
};

// Convert ValidationStatus to string for error messages
llvm::StringRef validationStatusToString(ValidationStatus status);

// Result of a single constraint validation test.
struct ValidationResult {
  ValidationStatus status = ValidationStatus::Success;

  // Index in reference configs vector.
  size_t configIndex = 0;

  // What the backend actually returned (only valid if status == Success).
  llvm::SmallVector<TTNNLayoutAttr> actualOutputLayouts;
  // For backward compatibility with existing code that expects a single output
  // layout.
  TTNNLayoutAttr firstActualOutputLayout;

  // L1 memory usage for the output tensor in bytes (only valid if status ==
  // Success). This is the per-core L1 footprint of the output tensor.
  uint64_t outputL1Usage = 0;

  // Error message if status != Success.
  std::string errorMessage;

  ValidationResult() = default;

  explicit ValidationResult(
      size_t configIndex, llvm::SmallVector<TTNNLayoutAttr> actualOutputLayouts,
      uint64_t outputL1Usage = 0)
      : configIndex(configIndex), actualOutputLayouts(actualOutputLayouts),
        firstActualOutputLayout(
            actualOutputLayouts.empty() ? nullptr : actualOutputLayouts[0]),
        outputL1Usage(outputL1Usage) {}

  static ValidationResult
  success(size_t configIndex,
          llvm::SmallVector<TTNNLayoutAttr> actualOutputLayouts,
          uint64_t outputL1Usage = 0) {
    return ValidationResult(configIndex, actualOutputLayouts, outputL1Usage);
  }

  static ValidationResult error(ValidationStatus status, std::string message) {
    ValidationResult result;
    result.status = status;
    result.errorMessage = std::move(message);
    return result;
  }

  static ValidationResult notImplemented(std::string message) {
    return error(ValidationStatus::NotImplemented, std::move(message));
  }

  static ValidationResult metalBackendError(std::string message) {
    return error(ValidationStatus::MetalBackendError, std::move(message));
  }

  static ValidationResult unmatchedReferenceConfig(std::string message) {
    return error(ValidationStatus::UnmatchedReferenceConfig,
                 std::move(message));
  }

  static ValidationResult outOfMemoryError(std::string message) {
    return error(ValidationStatus::OutOfMemoryError, std::move(message));
  }

  bool isSuccess() const { return status == ValidationStatus::Success; }
  bool isNotImplemented() const {
    return status == ValidationStatus::NotImplemented;
  }
  bool isError() const { return status != ValidationStatus::Success; }
};

// Simple validation with all layouts.
// op: Operation to validate.
// inputLayouts: Input tensor layouts for the operation.
// config: Operation configuration to test.
// additionalL1Usage: Additional L1 memory usage (in bytes) that must be
//                    accounted for during validation. This is used when
//                    validating ops that execute while other tensors occupy L1
//                    (e.g., during chain merging where Chain A's output stays
//                    in L1 while Chain B executes).
// Returns: ValidationResult where status indicates the outcome:
//   - status == Success: Operation validated successfully
//   - status == NotSupported: Operation not supported for validation (expected
//                             limitation)
//   - status == BackendError: Backend error occurred
//   - status == ValidationError: Validation failed (e.g., not enough L1)
// Callers should check result.status to handle each case appropriately.
// "NotSupported" is not an error - it indicates expected limitations.
ValidationResult validateOperation(Operation *op,
                                   llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                                   const OpConfig &config,
                                   uint64_t additionalL1Usage = 0);

// Test multiple attributes with all layouts.
// op: Operation to validate.
// inputLayouts: All input tensor layouts for the operation.
// opConfigs: List of op configs to test (they don't have to be full, i.e.
//            may contain only op-specific attrs).
// referenceConfigs: Reference configurations to search for matches. If empty,
//                   only validation is performed without matching to reference.
// Returns: Vector of ValidationResults, one per op config tested.
// Each ValidationResult can have different status - check individually.
std::vector<ValidationResult>
validateWithMultipleAttributes(Operation *op,
                               llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                               llvm::ArrayRef<OpConfig> opConfigs,
                               llvm::ArrayRef<OpConfig> referenceConfigs);

} // namespace op_constraint_validation

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATION_H
