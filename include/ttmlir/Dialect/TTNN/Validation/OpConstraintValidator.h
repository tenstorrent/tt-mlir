// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATOR_H
#define TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATOR_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace mlir::tt::ttnn {

// Context-agnostic utility library for systematic validation of TTNN operation
// configurations.
class OpConstraintValidator {
public:
  // Result of a single constraint validation test.
  struct ValidationResult {
    // Whether validation succeeded.
    bool success;

    // Index in reference configs vector (valid only if success).
    size_t configIndex;

    // What the backend actually returned.
    TTNNLayoutAttr actualOutputLayout;

    // Valid only if success=false.
    std::string errorMessage;

    ValidationResult() = default;
    ValidationResult(bool success, size_t configIndex,
                     TTNNLayoutAttr actualOutputLayout,
                     std::string errorMessage)
        : success(success), configIndex(configIndex),
          actualOutputLayout(actualOutputLayout),
          errorMessage(std::move(errorMessage)) {}
  };

  // Options for controlling validation behavior.
  struct ValidationOptions {
    // If true, calls report_fatal_error for non-OpModel ops.
    bool fatalErrorOnUnsupportedOp;

    // If true, compare provided output layout with backend output layout.
    bool compareOutputLayout;

    ValidationOptions() = default;
    ValidationOptions(bool fatalOnUnsupported, bool compareOutput = true)
        : fatalErrorOnUnsupportedOp(fatalOnUnsupported),
          compareOutputLayout(compareOutput) {}
  };

  // Constructor to create validator with specific options.
  explicit OpConstraintValidator(const ValidationOptions &options);

  // Simple validation with all layouts (PostOptimizer use case).
  // op: Operation to validate.
  // inputLayouts: Input tensor layouts for the operation.
  // config: Operation configuration to test.
  // Returns: Validation result with success/failure and details.
  ValidationResult
  validateOperation(Operation *op,
                    const std::vector<TTNNLayoutAttr> &inputLayouts,
                    const OpConfig &config);

  // Test multiple attributes with all layouts.
  // op: Operation to validate.
  // inputLayouts: All input tensor layouts for the operation.
  // opSpecificAttrs: List of op-specific attributes to test.
  // referenceConfigs: Reference configurations to search for matches. If null,
  // only
  //                   validation is performed without matching to reference.
  // Returns: Vector of validation results, one per op-specific attribute
  // tested.
  std::vector<ValidationResult> validateWithMultipleAttributes(
      Operation *op, const std::vector<TTNNLayoutAttr> &inputLayouts,
      const std::vector<OpConfig::OpSpecificAttrs> &opSpecificAttrs,
      const std::vector<OpConfig> *referenceConfigs = nullptr);

private:
  // Validation configuration.
  ValidationOptions options_;

  // Core constraint validation using OpModel interface.
  // op: The operation to validate.
  // inputLayouts: All input layouts for the operation.
  // config: Configuration for the operation.
  // Returns: Expected output layout if valid, error otherwise.
  llvm::Expected<TTNNLayoutAttr>
  validateConstraints(Operation *op,
                      const std::vector<TTNNLayoutAttr> &inputLayouts,
                      const OpConfig &config);
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATOR_H
