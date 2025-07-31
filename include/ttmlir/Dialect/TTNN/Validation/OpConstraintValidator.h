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
// configurations with automatic fallback strategies.
class OpConstraintValidator {
public:
  // Result of a single constraint validation test
  struct ValidationResult {
    bool success;       // Whether validation succeeded
    size_t configIndex; // Index in reference configs vector (valid only if
                        // success)
    TTNNLayoutAttr actualOutputLayout; // What the backend actually returned
    std::string errorMessage;          // Valid only if success=false

    ValidationResult() = default;
    ValidationResult(bool success, size_t configIndex,
                     TTNNLayoutAttr actualOutputLayout,
                     std::string errorMessage)
        : success(success), configIndex(configIndex),
          actualOutputLayout(actualOutputLayout),
          errorMessage(std::move(errorMessage)) {}
  };

  // Options for controlling validation behavior
  struct ValidationOptions {
    // If true, calls report_fatal_error for non-OpModel ops
    bool fatalErrorOnUnsupportedOp = false;

    ValidationOptions() = default;
    ValidationOptions(bool fatalOnUnsupported)
        : fatalErrorOnUnsupportedOp(fatalOnUnsupported) {}
  };

  // Factory method to create validator with specific options
  // options: Validation behavior configuration
  // Returns: Configured validator instance
  static OpConstraintValidator create(const ValidationOptions &options);

  // Test a single operation configuration
  // op: Operation to validate
  // inputLayouts: Input tensor layouts for the operation
  // config: Operation configuration to test
  // Returns: Validation result with success/failure and details
  ValidationResult
  validateSingleConfig(Operation *op,
                       const std::vector<TTNNLayoutAttr> &inputLayouts,
                       const OpConfig &config);

  // Test multiple op-specific attributes against reference configurations
  // This is the core method used by ShardSolver for constraint checking
  // op: Operation to validate
  // inputLayout: Input tensor layout to test
  // opSpecificAttrs: List of op-specific attributes to test
  // referenceConfigs: Reference configurations to search for matches
  // Returns: Vector of validation results, one per op-specific attribute tested
  std::vector<ValidationResult> validateWithMultipleAttributes(
      Operation *op, const TTNNLayoutAttr &inputLayout,
      const std::vector<OpConfig::OpSpecificAttrs> &opSpecificAttrs,
      const std::vector<OpConfig> &referenceConfigs);

private:
  // Private constructor - use factory method create()
  explicit OpConstraintValidator(const ValidationOptions &options);

  // Validation configuration
  ValidationOptions options_;

  // Core constraint validation using OpModel interface
  // This is the same logic as ShardSolver's checkShardCompatible
  // producerOperand: Value representing the producer operand
  // producerLayout: Layout of the producer tensor
  // consumerOp: Consumer operation
  // consumerConfig: Configuration for the consumer operation
  // Returns: Expected TTNNLayoutAttr on success, Error on failure
  llvm::Expected<TTNNLayoutAttr>
  validateConstraints(Value producerOperand,
                      const TTNNLayoutAttr &producerLayout,
                      Operation *consumerOp, const OpConfig &consumerConfig);

  // Core constraint validation using OpModel interface with all input layouts
  // consumerOp: The operation to validate
  // inputLayouts: All input layouts for the operation
  // consumerConfig: Configuration for the consumer operation
  // Returns: Expected output layout if valid, error otherwise
  llvm::Expected<TTNNLayoutAttr>
  validateConstraintsWithAllLayouts(Operation *consumerOp,
                                    const std::vector<TTNNLayoutAttr> &inputLayouts,
                                    const OpConfig &consumerConfig);

  // Extract input operand for constraint checking
  // op: Operation to extract from
  // operandIndex: Index of operand to extract (default: 0)
  // Returns: Value representing the input operand
  static Value extractInputOperand(Operation *op, size_t operandIndex = 0);

};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATOR_H
