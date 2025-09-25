// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATION_H
#define TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATION_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace mlir::tt::ttnn {

// Context-agnostic utility functions for systematic validation of TTNN
// operation configurations.
namespace op_constraint_validation {
// Result of a single constraint validation test.
struct ValidationResult {
  // Index in reference configs vector.
  size_t configIndex;

  // What the backend actually returned.
  TTNNLayoutAttr actualOutputLayout;

  ValidationResult() = default;
  explicit ValidationResult(size_t configIndex,
                            TTNNLayoutAttr actualOutputLayout)
      : configIndex(configIndex), actualOutputLayout(actualOutputLayout) {}
};

// Simple validation with all layouts (PostOptimizer use case).
// op: Operation to validate.
// inputLayouts: Input tensor layouts for the operation.
// config: Operation configuration to test.
// Returns: ValidationResult if valid, error otherwise.
llvm::Expected<ValidationResult>
validateOperation(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                  const OpConfig &config, float tensorL1UsageCap);

// Test multiple attributes with all layouts.
// op: Operation to validate.
// inputLayouts: All input tensor layouts for the operation.
// opSpecificAttrs: List of op configs to test (they don't have to be full, i.e.
//                  may contain only op-specific attrs).
// referenceConfigs: Reference configurations to search for matches. If empty,
//                   only validation is performed without matching to reference.
// Returns: Vector of ValidationResults, one per op config tested.
llvm::Expected<std::vector<ValidationResult>> validateWithMultipleAttributes(
    Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
    llvm::ArrayRef<OpConfig> opConfigs,
    llvm::ArrayRef<OpConfig> referenceConfigs, float tensorL1UsageCap);

// Core constraint validation using OpModel interface.
// op: The operation to validate.
// inputLayouts: All input layouts for the operation.
// config: Configuration for the operation.
// Returns: Expected output layout if valid, error otherwise.
llvm::Expected<TTNNLayoutAttr>
validateConstraints(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                    const OpConfig &config, float tensorL1UsageCap);

} // namespace op_constraint_validation

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATION_H
