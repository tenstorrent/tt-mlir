// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_POSTOPTIMIZERVALIDATIONANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_POSTOPTIMIZERVALIDATIONANALYSIS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidator.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

#include <optional>
#include <vector>

namespace mlir::tt::ttnn {

// Input for post-optimizer validation analysis
struct PostOptimizerValidationAnalysisInput {
  llvm::DenseMap<Operation *, OpConfig> chosenOpConfigs;

  PostOptimizerValidationAnalysisInput() : chosenOpConfigs() {}

  PostOptimizerValidationAnalysisInput(
      const llvm::DenseMap<Operation *, OpConfig> &chosenOpConfigs)
      : chosenOpConfigs(chosenOpConfigs) {}

  PostOptimizerValidationAnalysisInput(
      llvm::DenseMap<Operation *, OpConfig> &&chosenOpConfigs)
      : chosenOpConfigs(std::move(chosenOpConfigs)) {}

  bool operator==(const PostOptimizerValidationAnalysisInput &rhs) const {
    return chosenOpConfigs == rhs.chosenOpConfigs;
  }

  bool operator!=(const PostOptimizerValidationAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

// Changes needed for a single input operand to make validation pass
struct InputOperandChange {
  size_t operandIndex;

  // Layout changes
  std::optional<Layout> targetLayout;
  std::optional<TensorMemoryLayout> targetMemoryLayout;
  std::optional<BufferType> targetBufferType;

  // Data type changes
  std::optional<ttcore::DataType> targetDataType;

  // What the original values were (for debugging/logging)
  TTNNLayoutAttr originalLayout;

  InputOperandChange(size_t operandIndex, TTNNLayoutAttr originalLayout)
      : operandIndex(operandIndex), originalLayout(originalLayout) {}

  // Check if any changes are needed
  bool hasChanges() const {
    return targetLayout.has_value() || targetMemoryLayout.has_value() ||
           targetBufferType.has_value() || targetDataType.has_value();
  }
};

// Validation result for a single operation
struct OperationValidationResult {
  // If original config failed but we found a working fallback
  bool fixedWithFallback = false;

  // Input operand changes needed to make this operation work
  std::vector<InputOperandChange> inputOperandChanges;

  // Updated operation config if needed (contains the working output layout)
  std::optional<OpConfig> updatedOpConfig;

  OperationValidationResult() = default;

  // Helper methods to access layout information
  TTNNLayoutAttr getUpdatedOutputLayout() const {
    return updatedOpConfig ? updatedOpConfig->outputLayout : TTNNLayoutAttr{};
  }
  
  bool hasOutputLayoutChange() const {
    return updatedOpConfig.has_value();
  }
};

// Result of post-optimizer validation analysis
struct PostOptimizerValidationAnalysisResult {
  // Per-operation validation results
  llvm::DenseMap<Operation *, OperationValidationResult> operationResults;

  // Statistics
  size_t totalOperationsChecked = 0;
  size_t operationsValid = 0;
  size_t operationsFixed = 0;
  size_t operationsFailed = 0;

  // Operations that couldn't be fixed (for warnings)
  std::vector<Operation *> failedOperations;

  PostOptimizerValidationAnalysisResult() = default;

  // Helper method to get updated configuration for an operation
  OpConfig getUpdatedOpConfig(Operation *op, const PostOptimizerValidationAnalysisInput &input) const {
    auto it = operationResults.find(op);
    if (it != operationResults.end() && it->second.updatedOpConfig) {
      return *it->second.updatedOpConfig;
    }
    // Return original config from input
    return input.chosenOpConfigs.at(op);
  }

  // Check if an operation has an updated configuration
  bool hasUpdatedOpConfig(Operation *op) const {
    auto it = operationResults.find(op);
    return it != operationResults.end() && it->second.updatedOpConfig.has_value();
  }
};

// Post-optimizer validation analysis that verifies operation configurations
// will work at runtime and detects needed input operand changes automatically
class PostOptimizerValidationAnalysis
    : public TTNNAnalysis<PostOptimizerValidationAnalysisInput,
                          PostOptimizerValidationAnalysisResult> {

private:
  void analysisImplementation() override;
  bool applyOverrides() override;

  // Helper method to create fallback layouts directly
  std::vector<TTNNLayoutAttr>
  createFallbackTransforms(TTNNLayoutAttr originalLayout,
                           llvm::ArrayRef<int64_t> tensorShape);

  // Test provided fallback layouts for an operation
  // validator: Validator instance to use for testing
  // op: Operation to validate
  // originalConfig: Original configuration that failed
  // fallbackLayouts: List of layouts to test directly
  // Returns: Vector of validation results with early exit on first success
  std::vector<OpConstraintValidator::ValidationResult> testFallbackLayouts(
      OpConstraintValidator &validator, Operation *op,
      const OpConfig &originalConfig,
      const std::vector<TTNNLayoutAttr> &fallbackLayouts);

  // Test a specific combination of fallback layouts for an operation
  // validator: Validator instance to use for testing
  // op: Operation to validate
  // originalConfig: Original configuration that failed
  // inputLayouts: Combination of layouts to test
  // Returns: Single validation result
  OpConstraintValidator::ValidationResult testFallbackCombination(
      OpConstraintValidator &validator, Operation *op,
      const OpConfig &originalConfig,
      const std::vector<TTNNLayoutAttr> &inputLayouts);

  // Record a successful fallback combination
  void recordSuccessfulCombination(
      const std::vector<TTNNLayoutAttr> &originalLayouts,
      const std::vector<TTNNLayoutAttr> &workingLayouts,
      const OpConstraintValidator::ValidationResult &result,
      const OpConfig &originalConfig,
      OperationValidationResult &opResult);

  // Process fallback configurations for a failed operation
  void processFallbackConfigurations(OpConstraintValidator &validator,
                                     Operation *operation,
                                     const std::vector<TTNNLayoutAttr> &originalInputLayouts,
                                     const OpConfig &config,
                                     OperationValidationResult &opResult);

public:
  PostOptimizerValidationAnalysis(Operation *op) : TTNNAnalysis(op) {}

  // Getter for the analysis input (needed for accessing original configs)
  const PostOptimizerValidationAnalysisInput &getInput() const {
    return analysisInput;
  }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_POSTOPTIMIZERVALIDATIONANALYSIS_H
