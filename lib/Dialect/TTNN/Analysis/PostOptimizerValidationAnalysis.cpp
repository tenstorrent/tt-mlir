// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/PostOptimizerValidationAnalysis.h"

#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidator.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include <cstdlib>

namespace mlir::tt::ttnn {

void PostOptimizerValidationAnalysis::analysisImplementation() {
  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "Running post-optimizer validation analysis");

  // Create validator configured for non-fatal errors (post-optimizer context)
  auto validator = OpConstraintValidator::create(
      OpConstraintValidator::ValidationOptions(false));

  analysisResult = PostOptimizerValidationAnalysisResult();

  op->walk([&](func::FuncOp func) {
    if (ttmlir::utils::isConstEvalFunc(func)) {
      return;
    }

    func.walk([&](Operation *operation) {
      // Skip operations without results or not in chosenOpConfigs
      if (operation->getNumResults() == 0 ||
          analysisInput.chosenOpConfigs.find(operation) ==
              analysisInput.chosenOpConfigs.end()) {
        return;
      }

      analysisResult.totalOperationsChecked++;
      const OpConfig &config = analysisInput.chosenOpConfigs.at(operation);

      // Extract input layouts from the operation
      std::vector<TTNNLayoutAttr> inputLayouts;
      for (Value operand : operation->getOperands()) {
        if (auto tensorType =
                mlir::dyn_cast<RankedTensorType>(operand.getType())) {
          if (auto layoutAttr = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
                  tensorType.getEncoding())) {
            inputLayouts.push_back(layoutAttr);
          }
        }
      }

      if (inputLayouts.empty()) {
        return; // Skip operations without TTNN layout inputs
      }

      // Test original configuration
      OpConstraintValidator::ValidationResult originalResult =
          validator.validateSingleConfig(operation, inputLayouts, config);

      OperationValidationResult opResult;

      if (originalResult.success) {
        analysisResult.operationsValid++;
        TTMLIR_TRACE(
            ttmlir::LogComponent::Optimizer,
            "Operation {} at {} passed validation with original config",
            operation->getName(), operation->getLoc());
      } else {
        // Original config failed, try fallback configurations
        processFallbackConfigurations(validator, operation, inputLayouts,
                                      config, opResult);
      }

      // Store the operation result
      analysisResult.operationResults[operation] = opResult;
    });
  });

  // Log validation summary
  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "Post-optimizer validation complete: {} operations checked, "
               "{} valid, {} fixed, {} failed",
               analysisResult.totalOperationsChecked,
               analysisResult.operationsValid, analysisResult.operationsFixed,
               analysisResult.operationsFailed);
}

bool PostOptimizerValidationAnalysis::applyOverrides() {
  // No overrides for this analysis
  return false;
}

void PostOptimizerValidationAnalysis::processFallbackConfigurations(
    OpConstraintValidator &validator, Operation *operation,
    const std::vector<TTNNLayoutAttr> &originalInputLayouts,
    const OpConfig &config, OperationValidationResult &opResult) {

  // Extract tensor shapes for all input operands
  std::vector<llvm::ArrayRef<int64_t>> tensorShapes;

  for (Value operand : operation->getOperands()) {
    if (auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType())) {
      if (auto layoutAttr = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
              tensorType.getEncoding())) {
        tensorShapes.push_back(tensorType.getShape());
      }
    }
  }

  if (originalInputLayouts.empty()) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "No TTNN input layouts found for operation {} at {}",
                 operation->getName(), operation->getLoc());
    return;
  }

  // Create fallback combinations for each operand
  std::vector<std::vector<TTNNLayoutAttr>> operandFallbacks;
  for (size_t i = 0; i < originalInputLayouts.size(); ++i) {
    auto fallbacks =
        createFallbackTransforms(originalInputLayouts[i], tensorShapes[i]);
    // Insert original layout at the beginning
    fallbacks.insert(fallbacks.begin(), originalInputLayouts[i]);
    operandFallbacks.push_back(fallbacks);
  }

  TTMLIR_DEBUG(
      ttmlir::LogComponent::Optimizer,
      "Testing fallback combinations for operation {} at {} with {} operands",
      operation->getName(), operation->getLoc(), originalInputLayouts.size());

  // Try all combinations using recursive approach
  std::function<bool(size_t, std::vector<TTNNLayoutAttr> &, bool)>
      testCombinations;
  testCombinations = [&](size_t operandIdx,
                         std::vector<TTNNLayoutAttr> &currentLayouts,
                         bool hasTransformed) -> bool {
    if (operandIdx == operandFallbacks.size()) {
      // Base case: test this combination if it has at least one transformed
      // layout
      if (!hasTransformed) {
        return false;
      }

      auto result =
          testFallbackCombination(validator, operation, config, currentLayouts);
      if (result.success) {
        // Record the successful combination
        recordSuccessfulCombination(originalInputLayouts, currentLayouts,
                                    result, config, opResult);
        return true;
      }
      return false;
    }

    // Try each fallback for current operand
    for (size_t i = 0; i < operandFallbacks[operandIdx].size(); ++i) {
      currentLayouts[operandIdx] = operandFallbacks[operandIdx][i];
      bool isTransformed = (i > 0); // Index 0 is original layout

      if (testCombinations(operandIdx + 1, currentLayouts,
                           hasTransformed || isTransformed)) {
        return true; // Found working combination
      }
    }
    return false;
  };

  std::vector<TTNNLayoutAttr> testLayouts(originalInputLayouts.size());
  bool foundWorkingConfig = testCombinations(0, testLayouts, false);

  if (!foundWorkingConfig) {
    analysisResult.operationsFailed++;
    analysisResult.failedOperations.push_back(operation);

    TTMLIR_DEBUG(
        ttmlir::LogComponent::Optimizer,
        "Operation {} at {} FAILED validation - no working configuration found",
        operation->getName(), operation->getLoc());
  }
}

// Helper method to create fallback layouts directly
std::vector<TTNNLayoutAttr>
PostOptimizerValidationAnalysis::createFallbackTransforms(
    TTNNLayoutAttr originalLayout, llvm::ArrayRef<int64_t> tensorShape) {

  // Create systematic 2×4 combinations: 2 layouts × 4 data types = 8
  // combinations Layouts: RowMajor, Tile DataTypes: BFloat16, Float32, UInt32,
  // Int32
  std::vector<TTNNLayoutAttr> fallbackLayouts;

  // Define the 4 target data types for fallbacks
  std::vector<ttcore::DataType> targetDataTypes = {
      ttcore::DataType::BFloat16, ttcore::DataType::Float32,
      ttcore::DataType::UInt32, ttcore::DataType::Int32};

  // Define the 2 target layouts for fallbacks
  std::vector<Layout> targetLayouts = {Layout::RowMajor, Layout::Tile};

  int transformIndex = 0;
  for (Layout targetLayout : targetLayouts) {
    for (ttcore::DataType targetDataType : targetDataTypes) {
      // Skip if this is the same as original (already tested)
      if (targetLayout == originalLayout.getLayout() &&
          targetDataType == originalLayout.getDataType()) {
        continue;
      }

      TTMLIR_DEBUG(
          ttmlir::LogComponent::Optimizer,
          "Creating fallback {} (layout={}, dtype={}): Original layout: {}",
          transformIndex, static_cast<int>(targetLayout),
          static_cast<int>(targetDataType), originalLayout);

      // Start with original layout
      TTNNLayoutAttr result = originalLayout;

      // Apply layout transformation if needed
      if (targetLayout != originalLayout.getLayout()) {
        result = result.withLayout(targetLayout, tensorShape);
      }

      // Apply data type transformation if needed
      if (targetDataType != result.getDataType()) {
        auto targetElementType = ttnn::utils::getElementType(
            result.getContext(), result.getLayout(), targetDataType);
        result = result.withElementType(targetElementType, tensorShape);
      }

      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Created fallback {} result: {}, layout={}, dtype={}",
                   transformIndex, result, static_cast<int>(result.getLayout()),
                   static_cast<int>(result.getDataType()));

      fallbackLayouts.push_back(result);
      transformIndex++;
    }
  }

  return fallbackLayouts;
}

std::vector<OpConstraintValidator::ValidationResult>
PostOptimizerValidationAnalysis::testFallbackLayouts(
    OpConstraintValidator &validator, Operation *op,
    const OpConfig &originalConfig,
    const std::vector<TTNNLayoutAttr> &fallbackLayouts) {

  std::vector<OpConstraintValidator::ValidationResult> results;

  for (size_t i = 0; i < fallbackLayouts.size(); ++i) {
    const TTNNLayoutAttr &testInputLayout = fallbackLayouts[i];

    // For all fallbacks, don't constrain output layout - let backend decide
    OpConfig testConfig = originalConfig;
    testConfig.outputLayout =
        TTNNLayoutAttr{}; // Let backend choose output layout
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Fallback {}: testing layout={}, dtype={}", i,
                 static_cast<int>(testInputLayout.getLayout()),
                 static_cast<int>(testInputLayout.getDataType()));

    auto result =
        validator.validateSingleConfig(op, {testInputLayout}, testConfig);
    results.push_back(result);

    if (result.success) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Fallback {} succeeded with output layout: {}", i,
                   result.actualOutputLayout);
      break; // Early exit on first working configuration
    }
  }

  return results;
}

void PostOptimizerValidationAnalysis::recordSuccessfulCombination(
    const std::vector<TTNNLayoutAttr> &originalLayouts,
    const std::vector<TTNNLayoutAttr> &workingLayouts,
    const OpConstraintValidator::ValidationResult &result,
    const OpConfig &originalConfig, OperationValidationResult &opResult) {

  opResult.fixedWithFallback = true;

  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "Found working fallback combination with {} operands",
               workingLayouts.size());

  // Record changes for each operand that was transformed
  for (size_t i = 0; i < workingLayouts.size(); ++i) {
    TTNNLayoutAttr originalLayout = originalLayouts[i];
    TTNNLayoutAttr transformedLayout = workingLayouts[i];

    // Skip if layout wasn't actually transformed
    if (originalLayout == transformedLayout) {
      continue;
    }

    InputOperandChange change(i, originalLayout);

    // Detect layout changes
    if (originalLayout.getLayout() != transformedLayout.getLayout()) {
      change.targetLayout = transformedLayout.getLayout();
    }

    // Detect memory layout changes
    if (originalLayout.getMemLayout() != transformedLayout.getMemLayout()) {
      change.targetMemoryLayout = transformedLayout.getMemLayout().getValue();
    }

    // Detect buffer type changes
    if (originalLayout.getBufferType() != transformedLayout.getBufferType()) {
      change.targetBufferType = transformedLayout.getBufferType();
    }

    // Detect data type changes
    if (originalLayout.getDataType() != transformedLayout.getDataType()) {
      change.targetDataType = transformedLayout.getDataType();
    }

    if (change.hasChanges()) {
      opResult.inputOperandChanges.push_back(change);
    }
  }

  // Capture output layout changes
  if (result.actualOutputLayout != originalConfig.outputLayout) {
    OpConfig updatedConfig = originalConfig;
    updatedConfig.outputLayout = result.actualOutputLayout;
    opResult.updatedOpConfig = updatedConfig;
  }

  analysisResult.operationsFixed++;
}

OpConstraintValidator::ValidationResult
PostOptimizerValidationAnalysis::testFallbackCombination(
    OpConstraintValidator &validator, Operation *op,
    const OpConfig &originalConfig,
    const std::vector<TTNNLayoutAttr> &inputLayouts) {

  // For all fallbacks, don't constrain output layout - let backend decide
  OpConfig testConfig = originalConfig;
  testConfig.outputLayout =
      TTNNLayoutAttr{}; // Let backend choose output layout

  for (size_t i = 0; i < inputLayouts.size(); ++i) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "\t\tInput layout {}: {} ", i, inputLayouts[i]);
  }

  auto result = validator.validateSingleConfig(op, inputLayouts, testConfig);

  if (result.success) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Combination succeeded with output layout: {}",
                 result.actualOutputLayout);
  } else {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Combination failed, error: {}", result.errorMessage);
  }

  return result;
}

} // namespace mlir::tt::ttnn
