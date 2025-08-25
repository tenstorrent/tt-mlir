// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/PostOptimizerValidationAnalysis.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidator.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include <cstdlib>

namespace mlir::tt::ttnn {

// Calculate distance between two data types (lower = more similar)
static double calculateDataTypeDistance(ttcore::DataType from, ttcore::DataType to) {
  if (from == to) return 0.0;

  // Define data type similarity groups
  // Group 1: Integer types (closest to each other)
  bool fromIsInt = (from == ttcore::DataType::Int32 || from == ttcore::DataType::UInt32);
  bool toIsInt = (to == ttcore::DataType::Int32 || to == ttcore::DataType::UInt32);

  // Group 2: Floating point types
  bool fromIsFloat = (from == ttcore::DataType::Float32 || from == ttcore::DataType::BFloat16);
  bool toIsFloat = (to == ttcore::DataType::Float32 || to == ttcore::DataType::BFloat16);

  if (fromIsInt && toIsInt) {
    // Both integers: Int32 <-> UInt32 is close
    return 1.0;
  }
  if (fromIsFloat && toIsFloat) {
    // Both floating point: Float32 <-> BFloat16 is close
    return 1.5;
  }
  // Cross-category changes are more expensive
  return 3.0;
}

// Calculate distance between two layouts
static double calculateLayoutDistance(Layout from, Layout to) {
  if (from == to) return 0.0;
  // RowMajor <-> Tile conversion
  return 2.0;
}

// Calculate total distance for a layout transformation
static double calculateLayoutTransformDistance(TTNNLayoutAttr from, TTNNLayoutAttr to) {
  double distance = 0.0;

  // Add data type distance
  distance += calculateDataTypeDistance(from.getDataType(), to.getDataType());

  // Add layout distance
  distance += calculateLayoutDistance(from.getLayout(), to.getLayout());

  // Add memory layout distance (if different)
  if (from.getMemLayout() != to.getMemLayout()) {
    distance += 1.0;
  }

  // Add buffer type distance (if different)
  if (from.getBufferType() != to.getBufferType()) {
    distance += 4.0;
  }

  return distance;
}

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

      // Skip operations that are not OpModel castable (can't be validated)
      if (!mlir::dyn_cast<OpModel>(operation)) {
        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "Skipping operation {} at {} - not OpModel castable",
                     operation->getName(), operation->getLoc());
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
        processFallbackConfigurations(operation, inputLayouts, config,
                                      opResult);
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
    Operation *operation,
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

  // Don't compare output layout in this context - let backend decide. However,
  // we will send output config to the validator to ensure DRAM Interleaved
  // memory layout.
  OpConstraintValidator validator =
      OpConstraintValidator::create(OpConstraintValidator::ValidationOptions(
          /*fatalOnUnsupported=*/false, /*compareOutput=*/false));

  // Distance-based BFS: test combinations by total distance, starting from minimal
  struct CombinationCandidate {
    std::vector<TTNNLayoutAttr> layouts;
    double totalDistance;
  };

  // Generate all possible combinations and calculate their distances
  std::vector<CombinationCandidate> allCombinations;

  std::function<void(size_t, std::vector<TTNNLayoutAttr>&, double)> generateAllCombinations;
  generateAllCombinations = [&](size_t operandIdx, std::vector<TTNNLayoutAttr>& current, double currentDistance) {
    if (operandIdx == operandFallbacks.size()) {
      // Only include combinations that have at least one change
      if (currentDistance > 0.0) {
        allCombinations.push_back({current, currentDistance});
      }
      return;
    }

    // Try each fallback for current operand
    for (size_t i = 0; i < operandFallbacks[operandIdx].size(); ++i) {
      current[operandIdx] = operandFallbacks[operandIdx][i];

      double addedDistance = 0.0;
      if (i > 0) { // Not original layout
        addedDistance = calculateLayoutTransformDistance(
            originalInputLayouts[operandIdx], operandFallbacks[operandIdx][i]);
      }

      generateAllCombinations(operandIdx + 1, current, currentDistance + addedDistance);
    }
  };

  std::vector<TTNNLayoutAttr> current(originalInputLayouts.size());
  generateAllCombinations(0, current, 0.0);

  // Sort combinations by total distance (ascending)
  std::sort(allCombinations.begin(), allCombinations.end(),
            [](const CombinationCandidate& a, const CombinationCandidate& b) {
              return a.totalDistance < b.totalDistance;
            });

  bool foundWorkingConfig = false;
  double maxDistanceToTest = 0.0;

  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "Generated {} combinations, testing by distance", allCombinations.size());

  // Test combinations in order of increasing distance
  for (const auto& candidate : allCombinations) {
    // Early termination: if we found a solution and current distance is higher, stop
    if (foundWorkingConfig && candidate.totalDistance > maxDistanceToTest) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Stopping search - found solution at distance {}, current distance {}",
                   maxDistanceToTest, candidate.totalDistance);
      break;
    }

    auto result = testFallbackCombination(validator, operation, config, candidate.layouts);
    if (result.success) {
      // Found optimal solution with minimal distance
      recordSuccessfulCombination(originalInputLayouts, candidate.layouts, result, config, opResult);
      foundWorkingConfig = true;
      maxDistanceToTest = candidate.totalDistance;

      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Found optimal solution with distance {}", candidate.totalDistance);
      // Continue testing combinations with the same distance to see if there are multiple solutions
      // but stop when distance increases
    }
  }

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

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Recording operand {} change: {} -> {}", i, originalLayout,
                 transformedLayout);

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

    // Detect specific output property changes (what backend actually changed)
    TTNNLayoutAttr originalOutputLayout = originalConfig.outputLayout;
    TTNNLayoutAttr actualOutputLayout = result.actualOutputLayout;

    // Check for layout changes (RowMajor vs Tile)
    if (originalOutputLayout.getLayout() != actualOutputLayout.getLayout()) {
      opResult.outputLayoutChange = actualOutputLayout.getLayout();
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Backend changed output layout: {} -> {}",
                   static_cast<int>(originalOutputLayout.getLayout()),
                   static_cast<int>(actualOutputLayout.getLayout()));
    }

    // Check for memory layout changes
    if (originalOutputLayout.getMemLayout() !=
        actualOutputLayout.getMemLayout()) {
      opResult.outputMemoryLayoutChange =
          actualOutputLayout.getMemLayout().getValue();
      TTMLIR_DEBUG(
          ttmlir::LogComponent::Optimizer,
          "Backend changed output memory layout: {} -> {}",
          static_cast<int>(originalOutputLayout.getMemLayout().getValue()),
          static_cast<int>(actualOutputLayout.getMemLayout().getValue()));
    }

    // Check for buffer type changes
    if (originalOutputLayout.getBufferType() !=
        actualOutputLayout.getBufferType()) {
      opResult.outputBufferTypeChange = actualOutputLayout.getBufferType();
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Backend changed output buffer type: {} -> {}",
                   static_cast<int>(originalOutputLayout.getBufferType()),
                   static_cast<int>(actualOutputLayout.getBufferType()));
    }

    // Check for data type changes
    if (originalOutputLayout.getDataType() !=
        actualOutputLayout.getDataType()) {
      opResult.outputDataTypeChange = actualOutputLayout.getDataType();
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Backend changed output data type: {} -> {}",
                   static_cast<int>(originalOutputLayout.getDataType()),
                   static_cast<int>(actualOutputLayout.getDataType()));
    }
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

  // We want to force fallback to spill output tensor to DRAM Interleaved
  // anyway.
  testConfig.outputLayout = originalConfig.outputLayout;
  testConfig.outputLayout =
      testConfig.outputLayout.withBufferType(BufferType::DRAM);
  testConfig.outputLayout =
      testConfig.outputLayout.withMemoryLayout(TensorMemoryLayout::Interleaved);

  for (size_t i = 0; i < inputLayouts.size(); ++i) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "\t\tInput layout {}: {} ", i,
                 inputLayouts[i]);
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
