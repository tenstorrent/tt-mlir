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
      auto originalResult =
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
        processFallbackConfigurations(validator, operation, inputLayouts[0],
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
    TTNNLayoutAttr originalInputLayout, const OpConfig &config,
    OperationValidationResult &opResult) {

  // Create fallback transforms in priority order
  Value operand = operation->getOperand(0);
  auto operandType = mlir::cast<RankedTensorType>(operand.getType());
  auto tensorShape = operandType.getShape();

  auto transforms = createFallbackTransforms(originalInputLayout, tensorShape);
  auto fallbackResults = validator.testFallbackTransforms(
      operation, originalInputLayout, config, transforms);

  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "Testing {} fallback configurations for operation {} at {}, "
               "original output layout: {}",
               fallbackResults.size(), operation->getName(),
               operation->getLoc(), config.outputLayout);

  bool foundWorkingConfig = false;
  for (size_t i = 0; i < fallbackResults.size(); ++i) {
    const auto &result = fallbackResults[i];
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Fallback {} result: success={}", i, result.success);
    if (result.success) {
      // Found a working configuration - analyze what changed
      opResult.fixedWithFallback = true;
      foundWorkingConfig = true;

      // Use the transforms already created at the beginning of this function
      TTNNLayoutAttr transformedLayout = transforms[i](originalInputLayout);

      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Fallback {} worked for operation {} at {}. "
                   "Original layout: {}, Transformed layout: {}",
                   i, operation->getName(), operation->getLoc(),
                   originalInputLayout, transformedLayout);

      // Record the changes needed for the first input operand
      InputOperandChange change(0, originalInputLayout);

      // Detect layout changes (ROW_MAJOR vs TILE)
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Layout comparison: original.getLayout()={}, "
                   "transformed.getLayout()={}",
                   static_cast<int>(originalInputLayout.getLayout()),
                   static_cast<int>(transformedLayout.getLayout()));

      if (originalInputLayout.getLayout() != transformedLayout.getLayout()) {
        change.targetLayout = transformedLayout.getLayout();
        TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                     "Layout change detected: {} -> {}",
                     static_cast<int>(originalInputLayout.getLayout()),
                     static_cast<int>(transformedLayout.getLayout()));
      } else {
        TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                     "No layout change detected for fallback {}", i);
      }

      // Detect memory layout changes
      if (originalInputLayout.getMemLayout() !=
          transformedLayout.getMemLayout()) {
        change.targetMemoryLayout = transformedLayout.getMemLayout().getValue();
      }

      // Detect buffer type changes
      if (originalInputLayout.getBufferType() !=
          transformedLayout.getBufferType()) {
        change.targetBufferType = transformedLayout.getBufferType();
      }

      // Detect data type changes (for future extension)
      // Currently fallbacks don't change data types, but structure is ready

      if (change.hasChanges()) {
        opResult.inputOperandChanges.push_back(change);
      }

      // Capture the updated output layout if it's different from original
      if (result.actualOutputLayout != config.outputLayout) {
        TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                     "Fallback {} changed output layout from {} to {}", i,
                     config.outputLayout, result.actualOutputLayout);

        // Create updated OpConfig with the backend's actual output layout
        OpConfig updatedConfig = config;
        updatedConfig.outputLayout = result.actualOutputLayout;
        opResult.updatedOpConfig = updatedConfig;
      }

      analysisResult.operationsFixed++;
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Operation {} at {} can be fixed with input operand changes",
                   operation->getName(), operation->getLoc());
      break;
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

// Helper method to create fallback transform functions
std::vector<std::function<TTNNLayoutAttr(TTNNLayoutAttr)>>
PostOptimizerValidationAnalysis::createFallbackTransforms(
    TTNNLayoutAttr originalLayout, llvm::ArrayRef<int64_t> tensorShape) {

  return {// 1. Row-major layout fallback (skip identity since already tested by
          // validation)
          [tensorShape](TTNNLayoutAttr layout) {
            TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                         "Analysis fallback 0 (row-major): Input layout: {}",
                         layout);
            auto result = ttnn::utils::convertTTNNLayoutToRowMajor(
                layout.getContext(), layout, tensorShape);
            TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                         "Analysis fallback 0 result: {}, getLayout()={}",
                         result, static_cast<int>(result.getLayout()));
            return result;
          },

          // 2. Interleaved memory layout fallback
          [](TTNNLayoutAttr layout) {
            return layout.withMemoryLayout(TensorMemoryLayout::Interleaved);
          },

          // 3. L1 interleaved fallback
          [](TTNNLayoutAttr layout) {
            return layout.withBufferType(BufferType::L1)
                .withMemoryLayout(TensorMemoryLayout::Interleaved);
          },

          // 4. DRAM interleaved fallback
          [](TTNNLayoutAttr layout) {
            return layout.withBufferType(BufferType::DRAM)
                .withMemoryLayout(TensorMemoryLayout::Interleaved);
          }};
}

} // namespace mlir::tt::ttnn
