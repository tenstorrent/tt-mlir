// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstddef>
#include <vector>

using namespace mlir;
using namespace mlir::tt;
using namespace mlir::tt::ttnn;

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNOPERATIONVALIDATIONANDFALLBACK
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace fallbacks {
// Cost constants for layout transformation distances
static constexpr double NO_COST = 0.0;
static constexpr double LOW_COST = 1.0;
static constexpr double MID_COST = 2.0;
static constexpr double HIGH_COST = 3.0;

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

// Represents a combination candidate with layouts and total distance
struct CombinationCandidate {
  std::vector<TTNNLayoutAttr> layouts;
  double totalDistance;
};

// Forward declarations for helper functions
std::vector<TTNNLayoutAttr>
createFallbackTransforms(TTNNLayoutAttr originalLayout,
                         llvm::ArrayRef<int64_t> tensorShape);

double calculateDataTypeDistance(ttcore::DataType from, ttcore::DataType to);
double calculateLayoutDistance(Layout from, Layout to);
double calculateLayoutTransformDistance(TTNNLayoutAttr from, TTNNLayoutAttr to);

void generateAllCombinations(
    const std::vector<std::vector<TTNNLayoutAttr>> &operandFallbacks,
    const std::vector<TTNNLayoutAttr> &originalInputLayouts, size_t operandIdx,
    std::vector<TTNNLayoutAttr> &currentCombination, double currentDistance,
    std::vector<CombinationCandidate> &allCombinations);

llvm::Expected<op_constraint_validation::ValidationResult>
testFallbackCombination(Operation *op, const OpConfig &originalConfig,
                        const std::vector<TTNNLayoutAttr> &inputLayouts);

void applyFallbackTransformations(
    Operation *operation, const std::vector<TTNNLayoutAttr> &originalLayouts,
    const std::vector<TTNNLayoutAttr> &workingLayouts,
    const op_constraint_validation::ValidationResult &result,
    const OpConfig &config);

void applyInputOperandChange(Operation *operation,
                             const InputOperandChange &change);

void applyOutputLayoutRevert(Operation *operation,
                             TTNNLayoutAttr actualOutputLayout,
                             TTNNLayoutAttr expectedOutputLayout);

// Try fallback configurations for a failed operation
bool tryFallbacks(Operation *operation,
                  const std::vector<TTNNLayoutAttr> &originalInputLayouts,
                  const OpConfig &config);
} // namespace fallbacks

class TTNNOperationValidationAndFallback
    : public impl::TTNNOperationValidationAndFallbackBase<
          TTNNOperationValidationAndFallback> {

public:
  using impl::TTNNOperationValidationAndFallbackBase<
      TTNNOperationValidationAndFallback>::
      TTNNOperationValidationAndFallbackBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    size_t totalOperationsChecked = 0;
    size_t operationsValid = 0;
    size_t operationsFixed = 0;
    size_t operationsFailed = 0;

    moduleOp->walk([&](func::FuncOp func) {
      func.walk([&](Operation *operation) -> WalkResult {
        // Skip operations without results
        if (operation->getNumResults() == 0) {
          return WalkResult::skip();
        }

        // Skip operations that are not OpModel castable (can't be validated)
        // TODO(rpavlovic): we should have all ops implement OpModel interface
        // eventually. Then this check becomes an assert.
        if (!mlir::dyn_cast<OpModel>(operation)) {
          return WalkResult::skip();
        }

        totalOperationsChecked++;

        // Extract OpConfig from IR
        OpConfig config = extractOpConfigFromIR(operation);
        if (!config.outputLayout) {
          operation->emitError()
              << "OperationValidationAndFallback: No output layout found in "
                 "operation result type encoding";
          signalPassFailure();
          return WalkResult::interrupt();
        }

        // Extract input layouts from the operation
        std::vector<TTNNLayoutAttr> inputLayouts =
            utils::extractInputLayouts(operation);

        // Test original configuration
        llvm::Expected<op_constraint_validation::ValidationResult>
            originalResult = op_constraint_validation::validateOperation(
                operation, inputLayouts, config);

        if (originalResult) {
          if (originalResult->actualOutputLayout != config.outputLayout) {
            // Output layout mismatch - need to update the IR to match the
            // expected layout and insert necessary conversions back to the
            // expected layout.
            TTMLIR_DEBUG(
                ttmlir::LogComponent::OpValidation,
                "Operation {} at {} passed validation with original config "
                "but output layouts mismatch: expected output layout: {}, "
                "backend output layout: {}",
                operation->getName(), operation->getLoc(), config.outputLayout,
                originalResult->actualOutputLayout);
            // Passing inputLayouts as both original and working layouts
            // because we didn't change input layouts.
            fallbacks::applyFallbackTransformations(
                operation, inputLayouts, inputLayouts, originalResult.get(),
                config);
          } else {
            TTMLIR_TRACE(
                ttmlir::LogComponent::OpValidation,
                "Operation {} at {} passed validation with original config",
                operation->getName(), operation->getLoc());
          }
          operationsValid++;
        } else {
          llvm::consumeError(originalResult.takeError());

          // Original config failed, try fallback configurations
          if (fallbacks::tryFallbacks(operation, inputLayouts, config)) {
            operationsFixed++;
            TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                         "Operation {} at {} fixed with fallback configuration",
                         operation->getName(), operation->getLoc());
          } else {
            operationsFailed++;
            TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                         "Operation {} at {} FAILED validation - no working "
                         "configuration found",
                         operation->getName(), operation->getLoc());
          }
        }
        return WalkResult::advance();
      });
    });

    // Log validation summary
    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "Operation validation complete: {} operations checked, "
                 "{} valid, {} failed",
                 totalOperationsChecked, operationsValid, operationsFailed);

    // Fail the pass if we have operations that couldn't be validated
    if (operationsFailed > 0) {
      moduleOp->emitError()
          << "OperationValidationAndFallback: " << operationsFailed
          << " operations failed validation";
      signalPassFailure();
    }
  }

private:
  // Extract OpConfig from operation's IR
  OpConfig extractOpConfigFromIR(Operation *operation) {
    OpConfig config;

    assert(operation->getNumResults() == 1 &&
           "Expected operation with one result");

    // Extract output layout from result type
    if (auto tensorType =
            mlir::dyn_cast<RankedTensorType>(operation->getResultTypes()[0])) {
      if (auto layoutAttr = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
              tensorType.getEncoding())) {
        config.outputLayout = layoutAttr;
      }
    }

    // For Conv2d operations, extract op-specific attributes
    if (auto conv2dOp = mlir::dyn_cast<ttnn::Conv2dOp>(operation)) {
      config.opSpecificAttrs = Conv2dAttrs{conv2dOp.getConv2dConfigAttr(),
                                           conv2dOp.getComputeConfigAttr()};
    }

    return config;
  }
};

namespace fallbacks {

bool tryFallbacks(Operation *operation,
                  const std::vector<TTNNLayoutAttr> &originalInputLayouts,
                  const OpConfig &config) {

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
    return false;
  }

  TTMLIR_DEBUG(
      ttmlir::LogComponent::Optimizer,
      "Testing fallback combinations for operation {} at {} with {} operands",
      operation->getName(), operation->getLoc(), originalInputLayouts.size());

  // Create fallback combinations for each operand
  std::vector<std::vector<TTNNLayoutAttr>> operandFallbacks;
  for (size_t i = 0; i < originalInputLayouts.size(); ++i) {
    auto fallbacks =
        createFallbackTransforms(originalInputLayouts[i], tensorShapes[i]);
    // Insert original layout at the beginning
    fallbacks.insert(fallbacks.begin(), originalInputLayouts[i]);
    operandFallbacks.push_back(fallbacks);
  }

  // Generate all possible combinations and calculate their distances from the
  // original layouts.
  std::vector<CombinationCandidate> allCombinations;
  std::vector<TTNNLayoutAttr> currentCombination(originalInputLayouts.size());
  generateAllCombinations(operandFallbacks, originalInputLayouts,
                          /*operandIdx*/ 0, currentCombination,
                          /*currentDistance*/ 0.0, allCombinations);

  // Sort combinations by total distance (ascending)
  std::sort(allCombinations.begin(), allCombinations.end(),
            [](const CombinationCandidate &a, const CombinationCandidate &b) {
              return a.totalDistance < b.totalDistance;
            });

  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "Generated {} combinations, testing by distance",
               allCombinations.size());

  // Test combinations in order of increasing distance
  for (const auto &candidate : allCombinations) {
    auto result = testFallbackCombination(operation, config, candidate.layouts);
    if (auto error = result.takeError()) {
      // Combination failed
      llvm::consumeError(std::move(error));
      continue;
    }

    // Found working solution, apply transformations
    applyFallbackTransformations(operation, originalInputLayouts,
                                 candidate.layouts, result.get(), config);
    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "Found working fallback combination with {} operands",
                 candidate.layouts.size());
    return true;
  }

  return false;
}

// Helper method to create fallback layouts directly
std::vector<TTNNLayoutAttr>
createFallbackTransforms(TTNNLayoutAttr originalLayout,
                         llvm::ArrayRef<int64_t> tensorShape) {
  // Create systematic 2×4 combinations: 2 layouts × 4 data types = 8
  // combinations Layouts: RowMajor, Tile DataTypes: BFloat16, Float32,
  // UInt32, Int32
  // TODO(rpavlovicTT): Expand to more combinations if needed in the future.
  //                    E.g. Bfp8. At the moment we don't create new
  //                    MemoryLayouts as Interleaved should always work.
  std::vector<TTNNLayoutAttr> fallbackLayouts;

  // Define the 4 target data types for fallbacks
  std::vector<ttcore::DataType> targetDataTypes = {
      ttcore::DataType::BFloat16, ttcore::DataType::Float32,
      ttcore::DataType::UInt32, ttcore::DataType::Int32};

  // Define the 2 target layouts for fallbacks
  std::vector<Layout> targetLayouts = {Layout::RowMajor, Layout::Tile};

  // Define the 2 buffer types for fallbacks
  std::vector<BufferType> targetBufferTypes = {BufferType::DRAM,
                                               BufferType::SystemMemory};

  for (Layout targetLayout : targetLayouts) {
    for (ttcore::DataType targetDataType : targetDataTypes) {
      for (BufferType targetBufferType : targetBufferTypes) {
        // Skip if this is the same as original (already tested)
        if (targetLayout == originalLayout.getLayout() &&
            targetDataType == originalLayout.getDataType() &&
            targetBufferType == originalLayout.getBufferType()) {
          continue;
        }

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

        if (targetBufferType != result.getBufferType()) {
          result = result.withBufferType(targetBufferType);
        }

        fallbackLayouts.push_back(result);
      }
    }
  }

  return fallbackLayouts;
}

// Calculate distance between two data types (lower = more similar)
double calculateDataTypeDistance(ttcore::DataType from, ttcore::DataType to) {
  if (from == to) {
    return NO_COST;
  }

  // Define data type similarity groups
  bool fromIsInt =
      (from == ttcore::DataType::Int32 || from == ttcore::DataType::UInt32);
  bool toIsInt =
      (to == ttcore::DataType::Int32 || to == ttcore::DataType::UInt32);
  bool fromIsFloat =
      (from == ttcore::DataType::Float32 || from == ttcore::DataType::BFloat16);
  bool toIsFloat =
      (to == ttcore::DataType::Float32 || to == ttcore::DataType::BFloat16);

  // Helper function to check if conversion is an upcast (lower -> higher
  // precision)
  auto isUpcast = [](ttcore::DataType from, ttcore::DataType to) -> bool {
    // BFloat16 -> Float32 is upcast
    if (from == ttcore::DataType::BFloat16 && to == ttcore::DataType::Float32) {
      return true;
    }
    // Int32 -> UInt32 or UInt32 -> Int32 are lateral moves (not upcasts)
    return false;
  };

  if (fromIsInt && toIsInt) {
    return LOW_COST; // Both integers: Int32 <-> UInt32 is lateral move
  }
  if (fromIsFloat && toIsFloat) {
    if (isUpcast(from, to)) {
      return MID_COST; // Upcast (BFloat16 -> Float32) is more expensive
    }
    return LOW_COST; // Downcast (Float32 -> BFloat16) is cheaper
  }

  // Cross-category changes are more expensive as they are not safe conversions
  // wrt. precision
  return MID_COST;
}

// Calculate distance between two layouts
double calculateLayoutDistance(Layout from, Layout to) {
  if (from == to) {
    return NO_COST;
  }

  // RowMajor <-> Tile conversion is low cost as it's safer
  // than changing dtype
  return LOW_COST;
}

// Calculate total distance for a layout transformation
double calculateLayoutTransformDistance(TTNNLayoutAttr from,
                                        TTNNLayoutAttr to) {
  double distance = 0.0;

  // Add data type distance
  distance += calculateDataTypeDistance(from.getDataType(), to.getDataType());

  // Add layout distance
  distance += calculateLayoutDistance(from.getLayout(), to.getLayout());

  // Add buffer type distance (if different), make it more expensive
  if (from.getBufferType() != to.getBufferType()) {
    distance += HIGH_COST;
  }

  return distance;
}

// Generate all possible fallback combinations recursively
void generateAllCombinations(
    const std::vector<std::vector<TTNNLayoutAttr>> &operandFallbacks,
    const std::vector<TTNNLayoutAttr> &originalInputLayouts, size_t operandIdx,
    std::vector<TTNNLayoutAttr> &currentCombination, double currentDistance,
    std::vector<CombinationCandidate> &allCombinations) {

  if (operandIdx == operandFallbacks.size()) {
    // Reached the end, store the current combination if it has changes.
    if (currentDistance > 0.0) {
      allCombinations.push_back({currentCombination, currentDistance});
    }
    return;
  }

  // Try each fallback for the current operand
  for (size_t fallbackIdx = 0;
       fallbackIdx < operandFallbacks[operandIdx].size(); ++fallbackIdx) {
    currentCombination[operandIdx] = operandFallbacks[operandIdx][fallbackIdx];

    double addedDistance = calculateLayoutTransformDistance(
        originalInputLayouts[operandIdx],
        operandFallbacks[operandIdx][fallbackIdx]);

    // Go to the next operand, accumulating distance along the way.
    generateAllCombinations(operandFallbacks, originalInputLayouts,
                            operandIdx + 1, currentCombination,
                            currentDistance + addedDistance, allCombinations);
  }
}

// Test a specific combination of fallback layouts for an operation
llvm::Expected<op_constraint_validation::ValidationResult>
testFallbackCombination(Operation *op, const OpConfig &originalConfig,
                        const std::vector<TTNNLayoutAttr> &inputLayouts) {

  // For all fallbacks, constrain output layout to be DRAM Interleaved.
  OpConfig testConfig = originalConfig;
  testConfig.outputLayout = originalConfig.outputLayout;
  testConfig.outputLayout =
      testConfig.outputLayout.withBufferType(BufferType::DRAM);
  testConfig.outputLayout =
      testConfig.outputLayout.withMemoryLayout(TensorMemoryLayout::Interleaved);

  return op_constraint_validation::validateOperation(op, inputLayouts,
                                                     testConfig);
}

// Apply fallback transformations to the operation
void applyFallbackTransformations(
    Operation *operation, const std::vector<TTNNLayoutAttr> &originalLayouts,
    const std::vector<TTNNLayoutAttr> &workingLayouts,
    const op_constraint_validation::ValidationResult &result,
    const OpConfig &config) {

  // Apply input operand changes for each operand that was transformed
  for (size_t i = 0; i < workingLayouts.size(); ++i) {
    TTNNLayoutAttr originalLayout = originalLayouts[i];
    TTNNLayoutAttr transformedLayout = workingLayouts[i];

    // Skip if layout wasn't actually transformed
    if (originalLayout == transformedLayout) {
      continue;
    }

    // Create InputOperandChange object and populate it
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
      applyInputOperandChange(operation, change);
    }
  }

  // Handle output layout changes if backend produced different layout
  if (result.actualOutputLayout != config.outputLayout) {
    // Step 1: Update operation's result type to what backend actually
    // produced
    assert(operation->getNumResults() == 1 &&
           "Currently only single-result operations are supported");
    auto oldResultType =
        mlir::dyn_cast<RankedTensorType>(operation->getResult(0).getType());
    assert(oldResultType && "Operation result type must be RankedTensorType");
    auto newResultType =
        RankedTensorType::get(oldResultType.getShape(),
                              result.actualOutputLayout.getScalarElementType(),
                              result.actualOutputLayout);
    operation->getResult(0).setType(newResultType);

    // Step 2: Add revert ToLayoutOp to convert back to expected layout for
    // consumers
    applyOutputLayoutRevert(operation, result.actualOutputLayout,
                            config.outputLayout);
  }
}

// Apply a single input operand change by inserting ToLayoutOp
void applyInputOperandChange(Operation *operation,
                             const InputOperandChange &change) {
  Value operand = operation->getOperand(change.operandIndex);
  auto currentTensorType = mlir::cast<RankedTensorType>(operand.getType());
  auto currentLayoutAttr =
      mlir::cast<TTNNLayoutAttr>(currentTensorType.getEncoding());

  // Build the new layout attribute with the required changes
  TTNNLayoutAttr newLayoutAttr = currentLayoutAttr;

  // Handle layout changes (ROW_MAJOR vs TILE)
  if (change.targetLayout.has_value()) {
    newLayoutAttr = newLayoutAttr.withLayout(change.targetLayout.value(),
                                             currentTensorType.getShape());
  }

  if (change.targetMemoryLayout.has_value()) {
    newLayoutAttr =
        newLayoutAttr.withMemoryLayout(change.targetMemoryLayout.value());
  }

  if (change.targetBufferType.has_value()) {
    newLayoutAttr =
        newLayoutAttr.withBufferType(change.targetBufferType.value());
  }

  // Handle data type changes
  if (change.targetDataType.has_value()) {
    auto targetElementType = ttnn::utils::getElementType(
        newLayoutAttr.getContext(), newLayoutAttr.getLayout(),
        change.targetDataType.value());
    newLayoutAttr = newLayoutAttr.withElementType(targetElementType,
                                                  currentTensorType.getShape());
  }

  // Create new tensor type with updated layout
  // Use scalar element type for RankedTensorType, not tile type
  Type scalarElementType = mlir::tt::ttcore::dataTypeToElementType(
      operation->getContext(), newLayoutAttr.getDataType());
  RankedTensorType newTensorType = RankedTensorType::get(
      currentTensorType.getShape(), scalarElementType, newLayoutAttr);

  // Determine which layout to use for ToLayoutOp
  Layout targetLayoutValue =
      change.targetLayout.value_or(currentLayoutAttr.getLayout());

  // Insert ToLayout operation to perform the transformation
  OpBuilder builder(operation);

  // Create proper MemoryConfigAttr for the target layout
  MemoryConfigAttr memoryConfigAttr = MemoryConfigAttr::get(
      operation->getContext(), newLayoutAttr.getMemLayout(),
      BufferTypeAttr::get(operation->getContext(),
                          newLayoutAttr.getBufferType()),
      /*shardSpec=*/std::nullopt);

  // Prepare dtype attribute if data type changed
  ttcore::DataTypeAttr dtypeAttr = nullptr;
  if (change.targetDataType.has_value()) {
    dtypeAttr = ttcore::DataTypeAttr::get(operation->getContext(),
                                          change.targetDataType.value());
  } else {
    dtypeAttr = ttcore::DataTypeAttr::get(operation->getContext(),
                                          currentLayoutAttr.getDataType());
  }

  auto toLayoutOp = builder.create<ToLayoutOp>(
      operation->getLoc(), newTensorType, operand,
      LayoutAttr::get(operation->getContext(), targetLayoutValue), dtypeAttr,
      memoryConfigAttr);

  // Replace the operand with the result of ToLayout
  operation->setOperand(change.operandIndex, toLayoutOp.getResult());

  TTMLIR_DEBUG(
      ttmlir::LogComponent::OpValidation,
      "Applied input operand change for operation {} operand {}: "
      "layout {} -> {}, memory layout {} -> {}, buffer type {} -> {}, data "
      "type {} -> {}",
      operation->getName(), change.operandIndex,
      change.originalLayout.getLayout(),
      change.targetLayout.value_or(change.originalLayout.getLayout()),
      change.originalLayout.getMemLayout().getValue(),
      change.targetMemoryLayout.value_or(
          change.originalLayout.getMemLayout().getValue()),
      change.originalLayout.getBufferType(),
      change.targetBufferType.value_or(change.originalLayout.getBufferType()),
      static_cast<int>(change.originalLayout.getDataType()),
      static_cast<int>(
          change.targetDataType.value_or(change.originalLayout.getDataType())));
}

// Apply output layout revert after an operation
void applyOutputLayoutRevert(Operation *operation,
                             TTNNLayoutAttr actualOutputLayout,
                             TTNNLayoutAttr expectedOutputLayout) {
  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "Applying output layout revert for operation {}: "
               "actual layout {}, expected layout {}",
               operation->getName(), actualOutputLayout, expectedOutputLayout);

  Value result = operation->getResult(0);
  auto currentResultType = mlir::cast<RankedTensorType>(result.getType());

  // Save all uses of the operation's result before making changes
  llvm::SmallVector<std::pair<Operation *, unsigned>> uses;
  for (auto &use : result.getUses()) {
    uses.emplace_back(use.getOwner(), use.getOperandNumber());
  }

  // Insert ToLayoutOp after the operation to revert back to the original
  // expected layout
  OpBuilder builder(operation->getContext());
  builder.setInsertionPointAfter(operation);

  // Create the expected result type for the revert operation
  Type expectedElementType = mlir::tt::ttcore::dataTypeToElementType(
      operation->getContext(), expectedOutputLayout.getDataType());
  RankedTensorType expectedResultType = RankedTensorType::get(
      currentResultType.getShape(), expectedElementType, expectedOutputLayout);

  // Create proper MemoryConfigAttr for the expected layout
  MemoryConfigAttr expectedMemoryConfigAttr = MemoryConfigAttr::get(
      operation->getContext(), expectedOutputLayout.getMemLayout(),
      BufferTypeAttr::get(operation->getContext(),
                          expectedOutputLayout.getBufferType()),
      /*shardSpec=*/std::nullopt);

  auto revertToLayoutOp = builder.create<ToLayoutOp>(
      ttmlir::utils::appendLocationSuffix(operation->getLoc(),
                                          "_revert_layout"),
      expectedResultType, result,
      LayoutAttr::get(operation->getContext(),
                      expectedOutputLayout.getLayout()),
      ttcore::DataTypeAttr::get(operation->getContext(),
                                expectedOutputLayout.getDataType()),
      expectedMemoryConfigAttr);

  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "Inserted revert ToLayout op after operation {} to restore "
               "expected layout",
               operation->getName());

  // Update all saved uses to point to the revert operation instead
  for (auto &use : uses) {
    Operation *useOp = use.first;
    useOp->setOperand(use.second, revertToLayoutOp.getResult());
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Updated consumer {}@{} to use reverted layout",
                 useOp->getName(), useOp->getLoc());
  }
}
} // namespace fallbacks

} // namespace mlir::tt::ttnn
