// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/Conv2dConfigSearchSpace.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Conv2dConfigParams.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cassert>
#include <cstddef>
#include <optional>
#include <set>
#include <vector>

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

op_constraint_validation::ValidationResult
testFallbackCombination(Operation *op, const OpConfig &originalConfig,
                        const std::vector<TTNNLayoutAttr> &inputLayouts);

void applyFallbackTransformations(
    Operation *operation, const std::vector<TTNNLayoutAttr> &originalLayouts,
    const std::vector<TTNNLayoutAttr> &workingLayouts,
    const op_constraint_validation::ValidationResult &result,
    llvm::SmallVector<OpConfig> &configs);

void applyInputOperandChange(Operation *operation, size_t operandIndex,
                             TTNNLayoutAttr currentLayoutAttr,
                             TTNNLayoutAttr targetLayoutAttr);

void applyOutputLayoutRevert(Operation *operation, size_t resultIndex,
                             TTNNLayoutAttr actualOutputLayout,
                             TTNNLayoutAttr expectedOutputLayout);

ToLayoutOp createToLayoutOp(OpBuilder &builder, Location loc,
                            RankedTensorType resultType, Value inputValue,
                            TTNNLayoutAttr targetLayout);

// Try fallback configurations for a failed operation
bool tryFallbacks(Operation *operation,
                  const std::vector<TTNNLayoutAttr> &originalInputLayouts,
                  const OpConfig &config, uint32_t maxAttempts = 0);

// Try config fallbacks for Conv2d-like operations
bool tryConfigFallbacks(Operation *operation,
                        const std::vector<TTNNLayoutAttr> &originalInputLayouts,
                        const OpConfig &originalConfig,
                        uint32_t maxAttempts = 0);

void applyConfigChange(Operation *operation, Conv2dConfigAttr newConfig);
} // namespace fallbacks

class TTNNOperationValidationAndFallback
    : public impl::TTNNOperationValidationAndFallbackBase<
          TTNNOperationValidationAndFallback> {

public:
  using impl::TTNNOperationValidationAndFallbackBase<
      TTNNOperationValidationAndFallback>::
      TTNNOperationValidationAndFallbackBase;

  TTNNOperationValidationAndFallback(
      TTNNOperationValidationAndFallbackOptions options)
      : impl::TTNNOperationValidationAndFallbackBase<
            TTNNOperationValidationAndFallback>(std::move(options)) {}

  TTNNOperationValidationAndFallback() = default;

  void runOnOperation() override {
#ifndef TTMLIR_ENABLE_OPMODEL
    llvm::llvm_unreachable_internal(
        "TTNNOperationValidationAndFallback pass requires OpModel support to be"
        "enabled.");
#else
    // Device lifecycle is managed by OpModelDeviceWrapperPass in the pipeline,
    // but for standalone pass usage (e.g., in tests), the guard opens/closes
    // it.
    op_model::ScopedSingletonDeviceGuard deviceGuard;

    ModuleOp moduleOp = getOperation();

    size_t totalOperationsChecked = 0;
    size_t operationsFixed = 0;
    bool validationFailed = false;

    moduleOp->walk([&](func::FuncOp func) {
      func.walk([&](Operation *operation) -> WalkResult {
        if (auto toLayoutOp = mlir::dyn_cast<ttnn::ToLayoutOp>(operation)) {
          // Skip ToLayout operations - they will be decomposed later, so there
          // is no point in validating them here.
          return WalkResult::skip();
        }

        // Skip operations that are not OpModel castable (can't be validated)
        // TODO(rpavlovic): we should have all ops implement OpModel interface
        // eventually. Then this check becomes an assert.
        if (!mlir::dyn_cast<OpModel>(operation)) {
          return WalkResult::skip();
        }

        // Extract OpConfigs from IR
        llvm::SmallVector<OpConfig> configs = extractOpConfigsFromIR(operation);

        // Skip operations with no tensor results (e.g., GetDeviceOp)
        if (configs.empty()) {
          return WalkResult::skip();
        }

        totalOperationsChecked++;

        // Extract input layouts from the operation
        std::vector<TTNNLayoutAttr> inputLayouts =
            utils::extractInputLayouts(operation);

        TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                     "Validating operation {} at {} with {} input layouts, "
                     "with {} output layouts, {} first output layout",
                     operation->getName(), operation->getLoc(),
                     inputLayouts.size(), configs.size(),
                     configs[0].outputLayout);

        // Test original configuration
        op_constraint_validation::ValidationResult originalResult =
            op_constraint_validation::validateOperation(operation, inputLayouts,
                                                        configs[0]);
        if (originalResult.isNotImplemented()) {
          TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                       "Operation {} at {} not supported for validation: {}",
                       operation->getName(), operation->getLoc(),
                       originalResult.errorMessage);
          return WalkResult::skip();
        }

        if (originalResult.isSuccess()) {
          bool outputLayoutsChangeNeeded = false;
          for (size_t i = 0; i < configs.size(); ++i) {
            // Safety check if config has output layout before comparing
            if (configs[i].outputLayout &&
                originalResult.actualOutputLayouts[i] !=
                    configs[i].outputLayout) {
              // Output layout mismatch - need to update the IR to match the
              // expected layout and insert necessary conversions back to the
              // expected layout.
              TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                           "Operation {} at {} passed validation but output "
                           "layout index {} mismatch: expected output layout: "
                           "{}, backend output layout: {}",
                           operation->getName(), operation->getLoc(), i,
                           configs[i].outputLayout,
                           originalResult.actualOutputLayouts[i]);
              outputLayoutsChangeNeeded = true;
            }
          }
          if (outputLayoutsChangeNeeded) {
            // Passing inputLayouts as both original and working layouts
            // because we didn't change input layouts.
            fallbacks::applyFallbackTransformations(
                operation, inputLayouts, inputLayouts, originalResult, configs);
          } else {
            TTMLIR_TRACE(
                ttmlir::LogComponent::OpValidation,
                "Operation {} at {} passed validation with original config",
                operation->getName(), operation->getLoc());
          }
        } else {
          // Try fallback configurations
          // TODO(bmalesevic, #): Fallback paths only pass configs[0] to
          // tryFallbacks/tryConfigFallbacks, so multi-output ops will only
          // have the first output's layout revert handled. Extend to pass
          // all configs when multi-output fallback support is needed.
          // For OOM errors, try config fallbacks first as they're cheaper (no
          // ToLayout ops)
          bool fixed = false;
          if (originalResult.status ==
              op_constraint_validation::ValidationStatus::OutOfMemoryError) {
            TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                         "OOM error detected, trying config fallbacks first "
                         "for operation {} at {}",
                         operation->getName(), operation->getLoc());
            fixed = fallbacks::tryConfigFallbacks(
                operation, inputLayouts, configs[0], maxFallbackAttempts);
          }

          // If config fallbacks didn't work or it wasn't an OOM error, try
          // layout fallbacks
          if (!fixed) {
            fixed = fallbacks::tryFallbacks(operation, inputLayouts, configs[0],
                                            maxFallbackAttempts);
          }

          // For non-OOM errors, try config fallbacks as a last resort.
          // This can help with config-related failures like slice window
          // misalignment.
          if (!fixed && originalResult.status !=
                            op_constraint_validation::ValidationStatus::
                                OutOfMemoryError) {
            TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                         "Trying config fallbacks for non-OOM error (status: "
                         "{}) at operation {} at {}",
                         op_constraint_validation::validationStatusToString(
                             originalResult.status),
                         operation->getName(), operation->getLoc());
            fixed = fallbacks::tryConfigFallbacks(
                operation, inputLayouts, configs[0], maxFallbackAttempts);
          }

          if (fixed) {
            operationsFixed++;
            TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                         "Operation {} at {} fixed with fallback configuration",
                         operation->getName(), operation->getLoc());
          } else {
            emitValidationFailureError(operation, originalResult,
                                       maxFallbackAttempts);
            validationFailed = true;
            signalPassFailure();
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
    });

    // Log validation summary
    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "Operation validation {}: {} operations checked{}, {} fixed",
                 validationFailed ? "FAILED" : "complete",
                 totalOperationsChecked,
                 validationFailed ? " before failure" : "", operationsFixed);
#endif // TTMLIR_ENABLE_OPMODEL
  }

private:
  // We explicitly include the operation IR in the error message to ensure
  // visibility across all environments, while automatically added note in
  // emitError may not always be visible. This helps with debugging validation
  // failures, as well as the original error message and the maximum fallback
  // attempts limit.
  void emitValidationFailureError(
      Operation *operation,
      const op_constraint_validation::ValidationResult &originalResult,
      uint32_t maxAttempts) {
    std::string opStr;
    llvm::raw_string_ostream os(opStr);
    operation->print(os);

    emitError(operation->getLoc())
        << "OperationValidationAndFallback: Operation " << operation->getName()
        << " failed validation (original error: "
        << op_constraint_validation::validationStatusToString(
               originalResult.status)
        << (!originalResult.errorMessage.empty()
                ? " - " + originalResult.errorMessage
                : "")
        << "). No fallback configuration worked (tested up to "
        << std::to_string(maxAttempts) << " combinations)."
        << "\nOperation IR: " << os.str();
  }

  // Extract OpConfigs from operation's IR
  llvm::SmallVector<OpConfig> extractOpConfigsFromIR(Operation *operation) {
    llvm::SmallVector<OpConfig> configs;

    for (auto result : operation->getResults()) {
      if (auto tensorType =
              mlir::dyn_cast<RankedTensorType>(result.getType())) {
        if (auto layoutAttr = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
                tensorType.getEncoding())) {
          configs.emplace_back(OpConfig{layoutAttr});
        }
      }
    }

    // For Conv2d operations, extract op-specific attributes to first OpConfig
    // for validation and fallback purposes.
    llvm::TypeSwitch<Operation *>(operation)
        .Case<ttnn::Conv2dOp, ttnn::ConvTranspose2dOp>([&configs](auto convOp) {
          configs[0].opSpecificAttrs = Conv2dAttrs{
              convOp.getConv2dConfigAttr(), convOp.getComputeConfigAttr()};
        });

    return configs;
  }
};

namespace fallbacks {

bool tryFallbacks(Operation *operation,
                  const std::vector<TTNNLayoutAttr> &originalInputLayouts,
                  const OpConfig &config, uint32_t maxAttempts) {

  // Extract tensor shapes for all input operands
  std::vector<llvm::ArrayRef<int64_t>> tensorShapes;
  for (Value operand : operation->getOperands()) {
    if (auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType())) {
      tensorShapes.push_back(tensorType.getShape());
    }
  }

  if (originalInputLayouts.empty()) {
    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "No TTNN input layouts found for operation {} at {}",
                 operation->getName(), operation->getLoc());
    return false;
  }

  TTMLIR_DEBUG(
      ttmlir::LogComponent::OpValidation,
      "Testing fallback combinations for operation {} at {} with {} operands",
      operation->getName(), operation->getLoc(), originalInputLayouts.size());

  // Create fallback combinations for each operand
  std::vector<std::vector<TTNNLayoutAttr>> operandFallbacks;
  for (size_t i = 0; i < originalInputLayouts.size(); ++i) {
    auto fallbacks =
        createFallbackTransforms(originalInputLayouts[i], tensorShapes[i]);
    fallbacks.push_back(originalInputLayouts[i]);
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
  // Comparator includes full tiebreaker for deterministic ordering
  std::sort(allCombinations.begin(), allCombinations.end(),
            [](const CombinationCandidate &a, const CombinationCandidate &b) {
              if (a.totalDistance != b.totalDistance) {
                return a.totalDistance < b.totalDistance;
              }
              // Tiebreaker: content-based lexicographic comparison for full
              // determinism across runs when distances are equal
              for (size_t i = 0; i < a.layouts.size() && i < b.layouts.size();
                   ++i) {
                const auto &aLayout = a.layouts[i];
                const auto &bLayout = b.layouts[i];

                // Compare layout type first (RowMajor before Tile based on enum
                // order)
                if (aLayout.getLayout() != bLayout.getLayout()) {
                  return static_cast<int>(aLayout.getLayout()) <
                         static_cast<int>(bLayout.getLayout());
                }
                if (aLayout.getDataType() != bLayout.getDataType()) {
                  return static_cast<int>(aLayout.getDataType()) <
                         static_cast<int>(bLayout.getDataType());
                }
                if (aLayout.getBufferType() != bLayout.getBufferType()) {
                  return static_cast<int>(aLayout.getBufferType()) <
                         static_cast<int>(bLayout.getBufferType());
                }
              }
              // All layouts match - combinations are equivalent
              return false;
            });

  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "Generated {} combinations, testing by distance",
               allCombinations.size());

  // Track failed attempts
  size_t failedAttempts = 0;

  // Test combinations in order of increasing distance
  for (const auto &candidate : allCombinations) {
    auto result = testFallbackCombination(operation, config, candidate.layouts);

    if (!result.isSuccess()) {
      failedAttempts++;
      TTMLIR_TRACE(ttmlir::LogComponent::OpValidation,
                   "Combination failed (status: {}): {}",
                   static_cast<int>(result.status), result.errorMessage);

      // Check if we've exceeded the maximum attempts (if limit is set)
      if (maxAttempts > 0 &&
          failedAttempts >= static_cast<size_t>(maxAttempts)) {
        TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                     "Reached maximum fallback attempts ({}) for operation {} "
                     "at {}. Terminating early.",
                     maxAttempts, operation->getName(), operation->getLoc());
        return false;
      }
      continue;
    }
    llvm::SmallVector<OpConfig> outputConfigs({config});
    // Found working solution, apply transformations
    applyFallbackTransformations(operation, originalInputLayouts,
                                 candidate.layouts, result, outputConfigs);
    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "Found working fallback combination with {} operands after {} "
                 "failed attempts",
                 candidate.layouts.size(), failedAttempts);
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
  // Use set to automatically deduplicate during generation, for case
  // a target configuration isn't viable e.g. bfp8 and row-major combination
  // isn't possible. Using std::set instead of std::unordered_set provides
  // deterministic iteration order for consistent fallback selection.
  struct TTNNLayoutAttrCompare {
    bool operator()(const TTNNLayoutAttr &a, const TTNNLayoutAttr &b) const {
      // Content-based comparison for deterministic ordering across runs
      // Compare by: Layout -> DataType -> BufferType
      if (a.getLayout() != b.getLayout()) {
        return static_cast<int>(a.getLayout()) <
               static_cast<int>(b.getLayout());
      }
      if (a.getDataType() != b.getDataType()) {
        return static_cast<int>(a.getDataType()) <
               static_cast<int>(b.getDataType());
      }
      if (a.getBufferType() != b.getBufferType()) {
        return static_cast<int>(a.getBufferType()) <
               static_cast<int>(b.getBufferType());
      }
      // Layouts match - fallbacks are equivalent
      return false;
    }
  };
  std::set<TTNNLayoutAttr, TTNNLayoutAttrCompare> fallbackLayoutsSet;

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
        // Skip if this is the same as original
        if (targetLayout == originalLayout.getLayout() &&
            targetDataType == originalLayout.getDataType() &&
            targetBufferType == originalLayout.getBufferType()) {
          continue;
        }

        // Start with original layout
        TTNNLayoutAttr result = originalLayout;

        // Apply data type transformation first (before layout changes)
        // This prevents issues where layout changes force incompatible
        // combinations (e.g., BFP8 requires tile layout, so changing layout to
        // row-major first would auto-convert back to tile, preventing us from
        // trying row-major with other data types)
        if (targetDataType != result.getDataType()) {
          auto targetElementType = ttnn::utils::getElementType(
              result.getContext(), result.getLayout(), targetDataType);
          result = result.withElementType(targetElementType, tensorShape);
        }

        // Apply layout transformation if needed (after dtype change)
        if (targetLayout != result.getLayout()) {
          result = result.withLayout(targetLayout, tensorShape);
        }

        if (targetBufferType != result.getBufferType()) {
          result = result.withBufferType(targetBufferType);
        }

        fallbackLayoutsSet.insert(result);
      }
    }
  }

  TTMLIR_TRACE(
      ttmlir::LogComponent::OpValidation,
      "Generated {} unique fallback layouts from {} target combinations",
      fallbackLayoutsSet.size(),
      targetDataTypes.size() * targetLayouts.size() * targetBufferTypes.size());

  // Convert set to vector for return
  return std::vector<TTNNLayoutAttr>(fallbackLayoutsSet.begin(),
                                     fallbackLayoutsSet.end());
}

// Calculate distance between two data types (lower = more similar)
double calculateDataTypeDistance(ttcore::DataType from, ttcore::DataType to) {
  if (from == to) {
    return NO_COST;
  }

  // Helper to get type category and precision
  struct TypeInfo {
    enum Category { INTEGER, FLOATING } category;
    enum Precision { LOW = 1, HIGH = 2 } precision;
  };

  auto getTypeInfo = [](ttcore::DataType type) -> TypeInfo {
    switch (type) {
    case ttcore::DataType::Int32:
      return {TypeInfo::Category::INTEGER, TypeInfo::Precision::HIGH};
    case ttcore::DataType::UInt32:
      return {TypeInfo::Category::INTEGER, TypeInfo::Precision::HIGH};
    case ttcore::DataType::BFloat16:
      return {TypeInfo::Category::FLOATING, TypeInfo::Precision::LOW};
    case ttcore::DataType::Float32:
      return {TypeInfo::Category::FLOATING, TypeInfo::Precision::HIGH};
    default:
      // For unknown types, assume high precision floating point
      return {TypeInfo::Category::FLOATING, TypeInfo::Precision::HIGH};
    }
  };

  TypeInfo fromInfo = getTypeInfo(from);
  TypeInfo toInfo = getTypeInfo(to);

  // Same category transformations
  if (fromInfo.category == toInfo.category) {
    if (fromInfo.precision == toInfo.precision) {
      // Lateral moves like Int32 <-> UInt32
      return LOW_COST;
    }
    // Change in precision like Float32 <-> BFloat16. Increase in precision is
    // more expensive.
    return (fromInfo.precision < toInfo.precision) ? MID_COST : LOW_COST;
  }

  // Cross-category changes are more expensive
  return MID_COST;
}

// Calculate distance between two layouts
double calculateLayoutDistance(Layout from, Layout to) {
  if (from == to) {
    return NO_COST;
  }

  // RowMajor <-> Tile conversion is not cheap since it's not frequently needed
  return MID_COST;
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
    if (from.isDeviceBufferType() && to.isSystemBufferType()) {
      // Device <-> Host buffer type changes are more expensive
      distance += HIGH_COST;
    }
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
    allCombinations.push_back({currentCombination, currentDistance});
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
op_constraint_validation::ValidationResult
testFallbackCombination(Operation *op, const OpConfig &originalConfig,
                        const std::vector<TTNNLayoutAttr> &inputLayouts) {

  // For all fallbacks, constrain output layout to be DRAM Interleaved.
  OpConfig testConfig = originalConfig;
  if (testConfig.outputLayout) {
    testConfig.outputLayout = originalConfig.outputLayout;
    testConfig.outputLayout =
        testConfig.outputLayout.withBufferType(BufferType::DRAM);
    testConfig.outputLayout = testConfig.outputLayout.withMemoryLayout(
        TensorMemoryLayout::Interleaved);
  }

  return op_constraint_validation::validateOperation(op, inputLayouts,
                                                     testConfig);
}

// Apply fallback transformations to the operation
void applyFallbackTransformations(
    Operation *operation, const std::vector<TTNNLayoutAttr> &originalLayouts,
    const std::vector<TTNNLayoutAttr> &workingLayouts,
    const op_constraint_validation::ValidationResult &result,
    llvm::SmallVector<OpConfig> &configs) {

  // Apply input operand changes for each operand that was transformed
  for (size_t i = 0; i < workingLayouts.size(); ++i) {
    TTNNLayoutAttr originalLayout = originalLayouts[i];
    TTNNLayoutAttr transformedLayout = workingLayouts[i];

    // Skip if layout wasn't actually transformed
    if (originalLayout == transformedLayout) {
      continue;
    }

    applyInputOperandChange(operation, i, originalLayout, transformedLayout);
  }

  if (!configs[0].outputLayout) {
    // Current operation doesn't have any expected output layouts, nothing more
    // to do.
    return;
  }

  for (size_t i = 0; i < configs.size(); ++i) {
    if (!configs[i].outputLayout) {
      // This config doesn't have an expected output layout, skip it.
      continue;
    }

    assert(
        result.actualOutputLayouts.size() > i &&
        "Validation result must contain actual output layout for each output.");
    // Handle output layout changes if backend produced different layout
    if (result.actualOutputLayouts[i] != configs[i].outputLayout) {

      if (result.actualOutputLayouts[i].getLayout() ==
              configs[i].outputLayout.getLayout() &&
          result.actualOutputLayouts[i].getDataType() ==
              configs[i].outputLayout.getDataType() &&
          result.actualOutputLayouts[i].getBufferType() ==
              configs[i].outputLayout.getBufferType() &&
          result.actualOutputLayouts[i].getMemLayout() ==
              configs[i].outputLayout.getMemLayout()) {
        // This may happen if GridAttr is different, which should not matter for
        // any memory layout other than Sharded. Since fallbacks do not go into
        // Sharded memory layout, we can avoid this case for now.
        continue;
      }

      // Step 1: Update operation's result type to what backend actually
      // produced
      auto oldResultType =
          mlir::dyn_cast<RankedTensorType>(operation->getResult(i).getType());
      assert(oldResultType && "Operation result type must be RankedTensorType");
      auto newResultType = RankedTensorType::get(
          oldResultType.getShape(),
          result.actualOutputLayouts[i].getScalarElementType(),
          result.actualOutputLayouts[i]);
      operation->getResult(i).setType(newResultType);

      // Step 2: Add revert ToLayoutOp to convert back to expected layout for
      // consumers
      applyOutputLayoutRevert(operation, i, result.actualOutputLayouts[i],
                              configs[i].outputLayout);
    }

    // Update the layout attribute for ops that have one (e.g., creation ops).
    // The layout attribute must match the result type's layout.
    if (TTNNLayoutOpInterface opWithLayoutIF =
            mlir::dyn_cast<TTNNLayoutOpInterface>(operation)) {
      opWithLayoutIF.setLayoutAttr(LayoutAttr::get(
          operation->getContext(), result.firstActualOutputLayout.getLayout()));
    }
  }
}

// Apply a single input operand change by inserting ToLayoutOp
void applyInputOperandChange(Operation *operation, size_t operandIndex,
                             TTNNLayoutAttr currentLayoutAttr,
                             TTNNLayoutAttr targetLayoutAttr) {

  Value operand = operation->getOperand(operandIndex);
  auto currentTensorType = mlir::cast<RankedTensorType>(operand.getType());

  // Insert ToLayout operation to perform the transformation
  OpBuilder builder(operation);
  auto toLayoutOp =
      createToLayoutOp(builder, operation->getLoc(), currentTensorType, operand,
                       targetLayoutAttr);

  // Replace the operand with the result of ToLayout
  operation->setOperand(operandIndex, toLayoutOp.getResult());

  TTMLIR_DEBUG(
      ttmlir::LogComponent::OpValidation,
      "Applied input operand change for operation {} operand {}: "
      "layout {} -> {}, memory layout {} -> {}, buffer type {} -> {}, data "
      "type {} -> {}",
      operation->getName(), operandIndex, currentLayoutAttr.getLayout(),
      targetLayoutAttr.getLayout(),
      currentLayoutAttr.getMemLayoutOpt().value_or(
          TensorMemoryLayout::Interleaved),
      targetLayoutAttr.getMemLayoutOpt().value_or(
          TensorMemoryLayout::Interleaved),
      currentLayoutAttr.getBufferType(), targetLayoutAttr.getBufferType(),
      static_cast<int>(currentLayoutAttr.getDataType()),
      static_cast<int>(targetLayoutAttr.getDataType()));
}

// Apply output layout revert after an operation
void applyOutputLayoutRevert(Operation *operation, size_t resultIndex,
                             TTNNLayoutAttr actualOutputLayout,
                             TTNNLayoutAttr expectedOutputLayout) {
  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
               "Applying output layout revert for operation {} result {}: "
               "actual layout {}, expected layout {}",
               operation->getName(), resultIndex, actualOutputLayout,
               expectedOutputLayout);

  Value result = operation->getResult(resultIndex);
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

  auto revertToLayoutOp =
      createToLayoutOp(builder,
                       ttmlir::utils::appendLocationSuffix(operation->getLoc(),
                                                           "_revert_layout"),
                       currentResultType, result, expectedOutputLayout);

  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
               "Inserted revert ToLayout op after operation {} to restore "
               "expected layout",
               operation->getName());

  // Update all saved uses to point to the revert operation instead
  for (auto &use : uses) {
    Operation *useOp = use.first;
    useOp->setOperand(use.second, revertToLayoutOp.getResult());
    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "Updated consumer {}@{} to use reverted layout",
                 useOp->getName(), useOp->getLoc());
  }
}

ToLayoutOp createToLayoutOp(OpBuilder &builder, Location loc,
                            RankedTensorType currentResultType,
                            Value inputValue, TTNNLayoutAttr targetLayout) {

  // Create result type for ToLayoutOp, which has the same shape as
  // currentResultType but use scalar element type and encoding from
  // targetLayout
  Type scalarElementType = mlir::tt::ttcore::dataTypeToElementType(
      builder.getContext(), targetLayout.getDataType());
  RankedTensorType resultType = RankedTensorType::get(
      currentResultType.getShape(), scalarElementType, targetLayout);

  return builder.create<ToLayoutOp>(
      loc, resultType, inputValue,
      LayoutAttr::get(builder.getContext(), targetLayout.getLayout()),
      ttcore::DataTypeAttr::get(builder.getContext(),
                                targetLayout.getDataType()),
      MemoryConfigAttr::get(builder.getContext(), targetLayout.getMemLayout(),
                            BufferTypeAttr::get(builder.getContext(),
                                                targetLayout.getBufferType()),
                            /*shardSpec=*/std::nullopt));
}

// Try config fallbacks for Conv2d-like operations
bool tryConfigFallbacks(Operation *operation,
                        const std::vector<TTNNLayoutAttr> &originalInputLayouts,
                        const OpConfig &originalConfig, uint32_t maxAttempts) {
  // Only applicable to operations with Conv2dAttrs
  if (!std::holds_alternative<Conv2dAttrs>(originalConfig.opSpecificAttrs)) {
    return false;
  }

  const auto &conv2dAttrs =
      std::get<Conv2dAttrs>(originalConfig.opSpecificAttrs);

  // Use existing config or default if none exists
  Conv2dConfigAttr originalConfigAttr =
      conv2dAttrs.conv2dConfig.has_value()
          ? conv2dAttrs.conv2dConfig.value()
          : Conv2dConfigAttr::get(operation->getContext());

  // Create search spaces for act_block_h_override
  // L1Full supports 0; DRAM slices don't
  // TODO(bmalesevic, #6634): Remove when tt-metal backend supports
  // act_block_h=0 with DRAM slicing
  Conv2dConfigSearchSpace l1SearchSpace;
  l1SearchSpace.actBlockHOverride.push_back(0);
  for (uint32_t val = 1024; val >= 32; val -= 32) {
    l1SearchSpace.actBlockHOverride.push_back(val);
  }

  Conv2dConfigSearchSpace dramSearchSpace;
  for (uint32_t val = 1024; val >= 32; val -= 32) {
    dramSearchSpace.actBlockHOverride.push_back(val);
  }
  // Use Conv2dConfigParams to unset act_block_h_override so the generator
  // will iterate through the search space values
  Conv2dConfigParams configParams(originalConfigAttr);
  configParams.actBlockHOverride = std::nullopt; // Unset this field
  Conv2dConfigAttr baseConfig =
      configParams.buildConv2dConfigAttr(operation->getContext());

  // Use the Conv2dConfigGenerator to iterate through configs
  auto filterOutFn = [](const Conv2dConfigAttr &) { return false; };

  Conv2dConfigAttr workingConfig = nullptr;
  op_constraint_validation::ValidationResult workingResult;

  // Track failed attempts
  size_t failedAttempts = 0;

  // Use TypeSwitch to handle both Conv2dOp and ConvTranspose2dOp
  bool foundConfig =
      llvm::TypeSwitch<Operation *, bool>(operation)
          .Case<ttnn::Conv2dOp>([&](ttnn::Conv2dOp convOp) {
            // Conv2dOp supports all slice configs
            llvm::SmallVector<Conv2dSliceType> sliceConfigsToTry = {
                Conv2dSliceType::L1Full, Conv2dSliceType::DramWidth,
                Conv2dSliceType::DramHeight};

            for (Conv2dSliceType sliceType : sliceConfigsToTry) {
              // Use l1SearchSpace for L1Full (includes 0), dramSearchSpace
              // for DRAM slices (excludes 0)
              const Conv2dConfigSearchSpace &currentSearchSpace =
                  (sliceType == Conv2dSliceType::L1Full) ? l1SearchSpace
                                                         : dramSearchSpace;

              auto conv2dSliceConfig = Conv2dSliceConfigAttr::get(
                  convOp->getContext(), sliceType, 0);
              convOp.setConv2dSliceConfigAttr(conv2dSliceConfig);

              Conv2dConfigGenerator configGenerator(
                  &convOp, baseConfig, currentSearchSpace, filterOutFn);

              TTMLIR_TRACE(ttmlir::LogComponent::OpValidation,
                           "Trying slice config: {} with {} act_block_h values",
                           stringifyEnum(sliceType),
                           currentSearchSpace.actBlockHOverride.size());

              // Iterate through generated configs
              while (Conv2dConfigAttr configAttr =
                         configGenerator.getNextConfig()) {
                // Test this config
                OpConfig testConfig = originalConfig;
                auto &testConv2dAttrs =
                    std::get<Conv2dAttrs>(testConfig.opSpecificAttrs);
                testConv2dAttrs.conv2dConfig = configAttr;

                auto result = testFallbackCombination(operation, testConfig,
                                                      originalInputLayouts);

                if (result.isSuccess()) {
                  workingConfig = configAttr;
                  workingResult = result;
                  TTMLIR_DEBUG(
                      ttmlir::LogComponent::OpValidation,
                      "Found working config with slice type {} after {} "
                      "failed attempts",
                      stringifyEnum(sliceType), failedAttempts);
                  return true;
                }

                failedAttempts++;
                TTMLIR_TRACE(ttmlir::LogComponent::OpValidation,
                             "Config fallback failed (status: {}): {}",
                             static_cast<int>(result.status),
                             result.errorMessage);

                // Check if we've exceeded the maximum attempts (if limit is
                // set)
                if (maxAttempts > 0 &&
                    failedAttempts >= static_cast<size_t>(maxAttempts)) {
                  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                               "Reached maximum fallback attempts ({}) for "
                               "operation {} at {}. Terminating early.",
                               maxAttempts, operation->getName(),
                               operation->getLoc());
                  return false;
                }
              }
            }
            TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                         "No working config found after {} failed attempts",
                         failedAttempts);
            return false;
          })
          // ConvTranspose2dOp doesn't support slice config attribute yet
          // TODO(bmalesevic, #6639): Move ConvTranspose2dOp to case above
          // when slice config attribute is supported
          .Case<ttnn::ConvTranspose2dOp>([&](ttnn::ConvTranspose2dOp convOp) {
            Conv2dConfigGenerator configGenerator(&convOp, baseConfig,
                                                  l1SearchSpace, filterOutFn);

            TTMLIR_TRACE(ttmlir::LogComponent::OpValidation,
                         "Trying config fallback with {} act_block_h values",
                         l1SearchSpace.actBlockHOverride.size());

            // Iterate through generated configs
            while (Conv2dConfigAttr configAttr =
                       configGenerator.getNextConfig()) {
              // Test this config
              OpConfig testConfig = originalConfig;
              auto &testConv2dAttrs =
                  std::get<Conv2dAttrs>(testConfig.opSpecificAttrs);
              testConv2dAttrs.conv2dConfig = configAttr;

              auto result = testFallbackCombination(operation, testConfig,
                                                    originalInputLayouts);

              if (result.isSuccess()) {
                workingConfig = configAttr;
                workingResult = result;
                TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                             "Found working config after {} failed attempts",
                             failedAttempts);
                return true;
              }

              failedAttempts++;
              TTMLIR_TRACE(ttmlir::LogComponent::OpValidation,
                           "Config fallback failed (status: {}): {}",
                           static_cast<int>(result.status),
                           result.errorMessage);

              // Check if we've exceeded the maximum attempts (if limit is
              // set)
              if (maxAttempts > 0 &&
                  failedAttempts >= static_cast<size_t>(maxAttempts)) {
                TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                             "Reached maximum fallback attempts ({}) for "
                             "operation {} at {}. Terminating early.",
                             maxAttempts, operation->getName(),
                             operation->getLoc());
                return false;
              }
            }
            TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                         "No working config found after {} failed attempts",
                         failedAttempts);
            return false;
          })
          .Default([](Operation *) { return false; });

  if (!foundConfig) {
    return false;
  }

  if (workingConfig) {
    // Found a working config, apply it
    applyConfigChange(operation, workingConfig);

    if (originalConfig.outputLayout &&
        workingResult.firstActualOutputLayout != originalConfig.outputLayout) {
      applyOutputLayoutRevert(operation, 0,
                              workingResult.firstActualOutputLayout,
                              originalConfig.outputLayout);
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "Found working config fallback for operation {} at {}",
                 operation->getName(), operation->getLoc());
    return true;
  }

  return false;
}

// Apply config change to the operation
void applyConfigChange(Operation *operation, Conv2dConfigAttr newConfig) {
  if (auto conv2dOp = mlir::dyn_cast<ttnn::Conv2dOp>(operation)) {
    conv2dOp.setConv2dConfigAttr(newConfig);
  } else if (auto convTranspose2dOp =
                 mlir::dyn_cast<ttnn::ConvTranspose2dOp>(operation)) {
    convTranspose2dOp.setConv2dConfigAttr(newConfig);
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
               "Applied config change to operation {} at {}: new config = {}",
               operation->getName(), operation->getLoc(), newConfig);
}

} // namespace fallbacks

} // namespace mlir::tt::ttnn
