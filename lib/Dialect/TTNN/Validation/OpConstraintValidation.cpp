// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNTraits.h"
#include "ttmlir/Dialect/TTNN/Interfaces/OpModelError.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::ttnn {

namespace op_constraint_validation {

llvm::StringRef validationStatusToString(ValidationStatus status) {
  switch (status) {
  case ValidationStatus::Success:
    return "Success";
  case ValidationStatus::NotImplemented:
    return "NotImplemented";
  case ValidationStatus::MetalBackendError:
    return "MetalBackendError";
  case ValidationStatus::UnmatchedReferenceConfig:
    return "UnmatchedReferenceConfig";
  case ValidationStatus::OutOfMemoryError:
    return "OutOfMemoryError";
  }
  return "Unknown";
}

static ValidationResult validateConstraints(
    Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
    const OpConfig &config, uint64_t additionalL1Usage, bool useState = false,
    llvm::ArrayRef<op_model::OpModelAllocationRecord> liveRecords = {});

//----------- Public API implementations ----------

ValidationResult validateOperation(Operation *op,
                                   llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                                   const OpConfig &config,
                                   uint64_t additionalL1Usage) {
  return validateConstraints(op, inputLayouts, config, additionalL1Usage);
}

ValidationResult
validateOperation(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                  const OpConfig &config,
                  llvm::ArrayRef<op_model::OpModelAllocationRecord> liveRecords,
                  uint64_t additionalL1Usage) {
  return validateConstraints(op, inputLayouts, config, additionalL1Usage,
                             /*useState=*/true, liveRecords);
}

std::vector<ValidationResult>
validateWithMultipleAttributes(Operation *op,
                               llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                               llvm::ArrayRef<OpConfig> opConfigs,
                               llvm::ArrayRef<OpConfig> referenceConfigs) {

  std::vector<ValidationResult> results;
  for (const auto &testConfig : opConfigs) {
    // 1. Call core constraint checking.
    ValidationResult constraintResult = validateConstraints(
        op, inputLayouts, testConfig, /*additionalL1Usage=*/0);

    // If not supported, backend error, or validation error - add to results
    // and continue (don't fail early, collect all results)
    if (!constraintResult.isSuccess()) {
      results.push_back(constraintResult);
      continue;
    }

    // TODO(bmalesevic, #7108): propagate all output layouts once multi-output
    // matching is supported.
    const auto firstActualOutputLayout =
        constraintResult.checkAndGetFirstActualOutputLayout();

    // 2. Search referenceConfigs for matching (outputLayout + opSpecificAttr).
    if (!referenceConfigs.empty()) {
      bool foundMatch = false;
      for (size_t i = 0; i < referenceConfigs.size(); ++i) {
        if (referenceConfigs[i].outputLayout == firstActualOutputLayout &&
            referenceConfigs[i].opSpecificAttrs == testConfig.opSpecificAttrs) {
          results.push_back(
              ValidationResult::success(i, firstActualOutputLayout));
          foundMatch = true;
          break;
        }
      }

      if (!foundMatch) {
        results.push_back(ValidationResult::unmatchedReferenceConfig(
            "No matching reference config found"));
      }
    } else {
      // No reference configs to search - consider validation success as match.
      results.push_back(ValidationResult::success(0, firstActualOutputLayout));
    }
  }

  return results;
}

// ----------- Shared L1 budget check ----------

ValidationResult
checkConstraintsResult(Operation *contextOp,
                       llvm::Expected<op_model::OpConstraints> constraints,
                       uint64_t additionalL1Usage) {
  if (!constraints) {
    ValidationResult result;
    llvm::handleAllErrors(
        constraints.takeError(),
        [&](ttnn::detail::OpNotSupportedError &notSupportedErr) {
          result = ValidationResult::notImplemented(notSupportedErr.message());
        },
        [&](llvm::ErrorInfoBase &otherErr) {
          std::string errorMsg = otherErr.message();
          TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                       "OpModel constraints failed: {}",
                       ttmlir::utils::firstNLines(errorMsg, 8));
          // The stateful (build-from-records) query places the currently-live
          // tensors at real addresses, so an op that does not fit surfaces as a
          // tt-metal allocator exception ("Out of Memory: Not enough space
          // ...") from the backend rather than via the peak-usage budget check
          // below. Classify that as OOM (not a hard backend error) so the L1
          // spill pass takes the evict-and-refit recovery -- the same path the
          // scalar tracker's soft OOM takes. Demoting straight to DRAM instead
          // skips config fallback and can leave a numerically-wrong op config
          // (https://github.com/tenstorrent/tt-mlir/issues/9045). A genuine
          // backend constraint (unsupported config, etc.) carries no "Out of
          // Memory" marker and still routes to metalBackendError.
          if (errorMsg.find("Out of Memory") != std::string::npos) {
            result = ValidationResult::outOfMemoryError(
                ttmlir::utils::firstNLines(errorMsg, 8));
          } else {
            result = ValidationResult::metalBackendError(
                ttmlir::utils::firstNLines(errorMsg, 8));
          }
        });
    return result;
  }

  auto [cbPeakUsage, l1BuffersPeakUsage, overallPeakL1Usage,
        outputTensorUsagePerCore, outputLayouts, outputAllocations] =
      constraints.get();

  uint64_t effectiveL1Limit = utils::getUsableL1PerCore(contextOp);
  uint64_t totalL1Usage = overallPeakL1Usage + additionalL1Usage;

  if (totalL1Usage > effectiveL1Limit) {
    TTMLIR_DEBUG(
        ttmlir::LogComponent::OpValidation,
        "Not enough L1 memory. "
        "totalL1Usage: {} [overallPeakL1Usage={}, additionalL1Usage={}]"
        " [cbPeakUsage={}, l1BuffersPeakUsage={}] limit: {}",
        totalL1Usage, overallPeakL1Usage, additionalL1Usage, cbPeakUsage,
        l1BuffersPeakUsage, effectiveL1Limit);
    return ValidationResult::outOfMemoryError("Not enough L1 memory");
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
               "OpModel constraints valid. FirstOutputLayout: {}\n"
               "L1 usage: overallPeakL1Usage={}, cbPeakUsage={}, "
               "l1BuffersPeakUsage={}, outputTensorUsagePerCore={}",
               outputLayouts.empty() ? nullptr : outputLayouts[0],
               overallPeakL1Usage, cbPeakUsage, l1BuffersPeakUsage,
               outputTensorUsagePerCore);

  ValidationResult result = ValidationResult::success(
      0, outputLayouts, outputTensorUsagePerCore, cbPeakUsage);
  result.outputAllocations = std::move(outputAllocations);
  return result;
}

// ----------- Core constraint validation implementation ----------

static ValidationResult validateConstraints(
    Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
    const OpConfig &config, uint64_t additionalL1Usage, bool useState,
    llvm::ArrayRef<op_model::OpModelAllocationRecord> liveRecords) {

  // Check that operation supports OpModel interface.
  auto backend = mlir::dyn_cast<OpModel>(op);
  if (!backend) {
    // Ops marked with the OpModelExempt trait deliberately do not implement
    // the OpModel interface (e.g. CCL/multi-device, trace, generic, or other
    // ops without a metal-side definition). The optimizer relies on observing
    // a NotImplemented result for such ops so it can fall back gracefully
    // (e.g. evict L1 state) instead of treating the op as analyzable.
    if (op->hasTrait<OpModelExempt>()) {
      return ValidationResult::notImplemented(
          (llvm::Twine("OpModel interface not implemented for op ") +
           op->getName().getStringRef() + " (OpModelExempt)")
              .str());
    }
    llvm::reportFatalInternalError(llvm::Twine("Backend constraints are not "
                                               "implemented for op ")
                                       .concat(op->getName().getStringRef()));
  }

  // Constraints are implemented for this op.
  auto deviceAttr = ttcore::lookupDevice(op);
  if (!deviceAttr) {
    llvm::reportFatalInternalError(
        llvm::Twine("No device attribute found for operation ")
            .concat(op->getName().getStringRef()));
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
               "About to call getOpConstraints for {} with {} input layouts, "
               "additionalL1={}",
               ttmlir::opToString(op), inputLayouts.size(), additionalL1Usage);

  for (size_t i = 0; i < inputLayouts.size(); ++i) {
    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "Input layout {}: {}, getLayout()={}, dtype={}", i,
                 inputLayouts[i], static_cast<int>(inputLayouts[i].getLayout()),
                 static_cast<int>(inputLayouts[i].getDataType()));
  }
  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation, "Output config {}", config);

  // Stateful path (L1 spill): route through the uncached
  // getOpConstraintsWithState so the query is evaluated against the live
  // allocation set. Stateless path (beam search): the cached getOpConstraints.
  llvm::Expected<ttnn::op_model::OpConstraints> l1UsageExp =
      useState
          ? backend.getOpConstraintsWithState(inputLayouts, config, liveRecords)
          : backend.getOpConstraints(inputLayouts, config);

  return checkConstraintsResult(op, std::move(l1UsageExp), additionalL1Usage);
}

} // namespace op_constraint_validation
} // namespace mlir::tt::ttnn
