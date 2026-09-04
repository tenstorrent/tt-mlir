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
    const OpConfig &config, uint64_t additionalL1Usage,
    std::optional<llvm::ArrayRef<op_model::OpModelAllocationRecord>>
        liveRecords = std::nullopt);

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
                             liveRecords);
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
                       uint64_t additionalL1Usage, bool statefulQuery) {
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
          // tt-metal exception from the backend rather than via the peak-usage
          // budget check below. Two flavors are really L1-pressure conditions,
          // both recoverable by the L1 spill pass's evict-and-refit path (the
          // same path the scalar tracker's soft OOM takes):
          //   1. Allocator exhaustion: "Out of Memory: Not enough space ...".
          //   2. A CB-vs-L1 overlap: "Statically allocated circular buffers ...
          //      clash with L1 buffers ..." -- the op's static circular-buffer
          //      region collides with a still-live L1 input (e.g. a large
          //      height-sharded conv activation). Evicting that L1 input to
          //      DRAM and refitting resolves it.
          // Classify both as OOM so run() calls handleOOM (which spills the
          // offending L1 input) instead of the metalBackendError branch, which
          // only demotes the op's *output* to DRAM -- useless here, since the
          // clash is with the input -- leaving a layout that throws at runtime
          // (https://github.com/tenstorrent/tt-mlir/issues/9064). Demoting
          // straight to DRAM also skips config fallback and can leave a
          // numerically-wrong op config
          // (https://github.com/tenstorrent/tt-mlir/issues/9045). A genuine
          // backend constraint (unsupported config, etc.) carries neither
          // marker and still routes to metalBackendError.
          // Metal 2.0 program validation currently looks up a WORKER dispatch
          // grid for mock devices even when placement used the mock device's
          // ETH dispatch grid. Treat only that precise mock-only validation
          // failure as unsupported until the grids use the same dispatch
          // configuration. See
          // https://github.com/tenstorrent/tt-mlir/issues/9235.
          const bool mockProgramSpecGridMismatch =
              op_model::isMockDevice() &&
              errorMsg.find("WorkUnitSpec '") != std::string::npos &&
              errorMsg.find("targets node (") != std::string::npos &&
              errorMsg.find("which is out of bounds") != std::string::npos &&
              errorMsg.find("The compute worker grid on this device is ") !=
                  std::string::npos;
          if (mockProgramSpecGridMismatch) {
            result = ValidationResult::notImplemented(
                ttmlir::utils::firstNLines(errorMsg, 8));
          } else if (errorMsg.find("Out of Memory") != std::string::npos ||
                     errorMsg.find("clash with L1 buffers") !=
                         std::string::npos) {
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

  // Stateless-only L1 capacity model: both graph captures run NO_DISPATCH, so
  // nothing is allocated and only this comparison keeps the beam search off
  // illegal L1 layouts. The stateful path skips it -- tt-metal decides fit
  // there, and MockAllocatorL1Tracker::validate's ceiling owns the byte budget.
  if (!statefulQuery) {
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
    const OpConfig &config, uint64_t additionalL1Usage,
    std::optional<llvm::ArrayRef<op_model::OpModelAllocationRecord>>
        liveRecords) {

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

  // One query, one selector: an engaged `liveRecords` (L1 spill) evaluates the
  // op against the live allocation set through an uncached query; std::nullopt
  // (beam search) takes the cached stateless query. `statefulQuery` is DERIVED
  // from the same optional rather than passed alongside it, so the two can
  // never disagree -- a stateful capacity model paired with a stateless query
  // would leave the L1 byte budget unchecked on both sides.
  llvm::Expected<ttnn::op_model::OpConstraints> l1UsageExp =
      backend.getOpConstraints(inputLayouts, config, liveRecords);

  return checkConstraintsResult(op, std::move(l1UsageExp), additionalL1Usage,
                                /*statefulQuery=*/liveRecords.has_value());
}

} // namespace op_constraint_validation
} // namespace mlir::tt::ttnn
