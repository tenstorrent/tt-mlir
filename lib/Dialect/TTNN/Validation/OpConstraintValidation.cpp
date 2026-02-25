// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
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

static ValidationResult
validateConstraints(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                    const OpConfig &config, uint64_t additionalL1Usage);

//----------- Public API implementations ----------

ValidationResult validateOperation(Operation *op,
                                   llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                                   const OpConfig &config,
                                   uint64_t additionalL1Usage) {
  return validateConstraints(op, inputLayouts, config, additionalL1Usage);
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

    TTNNLayoutAttr actualOutput = constraintResult.actualOutputLayout;

    // 2. Search referenceConfigs for matching (outputLayout + opSpecificAttr).
    if (!referenceConfigs.empty()) {
      bool foundMatch = false;
      for (size_t i = 0; i < referenceConfigs.size(); ++i) {
        if (referenceConfigs[i].outputLayout == actualOutput &&
            referenceConfigs[i].opSpecificAttrs == testConfig.opSpecificAttrs) {
          results.push_back(ValidationResult::success(i, actualOutput));
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
      results.push_back(ValidationResult::success(0, actualOutput));
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
          result = ValidationResult::metalBackendError(
              ttmlir::utils::firstNLines(errorMsg, 8));
        });
    return result;
  }

  auto [cbPeakUsage, l1BuffersPeakUsage, overallPeakL1Usage,
        outputTensorUsagePerCore, outputLayout] = constraints.get();

  const float tensorL1UsageCap = utils::getTensorL1UsageCap(contextOp);

  ttcore::SystemDescAttr systemDesc = mlir::cast<ttcore::SystemDescAttr>(
      contextOp->getParentOfType<ModuleOp>()->getAttr(
          ttcore::SystemDescAttr::name));
  ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs()[0];
  uint64_t usableL1CacheSize = chipDesc.getUsableL1Size();

  uint64_t effectiveL1Limit =
      static_cast<uint64_t>(tensorL1UsageCap * usableL1CacheSize);
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
               "OpModel constraints valid. OutputLayout: {}\n"
               "L1 usage: overallPeakL1Usage={}, cbPeakUsage={}, "
               "l1BuffersPeakUsage={}, outputTensorUsagePerCore={}",
               outputLayout, overallPeakL1Usage, cbPeakUsage,
               l1BuffersPeakUsage, outputTensorUsagePerCore);

  return ValidationResult::success(0, outputLayout, outputTensorUsagePerCore);
}

// ----------- Core constraint validation implementation ----------

static ValidationResult
validateConstraints(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                    const OpConfig &config, uint64_t additionalL1Usage) {

  // Check that operation supports OpModel interface.
  auto backend = mlir::dyn_cast<OpModel>(op);
  if (!backend) {
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
               "About to call getOpConstraints for {} with {} input layouts",
               ttmlir::opToString(op), inputLayouts.size());

  for (size_t i = 0; i < inputLayouts.size(); ++i) {
    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "Input layout {}: {}, getLayout()={}, dtype={}", i,
                 inputLayouts[i], static_cast<int>(inputLayouts[i].getLayout()),
                 static_cast<int>(inputLayouts[i].getDataType()));
  }
  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation, "Output config {}", config);

  llvm::Expected<ttnn::op_model::OpConstraints> l1UsageExp =
      backend.getOpConstraints(inputLayouts, config);

  return checkConstraintsResult(op, std::move(l1UsageExp), additionalL1Usage);
}

} // namespace op_constraint_validation
} // namespace mlir::tt::ttnn
