// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidator.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::ttnn {

OpConstraintValidator::OpConstraintValidator(const ValidationOptions &options)
    : options_(options) {}

OpConstraintValidator::ValidationResult
OpConstraintValidator::validateOperation(
    Operation *op, const std::vector<TTNNLayoutAttr> &inputLayouts,
    const OpConfig &config) {

  if (inputLayouts.empty()) {
    return ValidationResult(false, 0, {}, "No input layouts provided");
  }

  // Call core constraint validation.
  auto constraintResult = validateConstraints(op, inputLayouts, config);

  if (constraintResult) {
    TTNNLayoutAttr actualOutput = constraintResult.get();
    return ValidationResult(true, 0, actualOutput, "");
  }
  std::string errorMsg = llvm::toString(constraintResult.takeError());
  return ValidationResult(false, 0, {}, errorMsg);
}

std::vector<OpConstraintValidator::ValidationResult>
OpConstraintValidator::validateWithMultipleAttributes(
    Operation *op, const std::vector<TTNNLayoutAttr> &inputLayouts,
    const std::vector<OpConfig::OpSpecificAttrs> &opSpecificAttrs,
    const std::vector<OpConfig> *referenceConfigs) {

  std::vector<ValidationResult> results;

  if (inputLayouts.empty()) {
    // Return failure for all attributes if no input layouts.
    for (size_t i = 0; i < opSpecificAttrs.size(); ++i) {
      results.push_back(
          ValidationResult(false, 0, {}, "No input layouts provided"));
    }
    return results;
  }

  for (const auto &opSpecificAttr : opSpecificAttrs) {
    // 1. Create test config with current op-specific attribute.
    OpConfig testConfig(nullptr, opSpecificAttr);

    // 2. Call core constraint checking.
    auto constraintResult = validateConstraints(op, inputLayouts, testConfig);

    if (constraintResult) {
      TTNNLayoutAttr actualOutput = constraintResult.get();

      // 3. Search referenceConfigs for matching (outputLayout +
      // opSpecificAttr).
      if (referenceConfigs != nullptr) {
        bool foundMatch = false;
        for (size_t i = 0; i < referenceConfigs->size(); ++i) {
          if ((*referenceConfigs)[i].outputLayout == actualOutput &&
              (*referenceConfigs)[i].opSpecificAttrs == opSpecificAttr) {
            results.push_back(ValidationResult(true, i, actualOutput, ""));
            foundMatch = true;
            break;
          }
        }

        if (!foundMatch) {
          // No matching config found.
          results.push_back(ValidationResult(false, 0, actualOutput,
                                             "No matching reference config"));
        }
      } else {
        // No reference configs to search - consider validation success as
        // match.
        results.push_back(ValidationResult(true, 0, actualOutput, ""));
      }
    } else {
      // Constraint checking failed.
      std::string errorMsg = llvm::toString(constraintResult.takeError());
      results.push_back(ValidationResult(false, 0, {}, errorMsg));
    }
  }

  return results;
}

llvm::Expected<TTNNLayoutAttr> OpConstraintValidator::validateConstraints(
    Operation *op, const std::vector<TTNNLayoutAttr> &inputLayouts,
    const OpConfig &config) {

  // Check that operation supports OpModel interface.
  auto backend = mlir::dyn_cast<OpModel>(op);
  if (!backend) {
    std::string errorMsg = "Backend constraints are not implemented for op " +
                           op->getName().getStringRef().str();

    if (options_.fatalErrorOnUnsupportedOp) {
      llvm::report_fatal_error(llvm::Twine(errorMsg));
    }

    return llvm::createStringError(errorMsg);
  }

  // Constraints are implemented for this op.
  auto deviceAttr = ttcore::lookupDevice(op);
  if (!deviceAttr) {
    return llvm::createStringError("No device attribute found for operation");
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "About to call getOpConstraints with {} input layouts",
               inputLayouts.size());

  for (size_t i = 0; i < inputLayouts.size(); ++i) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Input layout {}: {}, getLayout()={}, dtype={}", i,
                 inputLayouts[i], static_cast<int>(inputLayouts[i].getLayout()),
                 static_cast<int>(inputLayouts[i].getDataType()));
  }

  llvm::Expected<ttnn::op_model::OpConstraints> l1UsageExp =
      backend.getOpConstraints(inputLayouts, config);

  if (!l1UsageExp) {
    llvm::Error error = l1UsageExp.takeError();

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "OpModel constraints failed: {} @ {} :: {}, "
                 "config.outputLayout: {}",
                 op->getName(), op->getLoc(),
                 llvm::toStringWithoutConsuming(error), config.outputLayout);

    return llvm::Expected<TTNNLayoutAttr>(std::move(error));
  }

  auto [cBUsagePeak, tensorUsage, outputTensorUsage, outputLayout] =
      l1UsageExp.get();

  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "Backend returned output layout: {}, layout={}, dtype={}",
               outputLayout, static_cast<int>(outputLayout.getLayout()),
               static_cast<int>(outputLayout.getDataType()));

  if (options_.compareOutputLayout && config.outputLayout &&
      outputLayout != config.outputLayout) {
    std::string message = "Output layout mismatch: backend returned layout "
                          "doesn't match requested config layout";
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "{}", message);
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Config output layout: {}, backend output layout: {}",
                 config.outputLayout, outputLayout);
    return llvm::createStringError("[Optimizer] " + message);
  }

  // Get usable L1 cache size from device.
  ttcore::SystemDescAttr systemDesc = mlir::cast<ttcore::SystemDescAttr>(
      op->getParentOfType<ModuleOp>()->getAttr(ttcore::SystemDescAttr::name));
  ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs()[0];
  uint64_t usableL1CacheSize = chipDesc.getUsableL1Size();

  constexpr float tensorL1UsageCap = 0.8;

  // Calculate total L1 usage from all input layouts.
  uint64_t totalInputL1Usage = 0;
  for (const TTNNLayoutAttr &inputLayout : inputLayouts) {
    if (inputLayout.getBufferType() == BufferType::L1 ||
        inputLayout.getBufferType() == BufferType::L1Small) {
      totalInputL1Usage += inputLayout.getShardSizeInBytes();
    }
  }

  bool l1UsageValid = (totalInputL1Usage + tensorUsage + cBUsagePeak) <
                      tensorL1UsageCap * usableL1CacheSize;

  if (!l1UsageValid) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Not enough L1 memory. OpModel constraints failed: {} "
                 "totalInputL1Usage: {}, tensorUsage: {}, cBUsagePeak: {}, "
                 "total: {}, limit: {}",
                 op->getName(), totalInputL1Usage, tensorUsage, cBUsagePeak,
                 totalInputL1Usage + tensorUsage + cBUsagePeak,
                 static_cast<uint64_t>(tensorL1UsageCap * usableL1CacheSize));
    return llvm::createStringError("Not enough L1 memory");
  }

  TTMLIR_DEBUG(
      ttmlir::LogComponent::Optimizer,
      "OpModel constraints valid. Op: {}\nOutputLayout: {}\n"
      "L1 usage: cBUsagePeak: {}, tensorUsage: {}, outputTensorUsage: {}, "
      "totalInputL1Usage: {}, totalL1Usage: {}",
      op->getName(), outputLayout, cBUsagePeak, tensorUsage, outputTensorUsage,
      totalInputL1Usage, cBUsagePeak + tensorUsage + totalInputL1Usage);

  return outputLayout;
}

} // namespace mlir::tt::ttnn
