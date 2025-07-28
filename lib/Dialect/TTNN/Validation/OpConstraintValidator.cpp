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

// Factory method implementation
OpConstraintValidator
OpConstraintValidator::create(const ValidationOptions &options) {
  return OpConstraintValidator(options);
}

// Private constructor implementation
OpConstraintValidator::OpConstraintValidator(const ValidationOptions &options)
    : options_(options) {}

OpConstraintValidator::ValidationResult
OpConstraintValidator::validateSingleConfig(
    Operation *op, const std::vector<TTNNLayoutAttr> &inputLayouts,
    const OpConfig &config) {

  // Extract the input operand for constraint checking
  Value inputOperand = extractInputOperand(op);
  if (!inputOperand) {
    return ValidationResult(false, 0, {}, "No valid input operand found");
  }

  // Use the first input layout for validation
  if (inputLayouts.empty()) {
    return ValidationResult(false, 0, {}, "No input layouts provided");
  }

  // Call core constraint validation
  auto constraintResult =
      validateConstraints(inputOperand, inputLayouts[0], op, config);

  if (constraintResult) {
    TTNNLayoutAttr actualOutput = constraintResult.get();
    return ValidationResult(true, 0, actualOutput, "");
  }
  std::string errorMsg = llvm::toString(constraintResult.takeError());
  return ValidationResult(false, 0, {}, errorMsg);
}

std::vector<OpConstraintValidator::ValidationResult>
OpConstraintValidator::validateWithMultipleAttributes(
    Operation *op, const TTNNLayoutAttr &inputLayout,
    const std::vector<OpConfig::OpSpecificAttrs> &opSpecificAttrs,
    const std::vector<OpConfig> &referenceConfigs) {

  std::vector<ValidationResult> results;

  // Extract the input operand for constraint checking
  Value inputOperand = extractInputOperand(op);
  if (!inputOperand) {
    // Return failure for all attributes if no valid input operand
    for (size_t i = 0; i < opSpecificAttrs.size(); ++i) {
      results.push_back(
          ValidationResult(false, 0, {}, "No valid input operand found"));
    }
    return results;
  }

  for (const auto &opSpecificAttr : opSpecificAttrs) {
    // 1. Create test config with current op-specific attribute
    OpConfig testConfig(nullptr, opSpecificAttr);

    // 2. Call core constraint checking
    auto constraintResult =
        validateConstraints(inputOperand, inputLayout, op, testConfig);

    if (constraintResult) {
      // 3. Search referenceConfigs for matching (outputLayout + opSpecificAttr)
      TTNNLayoutAttr actualOutput = constraintResult.get();

      bool foundMatch = false;
      for (size_t i = 0; i < referenceConfigs.size(); ++i) {
        if (referenceConfigs[i].outputLayout == actualOutput &&
            referenceConfigs[i].opSpecificAttrs == opSpecificAttr) {
          results.push_back(ValidationResult(true, i, actualOutput, ""));
          foundMatch = true;
          break;
        }
      }

      if (!foundMatch) {
        // No matching config found
        results.push_back(ValidationResult(false, 0, actualOutput,
                                           "No matching reference config"));
      }
    } else {
      // Constraint checking failed
      std::string errorMsg = llvm::toString(constraintResult.takeError());
      results.push_back(ValidationResult(false, 0, {}, errorMsg));
    }
  }

  return results;
}

std::vector<OpConstraintValidator::ValidationResult>
OpConstraintValidator::testFallbackTransforms(
    Operation *op, const TTNNLayoutAttr &originalInputLayout,
    const OpConfig &originalConfig,
    const std::vector<std::function<TTNNLayoutAttr(TTNNLayoutAttr)>>
        &transforms) {

  std::vector<ValidationResult> results;

  for (size_t i = 0; i < transforms.size(); ++i) {
    const auto &transform = transforms[i];
    TTNNLayoutAttr testInputLayout = transform(originalInputLayout);

    // For all fallbacks, don't constrain output layout - let backend decide
    OpConfig testConfig = originalConfig;
    testConfig.outputLayout =
        TTNNLayoutAttr{}; // Let backend choose output layout
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Fallback {}: allowing backend to choose output layout", i);

    auto result = validateSingleConfig(op, {testInputLayout}, testConfig);
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

llvm::Expected<TTNNLayoutAttr> OpConstraintValidator::validateConstraints(
    Value producerOperand, const TTNNLayoutAttr &producerLayout,
    Operation *consumerOp, const OpConfig &consumerConfig) {

  // This is the same constraint checking logic as ShardSolver's
  // checkShardCompatible Figure out this const based on exec data, but will be
  // replaced with API.
  constexpr float tensorL1UsageCap = 0.8;

  auto backend = mlir::dyn_cast<OpModel>(consumerOp);
  if (!backend) {
    std::string errorMsg = "Backend constraints are not implemented for op " +
                           consumerOp->getName().getStringRef().str();

    if (options_.fatalErrorOnUnsupportedOp) {
      llvm::report_fatal_error(llvm::Twine(errorMsg));
    }

    return llvm::createStringError(errorMsg);
  }

  // Constraints are implemented for this op.
  auto deviceAttr = ttcore::lookupDevice(consumerOp);
  if (!deviceAttr) {
    return llvm::createStringError("No device attribute found for operation");
  }

  // Map consumer operands to DRAM interleave or provided producerLayout
  // only one operand can be mapped to producerLayout, it's picked as first
  // operand matching producerOp output shape.

  uint32_t numOperands = consumerOp->getNumOperands();
  // Discard DPS operand since it's not used in runtime.
  if (llvm::isa<DestinationStyleOpInterface>(consumerOp)) {
    numOperands = numOperands - 1;
  }

  std::vector<TTNNLayoutAttr> inputLayouts;

  bool inputUnderCheckFound = false;
  for (uint32_t i = 0; i < numOperands; i++) {
    auto operand = consumerOp->getOperand(i);

    if (mlir::isa<TypedValue<mlir::tt::ttnn::DeviceType>>(operand)) {
      // Skip device type operand.
      continue;
    }

    if (operand == producerOperand) {
      // This is the input we are checking compatibility for.
      inputLayouts.push_back(producerLayout);
      inputUnderCheckFound = true;
      continue;
    }

    RankedTensorType input = mlir::cast<RankedTensorType>(operand.getType());
    auto layout = mlir::cast<TTNNLayoutAttr>(input.getEncoding());

    if (!layout) {
      return llvm::createStringError("Input operand must have a layout");
    }
    inputLayouts.push_back(layout);
  }

  if (!inputUnderCheckFound) {
    return llvm::createStringError("Input under check not found");
  }

  TTMLIR_DEBUG(
      ttmlir::LogComponent::Optimizer,
      "About to call getOpConstraints with inputLayouts[0]: {}, getLayout()={}",
      inputLayouts[0], static_cast<int>(inputLayouts[0].getLayout()));

  llvm::Expected<op_model::ttnn::OpConstraints> l1UsageExp =
      backend.getOpConstraints(inputLayouts, consumerConfig);

  if (!l1UsageExp) {
    llvm::Error error = l1UsageExp.takeError();

    TTMLIR_DEBUG(
        ttmlir::LogComponent::Optimizer,
        "OpModel constraints failed: {0}->{1} :: {2}, \nproducerLayout: {3}, "
        "\nconsumerLayout: {4}",
        producerOperand.getLoc(), consumerOp->getName(),
        llvm::toStringWithoutConsuming(error), producerLayout,
        consumerConfig.outputLayout);

    return llvm::Expected<TTNNLayoutAttr>(std::move(error));
  }

  auto [cBUsagePeak, tensorUsage, outputTensorUsage, outputLayout] =
      l1UsageExp.get();

  if (consumerConfig.outputLayout &&
      outputLayout != consumerConfig.outputLayout) {
    std::string message = "Output layout mismatch: backend returned layout "
                          "doesn't match requested consumer layout";
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "{}", message);
    return llvm::createStringError("[Optimizer] " + message);
  }

  // Get usable L1 cache size from device
  ttcore::SystemDescAttr systemDesc = mlir::cast<ttcore::SystemDescAttr>(
      consumerOp->getParentOfType<ModuleOp>()->getAttr(
          ttcore::SystemDescAttr::name));
  ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs()[0];
  uint64_t usableL1CacheSize = chipDesc.getUsableL1Size();

  // Only include producer L1 output usage if producer is actually in L1, not
  // DRAM
  uint64_t producerL1OutputUsage = 0;
  if (producerLayout.getBufferType() == BufferType::L1 ||
      producerLayout.getBufferType() == BufferType::L1Small) {
    producerL1OutputUsage = producerLayout.getShardSizeInBytes();
  }

  bool l1UsageValid = (producerL1OutputUsage + tensorUsage + cBUsagePeak) <
                      tensorL1UsageCap * usableL1CacheSize;

  if (!l1UsageValid) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Not enough L1 memory. OpModel constraints failed: {0}->{1} "
                 "\n producerLayout: {2}, outputLayout: {3}, l1Usage: {4}, "
                 "producerL1OutputUsage: {5} (bufferType: {6}), "
                 "tensorUsage: {7}, cBUsagePeak: {8}",
                 producerOperand.getLoc(), consumerOp->getName(),
                 producerLayout, outputLayout,
                 cBUsagePeak + tensorUsage + producerL1OutputUsage,
                 producerL1OutputUsage, producerLayout.getBufferType(),
                 tensorUsage, cBUsagePeak);
    return llvm::createStringError("Not enough L1 memory");
  }

  TTMLIR_DEBUG(
      ttmlir::LogComponent::Optimizer,
      "OpModel constraints valid. Producer: {0} -> Consumer: {1}\n"
      "ProducerLayout: {2}\nOutputLayout: {3}\n"
      "L1 usage: cBUsagePeak: {4}, tensorUsage: {5}, outputTensorUsage: {6}, "
      "producerL1OutputUsage: {7} (bufferType: {8}), totalL1Usage: {9}",
      producerOperand.getLoc(), consumerOp->getName(), producerLayout,
      outputLayout, cBUsagePeak, tensorUsage, outputTensorUsage,
      producerL1OutputUsage, producerLayout.getBufferType(),
      cBUsagePeak + tensorUsage + producerL1OutputUsage);

  return outputLayout;
}

Value OpConstraintValidator::extractInputOperand(Operation *op,
                                                 size_t operandIndex) {
  if (operandIndex >= op->getNumOperands()) {
    return nullptr;
  }

  Value operand = op->getOperand(operandIndex);

  // Skip device type operands
  if (mlir::isa<TypedValue<mlir::tt::ttnn::DeviceType>>(operand)) {
    // Try next operand
    if (operandIndex + 1 < op->getNumOperands()) {
      return extractInputOperand(op, operandIndex + 1);
    }
    return nullptr;
  }

  // Check if it's a ranked tensor type with TTNN layout
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType())) {
    if (mlir::isa<TTNNLayoutAttr>(tensorType.getEncoding())) {
      return operand;
    }
  }

  // Try next operand if current doesn't match
  if (operandIndex + 1 < op->getNumOperands()) {
    return extractInputOperand(op, operandIndex + 1);
  }

  return nullptr;
}

} // namespace mlir::tt::ttnn
