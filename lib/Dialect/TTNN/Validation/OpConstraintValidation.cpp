// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::ttnn {

namespace op_constraint_validation {

llvm::Expected<ValidationResult>
validateOperation(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                  const OpConfig &config, float tensorL1UsageCap) {

  // Call core constraint validation.
  auto constraintResult =
      validateConstraints(op, inputLayouts, config, tensorL1UsageCap);

  if (constraintResult) {
    TTNNLayoutAttr actualOutput = constraintResult.get();
    return ValidationResult(0, actualOutput);
  }
  return constraintResult.takeError();
}

llvm::Expected<std::vector<ValidationResult>> validateWithMultipleAttributes(
    Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
    llvm::ArrayRef<OpConfig> opConfigs,
    llvm::ArrayRef<OpConfig> referenceConfigs, float tensorL1UsageCap) {

  std::vector<ValidationResult> results;
  for (const auto &testConfig : opConfigs) {
    // 1. Call core constraint checking.
    auto constraintResult =
        validateConstraints(op, inputLayouts, testConfig, tensorL1UsageCap);

    if (constraintResult) {
      TTNNLayoutAttr actualOutput = constraintResult.get();

      // 2. Search referenceConfigs for matching (outputLayout +
      // opSpecificAttr).
      if (!referenceConfigs.empty()) {
        bool foundMatch = false;
        for (size_t i = 0; i < referenceConfigs.size(); ++i) {
          if (referenceConfigs[i].outputLayout == actualOutput &&
              referenceConfigs[i].opSpecificAttrs ==
                  testConfig.opSpecificAttrs) {
            results.push_back(ValidationResult(i, actualOutput));
            foundMatch = true;
            break;
          }
        }

        if (!foundMatch) {
          // No matching config found - return early with error
          return llvm::createStringError("No matching reference config found");
        }
      } else {
        // No reference configs to search - consider validation success as
        // match.
        results.push_back(ValidationResult(0, actualOutput));
      }
    } else {
      // Constraint checking failed - return early with error
      return constraintResult.takeError();
    }
  }

  return results;
}

llvm::Expected<TTNNLayoutAttr>
validateConstraints(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                    const OpConfig &config, float tensorL1UsageCap) {

  // Check that operation supports OpModel interface.
  auto backend = mlir::dyn_cast<OpModel>(op);
  if (!backend) {
    std::string errorMsg = "Backend constraints are not implemented for op " +
                           op->getName().getStringRef().str();
    llvm::report_fatal_error(llvm::Twine(errorMsg));
  }

  // Constraints are implemented for this op.
  auto deviceAttr = ttcore::lookupDevice(op);
  if (!deviceAttr) {
    return llvm::createStringError("No device attribute found for operation");
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
               "About to call getOpConstraints with {} input layouts",
               inputLayouts.size());

  for (size_t i = 0; i < inputLayouts.size(); ++i) {
    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "Input layout {}: {}, getLayout()={}, dtype={}", i,
                 inputLayouts[i], static_cast<int>(inputLayouts[i].getLayout()),
                 static_cast<int>(inputLayouts[i].getDataType()));
  }

  llvm::Expected<ttnn::op_model::OpConstraints> l1UsageExp =
      backend.getOpConstraints(inputLayouts, config);

  if (!l1UsageExp) {
    llvm::Error error = l1UsageExp.takeError();

    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "OpModel constraints failed: {} @ {} :: {}, "
                 "config.outputLayout: {}",
                 op->getName(), op->getLoc(),
                 llvm::toStringWithoutConsuming(error), config.outputLayout);

    return llvm::Expected<TTNNLayoutAttr>(std::move(error));
  }

  auto [cBUsagePeak, tensorUsage, peakMemoryUsage, outputTensorUsage,
        outputLayout] = l1UsageExp.get();

  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
               "Backend returned output layout: {}, layout={}, dtype={}",
               outputLayout, static_cast<int>(outputLayout.getLayout()),
               static_cast<int>(outputLayout.getDataType()));

  // Get usable L1 cache size from device.
  ttcore::SystemDescAttr systemDesc = mlir::cast<ttcore::SystemDescAttr>(
      op->getParentOfType<ModuleOp>()->getAttr(ttcore::SystemDescAttr::name));
  ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs()[0];
  uint64_t usableL1CacheSize = chipDesc.getUsableL1Size();

  // Calculate total L1 usage from all input layouts.
  uint64_t totalInputL1Usage = 0;
  for (const TTNNLayoutAttr &inputLayout : inputLayouts) {
    if (inputLayout.getBufferType() == BufferType::L1) {
      totalInputL1Usage += inputLayout.getShardSizeInBytes();
    }
  }

  // TODO(rpavlovicTT): switch to new constraints usage API once it is ready.
  // https://github.com/tenstorrent/tt-mlir/issues/4143
  bool l1UsageValid = (totalInputL1Usage + tensorUsage + cBUsagePeak) <
                      tensorL1UsageCap * usableL1CacheSize;

  if (!l1UsageValid) {
    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "Not enough L1 memory. OpModel constraints failed: {} "
                 "totalInputL1Usage: {}, tensorUsage: {}, cBUsagePeak: {}, "
                 "total: {}, limit: {}",
                 op->getName(), totalInputL1Usage, tensorUsage, cBUsagePeak,
                 totalInputL1Usage + tensorUsage + cBUsagePeak,
                 static_cast<uint64_t>(tensorL1UsageCap * usableL1CacheSize));
    return llvm::createStringError("Not enough L1 memory");
  }

  TTMLIR_DEBUG(
      ttmlir::LogComponent::OpValidation,
      "OpModel constraints valid. Op: {}\nOutputLayout: {}\n"
      "L1 usage: cBUsagePeak: {}, tensorUsage: {}, outputTensorUsage: {}, "
      "totalInputL1Usage: {}, totalL1Usage: {}",
      op->getName(), outputLayout, cBUsagePeak, tensorUsage, outputTensorUsage,
      totalInputL1Usage, cBUsagePeak + tensorUsage + totalInputL1Usage);

  return outputLayout;
}

} // namespace op_constraint_validation
} // namespace mlir::tt::ttnn
