// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/L1InterleavedFallbackAnalysis.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace mlir::tt::ttnn {

void L1InterleavedFallbackAnalysis::analysisImplementation() {
  // Go through schedule in order using walk, trying to upgrade DRAM ops to L1
  // interleaved.
  analysisInput.funcOp->walk([&](Operation *op) {
    // Skip operations that have the row-major workaround later on in Optimizer.
    // TODO(bmalesevic,#3985): remove after this is fixed
    if (isa<ttnn::MaxPool2dOp, ttnn::UpsampleOp>(op)) {
      return;
    }

    // Skip operations that have DRAM output in runtime even when configured as
    // L1 via this analysis.
    // TODO(bmalesevic,#4505): remove this after they are supported in runtime
    if (isa<ttnn::SliceStaticOp, ttnn::TypecastOp>(op)) {
      return;
    }
    // Skip Matmul and Linear output, inefficient for L1 interleaved.
    if (isa<ttnn::MatmulOp, ttnn::LinearOp>(op)) {
      return;
    }
    // Skip output of Conv2D that uses matmul under the hood, inefficient for L1
    // interleaved.
    if (utils::isConv2DConvertibleToMatMul(op)) {
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "L1InterleavedFallbackAnalysis: Skipping {} - Conv2D "
                   "uses matmul, inefficient for L1 interleaved.",
                   op->getName());
      return;
    }

    for (auto *user : op->getUsers()) {
      // Skip operations that have DRAM input in runtime even when configured as
      // L1 via this analysis.
      // TODO(bmalesevic,#4505): remove this after they are supported in runtime
      if (isa<ttnn::SliceStaticOp, ttnn::TypecastOp>(user)) {
        return;
      }
      // Skip Matmul and Linear input, inefficient for L1 interleaved.
      if (isa<ttnn::MatmulOp, ttnn::LinearOp>(user)) {
        return;
      }
      // Skip input of Conv2D that uses matmul under the hood, inefficient for
      // L1 interleaved.
      if (utils::isConv2DConvertibleToMatMul(user)) {
        TTMLIR_TRACE(
            ttmlir::LogComponent::Optimizer,
            "L1InterleavedFallbackAnalysis: Skipping {} - Consumer Conv2D "
            "uses matmul, inefficient for L1 interleaved.",
            op->getName());
        return;
      }
    }

    // Skip if operation doesn't have L1 interleaved layout available.
    if (!hasL1InterleavedLegalLayout(op)) {
      return;
    }
    // Skip if operation doesn't use DRAM layout.
    if (!utils::producesDRAMLayout(op)) {
      return;
    }
    // Skip if operation doesn't have exactly one user.
    if (!op->hasOneUse()) {
      return;
    }
    // Skip if producer is not immediately consumed by consumer.
    if (!hasImmediateConsumer(op)) {
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "L1InterleavedFallbackAnalysis: Skipping {} - "
                   "consumer not scheduled immediately after.",
                   op->getName());
      return;
    }
    // Skip if operation output is returnOp input, no point in storing in L1, no
    // acceleration would happen but additional memory management risks would be
    // introduced.
    if (hasReturnOpUser(op)) {
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "L1InterleavedFallbackAnalysis: Skipping {} - output is "
                   "returnOp input.",
                   op->getName());
      return;
    }
    tryUpgradeToL1Interleaved(op);
  });
  TTMLIR_TRACE(
      ttmlir::LogComponent::Optimizer,
      "L1InterleavedFallbackAnalysis: Completed - upgraded {} operations.",
      analysisResult.upgradedConfigs.size());
}

bool L1InterleavedFallbackAnalysis::hasL1InterleavedLegalLayout(
    Operation *op) const {
  return analysisInput.legalL1InterleavedConfigs.find(op) !=
         analysisInput.legalL1InterleavedConfigs.end();
}

std::vector<OpConfig>
L1InterleavedFallbackAnalysis::getL1InterleavedLayoutConfigs(
    Operation *op) const {
  const auto it = analysisInput.legalL1InterleavedConfigs.find(op);
  assert(it != analysisInput.legalL1InterleavedConfigs.end());
  return it->second;
}

bool L1InterleavedFallbackAnalysis::hasImmediateConsumer(Operation *op) const {
  // Get the single user.
  Operation *userOp = *op->getUsers().begin();
  Operation *consumerOp = op->getNextNode();

  // User must not be scheduled at an earlier index than its operand.
  assert(consumerOp);

  // Check if the user is the immediate next operation in schedule.
  return consumerOp == userOp;
}

bool L1InterleavedFallbackAnalysis::hasReturnOpUser(Operation *op) const {
  // Check users to see if any is a return op.
  for (Operation *userOp : op->getUsers()) {
    if (isa<mlir::func::ReturnOp>(userOp)) {
      return true;
    }
  }
  return false;
}

void L1InterleavedFallbackAnalysis::tryUpgradeToL1Interleaved(Operation *op) {
  std::vector<OpConfig> opL1InterleavedConfigs =
      getL1InterleavedLayoutConfigs(op);

  bool isCurrentlyTiled = utils::producesTiledTensorLayout(op);

  // Partition configs to prioritize those matching current tiling preference.
  std::partition(opL1InterleavedConfigs.begin(), opL1InterleavedConfigs.end(),
                 [isCurrentlyTiled](const OpConfig &config) {
                   bool configTiled = config.outputLayout.isTiled();
                   return configTiled == isCurrentlyTiled;
                 });

  // Try both L1 interleaved configs until one works if there are multiple
  // (rowMajor and tiled).
  for (auto opL1InterleavedConfig : opL1InterleavedConfigs) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "=== Start of debug dump for op {} ===",
                 op->getName().getStringRef().data());
    llvm::Expected<TTNNLayoutAttr> possibleL1Layout =
        checkUpgradeToL1Interleaved(op, opL1InterleavedConfig,
                                    /*upgradedProducerOp=*/nullptr,
                                    /*upgradedProducerLayout=*/nullptr);

    if (!possibleL1Layout) {
      llvm::Error error = possibleL1Layout.takeError();
      std::string errorStr = llvm::toString(std::move(error));
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "L1InterleavedFallbackAnalysis: Invalid upgrade, error: {}",
                   errorStr);
      continue;
    }
    TTNNLayoutAttr l1InterleavedLayout = possibleL1Layout.get();
    assert(l1InterleavedLayout == opL1InterleavedConfig.outputLayout &&
           "Expected output layout to match the one in OpConfig");
    analysisResult.upgradedConfigs[op] = opL1InterleavedConfig;
    break;
  }
}

llvm::Expected<TTNNLayoutAttr>
L1InterleavedFallbackAnalysis::checkUpgradeToL1Interleaved(
    Operation *consumerOp, const OpConfig &consumerConfig,
    const Operation *upgradedProducerOp,
    const TTNNLayoutAttr upgradedProducerLayout) const {

  OpModel backend = mlir::dyn_cast<OpModel>(consumerOp);
  if (!backend) {
    // This function should not be called for ops without backend constraints.
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Backend constraints are not implemented for op %s",
        consumerOp->getName().getStringRef().data());
  }

  // Verify device attribute exists (will assert if not found).
  mlir::tt::ttcore::lookupDevice(consumerOp);

  uint32_t numOperands = consumerOp->getNumOperands();
  std::vector<TTNNLayoutAttr> inputLayouts;
  inputLayouts.reserve(numOperands);
  uint64_t producersL1OutputUsage = 0;

  for (uint32_t i = 0; i < numOperands; i++) {
    auto operand = consumerOp->getOperand(i);

    if (mlir::isa<TypedValue<mlir::tt::ttnn::DeviceType>>(operand)) {
      // Skip device type operand.
      continue;
    }

    if (operand.getDefiningOp()) {
      if (operand.getDefiningOp() == upgradedProducerOp) {
        // If it's a nested check of update candidate's (producer in this scope)
        // consumer's storage.
        inputLayouts.push_back(upgradedProducerLayout);
        continue;
      }
      auto it = analysisResult.upgradedConfigs.find(operand.getDefiningOp());
      if (it != analysisResult.upgradedConfigs.end()) {
        inputLayouts.push_back(it->second.outputLayout);
        continue;
      }
    }

    RankedTensorType input = mlir::cast<RankedTensorType>(operand.getType());

    auto layout = mlir::cast<TTNNLayoutAttr>(input.getEncoding());

    assert(layout && "Input operand must have a layout");
    inputLayouts.push_back(layout);
  }

  for (const auto &inputLayout : inputLayouts) {
    producersL1OutputUsage += utils::getOpOutputL1Usage(inputLayout);
  }

  llvm::Expected<op_model::OpConstraints> l1UsageExp =
      backend.getOpConstraints(inputLayouts, consumerConfig);

  if (!l1UsageExp) {
    llvm::Error error = l1UsageExp.takeError();
    std::string errorStr = llvm::toString(std::move(error));

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "OpModel constraints failed for op {0} :: {1},"
                 "\nconsumerLayout: {2}",
                 consumerOp->getName(), ttmlir::utils::firstNLines(errorStr, 4),
                 consumerConfig.outputLayout);

    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "OpModel constraints failed for op %s.",
                                   consumerOp->getName().getStringRef().data());
  }

  auto [cBUsagePeak, tensorUsage, outputTensorUsage, outputLayout] =
      l1UsageExp.get();

  if (outputLayout != consumerConfig.outputLayout) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Output layout mismatch for op {0}:"
                 "\nexpected: {1},\nactual: {2},",
                 consumerOp->getName(), consumerConfig.outputLayout,
                 outputLayout);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Output layout mismatch for op %s.",
                                   consumerOp->getName().getStringRef().data());
  }

  bool l1UsageValid = (producersL1OutputUsage + tensorUsage + cBUsagePeak) <
                      analysisInput.usableL1CacheSize;

  if (!l1UsageValid) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Not enough L1 memory. OpModel constraints failed: {0} "
                 "\n outputLayout: {1}, l1Usage: {2}, "
                 "producerL1OutputUsage: {3}, tensorUsage: {4}, "
                 "outputTensorUsage: {5}, cBUsagePeak: {6}",
                 consumerOp->getName(), outputLayout,
                 cBUsagePeak + tensorUsage + producersL1OutputUsage,
                 producersL1OutputUsage, tensorUsage, outputTensorUsage,
                 cBUsagePeak);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Not enough L1 memory for op %s.",
                                   consumerOp->getName().getStringRef().data());
  }
  // Check if upgrading this operation would cause memory conflicts with its
  // consumer. This is a single-level recursion check:
  // - First call: upgradedProducerOp=nullptr, checks if current op can be
  // upgraded.
  // - Recursive call: upgradedProducerOp=consumerOp, checks if consumer can
  // handle the upgrade.
  if (upgradedProducerOp) {
    TTMLIR_DEBUG(
        ttmlir::LogComponent::Optimizer,
        "OpModel constraints valid for input of consumer {0}:\n"
        "OutputLayout: {1}\n"
        "L1 usage: cBUsagePeak: {2}, tensorUsage: {3}, outputTensorUsage: {4}, "
        "producerL1OutputUsage: {5}, totalL1Usage: {6}",
        consumerOp->getName(), outputLayout, cBUsagePeak, tensorUsage,
        outputTensorUsage, producersL1OutputUsage,
        cBUsagePeak + tensorUsage + producersL1OutputUsage);

    return outputLayout;
  }

  assert(consumerOp->hasOneUse() && "Consumer must have exactly one user");
  Operation *nextConsumerOp = *consumerOp->getUsers().begin();
  assert(nextConsumerOp && "Operation must have a consumer");
  // If next consumer has TTNN layout output encoding, verify both operations
  // can coexist in L1.
  if (utils::producesTTNNLayoutEncoding(nextConsumerOp)) {
    const OpConfig &nextConsumerOpConfig =
        analysisInput.currentConfigs.at(nextConsumerOp);

    llvm::Expected<TTNNLayoutAttr> nextConsumerOpL1Layout =
        checkUpgradeToL1Interleaved(nextConsumerOp, nextConsumerOpConfig,
                                    consumerOp, outputLayout);

    if (!nextConsumerOpL1Layout) {
      llvm::Error error = nextConsumerOpL1Layout.takeError();
      std::string errorStr = llvm::toString(std::move(error));
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "L1 upgrade blocked for {} - for consumer {}: {}",
                   consumerOp->getName().getStringRef().data(),
                   nextConsumerOp->getName().getStringRef().data(), errorStr);
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "L1 upgrade blocked: can't be input of consumer %s.",
          nextConsumerOp->getName().getStringRef().data());
    }
    TTNNLayoutAttr nextConsumerOpLayout = nextConsumerOpL1Layout.get();
    assert(nextConsumerOpLayout == nextConsumerOpConfig.outputLayout &&
           "Expected consumer of updated op layout to match the one in "
           "OpConfig");
  }

  TTMLIR_DEBUG(
      ttmlir::LogComponent::Optimizer,
      "OpModel constraints valid {0}:\n"
      "OutputLayout: {1}\n"
      "L1 usage: cBUsagePeak: {2}, tensorUsage: {3}, outputTensorUsage: {4}, "
      "producerL1OutputUsage: {5}, totalL1Usage: {6}",
      consumerOp->getName(), outputLayout, cBUsagePeak, tensorUsage,
      outputTensorUsage, producersL1OutputUsage,
      cBUsagePeak + tensorUsage + producersL1OutputUsage);

  return outputLayout;
}

} // namespace mlir::tt::ttnn
