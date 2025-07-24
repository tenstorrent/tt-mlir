// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/L1InterleavedAnalysis.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace mlir::tt::ttnn {

void L1InterleavedAnalysis::analysisImplementation() {
  // Go through schedule in order using walk, trying to upgrade DRAM ops to L1
  // interleaved
  analysisInput.funcOp->walk([&](Operation *op) {
    // Skip operations that have the row-major workaround later on in Optimizer
    if (isa<ttnn::MaxPool2dOp>(op) || isa<ttnn::UpsampleOp>(op)) {
      return;
    }

    // Skip if operation doesn't have L1 interleaved layout available
    if (!hasL1InterleavedLegalLayout(op)) {
      return;
    }
    // Skip if operation doesn't use DRAM layout
    if (!usesDRAMLayout(op)) {
      return;
    }
    // Skip if producer is not immediately consumed by consumer
    if (!hasImmediateConsumer(op)) {
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "L1InterleavedAnalysis: Skipping {} - not singular user or "
                   "not immediately consumed",
                   op->getName());
      return;
    }
    std::vector<OpConfig> opL1InterleavedConfigs =
        getL1InterleavedLayoutConfigs(op);

    bool isCurrentlyTiled = isTiledTensorLayout(op);
    std::stable_sort(
        opL1InterleavedConfigs.begin(), opL1InterleavedConfigs.end(),
        [isCurrentlyTiled](const OpConfig &a, const OpConfig &b) {
          bool aTiled = a.outputLayout.isTiled();
          bool bTiled = b.outputLayout.isTiled();
          // Prioritize configs that match current tiling preference
          return (aTiled == isCurrentlyTiled) && (bTiled != isCurrentlyTiled);
        });

    // Try both L1 interleaved configs until one works if there are multiple
    // (rowMajor and tiled)
    for (auto opL1InterleavedConfig : opL1InterleavedConfigs) {
      llvm::Expected<TTNNLayoutAttr> possibleL1Layout =
          checkUpgradeToL1Interleaved(op, opL1InterleavedConfig);

      if (!possibleL1Layout) {
        llvm::Error error = possibleL1Layout.takeError();
        std::string errorStr = llvm::toString(std::move(error));
        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "L1InterleavedAnalysis: Invalid upgrade, error: {}",
                     errorStr);
        continue;
      }
      TTNNLayoutAttr l1InterleavedLayout = possibleL1Layout.get();
      assert(l1InterleavedLayout == opL1InterleavedConfig.outputLayout &&
             "Expected output layout to match the one in OpConfig");
      analysisResult.upgradedConfigs[op] = opL1InterleavedConfig;
      break;
    }
  });
  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "L1InterleavedAnalysis: Completed - upgraded {} operations",
               analysisResult.upgradedConfigs.size());
}

bool L1InterleavedAnalysis::hasL1InterleavedLegalLayout(Operation *op) const {
  const auto it = analysisInput.legalL1InterleavedConfigs.find(op);
  return it != analysisInput.legalL1InterleavedConfigs.end();
}

std::vector<OpConfig>
L1InterleavedAnalysis::getL1InterleavedLayoutConfigs(Operation *op) const {
  const auto it = analysisInput.legalL1InterleavedConfigs.find(op);
  assert(it != analysisInput.legalL1InterleavedConfigs.end());
  return it->second;
}

bool L1InterleavedAnalysis::usesDRAMLayout(Operation *op) const {
  // Check if the operation has a result type that is a tensor
  if (op->getNumResults() == 0) {
    return false;
  }

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType) {
    return false;
  }

  auto encoding = resultType.getEncoding();
  if (!encoding) {
    return false;
  }

  if (auto ttnnLayout = mlir::dyn_cast<TTNNLayoutAttr>(encoding)) {
    return ttnnLayout.hasDRAMBufferType();
  }

  return false;
}

bool L1InterleavedAnalysis::hasImmediateConsumer(Operation *op) const {
  // Check if operation has exactly one user
  if (!op->hasOneUse()) {
    return false;
  }

  // Get the single user
  Operation *userOp = *op->getUsers().begin();
  Operation *consumerOp = op->getNextNode();

  // User must not be scheduled at an earlier index than its operand
  assert(consumerOp);

  // Check if the user is the immediate next operation in schedule
  return consumerOp == userOp;
}

bool L1InterleavedAnalysis::isTiledTensorLayout(Operation *op) const {
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  auto encoding = resultType.getEncoding();
  if (!encoding) {
    return false;
  }
  if (auto ttnnLayout = mlir::dyn_cast<TTNNLayoutAttr>(encoding)) {
    return ttnnLayout.isTiled();
  }
  return false;
}

llvm::Expected<TTNNLayoutAttr>
L1InterleavedAnalysis::checkUpgradeToL1Interleaved(
    Operation *consumerOp, const OpConfig &consumerConfig) const {
  // Figure out this const based on exec data, but will be replaced
  // with API.
  constexpr float tensorL1UsageCap = 0.8;

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
  // Discard DPS operand since it's not used in runtime.
  // TODO(odjuricic,#2088): Remove once fix this on MLIR / runtime side.
  if (llvm::isa<DestinationStyleOpInterface>(consumerOp)) {
    numOperands = numOperands - 1;
  }

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
      auto it = analysisResult.upgradedConfigs.find(operand.getDefiningOp());
      if (it != analysisResult.upgradedConfigs.end()) {
        inputLayouts.push_back(it->second.outputLayout);
        producersL1OutputUsage +=
            utils::getOpOutputL1Usage(it->second.outputLayout);
        continue;
      }
    }

    RankedTensorType input = mlir::cast<RankedTensorType>(operand.getType());

    auto layout = mlir::cast<TTNNLayoutAttr>(input.getEncoding());

    assert(layout && "Input operand must have a layout");
    producersL1OutputUsage += utils::getOpOutputL1Usage(layout);
    inputLayouts.push_back(layout);
  }

  llvm::Expected<op_model::ttnn::OpConstraints> l1UsageExp =
      backend.getOpConstraints(inputLayouts, consumerConfig);

  if (!l1UsageExp) {
    llvm::Error error = l1UsageExp.takeError();
    std::string errorStr = llvm::toString(std::move(error));

    // early exit
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "OpModel constraints failed: {0} :: {1},"
                 "\nconsumerLayout: {2}",
                 consumerOp->getName(), ttmlir::utils::firstNLines(errorStr, 4),
                 consumerConfig.outputLayout);

    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "OpModel constraints failed: %s",
                                   errorStr.data());
  }

  auto [cBUsagePeak, tensorUsage, outputTensorUsage, outputLayout] =
      l1UsageExp.get();

  // Check if the output layout matches the expected interleaved L1 layout
  if (outputLayout != consumerConfig.outputLayout) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Output layout mismatch for op %s",
                                   consumerOp->getName().getStringRef().data());
  }

  bool l1UsageValid = (producersL1OutputUsage + tensorUsage + cBUsagePeak) <
                      tensorL1UsageCap * analysisInput.usableL1CacheSize;

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
                                   "Not enough L1 memory for op %s",
                                   consumerOp->getName().getStringRef().data());
  }

  TTMLIR_DEBUG(
      ttmlir::LogComponent::Optimizer,
      "OpModel constraints valid. Consumer: {0}\n"
      "OutputLayout: {1}\n"
      "L1 usage: cBUsagePeak: {2}, tensorUsage: {3}, outputTensorUsage: {4}, "
      "producerL1OutputUsage: {5}, totalL1Usage: {6}\n"
      "=== End of debug dump ===",
      consumerOp->getName(), outputLayout, cBUsagePeak, tensorUsage,
      outputTensorUsage, producersL1OutputUsage,
      cBUsagePeak + tensorUsage + producersL1OutputUsage);

  return outputLayout;
}

} // namespace mlir::tt::ttnn
