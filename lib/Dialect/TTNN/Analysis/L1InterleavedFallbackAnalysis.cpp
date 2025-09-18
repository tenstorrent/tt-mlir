// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/L1InterleavedFallbackAnalysis.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace mlir::tt::ttnn {

static size_t failedBackend = 0;
static size_t failedOpConstraints = 0;
static size_t failedOutputLayoutMismatch = 0;
static size_t failedL1Memory = 0;
static size_t failedConsumerCheck = 0;
static size_t failedConsumerBackend = 0;
static size_t failedConsumerOpConstraints = 0;
static size_t failedConsumerOutputLayoutMismatch = 0;
static size_t failedConsumerL1Memory = 0;
static size_t failedRuntimeImprovement = 0;

void L1InterleavedFallbackAnalysis::analysisImplementation() {
  // Counters for statistics
  size_t totalOps = 0;
  size_t skippedRowMajor = 0;
  size_t skippedDRAMOut = 0;
  size_t skippedMatmulLinear = 0;
  size_t skippedConv2DMatmul = 0;
  size_t skippedUserDRAMIn = 0;
  size_t skippedUserMatmulLinear = 0;
  size_t skippedUserConv2DMatmul = 0;
  size_t skippedNoL1Legal = 0;
  size_t skippedNotDRAM = 0;
  size_t skippedNotOneUser = 0;
  size_t skippedNotImmediate = 0;
  size_t skippedReturnOp = 0;
  size_t skippedReshapeNoOp = 0;
  size_t skippedUserReshapeNoOp = 0;
  size_t attemptedUpgrade = 0;
  // Failure reason counters for checkUpgradeToL1Interleaved
  failedBackend = 0;
  failedOpConstraints = 0;
  failedOutputLayoutMismatch = 0;
  failedL1Memory = 0;
  failedConsumerCheck = 0;
  // Next consumer specific failure breakdowns
  failedConsumerBackend = 0;
  failedConsumerOpConstraints = 0;
  failedConsumerOutputLayoutMismatch = 0;
  failedConsumerL1Memory = 0;
  failedRuntimeImprovement = 0;
  size_t failedUpgrade = 0;
  size_t upgraded = 0;

  // Go through schedule in order using walk, trying to upgrade DRAM ops to L1
  // interleaved.
  analysisInput.funcOp->walk([&](Operation *op) {
    llvm::outs().flush();
    op->dumpPretty();

    if (isa<ttcore::LoadCachedOp>(op)) {
      return;
    }
    ++totalOps;

    // Skip operations that have the row-major workaround later on in Optimizer.
    // TODO(bmalesevic,#3985): remove after this is fixed
    if (isa<ttnn::MaxPool2dOp, ttnn::UpsampleOp>(op)) {
      ++skippedRowMajor;
      llvm::outs() << "[L1IFA] Skipped op (row-major workaround): "
                   << op->getName() << "\n";
      return;
    }

    // Skip operations that have DRAM output in runtime even when configured as
    // L1 via this analysis.
    // TODO(bmalesevic,#4505): remove this after they are supported in runtime
    if (isa<ttnn::SliceStaticOp, ttnn::TypecastOp>(op)) {
      ++skippedDRAMOut;
      llvm::outs() << "[L1IFA] Skipped op (DRAM output in runtime): "
                   << op->getName() << "\n";
      return;
    }
    // Skip Matmul and Linear output, inefficient for L1 interleaved.
    if (isa<ttnn::MatmulOp, ttnn::LinearOp>(op)) {
      ++skippedMatmulLinear;
      llvm::outs() << "[L1IFA] Skipped op (Matmul/Linear output): "
                   << op->getName() << "\n";
      return;
    }
    // Skip output of Conv2D that uses matmul under the hood, inefficient for L1
    // interleaved.
    if (L1InterleavedFallbackAnalysis::isConv2DConvertibleToMatMul(op)) {
      ++skippedConv2DMatmul;
      llvm::outs() << "[L1IFA] Skipped op (Conv2D uses matmul): "
                   << op->getName() << "\n";
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
        ++skippedUserDRAMIn;
        llvm::outs() << "[L1IFA] Skipped op (user DRAM input): "
                     << op->getName() << "\n";
        return;
      }
      // Skip Matmul and Linear input, inefficient for L1 interleaved.
      if (isa<ttnn::MatmulOp, ttnn::LinearOp>(user)) {
        ++skippedUserMatmulLinear;
        llvm::outs() << "[L1IFA] Skipped op (user Matmul/Linear input): "
                     << op->getName() << "\n";
        return;
      }
      // Skip input of Conv2D that uses matmul under the hood, inefficient for
      // L1 interleaved.
      if (L1InterleavedFallbackAnalysis::isConv2DConvertibleToMatMul(user)) {
        ++skippedUserConv2DMatmul;
        llvm::outs() << "[L1IFA] Skipped op (user Conv2D uses matmul): "
                     << op->getName() << "\n";
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
      ++skippedNoL1Legal;
      llvm::outs() << "[L1IFA] Skipped op (no L1 interleaved legal layout): "
                   << op->getName() << "\n";
      return;
    }
    // Skip if operation doesn't use DRAM layout.
    if (!utils::producesDRAMLayout(op)) {
      ++skippedNotDRAM;
      llvm::outs() << "[L1IFA] Skipped op (not DRAM layout): " << op->getName()
                   << "\n";
      return;
    }
    // Skip if operation doesn't have exactly one user.
    if (!op->hasOneUse()) {
      ++skippedNotOneUser;
      llvm::outs() << "[L1IFA] Skipped op (not exactly one user): "
                   << op->getName() << "\n";
      return;
    }
    // Skip if producer is not immediately consumed by consumer.
    if (!hasImmediateConsumer(op)) {
      ++skippedNotImmediate;
      llvm::outs() << "[L1IFA] Skipped op (consumer not scheduled "
                      "immediately after): "
                   << op->getName() << "\n";
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "L1InterleavedFallbackAnalysis: Skipping {} - "
                   "consumer not scheduled immediately after.",
                   op->getName());
      return;
    }
    // Skip if operation output is returnOp input, no point in storing
    // in L1, no acceleration would happen but additional memory
    // management risks would be introduced.
    if (hasReturnOpUser(op)) {
      ++skippedReturnOp;
      llvm::outs() << "[L1IFA] Skipped op (output is returnOp input): "
                   << op->getName() << "\n";
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "L1InterleavedFallbackAnalysis: Skipping {} - output is "
                   "returnOp input.",
                   op->getName());
      return;
    }
    llvm::outs() << "[L1IFA] Attempting to upgrade op: " << op->getName()
                 << "\n";
    ++attemptedUpgrade;
    size_t beforeUpgrade = analysisResult.upgradedConfigs.size();
    tryUpgradeToL1Interleaved(op);
    size_t afterUpgrade = analysisResult.upgradedConfigs.size();

    if (afterUpgrade > beforeUpgrade) {
      ++upgraded;
    } else {
      ++failedUpgrade;
    }
  });

  // Print summary statistics
  llvm::outs() << "\n[L1IFA] Analysis Summary:\n";
  llvm::outs() << "[L1IFA]  Total ops: " << totalOps << "\n";
  auto percent = [](size_t n, size_t total) -> std::string {
    if (total == 0) {
      return "0.00";
    }
    double pct = 100.0 * n / static_cast<double>(total);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << pct;
    return oss.str();
  };
  llvm::outs() << "[L1IFA]  Skipped (row-major workaround): " << skippedRowMajor
               << " (" << percent(skippedRowMajor, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]  Skipped (DRAM output in runtime): "
               << skippedDRAMOut << " (" << percent(skippedDRAMOut, totalOps)
               << "%)\n";
  llvm::outs() << "[L1IFA]  Skipped (Matmul/Linear input): "
               << skippedMatmulLinear << " ("
               << percent(skippedMatmulLinear, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]  Skipped (Conv2D uses matmul): "
               << skippedConv2DMatmul << " ("
               << percent(skippedConv2DMatmul, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]  Skipped (user DRAM input): " << skippedUserDRAMIn
               << " (" << percent(skippedUserDRAMIn, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]  Skipped (user Matmul/Linear input): "
               << skippedUserMatmulLinear << " ("
               << percent(skippedUserMatmulLinear, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]  Skipped (user Conv2D uses matmul): "
               << skippedUserConv2DMatmul << " ("
               << percent(skippedUserConv2DMatmul, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]  Skipped (no L1 interleaved legal layout): "
               << skippedNoL1Legal << " ("
               << percent(skippedNoL1Legal, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]  Skipped (not DRAM layout): " << skippedNotDRAM
               << " (" << percent(skippedNotDRAM, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]  Skipped (not exactly one user): "
               << skippedNotOneUser << " ("
               << percent(skippedNotOneUser, totalOps) << "%)\n";
  llvm::outs()
      << "[L1IFA]  Skipped (consumer not scheduled immediately after): "
      << skippedNotImmediate << " (" << percent(skippedNotImmediate, totalOps)
      << "%)\n";
  llvm::outs() << "[L1IFA]  Skipped (output is returnOp input): "
               << skippedReturnOp << " (" << percent(skippedReturnOp, totalOps)
               << "%)\n";
  llvm::outs() << "[L1IFA]  Skipped (reshape no-op): " << skippedReshapeNoOp
               << " (" << percent(skippedReshapeNoOp, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]  Skipped (user reshape no-op): "
               << skippedUserReshapeNoOp << " ("
               << percent(skippedUserReshapeNoOp, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]  Attempted upgrades: " << attemptedUpgrade << " ("
               << percent(attemptedUpgrade, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]  Successful upgrades: " << upgraded << " ("
               << percent(upgraded, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]  Failed upgrades: " << failedUpgrade << " ("
               << percent(failedUpgrade, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]    - Failed due to backend constraints: "
               << failedBackend << " (" << percent(failedBackend, totalOps)
               << "%)\n";
  llvm::outs() << "[L1IFA]    - Failed due to OpModel constraints: "
               << failedOpConstraints << " ("
               << percent(failedOpConstraints, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]    - Failed due to output layout mismatch: "
               << failedOutputLayoutMismatch << " ("
               << percent(failedOutputLayoutMismatch, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]    - Failed due to not enough L1 memory: "
               << failedL1Memory << " (" << percent(failedL1Memory, totalOps)
               << "%)\n";
  llvm::outs() << "[L1IFA]    - Failed due to consumer check: "
               << failedConsumerCheck << " ("
               << percent(failedConsumerCheck, totalOps) << "%)\n";
  llvm::outs()
      << "[L1IFA]      - Of which, failed due to consumer backend constraints: "
      << failedConsumerBackend << " ("
      << percent(failedConsumerBackend, totalOps) << "%)\n";
  llvm::outs()
      << "[L1IFA]      - Of which, failed due to consumer OpModel constraints: "
      << failedConsumerOpConstraints << " ("
      << percent(failedConsumerOpConstraints, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]      - Of which, failed due to consumer output "
                  "layout mismatch: "
               << failedConsumerOutputLayoutMismatch << " ("
               << percent(failedConsumerOutputLayoutMismatch, totalOps)
               << "%)\n";
  llvm::outs() << "[L1IFA]      - Of which, failed due to consumer not enough "
                  "L1 memory: "
               << failedConsumerL1Memory << " ("
               << percent(failedConsumerL1Memory, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA]    - Failed due to no runtime improvement: "
               << failedRuntimeImprovement << " ("
               << percent(failedRuntimeImprovement, totalOps) << "%)\n";
  llvm::outs() << "[L1IFA] Analysis complete.\n\n";

  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "L1InterleavedFallbackAnalysis: Completed - upgraded {} "
               "operations.",
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
  assert(!it->second.empty());
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

bool L1InterleavedFallbackAnalysis::isConv2DConvertibleToMatMul(Operation *op) {
  auto conv2dOp = dyn_cast<ttnn::Conv2dOp>(op);
  if (!conv2dOp) {
    return false;
  }

  // Get weight tensor to check kernel size
  RankedTensorType weightType = conv2dOp.getWeight().getType();
  llvm::ArrayRef<int64_t> weightShape = weightType.getShape();

  // Check kernel size is 1x1
  if (weightShape[2] != 1 || weightShape[3] != 1) {
    return false;
  }

  // Check all stride values are 1
  auto stride = conv2dOp.getStride();
  if (llvm::any_of(stride, [](int32_t v) { return v != 1; })) {
    return false;
  }

  // Check all padding values are 0
  auto padding = conv2dOp.getPadding();
  if (llvm::any_of(padding, [](int32_t v) { return v != 0; })) {
    return false;
  }

  // Check dilation = 1
  auto dilation = conv2dOp.getDilation();
  if (llvm::any_of(dilation, [](int32_t v) { return v != 1; })) {
    return false;
  }

  return true;
}

void L1InterleavedFallbackAnalysis::tryUpgradeToL1Interleaved(Operation *op) {
  std::vector<OpConfig> opL1InterleavedConfigs =
      getL1InterleavedLayoutConfigs(op);

  bool isCurrentlyTiled = utils::producesTiledTensorLayout(op);

  // Partition configs to prioritize those matching current tiling
  // preference.
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
    llvm::Expected<std::tuple<TTNNLayoutAttr, int64_t>> checkUpgradeResult =
        checkUpgradeToL1Interleaved(op, opL1InterleavedConfig,
                                    /*upgradedProducerOp=*/nullptr,
                                    /*upgradedProducerLayout=*/nullptr);

    if (!checkUpgradeResult) {
      llvm::Error error = checkUpgradeResult.takeError();
      std::string errorStr = llvm::toString(std::move(error));
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "L1InterleavedFallbackAnalysis: Invalid upgrade, error: {}",
                   errorStr);
      continue;
    }
    auto [l1InterleavedLayout, runtimeGain] = checkUpgradeResult.get();
    assert(l1InterleavedLayout == opL1InterleavedConfig.outputLayout &&
           "Expected output layout to match the one in OpConfig");
    analysisResult.upgradedConfigs[op] = opL1InterleavedConfig;
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "L1InterleavedFallbackAnalysis: Upgraded op {} to L1 "
                 "interleaved layout {}, runtime gain {}",
                 op->getName().getStringRef().data(), l1InterleavedLayout,
                 runtimeGain);
    break;
  }
}

llvm::Expected<std::tuple<TTNNLayoutAttr, int64_t>>
L1InterleavedFallbackAnalysis::checkUpgradeToL1Interleaved(
    Operation *consumerOp, const OpConfig &consumerConfig,
    const Operation *upgradedProducerOp,
    const TTNNLayoutAttr upgradedProducerLayout) const {

  llvm::outs() << "[L1IFA]     checkUpgradeToL1Interleaved for op: "
               << consumerOp->getName() << "\n";

  OpModel backend = mlir::dyn_cast<OpModel>(consumerOp);
  if (!backend) {
    llvm::outs() << "[L1IFA]     FAILED: Backend constraints not implemented\n";
    upgradedProducerOp ? ++failedConsumerBackend : ++failedBackend;
    // This function should not be called for ops without backend
    // constraints.
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
  uint64_t changedInputLayoutIndex = -1;
  TTNNLayoutAttr oldChangedInputLayout = nullptr;

  for (uint32_t i = 0; i < numOperands; i++) {
    auto operand = consumerOp->getOperand(i);

    if (mlir::isa<TypedValue<mlir::tt::ttnn::DeviceType>>(operand)) {
      // Skip device type operand.
      continue;
    }

    if (operand.getDefiningOp()) {
      if (operand.getDefiningOp() == upgradedProducerOp) {
        // If it's a nested check of update candidate's (producer in this
        // scope) consumer's storage.
        changedInputLayoutIndex = inputLayouts.size();
        RankedTensorType input =
            mlir::cast<RankedTensorType>(operand.getType());
        oldChangedInputLayout = mlir::cast<TTNNLayoutAttr>(input.getEncoding());
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

    llvm::outs() << "[L1IFA]     FAILED: OpModel constraints failed - "
                 << errorStr << "\n";
    upgradedProducerOp ? ++failedConsumerOpConstraints : ++failedOpConstraints;
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "OpModel constraints failed for op {0} :: {1},"
                 "\nconsumerLayout: {2}",
                 consumerOp->getName(), ttmlir::utils::firstNLines(errorStr, 4),
                 consumerConfig.outputLayout);

    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "OpModel constraints failed for op %s.",
                                   consumerOp->getName().getStringRef().data());
  }

  auto [cBUsagePeak, tensorUsage, peakMemoryUsage, outputTensorUsage,
        outputLayout] = l1UsageExp.get();

  if (outputLayout != consumerConfig.outputLayout) {
    llvm::outs() << "[L1IFA]     FAILED: Output layout mismatch - expected: "
                 << consumerConfig.outputLayout << ", actual: " << outputLayout
                 << "\n";
    upgradedProducerOp ? ++failedConsumerOutputLayoutMismatch
                       : ++failedOutputLayoutMismatch;
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

  llvm::outs() << "[L1IFA]     L1 usage check: total="
               << (producersL1OutputUsage + tensorUsage + cBUsagePeak)
               << ", available=" << analysisInput.usableL1CacheSize
               << ", valid=" << (l1UsageValid ? "YES" : "NO") << "\n";
  if (!l1UsageValid) {
    llvm::outs() << "[L1IFA]     FAILED: Not enough L1 memory (producer="
                 << producersL1OutputUsage << ", tensor=" << tensorUsage
                 << ", cb=" << cBUsagePeak << ")\n";
    upgradedProducerOp ? ++failedConsumerL1Memory : ++failedL1Memory;
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
  // Check if upgrading this operation would cause memory conflicts with
  // its consumer. This is a single-level recursion check:
  // - First call: upgradedProducerOp=nullptr, checks if current op can be
  // upgraded.
  // - Recursive call: upgradedProducerOp=consumerOp, checks if consumer
  // can handle the upgrade.
  if (upgradedProducerOp) {
    llvm::outs()
        << "[L1IFA]     SUCCESS: Recursive check passed for consumer\n";
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "OpModel constraints valid for input of consumer {0}:\n"
                 "OutputLayout: {1}\n"
                 "L1 usage: cBUsagePeak: {2}, tensorUsage: {3}, "
                 "outputTensorUsage: {4}, "
                 "producerL1OutputUsage: {5}, totalL1Usage: {6}",
                 consumerOp->getName(), outputLayout, cBUsagePeak, tensorUsage,
                 outputTensorUsage, producersL1OutputUsage,
                 cBUsagePeak + tensorUsage + producersL1OutputUsage);

    auto [hasErrors, beforeRuntime, afterRuntime] = checkOpRuntimesInputChange(
        backend, inputLayouts, consumerConfig, changedInputLayoutIndex,
        oldChangedInputLayout);
    int64_t consumerRuntimeGain = 0;
    if (hasErrors) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Error getting before-after upgrade runtimes for "
                   "consumer op {}",
                   consumerOp->getName().getStringRef().data());
    } else {
      consumerRuntimeGain = beforeRuntime - afterRuntime;
    }
    return std::make_tuple(outputLayout, consumerRuntimeGain);
  }

  int64_t nextConsumerRuntimeGain = 0;
  assert(consumerOp->hasOneUse() && "Consumer must have exactly one user");
  Operation *nextConsumerOp = *consumerOp->getUsers().begin();
  assert(nextConsumerOp && "Operation must have a consumer");
  // If next consumer has TTNN layout output encoding, verify both
  // operations can coexist in L1.
  if (utils::producesTTNNLayoutEncoding(nextConsumerOp) &&
      !isa<ttnn::ToLayoutOp>(nextConsumerOp)) {
    llvm::outs() << "[L1IFA]     Checking consumer: "
                 << nextConsumerOp->getName() << "\n";
    const OpConfig &nextConsumerOpConfig =
        analysisInput.currentConfigs.at(nextConsumerOp);

    auto nextConsumerCheckUpgradeResult = checkUpgradeToL1Interleaved(
        nextConsumerOp, nextConsumerOpConfig, consumerOp, outputLayout);

    if (!nextConsumerCheckUpgradeResult) {
      llvm::Error error = nextConsumerCheckUpgradeResult.takeError();
      std::string errorStr = llvm::toString(std::move(error));
      llvm::outs() << "[L1IFA]     FAILED: Consumer check failed - " << errorStr
                   << "\n";
      ++failedConsumerCheck;
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "L1 upgrade blocked for {} - for consumer {}: {}",
                   consumerOp->getName().getStringRef().data(),
                   nextConsumerOp->getName().getStringRef().data(), errorStr);
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "L1 upgrade blocked: can't be input of consumer %s.",
          nextConsumerOp->getName().getStringRef().data());
    }
    TTNNLayoutAttr nextConsumerOpLayout =
        std::get<0>(nextConsumerCheckUpgradeResult.get());
    nextConsumerRuntimeGain = std::get<1>(nextConsumerCheckUpgradeResult.get());
    assert(nextConsumerOpLayout == nextConsumerOpConfig.outputLayout &&
           "Expected consumer of updated op layout to match the one in "
           "OpConfig");
  }
  llvm::outs() << "[L1IFA]     SUCCESS: All checks passed for op: "
               << consumerOp->getName() << "\n";
  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "OpModel constraints valid {0}:\n"
               "OutputLayout: {1}\n"
               "L1 usage: cBUsagePeak: {2}, tensorUsage: {3}, "
               "outputTensorUsage: {4}, "
               "producerL1OutputUsage: {5}, totalL1Usage: {6}",
               consumerOp->getName(), outputLayout, cBUsagePeak, tensorUsage,
               outputTensorUsage, producersL1OutputUsage,
               cBUsagePeak + tensorUsage + producersL1OutputUsage);

  const OpConfig &oldConsumerOpConfig =
      analysisInput.currentConfigs.at(consumerOp);
  auto [hasErrors, beforeRuntime, afterRuntime] = checkOpRuntimesOutputChange(
      backend, inputLayouts, consumerConfig, oldConsumerOpConfig);
  int64_t totalRuntimeGain = nextConsumerRuntimeGain;
  if (hasErrors) {
    TTMLIR_DEBUG(
        ttmlir::LogComponent::Optimizer,
        "Error getting before-after upgrade runtimes for producer op {}",
        consumerOp->getName().getStringRef().data());
  } else {
    totalRuntimeGain += beforeRuntime - afterRuntime;
  }
  if (totalRuntimeGain <= 0) {
    ++failedRuntimeImprovement;
    llvm::outs() << "[L1IFA]    FAILED: Runtime improvement check failed - "
                 << consumerOp->getName() << "->" << nextConsumerOp->getName()
                 << "\n";
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "L1 upgrade blocked for {} - no runtime gain: before {}, "
                 "after {}, next consumer gain {}, in total gain {}",
                 consumerOp->getName().getStringRef().data(), beforeRuntime,
                 afterRuntime, nextConsumerRuntimeGain, totalRuntimeGain);
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "L1 upgrade blocked: no runtime gain for op %s with consumer %s.",
        consumerOp->getName().getStringRef().data(),
        nextConsumerOp->getName().getStringRef().data());
  }
  llvm::outs() << "[L1IFA]    SUCCESS: Runtime improvement check passed - "
               << consumerOp->getName() << "->" << nextConsumerOp->getName()
               << "\n"
               << "    runtime producer before: " << beforeRuntime << ",\n"
               << "    runtime producer after: " << afterRuntime << ",\n"
               << "    runtime gain consumer: " << nextConsumerRuntimeGain
               << ",\n"
               << "    runtime gain total: " << totalRuntimeGain << "\n";

  return std::make_tuple(outputLayout, totalRuntimeGain);
}

std::tuple<bool, size_t, size_t>
L1InterleavedFallbackAnalysis::checkOpRuntimesInputChange(
    OpModel &backend, std::vector<TTNNLayoutAttr> &inputLayouts,
    const OpConfig &consumerConfig, size_t changedInputLayoutIndex,
    const TTNNLayoutAttr &oldChangedInputLayout) const {

  // Get runtime with upgraded layout
  llvm::Expected<size_t> afterBackendOpRuntime =
      backend.getOpRuntime(inputLayouts, consumerConfig);

  // Temporarily restore old layout and get runtime
  TTNNLayoutAttr upgradedLayout = inputLayouts[changedInputLayoutIndex];
  inputLayouts[changedInputLayoutIndex] = oldChangedInputLayout;
  llvm::Expected<size_t> beforeBackendOpRuntime =
      backend.getOpRuntime(inputLayouts, consumerConfig);
  // Restore upgraded layout for consistency
  inputLayouts[changedInputLayoutIndex] = upgradedLayout;

  // Check for errors
  if (!afterBackendOpRuntime || !beforeBackendOpRuntime) {
    return std::make_tuple(true, 0, 0);
  }

  return std::make_tuple(false, beforeBackendOpRuntime.get(),
                         afterBackendOpRuntime.get());
}

std::tuple<bool, size_t, size_t>
L1InterleavedFallbackAnalysis::checkOpRuntimesOutputChange(
    OpModel &backend, const std::vector<TTNNLayoutAttr> &inputLayouts,
    const OpConfig &consumerConfig, const OpConfig &oldConsumerConfig) const {

  // Get runtime with upgraded layout
  llvm::Expected<size_t> afterBackendOpRuntime =
      backend.getOpRuntime(inputLayouts, consumerConfig);

  // Get runtime with old layout
  llvm::Expected<size_t> beforeBackendOpRuntime =
      backend.getOpRuntime(inputLayouts, oldConsumerConfig);

  // Check for errors
  if (!afterBackendOpRuntime || !beforeBackendOpRuntime) {
    return std::make_tuple(true, 0, 0);
  }

  return std::make_tuple(false, beforeBackendOpRuntime.get(),
                         afterBackendOpRuntime.get());
}

} // namespace mlir::tt::ttnn
