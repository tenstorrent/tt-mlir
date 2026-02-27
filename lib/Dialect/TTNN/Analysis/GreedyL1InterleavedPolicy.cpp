// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/GreedyL1InterleavedPolicy.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Scheduler/Scheduler.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::tt::ttnn {

GreedyL1InterleavedPolicy::GreedyPolicyChoice
GreedyL1InterleavedPolicy::getGreedyConfig(
    Operation *baseOp, llvm::DenseMap<Operation *, L1Usage> &opsL1Usage) {
  uint64_t numOfOps, bitIndex, currentMask;
  uint64_t currentL1Usage, optimalL1Usage;
  llvm::DenseMap<Operation *, OpConfig> optimalConfigs;
  llvm::DenseMap<Operation *, OpConfig> currentConfigs;
  llvm::SmallVector<Operation *> optimalPrecedence;

  constexpr uint64_t maxNumOfOps = sizeof(numOfOps) * 8;
  numOfOps = opsL1Usage.size();
  assert(numOfOps <= maxNumOfOps);

  optimalL1Usage = 0;
  for (currentMask = 0; currentMask < (1 << numOfOps); currentMask++) {
    std::bitset<maxNumOfOps> bitset(currentMask);
    llvm::DenseMap<Operation *, OpConfig> currentConfigs;
    llvm::SmallVector<Operation *> currentPrecedence, optimalL1Precedence,
        L1Precedence;

    // Calculate the L1 usage of the current configuration.
    //
    currentL1Usage = 0;
    bitIndex = 0;
    for (const auto &[op, l1Usage] : opsL1Usage) {
      if (bitset[bitIndex]) {
        // In case we have an operand with L1 interleaved layout, we need to
        // figure out its schedule among the other operands with L1 interleaved
        // layout. Therefore, we insert all of them into the L1Precedence where
        // calculate the optimal L1Precedence and then concatenate it with the
        // currentPrecedence.
        //
        currentL1Usage += l1Usage.outputL1Usage;
        currentConfigs[op] = OpConfig(getL1InterleavedLayout(op));

        // Skip the baseOp.
        //
        if (baseOp != op) {
          L1Precedence.push_back(op);
        }
      } else {
        // It is optimal to first schedule all ops with DRAM output layout.
        // Therefore, we can directly insert them into the
        // currentOptimalPrecedence.
        //
        currentConfigs[op] = OpConfig(getDRAMLayout(op));

        // Skip the baseOp.
        //
        if (baseOp != op) {
          currentPrecedence.push_back(op);
        }
      }
      bitIndex += 1;
    }

    // Calculate the optimal L1Precedence.
    //
    bool isMaskLegal = false;
    uint64_t minRequiredL1Usage = usableL1CacheSize;

    std::sort(L1Precedence.begin(), L1Precedence.end());
    do {
      // Check if the current order of L1Precedence is legal.
      //
      bool isLegal = true;
      uint64_t intermediateL1Usage = 0;
      uint64_t intermediateRequiredL1Usage = 0;
      for (Operation *op : L1Precedence) {
        if (intermediateL1Usage + opsL1Usage[op].requiredL1Usage >
            usableL1CacheSize) {
          isLegal = false;
          break;
        }

        intermediateRequiredL1Usage =
            std::max(intermediateRequiredL1Usage,
                     intermediateL1Usage + opsL1Usage[op].requiredL1Usage);
        intermediateL1Usage += opsL1Usage[op].outputL1Usage;
      }

      // Pick optimal L1Precedence among all legal L1Precedence.
      // The one that requires the least amount of L1 cache overall is
      // considered optimal.
      //
      if (isLegal && intermediateRequiredL1Usage < minRequiredL1Usage) {
        isMaskLegal = true;
        minRequiredL1Usage = intermediateRequiredL1Usage;
        optimalL1Precedence = L1Precedence;
      }
    } while (std::next_permutation(L1Precedence.begin(), L1Precedence.end()));

    if (isMaskLegal && optimalL1Usage < currentL1Usage &&
        currentL1Usage <= usableL1CacheSize) {

      // Append the legal L1Precedence to the currentPrecedence and therefore
      // create a complete precedence for the baseOp and currentMask.
      //
      currentPrecedence.insert(currentPrecedence.end(),
                               optimalL1Precedence.begin(),
                               optimalL1Precedence.end());

      // Update the optimal configuration.
      //
      optimalL1Usage = currentL1Usage;
      optimalConfigs = std::move(currentConfigs);
      optimalPrecedence = std::move(currentPrecedence);
    }
  }

  // Create the optimal config.
  //
  GreedyPolicyChoice optimalConfig;
  optimalConfig.baseOp = baseOp;
  optimalConfig.configs = std::move(optimalConfigs);
  optimalConfig.precedence = std::move(optimalPrecedence);

  return optimalConfig;
}

void GreedyL1InterleavedPolicy::run() {
  // Calculate effective L1 limit from module attribute (single source of
  // truth). This matches the pattern used in OpConstraintValidation.
  const float tensorL1UsageCap = utils::getTensorL1UsageCap(rootOp);
  ttcore::SystemDescAttr systemDesc = mlir::cast<ttcore::SystemDescAttr>(
      rootOp->getParentOfType<ModuleOp>()->getAttr(
          ttcore::SystemDescAttr::name));
  ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs()[0];
  usableL1CacheSize =
      static_cast<uint64_t>(tensorL1UsageCap * chipDesc.getUsableL1Size());

  for (Operation &funcOp : rootOp->getRegion(0).getOps()) {
    func::FuncOp func = dyn_cast<func::FuncOp>(funcOp);
    if (!func) {
      continue;
    }

    // Start the policy.
    //
    llvm::DenseMap<Operation *, OpMemSpec> OpMemSpecMap;
    mlir::tt::scheduler::Scheduler scheduler(&func);
    llvm::SmallVector<Operation *> schedulableOps;

    while (scheduler.hasUnscheduledOps()) {
      schedulableOps = scheduler.getSchedulableOps();

      for (Operation *op : schedulableOps) {
        // Schedule the op.
        //
        scheduler.scheduleOp(op);

        // Find optimal configuration for the op.
        //
        llvm::DenseMap<Operation *, L1Usage> opsL1Usage;
        llvm::SmallVector<Operation *> opsPrecedence;

        // Generate optimal configuration for the current op based on the
        // outputs of its operands and its legal configs.
        //
        if (isAnalyzable(op)) {

          // Create the OpMemSpec.
          //
          OpMemSpec OpMemSpec;
          assert(hasDRAMBufferType(op));
          OpMemSpec.config = OpConfig(getDRAMLayout(op));
          OpMemSpec.requiredL1Usage = 0;
          OpMemSpecMap[op] = OpMemSpec;

          if (op->hasOneUse() && hasL1BufferType(op)) {
            L1Usage l1Usage;
            l1Usage.outputL1Usage =
                utils::getOpOutputL1Usage(getL1InterleavedLayout(op));
            l1Usage.requiredL1Usage = 0;
            opsL1Usage[op] = l1Usage;
          }
        }

        for (auto operand : op->getOperands()) {
          // Skip block arguments (%arg0, %arg1, ...)
          //
          if (::llvm::isa<mlir::BlockArgument>(operand)) {
            continue;
          }

          Operation *operandOp = operand.getDefiningOp();

          // Skip non-analyzable operands.
          //
          if (isAnalyzable(operandOp)) {
            OpConfig operandOpConfig = OpMemSpecMap[operandOp].config;

            // Take into consideration only the operands with L1 interleaved
            // memory space.
            //
            if (operandOpConfig.outputLayout
                    .hasInterleavedL1TensorMemoryLayout()) {
              L1Usage l1Usage;
              l1Usage.outputL1Usage =
                  utils::getOpOutputL1Usage(operandOpConfig.outputLayout);
              l1Usage.requiredL1Usage = OpMemSpecMap[operandOp].requiredL1Usage;
              opsL1Usage[operandOp] = l1Usage;
            }
            // In case the operand has DRAM layout, we can insert it into the
            // precedence directly. If the op is analyzable, it means that it
            // is definitely schedulable.
            //
            else {
              opsPrecedence.push_back(operandOp);
            }
          }
          // In case the operand is not analyzable, i.e. there are no legal
          // configs for this operand, we can insert it into the precedence
          // directly if it is schedulable since it does not use DRAM nor L1
          // memory.
          //
          else {
            if (scheduler.isTTSchedulableOp(operandOp)) {
              opsPrecedence.push_back(operandOp);
            }
          }
        }

        // Greedily find the optimal configuration.
        //
        GreedyPolicyChoice optimalConfig = getGreedyConfig(op, opsL1Usage);
        for (const auto &[innerOp, cfg] : optimalConfig.configs) {
          OpMemSpecMap[innerOp].config = cfg;
        }

        // Override op's precedence.
        //
        opsPrecedence.insert(opsPrecedence.end(),
                             optimalConfig.precedence.begin(),
                             optimalConfig.precedence.end());
        precedenceMap[op] = std::move(opsPrecedence);

        // Update op's requiredL1Usage if the op is analyzable.
        //
        if (isAnalyzable(op)) {
          uint64_t intermediateRequiredL1Usage = 0;
          uint64_t intermediateL1Usage = 0;
          for (auto operand : op->getOperands()) {
            // Skip block arguments (%arg0, %arg1, ...)
            //
            if (::llvm::isa<mlir::BlockArgument>(operand)) {
              continue;
            }

            Operation *operandOp = operand.getDefiningOp();

            // Skip non-analyzable operands.
            //
            if (isAnalyzable(operandOp)) {
              intermediateRequiredL1Usage =
                  std::max(intermediateRequiredL1Usage,
                           intermediateL1Usage +
                               OpMemSpecMap[operandOp].requiredL1Usage);
              intermediateL1Usage += utils::getOpOutputL1Usage(
                  OpMemSpecMap[operandOp].config.outputLayout);
            }
          }
          OpMemSpecMap[op].requiredL1Usage = std::max(
              intermediateRequiredL1Usage,
              intermediateL1Usage + utils::getOpOutputL1Usage(
                                        OpMemSpecMap[op].config.outputLayout));
        }
      }
    }

    // Construct the schedule.
    //
    constructSchedule(func);

    // Build, Resolve and Complete the L1 chain.
    // This implementation is only here unitl we are able to merge
    // L1ChainConfigs.
    // TODO(fbajraktari): Fix this hack.
    //
    l1ChainConfigs->push_back(L1ChainConfig());
    llvm::DenseMap<Operation *, OpConfig> selectedOpConfig;
    for (auto &OpMemSpec : OpMemSpecMap) {
      OpL1MemSpec opL1MemSpec;
      opL1MemSpec.op = OpMemSpec.first;
      opL1MemSpec.tensorSplitFactor = 1;
      selectedOpConfig[OpMemSpec.first] = OpMemSpec.second.config.outputLayout;
      l1ChainConfigs->back().addOpL1MemSpec(opL1MemSpec);
    }
    l1ChainConfigs->back().build();
    l1ChainConfigs->back().resolve();
    llvm::DenseMap<Edge, MemReconfigEntry> memReconfigMap;
    l1ChainConfigs->back().complete(selectedOpConfig, memReconfigMap);
  }
}

bool GreedyL1InterleavedPolicy::isAnalyzable(Operation *op) {
  // Skip operations that are not analyzed by the LegalLayoutAnalysis.
  //
  if (legalConfigs.count(op) > 0) {
    // Skip operations that are filterd out by the MemoryLayoutAnalysis.
    //
    return legalConfigs[op].size() > 0;
  }
  return false;
}

bool GreedyL1InterleavedPolicy::hasDRAMBufferType(Operation *op) {
  return std::find_if(legalConfigs[op].begin(), legalConfigs[op].end(),
                      [](OpConfig config) {
                        return config.outputLayout.hasDRAMBufferType();
                      }) != legalConfigs[op].end();
}

TTNNLayoutAttr GreedyL1InterleavedPolicy::getDRAMLayout(Operation *op) {
  assert(hasDRAMBufferType(op));
  auto configIter = std::find_if(
      legalConfigs[op].begin(), legalConfigs[op].end(),
      [](OpConfig config) { return config.outputLayout.hasDRAMBufferType(); });
  return configIter->outputLayout;
}

bool GreedyL1InterleavedPolicy::hasL1BufferType(Operation *op) {
  return std::find_if(
             legalConfigs[op].begin(), legalConfigs[op].end(),
             [](OpConfig config) {
               return config.outputLayout.hasInterleavedL1TensorMemoryLayout();
             }) != legalConfigs[op].end();
}

TTNNLayoutAttr
GreedyL1InterleavedPolicy::getL1InterleavedLayout(Operation *op) {
  assert(hasL1BufferType(op));
  auto l1InterleaveLayoutIter = std::find_if(
      legalConfigs[op].begin(), legalConfigs[op].end(), [](OpConfig config) {
        return config.outputLayout.hasInterleavedL1TensorMemoryLayout();
      });
  return l1InterleaveLayoutIter->outputLayout;
}

} // namespace mlir::tt::ttnn
