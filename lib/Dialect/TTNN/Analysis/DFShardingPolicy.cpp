// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/DFShardingPolicy.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysisProgressTracker.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Scheduler/Scheduler.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Diagnostics.h"

namespace mlir::tt::ttnn {

void DFShardingPolicy::run() {
  func::FuncOp funcToProcess = nullptr;

  rootOp->walk([&](func::FuncOp func) {
    if (ttmlir::utils::isConstEvalFunc(func)) {
      return;
    }

    funcToProcess = func;
    deviceAttr = ttcore::lookupDevice(func);
    mlir::tt::scheduler::Scheduler scheduler(&func);
    l1ChainConfigs->push_back(L1ChainConfig());
    llvm::SmallVector<mlir::Operation *> scheduleableOps;
    Operation *currentOp = nullptr;

    // Produce shard chain configs.
    // 1. Schedule ops in DFS order.
    // 2. Check if currentOp has a valid successor. (no forking for now)
    // 3. Check if currentOp/nextOp pair is valid for sharding.
    // 4. Op is considered sharded if its output is sharded to L1.
    //
    while (scheduler.hasUnscheduledOps()) {
      scheduleableOps = scheduler.getSchedulableOps();

      // Before starting a sharding chain, schedule layout/memory management ops
      // first until they are exhausted from schedulable ops.
      // TODO(nobradovic) :
      // We need to examine type of memory op and determine if for
      // example we have a space in DRAM to perform this?(system->dram, double
      // check this)
      //
      if (l1ChainConfigs->back().isEmpty()) {
        for (auto *op : scheduleableOps) {
          if (isa<ToLayoutOp>(op)) {
            currentOp = op;
            break;
          }
        }
      }

      if (currentOp == nullptr) {
        currentOp = scheduleableOps[0];
      }

      // Schedule currentOp.
      //
      scheduler.scheduleOp(currentOp);

      // Skip starting sharding chain if currentOp is a memory management op.
      //
      if (l1ChainConfigs->back().isEmpty() && isa<ToLayoutOp>(currentOp)) {
        currentOp = nullptr;
        continue;
      }

      if (scheduler.hasUnscheduledOps()) {
        scheduleableOps = scheduler.getSchedulableOps();

        // Check if currentOp has a valid successor.
        //
        Operation *nextOp = nullptr;
        for (auto *op : scheduleableOps) {
          for (auto operand : op->getOperands()) {
            if (operand.getDefiningOp() == currentOp) {
              nextOp = op;
              break;
            }
          }
        }

        // Consider sharding only if we found at least single legal config for
        // the current op.
        bool validForSharding =
            llvm::isa<ttnn::Conv2dOp, ttnn::AddOp, ttnn::MultiplyOp,
                      ttnn::ReluOp, ttnn::Relu6Op, ttnn::HardsigmoidOp,
                      ttnn::TypecastOp, ttnn::MinimumOp>(currentOp) &&
            legalConfigs.lookup(currentOp).size() > 0;

        if (validForSharding) {
          OpL1MemSpec shardSpec;
          shardSpec.op = currentOp;

          // Hardcoded tensor split factor for now, until pipeline OP
          // support is added.
          //
          shardSpec.tensorSplitFactor = 1;
          l1ChainConfigs->back().addOpL1MemSpec(std::move(shardSpec));

          if (nextOp && currentOp->hasOneUse()) {
            // Only if nextOp is valid and currentOp is not a fork keep
            // growing the chain.
            if (nextOp->getOperand(0).getDefiningOp() != currentOp) {
              // Only continue chain if nextOp uses currentOp as first operand.
              // Here we break the chain if not.
              TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                           "Breaking L1 chain at op {} as it is not first "
                           "operand of next op {}",
                           currentOp->getName(), nextOp->getName());
              currentOp = nullptr;
            } else {
              currentOp = nextOp;
              continue;
            }
          }
        }

        currentOp = nullptr;
      } // endif hasUnscheduledOps

      if (!l1ChainConfigs->back().isEmpty()) {
        l1ChainConfigs->back().build();
        l1ChainConfigs->push_back(L1ChainConfig());
      }
    }

    (*schedule)[func] = scheduler.getSchedule();
  });

  if (l1ChainConfigs->back().isEmpty()) {
    l1ChainConfigs->pop_back();
  }

  for ([[maybe_unused]] L1ChainConfig &l1ChainConfig : *l1ChainConfigs) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "L1 chain config {}",
                 l1ChainConfig);
  }

  // Resolve shard chain configs.
  //
  mlir::tt::ttnn::MemoryLayoutAnalysisProgressTracker progressTracker;
  progressTracker.startAnalysis(funcToProcess, l1ChainConfigs->size(),
                                "DFShardingPolicy");

  for (size_t chainIndex = 0; chainIndex < l1ChainConfigs->size();
       ++chainIndex) {
    L1ChainConfig &l1ChainConfig = (*l1ChainConfigs)[chainIndex];

    // Count operations in this chain
    size_t numOpsInChain = l1ChainConfig.getOpL1MemSpecs().size();
    Operation *firstOp = l1ChainConfig.getOpL1MemSpecs()[0].op;
    progressTracker.startL1Chain(firstOp, chainIndex, numOpsInChain);
    ShardSolver shardSolver = l1ChainConfig.resolveWithSolver(
        tensorTypePossibleLayouts, legalConfigs, usableL1CacheSize,
        overrideReshardEdges, overrideOutputLayout);

    if (l1ChainConfig.getState() == L1ChainState::Failed) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Failed to resolve L1 chain config {}", l1ChainConfig);
      progressTracker.finishL1Chain(firstOp, chainIndex, false);
      continue;
    }

    pickOpShardConfigs(shardSolver, l1ChainConfig);

    ShardSolverSolution resolvedShardSolution = shardSolver.finish();
    l1ChainConfig.complete(resolvedShardSolution.selectedOpConfig,
                           resolvedShardSolution.memReconfigEntryMap);

    // TODO(odjuricic): Add constraint check if op can write to dram.
    if (!resolvedShardSolution.selectedOpConfig[l1ChainConfig.getLastOp()]
             .outputLayout.hasDRAMBufferType()) {
      l1ChainConfig.spillEndToDRAM = true;
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "Resolved L1 chain config {}",
                 l1ChainConfig);

    progressTracker.finishL1Chain(firstOp, chainIndex, true);
  }

  progressTracker.finishAnalysis(funcToProcess);
}

void DFShardingPolicy::pickOpShardConfigs(ShardSolver &shardSolver,
                                          const L1ChainConfig &l1ChainConfig) {

  assert(l1ChainConfig.getState() == L1ChainState::Resolved);
  llvm::DenseMap<Operation *, SmallVector<float, 64>> accMaxCoreUsage =
      shardSolver.produceMaxCoreUsage();

  for (const auto &shardSpec : l1ChainConfig.getOpL1MemSpecs()) {
    Operation *op = shardSpec.op;
    ShardSolver::RemainingConfigAttrs validConfigs = shardSolver.at(op);
    const OpConfig *selectedConfig = validConfigs.begin().get();
    float maxCoreUsage = 0;
    for (auto configIterator = validConfigs.begin();
         configIterator != validConfigs.end(); ++configIterator) {
      if (accMaxCoreUsage[op][configIterator.index()] > maxCoreUsage) {
        maxCoreUsage = accMaxCoreUsage[op][configIterator.index()];
        selectedConfig = configIterator.get();
      } else if (accMaxCoreUsage[op][configIterator.index()] == maxCoreUsage) {
        assert(configIterator->outputLayout.getMemLayout() &&
               "TensorMemoryLayout is not set");
        // If we have a tie, prefer layout that is not BlockSharded.
        //
        if (configIterator->outputLayout.getMemLayout().getValue() !=
            ttnn::TensorMemoryLayout::BlockSharded) {
          selectedConfig = configIterator.get();
        }
      }
    }

    shardSolver.set(op, *selectedConfig);
  }
}

} // namespace mlir::tt::ttnn
