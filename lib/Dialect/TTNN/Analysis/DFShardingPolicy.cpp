// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/DFShardingPolicy.h"

#include "ttmlir/Dialect/TT/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Scheduler/Scheduler.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Diagnostics.h"

namespace mlir::tt::ttnn {

void DFShardingPolicy::run() {
  rootOp->walk([&](func::FuncOp func) {
    if (ttmlir::utils::isConstEvalFunc(func)) {
      return;
    }

    deviceAttr = lookupDevice(func);
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
      scheduleableOps = scheduler.getScheduleableOps();

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
        scheduleableOps = scheduler.getScheduleableOps();

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

        if (nextOp) {

          // V0 Before adding to shard chain config check that currentOp is not
          // fork/join op.
          //
          bool validForSharding = currentOp->hasOneUse() &&
                                  legalConfigs.lookup(currentOp).size() > 0 &&
                                  legalConfigs.lookup(nextOp).size() > 0;

          // TODO(odjuricic): Skip all ops that don't support sharding due to
          // bugs and incomplete implementation. This needs to be addressed in
          // the near future.
          //
          if (llvm::isa<
                  ttnn::ZerosOp, ttnn::OnesOp,
                  // TODO(#3242): Re-enable once we are able to query backend
                  // for matmul.
                  ttnn::MatmulOp,
                  // TODO(#2038): Remove this once this bug is fixed.
                  // TODO(#2042): And constraints are implemented.
                  ttnn::ReshapeOp,
                  // TODO(#2038): Remove this once this bug is fixed.
                  ttnn::ConcatOp,
                  // TODO(#2041): Remove once constraints are added for MeanOp.
                  // TODO(#2084): Remove once constraints are added for all
                  // Llama ops.
                  ttnn::MeanOp, ttnn::SinOp, ttnn::CosOp, ttnn::ReciprocalOp,

                  // Following binary eltwise ops are blocked by metal issue
                  // https://github.com/tenstorrent/tt-metal/issues/21846
                  ttnn::AddOp, ttnn::SubtractOp, ttnn::MultiplyOp,

                  // TODO(#2588): Blocked by graph capture issue.
                  ttnn::MaxPool2dOp>(currentOp)) {
            validForSharding = false;
          }

          if (validForSharding) {
            // Fetch largest legal sharded L1 layouts for currentOp and nextOp.
            //
            // Calculate L1 tensor memory usage based on :
            // currentOp output tensor shard spec, nextOp exec and nextOp output
            // tensor.
            //
            OpConfig currentOpConfig = legalConfigs.lookup(currentOp).front();
            assert(
                currentOpConfig.outputLayout.hasShardedL1TensorMemoryLayout());
            uint64_t currentOpL1OutputUsage =
                currentOpConfig.outputLayout.getShardSizeInBytes();

            OpConfig nextOpConfig = legalConfigs.lookup(nextOp).front();
            assert(nextOpConfig.outputLayout.hasShardedL1TensorMemoryLayout());
            uint64_t nextOpL1OutputUsage =
                nextOpConfig.outputLayout.getShardSizeInBytes();

            // Figure out this const based on exec data, but will be replaced
            // with API.
            //
            constexpr float tensorL1UsageCap = 0.8;
            bool l1UsageValid = (currentOpL1OutputUsage + nextOpL1OutputUsage) <
                                tensorL1UsageCap * usableL1CacheSize;

            if (l1UsageValid) {
              // TODO(nobradovic)
              // It seems that some TTNN ops have constraints which prevent
              // them from being sharded if both inputs are interleaved,
              // so proposal for now is starting a shard chain
              // with reshard op. For this reason we also need to validate that
              // currentOp can fit into L1 with its first input sharded.
              //
              bool firstInputL1UsageValid = true;
              if (l1ChainConfigs->back().isEmpty() &&
                  (overrideReshardEdges.count(
                       Edge(currentOp->getOperand(0).getDefiningOp(), currentOp,
                            0)) > 0)) {
                RankedTensorType firstOpInputTensorType =
                    mlir::cast<RankedTensorType>(currentOp->getOperand(0)
                                                     .getDefiningOp()
                                                     ->getResult(0)
                                                     .getType());
                TTNNLayoutAttr firstOpInputLayout = mlir::cast<TTNNLayoutAttr>(
                    firstOpInputTensorType.getEncoding());

                TTNNLayoutAttr firstOpInputShardedLayout =
                    firstOpInputLayout
                        .withBufferType(
                            currentOpConfig.outputLayout.getBufferType())
                        .withMemoryLayout(
                            currentOpConfig.outputLayout.getMemLayout())
                        .withGrid(firstOpInputTensorType,
                                  currentOpConfig.outputLayout.getGrid());

                uint64_t firstInputL1Usage =
                    firstOpInputShardedLayout.getShardSizeInBytes();

                firstInputL1UsageValid =
                    (firstInputL1Usage + currentOpL1OutputUsage) <
                    tensorL1UsageCap * usableL1CacheSize;
              }

              if (firstInputL1UsageValid) {
                // Add to shard chain config.
                //
                OpL1MemSpec shardSpec;
                shardSpec.op = currentOp;

                // Hardcoded tensor split factor for now, until pipeline OP
                // support is added.
                //
                shardSpec.tensorSplitFactor = 1;
                l1ChainConfigs->back().addOpL1MemSpec(std::move(shardSpec));
                currentOp = nextOp;
                continue;
              }
            }
          }
        }

        currentOp = nullptr;
      }

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

  // Resolve shard chain configs.
  //
  for (L1ChainConfig &l1ChainConfig : *l1ChainConfigs) {
    ShardSolver shardSolver = l1ChainConfig.resolveWithSolver(
        tensorTypePossibleLayouts, legalConfigs, usableL1CacheSize,
        overrideReshardEdges);

    if (l1ChainConfig.getState() == L1ChainState::Failed) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Failed to resolve L1 chain config {}", l1ChainConfig);
      continue;
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "Resolved L1 chain config {}",
                 l1ChainConfig);

    pickOpShardConfigs(shardSolver, l1ChainConfig);

    ShardSolverSolution resolvedShardSolution = shardSolver.finish();
    l1ChainConfig.complete(resolvedShardSolution.selectedOpConfig,
                           resolvedShardSolution.memReconfigEntryMap);

    // TODO(odjuricic): Add constraint check if op can write to dram.
    if (!resolvedShardSolution.selectedOpConfig[l1ChainConfig.getLastOp()]
             .outputLayout.hasDRAMBufferType()) {
      l1ChainConfig.spillEndToDRAM = true;
    }
  }
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
