// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/DFShardingPolicy.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Scheduler/Scheduler.h"

namespace mlir::tt::ttnn {

void DFShardingPolicy::run() {
  rootOp->walk([&](func::FuncOp func) {
    DeviceAttr deviceAttr = getCurrentScopeDevice(func);
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
                                  legalLayouts.lookup(currentOp).size() > 0 &&
                                  legalLayouts.lookup(nextOp).size() > 0;

          if (validForSharding) {
            // Fetch largest legal sharded L1 layouts for currentOp and nextOp.
            //
            // Calculate L1 tensor memory usage based on :
            // currentOp output tensor shard spec, nextOp exec and nextOp output
            // tensor.
            //
            tt::LayoutAttr currentOpLayout =
                legalLayouts.lookup(currentOp).front();
            assert(currentOpLayout.hasShardedL1TensorMemoryLayout());
            llvm::ArrayRef<int64_t> currentOpOutputTensorShape =
                mlir::cast<RankedTensorType>(currentOp->getResult(0).getType())
                    .getShape();
            uint64_t currentOpL1OutputUsage = deviceAttr.getLayoutSizeBytes(
                currentOpOutputTensorShape, currentOpLayout,
                currentOpLayout.getMemorySpace());

            tt::LayoutAttr nextOpLayout = legalLayouts.lookup(nextOp).front();
            assert(nextOpLayout.hasShardedL1TensorMemoryLayout());
            llvm::ArrayRef<int64_t> nextOpOutputTensorShape =
                mlir::cast<RankedTensorType>(nextOp->getResult(0).getType())
                    .getShape();
            uint64_t nextOpL1OutputUsage = deviceAttr.getLayoutSizeBytes(
                nextOpOutputTensorShape, nextOpLayout,
                nextOpLayout.getMemorySpace());

            // Figure out this const based on exec data, but will be replaced
            // with API.
            //
            constexpr float tensorL1UsageCap = 0.8;
            bool l1UsageValid = (currentOpL1OutputUsage + nextOpL1OutputUsage) <
                                tensorL1UsageCap * usableL1CacheSize;

            if (l1UsageValid) {
              // TODO(nobradovic)
              // It seems that bunch of TTNN ops have constraints which prevent
              // them from being sharded if both inputs are interleaved,
              // so proposal for now is starting a shard chain
              // with reshard op(at later phase only when necessary based on op
              // type) For this reason we also need to validate that currentOp
              // can fit into L1 with its first input sharded.
              //
              bool firstInputL1UsageValid = true;
              if (l1ChainConfigs->back().isEmpty()) {
                RankedTensorType firstOpInputTensorType =
                    mlir::cast<RankedTensorType>(currentOp->getOperand(0)
                                                     .getDefiningOp()
                                                     ->getResult(0)
                                                     .getType());
                tt::LayoutAttr firstOpInputLayout = mlir::cast<tt::LayoutAttr>(
                    firstOpInputTensorType.getEncoding());

                tt::LayoutAttr firstOpInputShardedLayout =
                    firstOpInputLayout
                        .withMemorySpace(currentOp->getContext(),
                                         currentOpLayout.getMemorySpace())
                        .withMemoryLayout(currentOp->getContext(),
                                          currentOpLayout.getMemLayout())
                        .withGrid(currentOp->getContext(),
                                  firstOpInputTensorType,
                                  currentOpLayout.getGrid());

                uint64_t firstInputL1Usage = deviceAttr.getLayoutSizeBytes(
                    firstOpInputTensorType.getShape(),
                    firstOpInputShardedLayout,
                    firstOpInputShardedLayout.getMemorySpace());

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
  for (auto &l1ChainConfig : *l1ChainConfigs) {
    ShardSolver shardSolver = l1ChainConfig.resolveWithSolver(
        legalLayouts, usableL1CacheSize, overrideReshardEdges);

    // TODO(nobradovic)
    // For now dummy fetch first legal(largest grid) for shard spec.
    //
    for (const auto &shardSpec : l1ChainConfig.getOpL1MemSpecs()) {
      Operation *op = shardSpec.op;
      auto validLayouts = shardSolver.at(op);
      shardSolver.set(op, *validLayouts.begin());
    }

    ShardSolverSolution resolvedShardSolution = shardSolver.finish();
    l1ChainConfig.complete(resolvedShardSolution.selectedOpLayout,
                           resolvedShardSolution.memReconfigEdges);
  }
}

} // namespace mlir::tt::ttnn
