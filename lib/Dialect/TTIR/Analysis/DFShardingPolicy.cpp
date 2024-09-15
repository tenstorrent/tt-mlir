// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/DFShardingPolicy.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Scheduler/Scheduler.h"

namespace mlir::tt::ttir {

void DFShardingPolicy::run() {
  rootOp->walk([&](func::FuncOp func) {
    mlir::tt::scheduler::Scheduler scheduler(&func);
    shardChainConfigs->push_back(ShardChainConfig());
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

      // Before starting a sharding chain, schedule ttir.to_layout ops first
      // until they are exhausted from schedulable ops.
      // TODO(nobradovic) :
      // We need to examine type of to_layout op and determine if for
      // example we have a space in DRAM to perform this?(system->dram, double
      // check this)
      //
      if (shardChainConfigs->back().isEmpty()) {
        for (auto *op : scheduleableOps) {
          if (isa<ttir::ToLayoutOp>(op)) {
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

      // Skip sharding process if currentOp is a ttir.to_layout op.
      //
      if (shardChainConfigs->back().isEmpty() &&
          isa<ttir::ToLayoutOp>(currentOp)) {
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
                                  legalGrids.lookup(currentOp).size() > 0 &&
                                  legalGrids.lookup(nextOp).size() > 0;

          // TODO(nobradovic)
          // It seems that bunch of TTNN ops have constraints which prevent
          // them from being sharded if both inputs are interleaved, so my
          // proposal is to try to resolve this by starting a shard chain with
          // reshard op(at later phase only when necessary)
          //

          if (validForSharding) {
            // Fetch largest legal sharded L1 grids for currentOp and nextOp.
            //
            // Calculate L1 tensor memory usage based on :
            // currentOp output tensor shard spec, nextOp exec and nextOp output
            // tensor.
            //
            LayoutAttr currentOpLayout = legalGrids.lookup(currentOp).front();
            llvm::SmallVector<int64_t> currentOpShardShape =
                currentOpLayout.getShardShape(false /*convertTileToScalar*/);
            uint64_t l1InputUsage = currentOpLayout.getElementSizeBytes();
            for (int64_t dim : currentOpShardShape) {
              l1InputUsage *= dim;
            }

            LayoutAttr nextOpLayout = legalGrids.lookup(nextOp).front();
            llvm::SmallVector<int64_t> nextOpShardShape =
                nextOpLayout.getShardShape(false /*convertTileToScalar*/);
            uint64_t l1OutputUsage = nextOpLayout.getElementSizeBytes();
            for (int64_t dim : nextOpShardShape) {
              l1OutputUsage *= dim;
            }

            // Figure out this const based on exec data, but will be replaced
            // with API.
            //
            constexpr float tensorL1UsageCap = 0.8;
            bool l1UsageValid = (l1InputUsage + l1OutputUsage) <
                                tensorL1UsageCap * usableL1CacheSize;

            if (l1UsageValid) {
              // Add to shard chain config.
              //
              ShardSpec shardSpec;
              shardSpec.op = currentOp;

              // Hardcoded tensor split factor for now, until pipeline OP
              // support is added.
              //
              shardSpec.tensorSplitFactor = 1;
              shardChainConfigs->back().addShardSpec(std::move(shardSpec));
              currentOp = nextOp;
              continue;
            }
          }
        }

        currentOp = nullptr;
      }

      if (!shardChainConfigs->back().isEmpty()) {
        shardChainConfigs->back().build();
        shardChainConfigs->push_back(ShardChainConfig());
      }
    }

    (*schedule)[func] = scheduler.getSchedule();
  });

  if (shardChainConfigs->back().isEmpty()) {
    shardChainConfigs->pop_back();
  }

  // Resolve shard chain configs.
  //
  for (auto &shardChainConfig : *shardChainConfigs) {
    ShardSolver shardSolver =
        shardChainConfig.resolve(legalGrids, usableL1CacheSize);

    // TODO(nobradovic)
    // For now dummy fetch first legal(largest grid) for shard spec.
    //
    for (const auto &shardSpec : shardChainConfig.getShardSpecs()) {
      Operation *op = shardSpec.op;
      auto validLayouts = shardSolver.at(op);
      shardSolver.set(op, *validLayouts.begin());
    }

    ShardSolverSolution resolvedShardSolution = shardSolver.finish();
    shardChainConfig.complete(resolvedShardSolution.selectedOpLayout,
                              resolvedShardSolution.reshardedEdges);
  }
}

} // namespace mlir::tt::ttir
