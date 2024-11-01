// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/L1InterleavedPolicy.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Scheduler/Scheduler.h"

namespace mlir::tt::ttnn {

uint64_t getOpOutputLayoutUsage(
    Operation *op,
    llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>> &legalLayouts,
    DeviceAttr &deviceAttr) {
  tt::LayoutAttr opLayout = legalLayouts.lookup(op).front();
  assert(opLayout.hasInterleavedL1TensorMemoryLayout());

  llvm::ArrayRef<int64_t> opOutputTensorShape =
      mlir::cast<RankedTensorType>(op->getResult(0).getType()).getShape();

  uint64_t opL1OutputUsage = deviceAttr.getLayoutSizeBytes(
      opOutputTensorShape, opLayout, opLayout.getMemorySpace());
  return opL1OutputUsage;
}

void L1InterleavedPolicy::run() {
  rootOp->walk([&](func::FuncOp func) {
    DeviceAttr deviceAttr = getCurrentScopeDevice(func);
    mlir::tt::scheduler::Scheduler scheduler(&func);
    llvm::SmallVector<mlir::Operation *> scheduleableOps;
    llvm::DenseMap<Operation *, tt::LayoutAttr> selectedOpLayout;
    Operation *currentOp = nullptr;

    // TODO(fbajraktari):
    //
    l1ChainConfigs->push_back(L1ChainConfig());
    while (scheduler.hasUnscheduledOps()) {
      scheduleableOps = scheduler.getScheduleableOps();

      // Before starting a l1 chain, schedule layout/memory management ops
      // first until they are exhausted from schedulable ops.
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

          // V1: Check that currentOp is not fork/join op.
          //
          bool validForL1Interleaved =
              currentOp->hasOneUse() &&
              legalLayouts.lookup(currentOp).size() > 0 &&
              legalLayouts.lookup(nextOp).size() > 0;

          if (validForL1Interleaved) {
            // Figure out this const based on exec data, but will be replaced
            // with API.
            //
            constexpr float tensorL1UsageCap = 0.8;
            uint64_t currentOpL1OutputUsage =
                getOpOutputLayoutUsage(currentOp, legalLayouts, deviceAttr);
            uint64_t nextOpL1OutputUsage =
                getOpOutputLayoutUsage(nextOp, legalLayouts, deviceAttr);
            bool l1UsageValid = (currentOpL1OutputUsage + nextOpL1OutputUsage) <
                                tensorL1UsageCap * usableL1CacheSize;

            if (l1UsageValid) {
              selectedOpLayout[currentOp] =
                  legalLayouts.lookup(currentOp).front();

              // Add currentOp to l1 chain config.
              //
              OpL1MemSpec shardSpec;
              shardSpec.op = currentOp;

              // Hardcoded tensor split factor for now, until pipeline OP
              // support is added.
              //
              shardSpec.tensorSplitFactor = 1;
              l1ChainConfigs->back().addOpL1MemSpec(std::move(shardSpec));

              // Update currentOp pointer.
              //
              currentOp = nextOp;
              continue;
            }
          }
        }

        currentOp = nullptr;
        if (!l1ChainConfigs->back().isEmpty()) {
          l1ChainConfigs->back().build();
          l1ChainConfigs->push_back(L1ChainConfig());
        }
      }
    }

    if (l1ChainConfigs->back().isEmpty()) {
      l1ChainConfigs->pop_back();
    }

    // Schedule
    //
    (*schedule)[func] = scheduler.getSchedule();

    // Resolve l1 chain configs.
    //
    for (auto &l1ChainConfig : *l1ChainConfigs) {
      l1ChainConfig.resolve();

      std::unordered_set<Edge> memReconfigEdges;
      l1ChainConfig.complete(selectedOpLayout, memReconfigEdges);
    }
  });
}

} // namespace mlir::tt::ttnn
