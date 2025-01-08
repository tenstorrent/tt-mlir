// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/BFInterleavedPolicy.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Scheduler/Scheduler.h"

namespace mlir::tt::ttnn {

void BFInterleavedPolicy::run() {
  for (Operation &funcOp : rootOp->getRegion(0).getOps()) {
    func::FuncOp func = dyn_cast<func::FuncOp>(funcOp);
    mlir::tt::scheduler::Scheduler scheduler(&func);

    // Initialize the policy.
    //
    llvm::DenseMap<Operation *, OpL1MemUsage> currentL1UsagePerOp;
    uint64_t currentL1Usage = 0;
    l1ChainConfigs->push_back(L1ChainConfig());

    while (scheduler.hasUnscheduledOps()) {
      uint64_t minimalChangeInL1Usage;
      Operation *nextOpForScheduling;
      BufferType nextOpForSchedulingBufferType;

      nextOpForScheduling = nullptr;
      minimalChangeInL1Usage = std::numeric_limits<uint64_t>::max();
      for (Operation *op : scheduler.getScheduleableOps()) {
        uint64_t deallocOfL1Mem, allocOfL1Mem, changeInL1Usage;
        BufferType opBufferType;

        // Calculate the L1 memory usage of the op's operands.
        //
        deallocOfL1Mem = 0;
        walkOnAnalyzableOperands(op, [&](Operation *operandOp) {
          if (currentL1UsagePerOp.count(operandOp)) {
            deallocOfL1Mem +=
                (currentL1UsagePerOp[operandOp].numOfUnscheduledUsers == 1) *
                currentL1UsagePerOp[operandOp].l1MemUsagePerUser;
          }
        });

        // Default setup for all DRAM buffer type ops.
        //
        allocOfL1Mem = 0;
        opBufferType = BufferType::DRAM;

        // Analyse the possibility of scheduling the op with L1 memory layout.
        //
        if (hasL1BufferType(op)) {
          TTNNLayoutAttr layout = getL1InterleavedLayout(op);
          uint64_t opOutputL1Usage = utils::getOpOutputL1Usage(layout);

          if (currentL1Usage + opOutputL1Usage <= getAvailableL1CacheSize()) {
            allocOfL1Mem = opOutputL1Usage;
            opBufferType = BufferType::L1;
          }
        }

        // Check if the scheduling of the op is consuming the least amount of L1
        // memory among all the scheduleable ops.
        //
        changeInL1Usage = allocOfL1Mem - deallocOfL1Mem;
        if (changeInL1Usage < minimalChangeInL1Usage) {
          nextOpForScheduling = op;
          nextOpForSchedulingBufferType = opBufferType;
          minimalChangeInL1Usage = changeInL1Usage;
        }
      }

      // In case we picked the L1 layout for the nextOpForScheduling, we need
      // to add the OpL1MemSpec to the L1ChainConfig and update the state of L1
      // memory.
      //
      if (nextOpForSchedulingBufferType == BufferType::L1) {

        // Construct OpL1MemSpec for the nextOpForScheduling.
        //
        OpL1MemSpec opL1MemSpec;
        opL1MemSpec.op = nextOpForScheduling;
        opL1MemSpec.layout = getL1InterleavedLayout(nextOpForScheduling);
        l1ChainConfigs->back().addOpL1MemSpec(opL1MemSpec);

        // Update the state of L1 memory by allocating the nextOpForScheduling's
        // output tensor in L1 memory.
        //
        uint64_t numOfUsers = std::distance(nextOpForScheduling->user_begin(),
                                            nextOpForScheduling->user_end());
        currentL1UsagePerOp[nextOpForScheduling].l1MemUsagePerUser =
            utils::getOpOutputL1Usage(opL1MemSpec.layout);
        currentL1UsagePerOp[nextOpForScheduling].numOfUnscheduledUsers =
            numOfUsers;
        currentL1Usage +=
            currentL1UsagePerOp[nextOpForScheduling].l1MemUsagePerUser;
      }

      // Update the state of L1 memory.
      //
      walkOnAnalyzableOperands(nextOpForScheduling, [&](Operation *operandOp) {
        if (currentL1UsagePerOp.count(operandOp)) {
          currentL1UsagePerOp[operandOp].numOfUnscheduledUsers -= 1;
          if (currentL1UsagePerOp[operandOp].numOfUnscheduledUsers == 0) {
            // Only once we scheduled all the users of the operandOp, we can
            // free its output tensor from L1 memory.
            //
            currentL1Usage -= currentL1UsagePerOp[operandOp].l1MemUsagePerUser;
            currentL1UsagePerOp.erase(operandOp);
          }
        }
      });

      // Schedule the nextOpForScheduling and update currentL1Usage.
      //
      scheduler.scheduleOp(nextOpForScheduling);
    }

    // TODO: This is a temporary solution
    // Currently ReturnOps are not considered when calculating L1 usage
    llvm::SmallVector<mlir::Operation *> eraseableL1UsageOps;
    for (auto &[op, usage] : currentL1UsagePerOp) {
      for (Operation *user : op->getUsers()) {
        if (isa<mlir::func::ReturnOp>(user)) {
          usage.numOfUnscheduledUsers -= 1;
        }
      }
      if (usage.numOfUnscheduledUsers == 0) {
        eraseableL1UsageOps.push_back(op);
      }
    }

    for (Operation *op : eraseableL1UsageOps) {
      currentL1Usage -= currentL1UsagePerOp[op].l1MemUsagePerUser;
      currentL1UsagePerOp.erase(op);
    }

    assert(currentL1Usage == 0);
    assert(currentL1UsagePerOp.size() == 0);

    (*schedule)[func] = scheduler.getSchedule();

    // Build, Resolve and Complete all L1ChainConfigs.
    //
    for (L1ChainConfig &l1ChainConfig : *l1ChainConfigs) {
      l1ChainConfig.build();
      l1ChainConfig.resolve();
      l1ChainConfig.complete();
    }
  }
}

bool BFInterleavedPolicy::isAnalyzable(Operation *op) {
  // Skip operations that are not analyzed by the LegalGridAnalysis.
  //
  if (legalLayouts.count(op) > 0) {
    // Skip operations that are filterd out by the MemoryLayoutAnalysis.
    //
    return legalLayouts[op].size() > 0;
  }
  return false;
}

void BFInterleavedPolicy::walkOnAnalyzableOperands(
    Operation *op, function_ref<void(Operation *)> callback) {
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
      callback(operandOp);
    }
  }
}

bool BFInterleavedPolicy::hasDRAMBufferType(Operation *op) {
  if (legalLayouts.count(op)) {
    return std::find_if(legalLayouts[op].begin(), legalLayouts[op].end(),
                        [](TTNNLayoutAttr layout) {
                          return layout.hasDRAMBufferType();
                        }) != legalLayouts[op].end();
  }
  return false;
}

TTNNLayoutAttr BFInterleavedPolicy::getDRAMLayout(Operation *op) {
  assert(hasDRAMBufferType(op));
  auto dramLayoutIter = std::find_if(
      legalLayouts[op].begin(), legalLayouts[op].end(),
      [](TTNNLayoutAttr layout) { return layout.hasDRAMBufferType(); });
  return *dramLayoutIter;
}

bool BFInterleavedPolicy::hasL1BufferType(Operation *op) {
  if (legalLayouts.count(op)) {
    return std::find_if(legalLayouts[op].begin(), legalLayouts[op].end(),
                        [](TTNNLayoutAttr layout) {
                          return layout.hasInterleavedL1TensorMemoryLayout();
                        }) != legalLayouts[op].end();
  }
  return false;
}

TTNNLayoutAttr BFInterleavedPolicy::getL1InterleavedLayout(Operation *op) {
  assert(hasL1BufferType(op));
  auto l1InterleaveLayoutIter =
      std::find_if(legalLayouts[op].begin(), legalLayouts[op].end(),
                   [](TTNNLayoutAttr layout) {
                     return layout.hasInterleavedL1TensorMemoryLayout();
                   });
  return *l1InterleaveLayoutIter;
}

} // namespace mlir::tt::ttnn
