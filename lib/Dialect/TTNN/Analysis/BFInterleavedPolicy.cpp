// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/BFInterleavedPolicy.h"
#include "ttmlir/Dialect/TTNN/Analysis/DisjointL1ChainConfigsUnion.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Scheduler/Scheduler.h"

namespace mlir::tt::ttnn {

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

void BFInterleavedPolicy::run() {
  for (Operation &funcOp : rootOp->getRegion(0).getOps()) {
    func::FuncOp func = dyn_cast<func::FuncOp>(funcOp);
    DeviceAttr deviceAttr = getCurrentScopeDevice(func);
    mlir::tt::scheduler::Scheduler scheduler(&func);

    // Initialize the policy.
    //
    llvm::DenseMap<Operation *, OpL1MemUsage> currentL1UsagePerOp;
    uint64_t currentL1Usage = 0;
    DisjoinL1ChainConfigsUnion disjointL1ChainConfigsUnion;

    while (scheduler.hasUnscheduledOps()) {
      uint64_t minimalChangeInL1Usage;
      Operation *nextOpForScheduling;
      NextOpType nextOpForSchedulingType;

      nextOpForScheduling = nullptr;
      minimalChangeInL1Usage = std::numeric_limits<uint64_t>::max();
      for (Operation *op : scheduler.getScheduleableOps()) {
        bool hasL1Operands;
        uint64_t deallocOfL1Mem, allocOfL1Mem, changeInL1Usage;
        NextOpType opType;

        // In case there are no ops whose output tensor can be stored in L1
        // memory buffer, we have to pick one op to schedule and place its
        // output tensor in DRAM memory buffer.
        //
        if (nextOpForScheduling == nullptr) {
          nextOpForScheduling = op;
          nextOpForSchedulingType = NextOpType::NoL1ChainConfigOp;
        }

        // Calculate the L1 memory usage of the op's operands.
        //
        hasL1Operands = false;
        deallocOfL1Mem = 0;
        walkOnAnalyzableOperands(op, [&](Operation *operandOp) {
          if (currentL1UsagePerOp.count(operandOp)) {
            hasL1Operands = true;
            deallocOfL1Mem +=
                (currentL1UsagePerOp[operandOp].numOfUnscheduledUsers == 1) *
                currentL1UsagePerOp[operandOp].l1MemUsagePerUser;
          }
        });

        // In case the op has no legal L1 memory layout, the op can only be of
        // type NextOpType::NoL1ChainConfigOp.
        //
        allocOfL1Mem = 0;
        opType = NextOpType::NoL1ChainConfigOp;
        if (hasL1BufferType(op)) {
          TTNNLayoutAttr layout = getL1InterleavedLayout(op);
          allocOfL1Mem = utils::getOpOutputL1Usage(op, layout, deviceAttr);

          // Determine the type of the op based on the operandsL1Usage. If
          // there is at least one op's operand whose tensor is in L1 memory,
          // then the op can be assigned to the existing L1ChainConfig and in
          // that case the op's type should be
          // NextOpType::MergeL1ChainConfigsOp. Otherwise, new L1ChainConfig
          // should be created and NextOpType::NewL1ChainConfigOp assigned as
          // the op's type.
          //
          opType = (hasL1Operands ? NextOpType::MergeL1ChainConfigsOp
                                  : NextOpType::NewL1ChainConfigOp);
        }

        // Check if the scheduling of the op is consumes the least amount of L1
        // memory among all the scheduleable ops.
        //
        changeInL1Usage = allocOfL1Mem - deallocOfL1Mem;
        bool isOpExecutionLegal =
            currentL1Usage + allocOfL1Mem <= getAvailableL1CacheSize();
        bool isBestOptionSoFar = changeInL1Usage < minimalChangeInL1Usage;
        if (isOpExecutionLegal && isBestOptionSoFar) {
          nextOpForScheduling = op;
          nextOpForSchedulingType = opType;
          minimalChangeInL1Usage = changeInL1Usage;
        }
      }

      // Update DisjointL1ChainConfigsUnion with the nextOpForScheduling.
      //
      if (nextOpForSchedulingType != NextOpType::NoL1ChainConfigOp) {

        // Construct OpL1MemSpec for the nextOpForScheduling.
        //
        OpL1MemSpec opL1MemSpec;
        opL1MemSpec.op = nextOpForScheduling;
        opL1MemSpec.layout = getL1InterleavedLayout(nextOpForScheduling);

        // Update the state of L1 memory by allocating the nextOpForScheduling's
        // output tensor in L1 memory.
        //
        uint64_t numOfUsers = std::distance(nextOpForScheduling->user_begin(),
                                            nextOpForScheduling->user_end());
        currentL1UsagePerOp[nextOpForScheduling].l1MemUsagePerUser =
            utils::getOpOutputL1Usage(nextOpForScheduling, opL1MemSpec.layout,
                                      deviceAttr);
        currentL1UsagePerOp[nextOpForScheduling].numOfUnscheduledUsers =
            numOfUsers;
        currentL1Usage +=
            currentL1UsagePerOp[nextOpForScheduling].l1MemUsagePerUser;

        // Merge L1ChainConfigs of the operands of the nextOpForScheduling if
        // there are op's operands that are in L1 memory.
        //
        Operation *opIter = nullptr;
        walkOnAnalyzableOperands(
            nextOpForScheduling, [&](Operation *operandOp) {
              if (currentL1UsagePerOp.count(operandOp)) {
                // Merge L1ChainConfigs of the op's operands.
                //
                opIter = disjointL1ChainConfigsUnion.mergeL1ChainConfigs(
                    opIter, operandOp);
              }
            });

        // Insert the nextOpForScheduling's OpL1MemSpec in the
        // DisjointL1ChainConfigsUnion.
        //
        disjointL1ChainConfigsUnion.insertOpL1MemSpec(opL1MemSpec, opIter);
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
      nextOpForScheduling = nullptr;
    }

    assert(currentL1Usage == 0);
    assert(currentL1UsagePerOp.size() == 0);

    (*schedule)[func] = scheduler.getSchedule();

    // Build, Resolve and Complete all L1ChainConfigs.
    //
    for (auto &[op, l1ChainConfig] :
         disjointL1ChainConfigsUnion.getL1ChainConfigs()) {
      l1ChainConfig.build();
      l1ChainConfig.resolve();
      l1ChainConfig.complete();
      l1ChainConfigs->push_back(l1ChainConfig);
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
