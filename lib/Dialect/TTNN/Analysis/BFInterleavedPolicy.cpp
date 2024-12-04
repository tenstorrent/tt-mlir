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

void BFInterleavedPolicy::run() {
  for (Operation &funcOp : rootOp->getRegion(0).getOps()) {
    func::FuncOp func = dyn_cast<func::FuncOp>(funcOp);
    DeviceAttr deviceAttr = getCurrentScopeDevice(func);
    DisjoinL1ChainConfigsUnion disjointL1ChainConfigsUnion;

    // Start the policy.
    //
    mlir::tt::scheduler::Scheduler scheduler(&func);
    llvm::SmallVector<Operation *> scheduleableOps;
    llvm::DenseMap<Operation *, uint64_t> currentL1Usage;

    while (scheduler.hasUnscheduledOps()) {
      uint64_t optimalChangeInL1Usage, currentChangeInL1Usage;
      Operation *nextOpForScheduling;
      TTNNLayoutAttr nextOpLayout;

      scheduleableOps = scheduler.getScheduleableOps();

      optimalChangeInL1Usage = std::numeric_limits<uint64_t>::max();
      for (Operation *op : scheduleableOps) {
        uint64_t operandsL1Usage, opL1OutputUsage;
        TTNNLayoutAttr opLayout;

        // Calculate the L1 memory usage of the op's operands.
        //
        operandsL1Usage = 0;
        for (auto operand : op->getOperands()) {
          // Skip block arguments (%arg0, %arg1, ...)
          //
          if (::llvm::isa<mlir::BlockArgument>(operand)) {
            continue;
          }

          Operation *operandOp = operand.getDefiningOp();

          // Skip non-analyzable operands.
          //
          if (isAnalyzable(operandOp) && currentL1Usage.count(operandOp)) {
            operandsL1Usage += currentL1Usage[operandOp];
          }
        }

        // Calculate the L1 memory usage of the op's output.
        //
        opL1OutputUsage = 0;
        opLayout = getDRAMLayout(op);
        if (isAnalyzable(op) && hasL1BufferType(op)) {
          opL1OutputUsage = utils::getOpOutputL1Usage(
              op, getL1InterleavedLayout(op), deviceAttr);
          opLayout = getL1InterleavedLayout(op);
        }

        // Calculate the change in L1 memory usage if the op is scheduled and
        // check if scheduling of this op leads to optimal change in L1 memory.
        //
        currentChangeInL1Usage = opL1OutputUsage - operandsL1Usage;
        if (currentChangeInL1Usage < optimalChangeInL1Usage) {
          optimalChangeInL1Usage = currentChangeInL1Usage;
          nextOpForScheduling = op;
          nextOpLayout = opLayout;
        }
      }

      // Schedule the nextOpForScheduling and update currentL1Usage.
      //
      scheduler.scheduleOp(nextOpForScheduling);
      for (auto operand : nextOpForScheduling->getOperands()) {
        // Skip block arguments (%arg0, %arg1, ...)
        //
        if (::llvm::isa<mlir::BlockArgument>(operand)) {
          continue;
        }

        Operation *operandOp = operand.getDefiningOp();

        // Skip non-analyzable operands.
        //
        if (isAnalyzable(operandOp) && currentL1Usage.count(operandOp)) {
          currentL1Usage.erase(operandOp);
        }
      }
      currentL1Usage[nextOpForScheduling] = utils::getOpOutputL1Usage(
          nextOpForScheduling, nextOpLayout, deviceAttr);
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
  return std::find_if(legalLayouts[op].begin(), legalLayouts[op].end(),
                      [](TTNNLayoutAttr layout) {
                        return layout.hasDRAMBufferType();
                      }) != legalLayouts[op].end();
}

TTNNLayoutAttr BFInterleavedPolicy::getDRAMLayout(Operation *op) {
  assert(hasDRAMBufferType(op));
  auto dramLayoutIter = std::find_if(
      legalLayouts[op].begin(), legalLayouts[op].end(),
      [](TTNNLayoutAttr layout) { return layout.hasDRAMBufferType(); });
  return *dramLayoutIter;
}

bool BFInterleavedPolicy::hasL1BufferType(Operation *op) {
  return std::find_if(legalLayouts[op].begin(), legalLayouts[op].end(),
                      [](TTNNLayoutAttr layout) {
                        return layout.hasInterleavedL1TensorMemoryLayout();
                      }) != legalLayouts[op].end();
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
