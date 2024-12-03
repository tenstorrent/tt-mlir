// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/BFInterleavedPolicy.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Scheduler/Scheduler.h"

namespace mlir::tt::ttnn {

void BFInterleavedPolicy::run() {
  for (Operation &funcOp : rootOp->getRegion(0).getOps()) {
    func::FuncOp func = dyn_cast<func::FuncOp>(funcOp);
    // DeviceAttr deviceAttr = getCurrentScopeDevice(func);

    // Start the policy.
    //
    mlir::tt::scheduler::Scheduler scheduler(&func);
    llvm::SmallVector<Operation *> scheduleableOps;

    while (scheduler.hasUnscheduledOps()) {
      scheduleableOps = scheduler.getScheduleableOps();
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
