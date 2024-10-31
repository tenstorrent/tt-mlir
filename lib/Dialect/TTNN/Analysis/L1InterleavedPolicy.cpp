// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/L1InterleavedPolicy.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Scheduler/Scheduler.h"

namespace mlir::tt::ttnn {

void L1InterleavedPolicy::run(
    const std::unordered_set<Edge> &overrideReshardEdges) {
  rootOp->walk([&](func::FuncOp func) {
    mlir::tt::scheduler::Scheduler scheduler(&func);
    llvm::SmallVector<mlir::Operation *> scheduleableOps;
    Operation *currentOp = nullptr;
    llvm::DenseMap<Operation *, tt::LayoutAttr> selectedOpLayout;

    // TODO(fbajraktari): Algo
    //
    l1ChainConfigs->push_back(L1ChainConfig());
    while (scheduler.hasUnscheduledOps()) {
      scheduleableOps = scheduler.getScheduleableOps();
      currentOp = scheduleableOps[0];

      // Schedule currentOp.
      //
      scheduler.scheduleOp(currentOp);

      // Check if currentOp is valid l1 interleaved op.
      //
      if (legalLayouts.lookup(currentOp).size() > 0) {
        selectedOpLayout[currentOp] = legalLayouts.lookup(currentOp).front();

        // Add currentOp to l1 chain config.
        //
        OpL1MemSpec shardSpec;
        shardSpec.op = currentOp;

        // Hardcoded tensor split factor for now, until pipeline OP
        // support is added.
        //
        shardSpec.tensorSplitFactor = 1;
        l1ChainConfigs->back().addOpL1MemSpec(std::move(shardSpec));
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
      l1ChainConfig.build();
      l1ChainConfig.resolve();

      std::unordered_set<Edge> memReconfigEdges;
      l1ChainConfig.complete(selectedOpLayout, memReconfigEdges);
    }
  });
  llvm::errs() << "usableL1CacheSize: " << usableL1CacheSize << "\n";
}

} // namespace mlir::tt::ttnn
