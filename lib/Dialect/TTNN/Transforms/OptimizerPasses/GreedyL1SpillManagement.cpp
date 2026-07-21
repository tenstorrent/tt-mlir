// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTNNGreedyL1SpillManagement: the follow-up stage of the greedy memory-layout
// optimizer that runs after layout propagation has assigned L1/DRAM layouts.
// Its job is to enforce the per-core L1 memory budget: layout propagation may
// have placed more tensors in L1 than physically fit at some point in the
// schedule, so this pass walks each forward device function and, using a
// farthest-last-use eviction strategy, spills L1 tensors back to DRAM until
// every op re-validates within budget. The eviction and validation logic lives
// in the L1SpillManagement analysis; this pass computes the budget, runs the
// analysis per function, and propagates its result and failure state.
//
// Key inputs:
//   - The ModuleOp IR; only forward device functions are processed.
//   - The worker grid from the device description.
//   - The usable L1 budget per core, derived from the chip descriptor's usable
//     L1 size minus the reserved usage and tensor usage cap.
//   - Backend op-constraint validation, which re-checks each op against the
//     running L1 pressure; the pass is only functional when built with
//     TTMLIR_ENABLE_OPMODEL.
//   - Pass options enableDecisionTrace and decisionTraceDir.
//
// Key outputs (all mutations of the module in place):
//   - ToMemoryConfigOp ops inserted after spilled ops to move tensors from L1
//     to DRAM, with consuming uses reconnected to read the spilled tensor.
//   - D2M subgraph function types re-synced to match dispatch operand types
//     that changed to DRAM as a result of spilling.
//   - A pass failure signalled (and the module walk interrupted) when the
//     analysis reports an unrecoverable condition.
//   - Optionally, spill-management data merged into the decision-trace JSON
//     produced earlier by layout propagation.

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h"
#include "ttmlir/Dialect/TTNN/Diagnostics/DecisionTrace.h"
#include "ttmlir/Dialect/TTNN/Utils/D2MOptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNGREEDYL1SPILLMANAGEMENT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNGreedyL1SpillManagement
    : public impl::TTNNGreedyL1SpillManagementBase<
          TTNNGreedyL1SpillManagement> {
public:
  using impl::TTNNGreedyL1SpillManagementBase<
      TTNNGreedyL1SpillManagement>::TTNNGreedyL1SpillManagementBase;

  void runOnOperation() final {
#ifndef TTMLIR_ENABLE_OPMODEL
    llvm::llvm_unreachable_internal(
        "TTNNGreedyL1SpillManagement pass requires OpModel support to be "
        "enabled.");
#else
    op_model::ScopedSingletonDeviceGuard deviceGuard(getOperation());

    ModuleOp moduleOp = getOperation();

    ttcore::GridAttr deviceGrid =
        ttcore::lookupDevice(moduleOp).getWorkerGrid();
    uint64_t l1BudgetPerCore = utils::getUsableL1PerCore(moduleOp);

    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "L1 spill management: budget per core = {0} bytes "
                 "(usable = {1}, cap = {2}, reserved = {3})",
                 l1BudgetPerCore,
                 ttcore::getOpChipDescAttr(moduleOp).getUsableL1Size(),
                 utils::getTensorL1UsageCap(moduleOp),
                 utils::getReservedL1Usage(moduleOp));

    moduleOp->walk([&](func::FuncOp func) -> WalkResult {
      if (!ttmlir::utils::isForwardDeviceFunc(func)) {
        return WalkResult::advance();
      }

      // Create observer if tracing is enabled.
      std::unique_ptr<L1SpillObserver> observer;
      if (enableDecisionTrace) {
        observer = std::make_unique<DecisionTraceObserver>();
      }

      L1SpillManagement<SumL1MemoryTracker> spill(
          func, deviceGrid, l1BudgetPerCore, std::move(observer));
      spill.run();

      // run() emits a diagnostic but cannot fail the pass on its own; surface
      // any unrecoverable condition (e.g. an op whose CBs overlap a required
      // inserted-reshard input) as a pass failure. Interrupt the walk too: the
      // pass is already doomed, so don't keep mutating later forward-device
      // funcs.
      if (spill.hasFailed()) {
        signalPassFailure();
        return WalkResult::interrupt();
      }

      // Sync D2M subgraph function types to match dispatch op's current inputs
      // (e.g. after spill, operand types may have changed to DRAM).
      d2m_optimizer_utils::syncAllD2MFuncTypes(func);

      // Merge spill management data into the existing decision trace JSON.
      if (enableDecisionTrace) {
        if (const DecisionTrace *dt = spill.getObserver()->getDecisionTrace()) {
          if (DecisionTrace::mergeSpillTrace(decisionTraceDir, func.getName(),
                                             *dt)) {
            TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                         "Merged spill management trace into {0}/{1}",
                         decisionTraceDir, func.getName());
          } else {
            TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                         "Failed to merge spill management trace for func "
                         "{0}; layout propagation may not have written a "
                         "decision trace to {1}.",
                         func.getName(), decisionTraceDir);
          }
        }
      }

      return WalkResult::advance();
    });
#endif
  }
};

} // namespace mlir::tt::ttnn
