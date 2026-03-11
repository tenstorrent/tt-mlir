// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"
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
    ModuleOp moduleOp = getOperation();

    // Get L1 budget from system description and usage cap.
    ttcore::GridAttr deviceGrid =
        ttcore::lookupDevice(moduleOp).getWorkerGrid();
    ttcore::ChipDescAttr chipDesc = ttcore::getOpChipDescAttr(moduleOp);
    float tensorL1UsageCap = utils::getTensorL1UsageCap(moduleOp);
    uint64_t l1BudgetPerCore =
        static_cast<uint64_t>(tensorL1UsageCap * chipDesc.getUsableL1Size());

    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "L1 spill management: budget per core = {0} bytes "
                 "(usable = {1}, cap = {2})",
                 l1BudgetPerCore, chipDesc.getUsableL1Size(), tensorL1UsageCap);

    moduleOp->walk([&](func::FuncOp func) {
      if (!ttmlir::utils::isForwardDeviceFunc(func)) {
        return;
      }

      L1SpillManagement<SumL1MemoryTracker> spill(func, deviceGrid,
                                                  l1BudgetPerCore);
      spill.run();
    });
  }
};

} // namespace mlir::tt::ttnn
