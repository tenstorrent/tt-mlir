// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Pipelines/StableHLOPipelines.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"

namespace mlir::tt::stablehlo {

#define GEN_PASS_DEF_INSERTEXPLICITRESHARDSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Template for conditional SDY pass wrappers
//===----------------------------------------------------------------------===//

template <typename PassBase>
class ConditionalSdyPassWrapper : public PassBase {
public:
  using PassBase::PassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = this->getOperation();

    // Check if the graph is already solved
    if (shardy_utils::isGraphSolved(module)) {
      // If the graph is solved, skip running the SDY pass
      return;
    }

    // If the graph is not solved, run the SDY pass
    mlir::PassManager pm(&this->getContext());
    if (mlir::failed(addSdyPass(pm))) {
      return this->signalPassFailure();
    }

    if (mlir::failed(pm.run(module))) {
      return this->signalPassFailure();
    }
  }

protected:
  virtual mlir::LogicalResult addSdyPass(mlir::PassManager &pm) = 0;
};

//===----------------------------------------------------------------------===//
// Specialized pass implementations using the template
//===----------------------------------------------------------------------===//

// Module-level wrapper for sdy::createInsertExplicitReshardsPass
class InsertExplicitReshardsPass
    : public ConditionalSdyPassWrapper<
          impl::InsertExplicitReshardsPassBase<InsertExplicitReshardsPass>> {
protected:
  mlir::LogicalResult addSdyPass(mlir::PassManager &pm) override {
    mlir::sdy::InsertExplicitReshardsPassOptions options;
    options.enableFullVersion = true;
    pm.nest<mlir::func::FuncOp>().addPass(
        mlir::sdy::createInsertExplicitReshardsPass(options));
    return mlir::success();
  }
};

} // namespace mlir::tt::stablehlo
