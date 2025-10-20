// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"

#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"
#include "ttmlir/Dialect/StableHLO/Pipelines/StableHLOPipelines.h"

namespace mlir::tt::stablehlo {

#define GEN_PASS_DEF_APPLYSHARDINGCONSTRAINTSPASS
#define GEN_PASS_DEF_AGGRESSIVEPROPAGATIONPASS
#define GEN_PASS_DEF_SHARDINGCONSTRAINTTORESHARDPASS
#define GEN_PASS_DEF_INSERTEXPLICITRESHARDSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// Module-level wrapper for sdy::createApplyShardingConstraintsPass
class ApplyShardingConstraintsPass
    : public impl::ApplyShardingConstraintsPassBase<
          ApplyShardingConstraintsPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Check if the graph is already solved
    if (shardy_utils::isGraphSolved(module)) {
      // Graph is already solved, skip this pass
      return;
    }

    mlir::PassManager pm(&getContext());
    pm.addPass(mlir::sdy::createApplyShardingConstraintsPass());

    if (failed(pm.run(module))) {
      return signalPassFailure();
    }
  }
};

// Wrapper for sdy::createAggressivePropagationPass
class AggressivePropagationPass
    : public impl::AggressivePropagationPassBase<AggressivePropagationPass> {

public:
  using impl::AggressivePropagationPassBase<
      AggressivePropagationPass>::AggressivePropagationPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Check if the graph is already solved
    if (shardy_utils::isGraphSolved(module)) {
      // Graph is already solved, skip this pass
      return;
    }

    // Run the actual SDY pass using a nested PassManager
    mlir::PassManager pm(&getContext());
    mlir::sdy::PropagationOptions propagationOptions;
    mlir::sdy::PropagationStrategy propagationStrategy =
        mlir::sdy::PropagationStrategy::Aggressive;
    propagationOptions.conservativePropagation =
        true; // Use default conservative setting
    pm.addPass(mlir::sdy::createAggressivePropagationPass(propagationOptions,
                                                          propagationStrategy));
    if (failed(pm.run(module))) {
      return signalPassFailure();
    }
  }
};

// Module-level wrapper for sdy::createShardingConstraintToReshardPass
class ShardingConstraintToReshardPass
    : public impl::ShardingConstraintToReshardPassBase<
          ShardingConstraintToReshardPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    if (shardy_utils::isGraphSolved(module)) {
      return;
    }

    mlir::PassManager pm(&getContext());
    pm.nest<mlir::func::FuncOp>().addPass(
        mlir::sdy::createShardingConstraintToReshardPass());

    if (failed(pm.run(module))) {
      return signalPassFailure();
    }
  }
};

// Module-level wrapper for sdy::createInsertExplicitReshardsPass
class InsertExplicitReshardsPass
    : public impl::InsertExplicitReshardsPassBase<InsertExplicitReshardsPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    if (shardy_utils::isGraphSolved(module)) {
      return;
    }

    mlir::PassManager pm(&getContext());
    mlir::sdy::InsertExplicitReshardsPassOptions options;
    options.enableFullVersion = true;
    pm.nest<mlir::func::FuncOp>().addPass(
        mlir::sdy::createInsertExplicitReshardsPass(options));

    if (failed(pm.run(module))) {
      return signalPassFailure();
    }
  }
};

} // namespace mlir::tt::stablehlo
