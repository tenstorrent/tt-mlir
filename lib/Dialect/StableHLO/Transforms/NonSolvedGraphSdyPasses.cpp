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
#define GEN_PASS_DEF_RESHARDTOCOLLECTIVESPASS
#define GEN_PASS_DEF_CLOSESHARDINGSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// Wrapper for sdy::createApplyShardingConstraintsPass
class ApplyShardingConstraintsPass
    : public impl::ApplyShardingConstraintsPassBase<
          ApplyShardingConstraintsPass> {

public:
  using impl::ApplyShardingConstraintsPassBase<
      ApplyShardingConstraintsPass>::ApplyShardingConstraintsPassBase;

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    mlir::ModuleOp module = funcOp->getParentOfType<mlir::ModuleOp>();

    // Check if the graph is already solved
    if (shardy_utils::isGraphSolved(module)) {
      // Graph is already solved, skip this pass
      return;
    }

    // Run the actual SDY pass using a nested PassManager
    mlir::PassManager pm(&getContext());
    pm.addPass(mlir::sdy::createApplyShardingConstraintsPass());
    if (failed(pm.run(funcOp))) {
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

// Wrapper for sdy::createShardingConstraintToReshardPass
class ShardingConstraintToReshardPass
    : public impl::ShardingConstraintToReshardPassBase<
          ShardingConstraintToReshardPass> {

public:
  using impl::ShardingConstraintToReshardPassBase<
      ShardingConstraintToReshardPass>::ShardingConstraintToReshardPassBase;

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    mlir::ModuleOp module = funcOp->getParentOfType<mlir::ModuleOp>();

    // Check if the graph is already solved
    if (shardy_utils::isGraphSolved(module)) {
      // Graph is already solved, skip this pass
      return;
    }

    // Run the actual SDY pass using a nested PassManager
    mlir::PassManager pm(&getContext());
    pm.addPass(mlir::sdy::createShardingConstraintToReshardPass());
    if (failed(pm.run(funcOp))) {
      return signalPassFailure();
    }
  }
};

// Wrapper for sdy::createInsertExplicitReshardsPass
class InsertExplicitReshardsPass
    : public impl::InsertExplicitReshardsPassBase<InsertExplicitReshardsPass> {

public:
  using impl::InsertExplicitReshardsPassBase<
      InsertExplicitReshardsPass>::InsertExplicitReshardsPassBase;

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    mlir::ModuleOp module = funcOp->getParentOfType<mlir::ModuleOp>();

    // Check if the graph is already solved
    if (shardy_utils::isGraphSolved(module)) {
      // Graph is already solved, skip this pass
      return;
    }

    // Run the actual SDY pass using a nested PassManager
    mlir::PassManager pm(&getContext());
    mlir::sdy::InsertExplicitReshardsPassOptions options;
    options.enableFullVersion = true; // Use default full version setting
    pm.addPass(mlir::sdy::createInsertExplicitReshardsPass(options));
    if (failed(pm.run(funcOp))) {
      return signalPassFailure();
    }
  }
};

// Wrapper for sdy::createReshardToCollectivesPass
class ReshardToCollectivesPass
    : public impl::ReshardToCollectivesPassBase<ReshardToCollectivesPass> {

public:
  using impl::ReshardToCollectivesPassBase<
      ReshardToCollectivesPass>::ReshardToCollectivesPassBase;

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    mlir::ModuleOp module = funcOp->getParentOfType<mlir::ModuleOp>();

    // Check if the graph is already solved
    if (shardy_utils::isGraphSolved(module)) {
      // Graph is already solved, skip this pass
      return;
    }

    // Run the actual SDY pass using a nested PassManager
    mlir::PassManager pm(&getContext());
    pm.addPass(mlir::sdy::createReshardToCollectivesPass());
    if (failed(pm.run(funcOp))) {
      return signalPassFailure();
    }
  }
};

// Wrapper for sdy::createCloseShardingsPass
class CloseShardingsPass
    : public impl::CloseShardingsPassBase<CloseShardingsPass> {

public:
  using impl::CloseShardingsPassBase<
      CloseShardingsPass>::CloseShardingsPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Check if the graph is already solved
    if (shardy_utils::isGraphSolved(module)) {
      // Graph is already solved, skip this pass
      return;
    }

    // Run the actual SDY pass using a nested PassManager
    mlir::PassManager pm(&getContext());
    pm.addPass(mlir::sdy::createCloseShardingsPass());
    if (failed(pm.run(module))) {
      return signalPassFailure();
    }
  }
};

} // namespace mlir::tt::stablehlo
