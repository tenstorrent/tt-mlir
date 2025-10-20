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

#define GEN_PASS_DEF_AGGRESSIVEPROPAGATIONPASS
#define GEN_PASS_DEF_SHARDINGCONSTRAINTTORESHARDPASS
#define GEN_PASS_DEF_INSERTEXPLICITRESHARDSPASS
#define GEN_PASS_DEF_RESHARDTOCOLLECTIVESPASS
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

// Module-level wrapper for sdy::createAggressivePropagationPass
class AggressivePropagationPass
    : public ConditionalSdyPassWrapper<
          impl::AggressivePropagationPassBase<AggressivePropagationPass>> {
protected:
  mlir::LogicalResult addSdyPass(mlir::PassManager &pm) override {
    // This propagation is taken from
    // https://github.com/openxla/shardy/blob/0b8873d121008abc3edf7db2281f2b48cc647978/docs/sdy_propagation_passes.md?plain=1#L27.
    // Aggressive propagation is a wrapper ontop of basic propagation with
    // additional options user can set. With basic propagation, only shardings
    // that have no conflicts are propagated. With aggressive propagation, we
    // can set options to resolve conflicts and propagate more shardings.
    // However, sometimes, the propagation algorithm can be too aggressive and
    // propagate shardings that are not valid. To mitigate this, we set
    // conservativePropagation to true, which ensures that only shardings that
    // are valid are propagated.
    mlir::sdy::PropagationOptions propagationOptions;
    mlir::sdy::PropagationStrategy propagationStrategy =
        mlir::sdy::PropagationStrategy::Aggressive;
    propagationOptions.conservativePropagation = true;

    pm.addPass(mlir::sdy::createAggressivePropagationPass(propagationOptions,
                                                          propagationStrategy));
    return mlir::success();
  }
};

// Module-level wrapper for sdy::createShardingConstraintToReshardPass
class ShardingConstraintToReshardPass
    : public ConditionalSdyPassWrapper<
          impl::ShardingConstraintToReshardPassBase<
              ShardingConstraintToReshardPass>> {
protected:
  mlir::LogicalResult addSdyPass(mlir::PassManager &pm) override {
    pm.nest<mlir::func::FuncOp>().addPass(
        mlir::sdy::createShardingConstraintToReshardPass());
    return mlir::success();
  }
};

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

// Module-level wrapper for sdy::createReshardToCollectivesPass
class ReshardToCollectivesPass
    : public ConditionalSdyPassWrapper<
          impl::ReshardToCollectivesPassBase<ReshardToCollectivesPass>> {
protected:
  mlir::LogicalResult addSdyPass(mlir::PassManager &pm) override {
    pm.nest<mlir::func::FuncOp>().addPass(
        mlir::sdy::createReshardToCollectivesPass());
    return mlir::success();
  }
};

} // namespace mlir::tt::stablehlo
