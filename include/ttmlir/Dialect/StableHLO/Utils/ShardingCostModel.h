// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGCOSTMODEL_H
#define TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGCOSTMODEL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::stablehlo {

// Tier 2: intermediate op result eligible for sdy.sharding_constraint.
struct ConstraintCandidate {
  size_t opIndex;
  int64_t rank;
};

struct ShardingConfig {
  // Tier 1: per-arg per-dim sharding (true = sharded on the axis).
  llvm::SmallVector<llvm::SmallVector<bool>> argDimSharded;

  // Tier 2: per-constraint-candidate target sharding.
  // std::nullopt = no constraint at this point.
  // SmallVector<bool> = insert sdy.sharding_constraint with this target.
  llvm::SmallVector<std::optional<llvm::SmallVector<bool>>> constraintTargets;
};

struct ShardingResult {
  double communicationCost;
  double memoryBenefit;
  double netCost;
};

// Tunable parameters for ShardingCostModel.
struct ShardingCostModelOptions {
  double baseCCLLatency = 1.0;
  double parameterMultiplier = 3.0;
  // Weight for compute savings from sharding (FLOPs reduction).
  double computeBenefitWeight = 1.0;
  // Cost penalty for sharded function outputs that need gathering.
  double outputGatherCostWeight = 1.0;
};

// Cost model for evaluating sharding configurations by estimating communication
// overhead from CCL ops and memory benefit from tensor distribution across a
// mesh. Designed to be swappable (e.g., with a lookup-table approach) without
// changing the AutoSharding pass.
class ShardingCostModel {
public:
  using Options = ShardingCostModelOptions;

  ShardingCostModel();
  explicit ShardingCostModel(Options options);

  ShardingResult evaluate(ModuleOp module, const ShardingConfig &config,
                          func::FuncOp originalFuncOp,
                          int64_t meshAxisSize) const;

  static int64_t computeMaxElements(func::FuncOp funcOp);

private:
  Options options;

  double evaluateCommunicationCost(ModuleOp module, int64_t maxElements) const;

  double evaluateMemoryBenefit(const ShardingConfig &config,
                               func::FuncOp funcOp, int64_t meshAxisSize,
                               int64_t maxElements) const;

  double evaluateComputeBenefit(ModuleOp module, func::FuncOp originalFuncOp,
                                int64_t maxElements) const;

  double evaluateOutputShardingCost(ModuleOp module,
                                    int64_t maxElements) const;
};

} // namespace mlir::tt::stablehlo

#endif // TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGCOSTMODEL_H
