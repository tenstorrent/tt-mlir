// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGCOSTMODEL_H
#define TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGCOSTMODEL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt::stablehlo {

struct ShardingConfig {
  // Per-arg per-dim sharding (true = sharded on the axis).
  llvm::SmallVector<llvm::SmallVector<bool>> argDimSharded;
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
};

} // namespace mlir::tt::stablehlo

#endif // TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGCOSTMODEL_H
