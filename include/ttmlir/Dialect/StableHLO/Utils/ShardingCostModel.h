// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGCOSTMODEL_H
#define TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGCOSTMODEL_H

#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::stablehlo {

// Intermediate op result eligible for sdy.sharding_constraint.
struct ConstraintCandidate {
  size_t opIndex;
  int64_t rank;
};

// Per-dimension sharding: indices of mesh axes assigned to this dimension.
// Empty = replicated on all axes for this dim.
using DimShardSpec = llvm::SmallVector<size_t>;

// Per-tensor sharding option: one DimShardSpec per dimension.
using TensorShardOption = llvm::SmallVector<DimShardSpec>;

// Full sharding configuration for a module.
struct ShardingConfig {
  // Per-arg per-dim axis assignments.
  // argDimSharding[arg][dim] = list of mesh axis indices sharding that dim.
  llvm::SmallVector<TensorShardOption> argDimSharding;

  // Per-constraint-candidate target sharding.
  // std::nullopt = no constraint at this point.
  llvm::SmallVector<std::optional<TensorShardOption>> constraintTargets;
};

// Result of evaluating a sharding configuration's cost.
struct ShardingResult {
  double communicationCost; // Estimated cost of CCL ops in the lowered graph.
  double memoryBenefit;     // Estimated memory saved by distributing tensors.
  double netCost;           // communicationCost - memoryBenefit (lower is
                            // better).
};

// Heuristic weight for each CCL op type, reflecting relative communication
// cost. These are experimentally-driven estimates, not measured values.
//
// all_gather / collective_permute: 1.0 — single data movement, no reduction.
// reduce_scatter / all_reduce / all_to_all: 1.5 — reduction + data movement.
//
// A weight of 0.0 means the op is not a CCL and should be skipped.
inline double getCCLWeight(Operation *op) {
  return llvm::TypeSwitch<Operation *, double>(op)
      .Case<mlir::stablehlo::AllGatherOp>([](auto) { return 1.0; })
      .Case<mlir::stablehlo::ReduceScatterOp>([](auto) { return 1.5; })
      .Case<mlir::stablehlo::AllReduceOp>([](auto) { return 1.5; })
      .Case<mlir::stablehlo::AllToAllOp>([](auto) { return 1.5; })
      .Case<mlir::stablehlo::CollectivePermuteOp>([](auto) { return 1.0; })
      .Default([](Operation *) { return 0.0; });
}

// Returns true if the operation is a CCL (collective communication) op.
inline bool isCCLOp(Operation *op) { return getCCLWeight(op) > 0.0; }

// Return the element count of the largest argument tensor in `funcOp`.
int64_t computeMaxElements(func::FuncOp funcOp);

// Evaluate the cost of a sharding configuration by estimating communication
// overhead from CCL ops and memory benefit from tensor distribution across a
// mesh.
//
// `config` encodes per-argument, per-dimension sharding decisions using axis
// indices into the mesh.
//
// Example:
//   func @main(%arg0: tensor<32x32xf32>, %arg1: tensor<64xf32>)
//   config.argDimSharding = [
//     [{0}, {}],   // %arg0 sharded on dim 0 via axis 0, dim 1 replicated
//     [{}]         // %arg1 replicated
//   ]
//
// `module` is the lowered module (after shardy propagation + CCL insertion).
// `originalFuncOp` is the pre-lowering function (used for tensor metadata).
// `axisSizes` contains the size of each shardable mesh axis.
// `maxElementsOverride` overrides the normalization constant (useful for
// multi-layer models where a single outlier arg would dominate).
ShardingResult evaluate(ModuleOp module, const ShardingConfig &config,
                        func::FuncOp originalFuncOp,
                        llvm::ArrayRef<int64_t> axisSizes,
                        int64_t maxElementsOverride = 0);

} // namespace mlir::tt::stablehlo

#endif // TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGCOSTMODEL_H
