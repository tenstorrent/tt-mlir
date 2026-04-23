// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGCOSTMODEL_H
#define TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGCOSTMODEL_H

#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>

namespace mlir::tt::stablehlo {

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

// Evaluate the cost of a sharding configuration by estimating communication
// overhead from CCL ops and memory benefit from tensor distribution across a
// mesh.
//
// `argDimSharded` encodes per-argument, per-dimension sharding decisions.
// Each inner vector corresponds to one function argument; each bool indicates
// whether that dimension is sharded on the mesh axis.
//
// Example:
//   func @main(%arg0: tensor<32x32xf32>, %arg1: tensor<64xf32>)
//   argDimSharded = [[true, false], [false]]
//     -> %arg0 is sharded on dim 0 (e.g. sdy.sharding<@mesh, [{"batch"}, {}]>)
//     -> %arg1 is replicated
//
// `module` is the lowered module (after shardy propagation + CCL insertion).
// `originalFuncOp` is the pre-lowering function (used for tensor metadata).
// `meshAxisSize` is the number of devices along the sharding axis.
ShardingResult
evaluate(ModuleOp module,
         const llvm::SmallVector<llvm::SmallVector<bool>> &argDimSharded,
         func::FuncOp originalFuncOp, int64_t meshAxisSize);

} // namespace mlir::tt::stablehlo

#endif // TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGCOSTMODEL_H
