// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// Sharding Cost Model
// =============================================================================
//
// Scores a sharding configuration by estimating two competing factors:
//
//   net_cost = communication_cost - memory_benefit
//
// Communication cost: walks the lowered module, identifies stablehlo CCL ops
// (all_gather, reduce_scatter, etc.), and sums a per-op cost of
// (fixed_latency + weight * volume_fraction). This penalizes configs that
// introduce many or large collective operations.
//
// Memory benefit: estimates how much memory is saved by sharding argument
// tensors across the mesh. Weight/parameter tensors get a higher multiplier
// since they are persistent on-device.
//
// Notes:
// - CCL weights and magic numbers are heuristic, not hardware-calibrated.
//
// =============================================================================

#include "ttmlir/Dialect/StableHLO/Utils/ShardingCostModel.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::stablehlo {

// Return the element count of the largest argument tensor in `funcOp`.
// Used as a normalization factor so that costs are relative to the model's
// largest tensor rather than absolute element counts.
static int64_t computeMaxElements(func::FuncOp funcOp) {
  int64_t maxElements = 1;
  for (auto arg : funcOp.getArguments()) {
    if (auto tt = dyn_cast<RankedTensorType>(arg.getType())) {
      maxElements = std::max(maxElements, tt.getNumElements());
    }
  }
  return maxElements;
}

// Evaluate communication cost of a lowered module by counting and weighting
// CCL ops.
//
// Cost formula per CCL op:
//   cost_i = baseCCLLatency + cclWeight_i * (commElements_i / maxElements)
//
// - baseCCLLatency: fixed overhead per CCL (synchronization, kernel launch).
//   A graph with zero CCLs incurs zero latency. This is a heuristic constant
//   (experimentally estimated, not measured).
// - cclWeight: reflects op complexity (see getCCLWeight).
// - volumeFactor: fraction of data communicated relative to the largest tensor,
//   approximating bandwidth cost.
//
// Total communication cost is the sum over all CCL ops in the module.
static double evaluateCommunicationCost(ModuleOp module, int64_t maxElements) {
  // Heuristic constant: fixed per-CCL overhead for synchronization and setup.
  // Every CCL op incurs at least this much cost regardless of data volume,
  // ensuring that even a tiny CCL is never "free".
  // Experimentally estimated — not derived from hardware measurements.
  constexpr double baseCCLLatency = 1.0;

  double totalCost = 0.0;

  // module.walk is recursive across all functions in the module, so CCL ops
  // inside stablehlo.composite decomposition functions (created by
  // ReoutlineComposite) are counted as well.
  module.walk([&](Operation *op) {
    if (!isCCLOp(op)) {
      return;
    }

    double opWeight = getCCLWeight(op);

    int64_t commElements = 1;
    if (op->getNumOperands() > 0) {
      if (auto tt = dyn_cast<RankedTensorType>(op->getOperand(0).getType())) {
        commElements = tt.getNumElements();
      }
    }
    double volumeFactor =
        static_cast<double>(commElements) / static_cast<double>(maxElements);
    totalCost += baseCCLLatency + opWeight * volumeFactor;
  });

  return totalCost;
}

// Compute the memory benefit of a sharding configuration.
static double evaluateMemoryBenefit(
    const llvm::SmallVector<llvm::SmallVector<bool>> &argDimSharded,
    func::FuncOp funcOp, int64_t meshAxisSize, int64_t maxElements) {
  // With a single device, there is no distribution and fractionSaved = 0.
  if (meshAxisSize <= 1) {
    return 0.0;
  }

  // Heuristic multiplier: sharding weights/parameters is ~3x more valuable
  // than sharding activations because weights are persistent on-device and
  // dominate peak memory. Experimentally estimated, not measured.
  constexpr double parameterMultiplier = 3.0;

  double benefit = 0.0;

  // Fraction of memory saved per sharded tensor: (N-1)/N where N = mesh size.
  // When a tensor is sharded across N devices (meshAxisSize = N), each device
  // holds 1/N of the data, saving (1 - 1/N) of the original memory.
  double fractionSaved = 1.0 - 1.0 / static_cast<double>(meshAxisSize);

  for (size_t argIdx = 0; argIdx < argDimSharded.size(); ++argIdx) {
    bool isSharded =
        llvm::any_of(argDimSharded[argIdx], [](bool s) { return s; });
    if (!isSharded) {
      continue;
    }

    auto tensorType =
        cast<RankedTensorType>(funcOp.getArgument(argIdx).getType());
    int64_t numElements = tensorType.getNumElements();

    // Heuristic: tensors with rank <= 4 and > 1024 elements are likely model
    // weights/parameters rather than activations.
    double typeMultiplier = 1.0;
    if (tensorType.getRank() <= 4 && numElements > 1024) {
      typeMultiplier = parameterMultiplier;
    }

    benefit += typeMultiplier *
               (static_cast<double>(numElements) / maxElements) * fractionSaved;
  }

  return benefit;
}

ShardingResult
evaluate(ModuleOp module,
         const llvm::SmallVector<llvm::SmallVector<bool>> &argDimSharded,
         func::FuncOp originalFuncOp, int64_t meshAxisSize) {
  int64_t maxElements = computeMaxElements(originalFuncOp);
  double commCost = evaluateCommunicationCost(module, maxElements);
  double memBenefit = evaluateMemoryBenefit(argDimSharded, originalFuncOp,
                                            meshAxisSize, maxElements);
  return {commCost, memBenefit, commCost - memBenefit};
}

} // namespace mlir::tt::stablehlo
