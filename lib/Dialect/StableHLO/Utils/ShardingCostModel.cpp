// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// Sharding Cost Model
// =============================================================================
//
// Scores a sharding configuration by estimating four competing factors:
//
//   net_cost = (communication_cost + output_gather_cost)
//            - (memory_benefit + compute_benefit)
//
// Communication cost: walks the lowered module, identifies stablehlo CCL ops
// (all_gather, reduce_scatter, etc.) and composite CCLs (sdy.all_slice, etc.),
// sums a per-op cost of (baseCCLLatency + weight) * volume_factor.
// CCLs on the critical path before heavy compute get an additional penalty.
//
// Output gather cost: penalizes sharded function outputs that need implicit
// gathering at the function boundary.
//
// Memory benefit: estimates how much memory is saved by sharding argument
// tensors across the mesh. When ttcore.argument_type annotations are present,
// uses them to distinguish parameters (high multiplier) from activations
// (baseline). Falls back to a sigmoid heuristic for unannotated modules.
//
// Compute benefit: estimates FLOPs reduction from sharding by comparing
// dot_general operations between sharded and original modules.
//
// Notes:
// - CCL weights and magic numbers are heuristic, not hardware-calibrated.
//
// =============================================================================

#include "ttmlir/Dialect/StableHLO/Utils/ShardingCostModel.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "shardy/dialect/sdy/ir/dialect.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"

#include <cmath>

namespace mlir::tt::stablehlo {

//===----------------------------------------------------------------------===//
// CCL weight lookup by op name (for composite ops).
//===----------------------------------------------------------------------===//

static double getCCLWeight(llvm::StringRef opName,
                           llvm::StringRef compositeName) {
  if (opName == "stablehlo.all_gather") {
    return 1.0;
  }
  if (opName == "stablehlo.reduce_scatter" ||
      opName == "stablehlo.all_reduce" || opName == "stablehlo.all_to_all") {
    return 1.5;
  }
  if (opName == "stablehlo.collective_permute") {
    return 1.0;
  }
  if (!compositeName.empty()) {
    if (compositeName == "sdy.all_slice") {
      return 0.5;
    }
    if (compositeName == "sdy.all_gather") {
      return 1.0;
    }
    if (compositeName == "sdy.reduce_scatter" ||
        compositeName == "sdy.all_reduce") {
      return 1.5;
    }
  }
  return 0.0;
}

//===----------------------------------------------------------------------===//
// Utility.
//===----------------------------------------------------------------------===//

// Return the element count of the largest argument tensor in `funcOp`.
// Used as a normalization factor so that costs are relative to the model's
// largest tensor rather than absolute element counts.
int64_t computeMaxElements(func::FuncOp funcOp) {
  int64_t maxElements = 1;
  for (auto arg : funcOp.getArguments()) {
    if (auto tt = dyn_cast<RankedTensorType>(arg.getType())) {
      maxElements = std::max(maxElements, tt.getNumElements());
    }
  }
  return maxElements;
}

//===----------------------------------------------------------------------===//
// Internal cost model components.
//===----------------------------------------------------------------------===//

// Check if a CCL op feeds directly into a heavy compute op (dot_general).
// CCLs on the critical path before matmuls serialize communication and
// compute — these deserve a penalty.
static bool isCCLBeforeCompute(Operation *cclOp) {
  if (cclOp->getNumResults() == 0) {
    return false;
  }
  Value result = cclOp->getResult(0);
  for (OpOperand &use : result.getUses()) {
    if (isa<mlir::stablehlo::DotGeneralOp>(use.getOwner())) {
      return true;
    }
  }
  return false;
}

// Evaluate communication cost by counting and weighting CCL ops.
// Each CCL incurs a fixed latency overhead plus a bandwidth cost proportional
// to the volume of data communicated. CCLs that feed directly into heavy
// compute (dot_general) get a position penalty.
//
// Cost formula per CCL op:
//   cost_i = (baseCCLLatency + cclWeight_i) * (commElements_i / maxElements)
//
// - baseCCLLatency: fixed overhead per CCL (synchronization, kernel launch).
// - cclWeight: reflects op complexity (see getCCLWeight).
// - volumeFactor: ratio of data communicated to the largest *argument* tensor,
//   approximating bandwidth cost. This can exceed 1.0 when CCL ops operate on
//   intermediate tensors larger than any input.
static double evaluateCommunicationCost(ModuleOp module, int64_t maxElements) {
  // Heuristic constant: fixed per-CCL overhead for synchronization and setup.
  constexpr double baseCCLLatency = 1.0;
  constexpr double kCriticalPathPenalty = 0.5;
  double totalCost = 0.0;

  module.walk([&](Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();
    llvm::StringRef compositeName;
    if (opName == "stablehlo.composite") {
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name")) {
        compositeName = nameAttr.getValue();
      }
    }

    double opWeight = getCCLWeight(opName, compositeName);
    if (opWeight <= 0.0 || maxElements <= 0) {
      return;
    }

    int64_t commElements = 1;
    if (op->getNumOperands() > 0) {
      if (auto tt = dyn_cast<RankedTensorType>(op->getOperand(0).getType())) {
        commElements = tt.getNumElements();
      }
    }
    double volumeFactor =
        static_cast<double>(commElements) / static_cast<double>(maxElements);
    double cost = (baseCCLLatency + opWeight) * volumeFactor;

    if (isCCLBeforeCompute(op)) {
      cost += kCriticalPathPenalty;
    }

    totalCost += cost;
  });

  return totalCost;
}

// Compute memory benefit of a sharding config. For multi-axis meshes, the
// split factor is the product of all axis sizes assigned to any dimension.
//
// When ttcore.argument_type annotations are present, they drive the type
// multiplier: Parameter/Constant args get the full parameterMultiplier while
// Input args get a baseline of 1.0. When no annotations exist (standalone
// subgraphs), falls back to a sigmoid heuristic based on tensor size.
static double evaluateMemoryBenefit(const ShardingConfig &config,
                                    func::FuncOp funcOp,
                                    llvm::ArrayRef<int64_t> axisSizes,
                                    int64_t maxElements) {
  // Heuristic multiplier: sharding weights/parameters is ~3x more valuable
  // than sharding activations because weights are persistent on-device and
  // dominate peak memory.
  constexpr double parameterMultiplier = 3.0;

  bool hasAnnotations = false;
  for (size_t i = 0; i < funcOp.getNumArguments(); ++i) {
    if (funcOp.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
            i, ttcore::ArgumentTypeAttr::name)) {
      hasAnnotations = true;
      break;
    }
  }

  double benefit = 0.0;

  for (size_t argIdx = 0; argIdx < config.argDimSharding.size(); ++argIdx) {
    int64_t splitFactor = 1;
    for (const auto &dimAxes : config.argDimSharding[argIdx]) {
      for (size_t axisIdx : dimAxes) {
        if (axisIdx < axisSizes.size()) {
          splitFactor *= axisSizes[axisIdx];
        }
      }
    }

    if (splitFactor <= 1) {
      continue;
    }

    double fractionSaved = 1.0 - 1.0 / static_cast<double>(splitFactor);

    auto tensorType =
        cast<RankedTensorType>(funcOp.getArgument(argIdx).getType());
    int64_t numElements = tensorType.getNumElements();

    double typeMultiplier;
    if (hasAnnotations) {
      typeMultiplier = 1.0;
      if (ttcore::isConstantOrParameterArgumentType(funcOp, argIdx)) {
        typeMultiplier = parameterMultiplier;
      }
    } else {
      double ratio = static_cast<double>(numElements) / maxElements;
      constexpr double kSigmoidSteepness = 20.0;
      constexpr double kSigmoidCenter = 0.25;
      double sigmoid = 1.0 / (1.0 + std::exp(-kSigmoidSteepness *
                                              (ratio - kSigmoidCenter)));
      typeMultiplier = 1.0 + (parameterMultiplier - 1.0) * sigmoid;
    }

    benefit += typeMultiplier *
               (static_cast<double>(numElements) / maxElements) * fractionSaved;
  }

  return benefit;
}

// Estimate compute benefit from sharding. Compares total dot_general FLOPs
// in the sharded module vs the original (unsharded) module. Models with
// sharded matmuls see FLOPs reduction proportional to the split factor,
// e.g. sharding a [M,K]x[K,N] matmul on the M dimension halves the FLOPs.
// Returns a normalized benefit in [0, 1] scaled by computeBenefitWeight.
static double evaluateComputeBenefit(ModuleOp module,
                                     func::FuncOp originalFuncOp) {
  constexpr double computeBenefitWeight = 1.0;

  auto sumDotFlops = [](Operation *rootOp) -> double {
    double totalFlops = 0.0;
    rootOp->walk([&](mlir::stablehlo::DotGeneralOp dotOp) {
      auto resultType = dyn_cast<RankedTensorType>(dotOp.getResult().getType());
      if (!resultType) {
        return;
      }
      int64_t resultElements = resultType.getNumElements();

      auto dotDimNumbers = dotOp.getDotDimensionNumbers();
      int64_t contractingSize = 1;
      auto lhsType = dyn_cast<RankedTensorType>(dotOp.getLhs().getType());
      if (lhsType) {
        for (int64_t dim : dotDimNumbers.getLhsContractingDimensions()) {
          if (dim < lhsType.getRank()) {
            contractingSize *= lhsType.getDimSize(dim);
          }
        }
      }
      totalFlops += 2.0 * static_cast<double>(resultElements) *
                    static_cast<double>(contractingSize);
    });
    return totalFlops;
  };

  double shardedFlops = sumDotFlops(module);
  double originalFlops = sumDotFlops(originalFuncOp);

  if (originalFlops <= 0.0) {
    return 0.0;
  }

  double flopsSaved = originalFlops - shardedFlops;
  if (flopsSaved <= 0.0) {
    return 0.0;
  }

  return computeBenefitWeight * (flopsSaved / originalFlops);
}

// Estimate the cost of gathering sharded outputs at the function boundary.
// When a manual_computation's out_shardings have sharded dimensions, the
// runtime must perform an implicit all-gather to reconstruct the full tensor.
// This penalizes shardings that distribute final outputs, since the gather
// latency is paid at inference time and cannot be overlapped with compute.
// Cost is normalized by maxElements to stay comparable with communication cost.
static double evaluateOutputShardingCost(ModuleOp module, int64_t maxElements) {
  constexpr double outputGatherCostWeight = 1.0;
  constexpr double baseCCLLatency = 1.0;

  double totalCost = 0.0;

  module.walk([&](mlir::sdy::ManualComputationOp manualOp) {
    auto outShardingAttrs = manualOp.getOutShardings().getShardings();
    for (auto [i, sharding] : llvm::enumerate(outShardingAttrs)) {
      bool hasShardedDim = false;
      for (auto dimSharding : sharding.getDimShardings()) {
        if (!dimSharding.getAxes().empty()) {
          hasShardedDim = true;
          break;
        }
      }
      if (!hasShardedDim) {
        continue;
      }

      if (i < manualOp.getNumResults()) {
        if (auto tt = dyn_cast<RankedTensorType>(
                manualOp.getResult(i).getType())) {
          int64_t numElements = tt.getNumElements();
          double volumeFactor = static_cast<double>(numElements) /
                                static_cast<double>(maxElements);
          totalCost += outputGatherCostWeight *
                       (baseCCLLatency + 1.0) * volumeFactor;
        }
      }
    }
  });

  return totalCost;
}

//===----------------------------------------------------------------------===//
// Public API.
//===----------------------------------------------------------------------===//

ShardingResult evaluate(ModuleOp module, const ShardingConfig &config,
                        func::FuncOp originalFuncOp,
                        llvm::ArrayRef<int64_t> axisSizes,
                        int64_t maxElementsOverride) {
  int64_t maxElements = maxElementsOverride > 0
                            ? maxElementsOverride
                            : computeMaxElements(originalFuncOp);
  double commCost = evaluateCommunicationCost(module, maxElements);
  commCost += evaluateOutputShardingCost(module, maxElements);
  double memBenefit =
      evaluateMemoryBenefit(config, originalFuncOp, axisSizes, maxElements);
  memBenefit += evaluateComputeBenefit(module, originalFuncOp);
  return {commCost, memBenefit, commCost - memBenefit};
}

} // namespace mlir::tt::stablehlo
