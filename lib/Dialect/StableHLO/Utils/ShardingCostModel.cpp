// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Utils/ShardingCostModel.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::stablehlo {

ShardingCostModel::ShardingCostModel() = default;

ShardingCostModel::ShardingCostModel(Options options)
    : options(std::move(options)) {}

ShardingResult ShardingCostModel::evaluate(ModuleOp module,
                                           const ShardingConfig &config,
                                           func::FuncOp originalFuncOp,
                                           int64_t meshAxisSize) const {
  int64_t maxElements = computeMaxElements(originalFuncOp);
  double commCost = evaluateCommunicationCost(module, maxElements);
  commCost += evaluateOutputShardingCost(module, maxElements);
  double memBenefit =
      evaluateMemoryBenefit(config, originalFuncOp, meshAxisSize, maxElements);
  memBenefit += evaluateComputeBenefit(module, originalFuncOp);
  return {commCost, memBenefit, commCost - memBenefit};
}

int64_t ShardingCostModel::computeMaxElements(func::FuncOp funcOp) {
  int64_t maxElements = 1;
  for (auto arg : funcOp.getArguments()) {
    if (auto tt = dyn_cast<RankedTensorType>(arg.getType())) {
      maxElements = std::max(maxElements, tt.getNumElements());
    }
  }
  return maxElements;
}

// Check if a CCL op feeds directly into a heavy compute op (dot_general)
// without other intervening operations. CCLs on the critical path before
// matmuls serialize communication and compute — these deserve a penalty.
static bool isCCLBeforeCompute(Operation *cclOp) {
  if (cclOp->getNumResults() == 0) {
    return false;
  }
  Value result = cclOp->getResult(0);
  for (OpOperand &use : result.getUses()) {
    Operation *user = use.getOwner();
    if (isa<mlir::stablehlo::DotGeneralOp>(user)) {
      return true;
    }
  }
  return false;
}

static double getCCLWeight(StringRef opName, StringRef compositeName = "") {
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

// Evaluate cost of a lowered StableHLO module by counting and weighting CCL
// ops. Each CCL op incurs a fixed latency overhead (setup, synchronization)
// plus a bandwidth cost proportional to the volume of data communicated.
// CCLs that feed directly into heavy compute (dot_general) get a position
// penalty since they serialize communication and compute.
double ShardingCostModel::evaluateCommunicationCost(ModuleOp module,
                                                    int64_t maxElements) const {
  double totalCost = 0.0;
  constexpr double kCriticalPathPenalty = 0.5;

  module.walk([&](Operation *op) {
    StringRef opName = op->getName().getStringRef();
    StringRef compositeName;
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
    double cost = options.baseCCLLatency + opWeight * volumeFactor;

    if (isCCLBeforeCompute(op)) {
      cost += kCriticalPathPenalty;
    }

    totalCost += cost;
  });

  return totalCost;
}

// Compute the memory benefit of a sharding config. Sharding large argument
// tensors across the mesh saves memory proportional to the element count.
// Benefit is normalized relative to the largest argument.
//
// Large tensors (at least 25% of the largest argument) get a higher multiplier
// because sharding them yields significant memory and compute savings. This
// applies to both weight matrices (rank 2) and attention Q/K/V tensors
// (rank 4 with head dimensions). Small tensors get a base multiplier of 1.0.
double ShardingCostModel::evaluateMemoryBenefit(const ShardingConfig &config,
                                                func::FuncOp funcOp,
                                                int64_t meshAxisSize,
                                                int64_t maxElements) const {
  if (meshAxisSize <= 1) {
    return 0.0;
  }

  double benefit = 0.0;
  double fractionSaved = 1.0 - 1.0 / static_cast<double>(meshAxisSize);

  for (size_t argIdx = 0; argIdx < config.argDimSharded.size(); ++argIdx) {
    bool isSharded =
        llvm::any_of(config.argDimSharded[argIdx], [](bool s) { return s; });
    if (!isSharded) {
      continue;
    }

    auto tensorType =
        cast<RankedTensorType>(funcOp.getArgument(argIdx).getType());
    int64_t numElements = tensorType.getNumElements();

    double typeMultiplier = 1.0;
    if (numElements > 1024 && numElements * 4 >= maxElements) {
      typeMultiplier = options.parameterMultiplier;
    }

    benefit += typeMultiplier *
               (static_cast<double>(numElements) / maxElements) * fractionSaved;
  }

  return benefit;
}

// Estimate compute benefit from sharding by comparing total dot_general FLOPs
// between the sharded module and the original (unsharded) module.
// Sharding reduces tensor dimensions, producing smaller matmuls. The benefit
// is the fraction of FLOPs saved, which directly represents the compute
// speedup from parallelism (e.g., halving heads in SDPA saves 50% of FLOPs).
double ShardingCostModel::evaluateComputeBenefit(
    ModuleOp module, func::FuncOp originalFuncOp) const {
  if (options.computeBenefitWeight <= 0.0) {
    return 0.0;
  }

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

  double normalizedSavings = flopsSaved / originalFlops;
  return options.computeBenefitWeight * normalizedSavings;
}

// Estimate cost of sharded function outputs that need gathering.
// If the manual_computation's out_shardings have any sharded dimension,
// the output needs an implicit gather at the function boundary.
double ShardingCostModel::evaluateOutputShardingCost(
    ModuleOp module, int64_t maxElements) const {
  if (options.outputGatherCostWeight <= 0.0) {
    return 0.0;
  }

  double totalCost = 0.0;

  module.walk([&](mlir::sdy::ManualComputationOp manualOp) {
    auto outShardingAttrs =
        manualOp.getOutShardings().getShardings();
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
          totalCost +=
              options.outputGatherCostWeight *
              (options.baseCCLLatency + 1.0 * volumeFactor);
        }
      }
    }
  });

  return totalCost;
}

} // namespace mlir::tt::stablehlo
