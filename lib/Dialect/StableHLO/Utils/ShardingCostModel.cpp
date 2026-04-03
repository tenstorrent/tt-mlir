// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Utils/ShardingCostModel.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/StringMap.h"

namespace mlir::tt::stablehlo {

// Relative cost weights for collective communication ops.
// all_gather / collective_permute: 1.0 (single data movement, no reduction)
// reduce_scatter / all_reduce / all_to_all: 1.5 (reduction + data movement)
// all_slice: 0.5 (local slicing, minimal communication)
static const llvm::StringMap<double> cclWeights = {
    {"stablehlo.all_gather", 1.0},         {"stablehlo.reduce_scatter", 1.5},
    {"stablehlo.all_reduce", 1.5},         {"stablehlo.all_to_all", 1.5},
    {"stablehlo.collective_permute", 1.0},
};

static const llvm::StringMap<double> compositeWeights = {
    {"sdy.all_slice", 0.5},
    {"sdy.all_gather", 1.0},
    {"sdy.reduce_scatter", 1.5},
    {"sdy.all_reduce", 1.5},
};

ShardingCostModel::ShardingCostModel() = default;

ShardingCostModel::ShardingCostModel(Options options)
    : options(std::move(options)) {}

ShardingResult ShardingCostModel::evaluate(ModuleOp module,
                                           const ShardingConfig &config,
                                           func::FuncOp originalFuncOp,
                                           int64_t meshAxisSize) const {
  int64_t maxElements = computeMaxElements(originalFuncOp);
  double commCost = evaluateCommunicationCost(module, maxElements);
  double memBenefit =
      evaluateMemoryBenefit(config, originalFuncOp, meshAxisSize, maxElements);
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

// Evaluate cost of a lowered StableHLO module by counting and weighting CCL
// ops. Each CCL op incurs a fixed latency overhead (setup, synchronization)
// plus a bandwidth cost proportional to the volume of data communicated.
double ShardingCostModel::evaluateCommunicationCost(ModuleOp module,
                                                    int64_t maxElements) const {
  double totalCost = 0.0;

  module.walk([&](Operation *op) {
    StringRef opName = op->getName().getStringRef();
    double opWeight = 0.0;

    auto it = cclWeights.find(opName);
    if (it != cclWeights.end()) {
      opWeight = it->second;
    } else if (opName == "stablehlo.composite") {
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name")) {
        auto cit = compositeWeights.find(nameAttr.getValue());
        if (cit != compositeWeights.end()) {
          opWeight = cit->second;
        }
      }
    }

    if (opWeight > 0.0 && maxElements > 0) {
      int64_t commElements = 1;
      if (op->getNumOperands() > 0) {
        if (auto tt = dyn_cast<RankedTensorType>(op->getOperand(0).getType())) {
          commElements = tt.getNumElements();
        }
      }
      double volumeFactor =
          static_cast<double>(commElements) / static_cast<double>(maxElements);
      totalCost += options.baseCCLLatency + opWeight * volumeFactor;
    }
  });

  return totalCost;
}

// Compute the memory benefit of a sharding config. Sharding large argument
// tensors across the mesh saves memory proportional to the element count.
// Benefit is normalized relative to the largest argument.
//
// Weight/parameter tensors (heuristic: rank <= 2 with > 1024 elements) get a
// higher multiplier because they are persistent on device and dominate peak
// memory in large models. Activations (rank > 2, typically with a batch dim)
// are transient and less valuable to shard for memory.
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

    // Heuristic: tensors with rank <= 4 and > 1024 elements are likely
    // model weights/parameters rather than activations.
    double typeMultiplier = 1.0;
    if (tensorType.getRank() <= 4 && numElements > 1024) {
      typeMultiplier = options.parameterMultiplier;
    }

    benefit += typeMultiplier *
               (static_cast<double>(numElements) / maxElements) * fractionSaved;
  }

  return benefit;
}

} // namespace mlir::tt::stablehlo
