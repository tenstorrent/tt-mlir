// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttnn {

/// Filter legalConfigs to only include non-sharded (DRAM or L1-interleaved)
/// configs.
static std::vector<OpConfig>
filterNonSharded(const std::vector<OpConfig> &legalConfigs) {
  std::vector<OpConfig> result;
  for (const auto &config : legalConfigs) {
    if (!config.outputLayout) {
      result.push_back(config);
      continue;
    }
    auto memLayout = config.outputLayout.getMemLayout();
    if (!memLayout || !isShardedMemoryLayout(memLayout.getValue())) {
      result.push_back(config);
    }
  }
  return result;
}

/// Check if a config is L1-interleaved.
static bool isL1Interleaved(const OpConfig &config) {
  if (!config.outputLayout) {
    return false;
  }
  auto memLayout = config.outputLayout.getMemLayout();
  return config.outputLayout.getBufferType() == BufferType::L1 && memLayout &&
         memLayout.getValue() == TensorMemoryLayout::Interleaved;
}

/// Check if a config is DRAM.
static bool isDRAM(const OpConfig &config) {
  if (!config.outputLayout) {
    return false;
  }
  return config.outputLayout.getBufferType() == BufferType::DRAM;
}

OutputHints getOutputHints(Operation *op,
                           const std::vector<OpConfig> &legalConfigs) {
  return llvm::TypeSwitch<Operation *, OutputHints>(op)
      .Case<MatmulOp, LinearOp>([&](auto) {
        // Must pass full configs (carry MatmulProgramConfig tied to sharded
        // output hint). Without hints, matmul defaults to DRAM interleaved.
        return OutputHints{legalConfigs, /*attemptL1Sharding=*/true};
      })
      .Case<Conv2dOp, ConvTranspose2dOp>([&](auto) {
        // Conv2d configs carry Conv2dConfig tied to output hint.
        return OutputHints{legalConfigs, /*attemptL1Sharding=*/true};
      })
      .Case<ReshapeOp, PermuteOp>([&](auto) {
        auto dramConfigs = filterNonSharded(legalConfigs);
        return OutputHints{dramConfigs, /*attemptL1Sharding=*/false};
      })
      .Default([&](Operation *) {
        // Default: NULL hint (backend decides sharding) + L1-interleaved
        // (explicit) + DRAM configs.
        OutputHints result;
        result.attemptL1Sharding = true;

        // NULL hint: backend decides output sharding from inputs.
        result.hints.push_back(OpConfig(TTNNLayoutAttr()));

        // Add L1-interleaved and DRAM configs from legalConfigs.
        for (const auto &cfg : legalConfigs) {
          if (isL1Interleaved(cfg) || isDRAM(cfg)) {
            result.hints.push_back(cfg);
          }
        }
        return result;
      });
}

bool shouldExploreReshards(Operation *op) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<ReshapeOp, PermuteOp>([](auto) { return false; })
      .Default([](Operation *) { return true; });
}

//--- Scoring ---

bool LayoutScore::operator>(const LayoutScore &other) const {
  // 1. L1 > DRAM (highest priority).
  if (isL1 != other.isL1) {
    return isL1;
  }

  // 2. Sharded > Interleaved.
  if (isSharded != other.isSharded) {
    return isSharded;
  }

  // 3. No reshard > reshard.
  if (requiresReshard != other.requiresReshard) {
    return !requiresReshard;
  }

  // 4. More cores > fewer cores.
  if (coreCount != other.coreCount) {
    return coreCount > other.coreCount;
  }

  // 5. Lower L1 usage is better (leaves more room for other tensors).
  return outputL1Usage < other.outputL1Usage;
}

bool LayoutScore::operator==(const LayoutScore &other) const {
  return isL1 == other.isL1 && isSharded == other.isSharded &&
         requiresReshard == other.requiresReshard &&
         coreCount == other.coreCount && outputL1Usage == other.outputL1Usage;
}

LayoutScore
scoreCandidate(Operation *op, const OpConfig &config,
               const op_constraint_validation::ValidationResult &result,
               bool requiresReshard) {
  LayoutScore score;
  score.requiresReshard = requiresReshard;
  score.outputL1Usage = result.outputL1Usage;

  TTNNLayoutAttr layout = result.actualOutputLayout;
  if (!layout) {
    // No layout returned; treat as DRAM fallback.
    return score;
  }

  score.isL1 = layout.hasL1BufferType();
  auto memLayout = layout.getMemLayout();
  score.isSharded = memLayout && isShardedMemoryLayout(memLayout.getValue());

  // Extract core count from grid shape.
  if (auto grid = layout.getGrid()) {
    auto gridShape = grid.getShape();
    score.coreCount = 1;
    for (auto dim : gridShape) {
      score.coreCount *= dim;
    }
  }

  return score;
}

} // namespace mlir::tt::ttnn
