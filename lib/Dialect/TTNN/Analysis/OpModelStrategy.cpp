// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttnn {

/// Filter legalConfigs to only include non-sharded (DRAM or L1-interleaved)
/// configs.
[[maybe_unused]] static std::vector<OpConfig>
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
[[maybe_unused]] static bool isL1Interleaved(const OpConfig &config) {
  if (!config.outputLayout) {
    return false;
  }
  auto memLayout = config.outputLayout.getMemLayout();
  return config.outputLayout.getBufferType() == BufferType::L1 && memLayout &&
         memLayout.getValue() == TensorMemoryLayout::Interleaved;
}

OutputHints getOutputHints(Operation *op,
                           const std::vector<OpConfig> &legalConfigs) {
  return llvm::TypeSwitch<Operation *, OutputHints>(op)
      .Case<MatmulOp, LinearOp>([&](auto) {
        // Use partial configs: deduplicate by (bufferType, memLayout),
        // set ignorePhysicalLayout=true. Backend decides physical layout.
        auto partialConfigs =
            optimizer_utils::getUniqueTestConfigsForMatmulLinear(legalConfigs);

        // Remove L1-interleaved hints for matmul/linear output.
        //
        // L1-interleaved output is "worst of both worlds" for matmul:
        //  - generateMatmulProgramConfig() returns nullopt for non-sharded
        //    output, so no program config is emitted by the compiler.
        //  - tt-metal runtime falls back to MatmulMultiCoreProgramConfig{}
        //    with hardcoded HiFi4 math fidelity (the slowest kernel path).
        //  - Same NOC write overhead as DRAM interleaved (not eliminated
        //    like sharded), loses bias fusion, optimized mcast program
        //    configs, and narrow-shape 1D optimization.
        //  - Subject to L1 capacity constraints unlike DRAM interleaved.
        //
        // DRAM-interleaved is the safe default: full bias fusion, optimized
        // 1D/2D mcast configs, and no L1 pressure from the output tensor.
        // L1-sharded is best when applicable (eliminates NOC writes entirely).
        std::vector<OpConfig> filtered;
        for (const auto &cfg : partialConfigs) {
          if (isL1Interleaved(cfg)) {
            continue;
          }
          filtered.push_back(cfg);
        }

        return OutputHints{filtered, {}, /*attemptL1Sharding=*/true};
      })
      .Case<Conv2dOp, ConvTranspose2dOp>([&](auto) {
        // Conv2d configs carry Conv2dConfig tied to output hint.
        return OutputHints{legalConfigs, {}, /*attemptL1Sharding=*/true};
      })
      // PadOp: ttnn::pad only supports sharded output for
      // HEIGHT_SHARDED + ROW_MAJOR inputs. Its tile-layout path
      // (invoke_tile) crashes when given a sharded output memory config
      // with interleaved input because it unconditionally dereferences
      // input_tensor.shard_spec()->grid, which is nullopt for
      // interleaved tensors. The op model validation doesn't catch this
      // since it only checks sharding constraints when the input is
      // already sharded. Until tt-metal fixes this, forbid sharded
      // output hints for PadOp.
      .Case<ReshapeOp, PermuteOp, ConcatenateHeadsOp, PadOp>([&](auto) {
        auto nonShardedConfigs = filterNonSharded(legalConfigs);
        return OutputHints{nonShardedConfigs, {}, /*attemptL1Sharding=*/false};
      })
      .Default([&](Operation *) {
        // Primary: NULL hint only -- let the backend decide output from inputs.
        // Fallback: sharded configs -- tried only when NULL yields non-sharded
        // output (e.g., when inputs are interleaved and the backend mirrors
        // the interleaved layout on the output).
        OutputHints result;
        result.attemptL1Sharding = true;

        // NULL hint: backend decides output sharding from inputs.
        result.hints.push_back(OpConfig(TTNNLayoutAttr()));

        // Sharded configs as fallback.
        for (const auto &cfg : legalConfigs) {
          if (!cfg.outputLayout) {
            continue;
          }
          auto memLayout = cfg.outputLayout.getMemLayout();
          if (memLayout && isShardedMemoryLayout(memLayout.getValue())) {
            result.fallbackHints.push_back(cfg);
          }
        }
        return result;
      });
}

bool shouldExploreReshards(Operation *op) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<ReshapeOp, PermuteOp, ConcatenateHeadsOp>([](auto) {
        return false;
      })
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
