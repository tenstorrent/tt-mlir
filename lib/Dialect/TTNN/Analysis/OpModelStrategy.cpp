// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace mlir::tt::ttnn {

// Returns true if the optimizer should reject any candidate whose backend-
// captured output layout is sharded for this op.
//
// Always-on for ops with known kernel hangs at certain sharded layouts:
//   - ttnn.rotary_embedding: sharded outputs leave Q/K in a state that
//     deadlocks the downstream all_gather ring kernel in gpt-oss-20b TP=8
//     decode (bisection-confirmed; rotary's hint is already nullHintOnly()
//     but the backend still captures sharded for these decode shapes).
//
// Plus env-var-driven extension TT_NO_SHARD_OPS (comma-separated op names)
// for bisecting other suspected ops without recompiling.
//
// Example: TT_NO_SHARD_OPS="ttnn.rms_norm,ttnn.repeat"
bool shouldDisableShardedForOp(Operation *op) {
  llvm::StringRef opName = op->getName().getStringRef();

  // Always-on entries.
  if (opName == "ttnn.rotary_embedding") {
    return true;
  }

  // Env-var-driven extension.
  static const std::string envValue = []() {
    if (const char *v = std::getenv("TT_NO_SHARD_OPS")) {
      return std::string(v);
    }
    return std::string();
  }();
  if (envValue.empty()) {
    return false;
  }
  size_t pos = 0;
  while (pos <= envValue.size()) {
    size_t end = envValue.find(',', pos);
    if (end == std::string::npos) {
      end = envValue.size();
    }
    llvm::StringRef token(envValue.data() + pos, end - pos);
    token = token.trim();
    if (!token.empty() && token == opName) {
      return true;
    }
    if (end == envValue.size()) {
      break;
    }
    pos = end + 1;
  }
  return false;
}

static bool isShardedConfig(const OpConfig &cfg) {
  if (!cfg.outputLayout) {
    return false;
  }
  auto memLayout = cfg.outputLayout.getMemLayout();
  return memLayout && isShardedMemoryLayout(memLayout.getValue());
}

OutputHints getOutputHints(Operation *op,
                           const std::vector<OpConfig> &legalConfigs) {
  OutputHints hints = getRuleBook(op).getOutputHints(op, legalConfigs);

  if (!shouldDisableShardedForOp(op)) {
    return hints;
  }

  // Drop sharded hints from both primary and fallback.
  hints.hints.erase(std::remove_if(hints.hints.begin(), hints.hints.end(),
                                   isShardedConfig),
                    hints.hints.end());
  hints.fallbackHints.clear();

  // If everything was sharded, supply DRAM-interleaved configs from
  // legalConfigs as primary hints.
  if (hints.hints.empty()) {
    for (const auto &cfg : legalConfigs) {
      if (cfg.outputLayout && !isShardedConfig(cfg) &&
          cfg.outputLayout.getBufferType() == BufferType::DRAM) {
        hints.hints.push_back(cfg);
      }
    }
  }

  // Last resort: NULL hint (backend decides).
  if (hints.hints.empty()) {
    hints.hints.push_back(OpConfig(TTNNLayoutAttr()));
  }

  return hints;
}

bool shouldExploreReshards(Operation *op) {
  return getRuleBook(op).shouldExploreReshards();
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

  // 3. Less DRAM input transfer > more DRAM input transfer.
  if (inputDramBytes != other.inputDramBytes) {
    return inputDramBytes < other.inputDramBytes;
  }

  // 4. No reshard > reshard.
  if (requiresReshard != other.requiresReshard) {
    return !requiresReshard;
  }

  // 5. More cores > fewer cores.
  if (coreCount != other.coreCount) {
    return coreCount > other.coreCount;
  }

  // 6. Lower L1 usage is better (leaves more room for other tensors).
  return outputL1Usage < other.outputL1Usage;
}

bool LayoutScore::operator==(const LayoutScore &other) const {
  return isL1 == other.isL1 && isSharded == other.isSharded &&
         inputDramBytes == other.inputDramBytes &&
         requiresReshard == other.requiresReshard &&
         coreCount == other.coreCount && outputL1Usage == other.outputL1Usage;
}

LayoutScore
scoreCandidate(Operation *op, const OpConfig &config,
               const op_constraint_validation::ValidationResult &result,
               bool requiresReshard,
               llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) {
  LayoutScore score;
  score.requiresReshard = requiresReshard;
  score.outputL1Usage = result.outputL1Usage;

  TTNNLayoutAttr layout = result.getFirstActualOutputLayout();
  if (!layout) {
    // No layout returned; treat as DRAM fallback.
    return getRuleBook(op).adjustScore(op, score, config, inputLayouts,
                                       requiresReshard);
  }

  score.isL1 = layout.hasL1BufferType();
  auto memLayout = layout.getMemLayout();
  score.isSharded = memLayout && isShardedMemoryLayout(memLayout.getValue());

  // Extract core count from grid shape.
  score.coreCount = 1;
  for (auto dim : layout.getGridShape()) {
    score.coreCount *= dim;
  }

  return getRuleBook(op).adjustScore(op, score, config, inputLayouts,
                                     requiresReshard);
}

bool preferCandidate(Operation *op, const BeamCandidate &a,
                     const BeamCandidate &b) {
  return getRuleBook(op).preferCandidate(op, a, b);
}

} // namespace mlir::tt::ttnn
