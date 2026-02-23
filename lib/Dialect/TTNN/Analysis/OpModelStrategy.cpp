// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

OutputHints getOutputHints(Operation *op,
                           const std::vector<OpConfig> &legalConfigs) {
  return getRuleBook(op).getOutputHints(op, legalConfigs);
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

  TTNNLayoutAttr layout = result.getFirstActualOutputLayout();
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

bool preferCandidate(Operation *op, const BeamCandidate &a,
                     const BeamCandidate &b) {
  return getRuleBook(op).preferCandidate(op, a, b);
}

} // namespace mlir::tt::ttnn
