// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Roofline.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttnn {

OutputHints getOutputHints(Operation *op,
                           const std::vector<OpConfig> &legalConfigs) {
  return getRuleBook(op).getOutputHints(op, legalConfigs);
}

bool shouldExploreReshards(Operation *op) {
  return getRuleBook(op).shouldExploreReshards();
}

bool generatesRowMajorInputSiblings(Operation *op, unsigned operandIdx) {
  return getRuleBook(op).generatesRowMajorInputSiblings(operandIdx);
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

  // Matmul compute work (tile-multiplies), layout-independent. Lets the
  // analytical-time cost model price the compute roofline so it does not trade
  // away matmul parallelism. 0 for non-matmul ops.
  score.tileMuls =
      llvm::TypeSwitch<Operation *, uint64_t>(op)
          .Case<MatmulOp, LinearOp>([](auto m) {
            return roofline::getNumTileMatmuls(m.getA(), m.getResult(),
                                               m.getTransposeA());
          })
          .Default(uint64_t{0});

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

  // Total output footprint (shard size * number of shards), independent of
  // buffer type. For L1 outputs this matches outputL1Usage; for DRAM outputs
  // outputL1Usage is 0 but this is the real tensor size.
  score.outputBytes = layout.getShardSizeInBytes();
  for (auto dim : layout.getGridShape()) {
    score.outputBytes *= dim;
  }

  return getRuleBook(op).adjustScore(op, score, config, inputLayouts,
                                     requiresReshard);
}

bool preferCandidate(Operation *op, const BeamCandidate &a,
                     const BeamCandidate &b) {
  return getRuleBook(op).preferCandidate(op, a, b);
}

} // namespace mlir::tt::ttnn
