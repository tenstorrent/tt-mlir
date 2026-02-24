// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPMODELSTRATEGY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPMODELSTRATEGY_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <vector>

namespace mlir::tt::ttnn {

/// Per-op output hint strategy: what output config hints to try for a given
/// set of input layouts. These are secondary to the input candidates.
///
/// Default ops: {NULL} as primary hint; sharded configs as fallback (tried
///   only when NULL doesn't produce a sharded result for a given input combo).
/// Matmul: partial legalConfigs (each carries MatmulProgramConfig tied to
///   sharded hint, L1-interleaved filtered out)
/// Conv2d: full legalConfigs (each carries Conv2dConfig)
/// Reshape/Permute: non-sharded configs only (skip L1 sharding entirely)
struct OutputHints {
  /// Primary output configs to pass as hints to backend (always tried).
  std::vector<OpConfig> hints;

  /// Fallback output configs tried only when no primary hint produces a
  /// sharded result for a given input combination. This avoids trying all
  /// sharded configs when the NULL hint already yields a sharded output
  /// (e.g., when inputs are already sharded), keeping the search space small.
  std::vector<OpConfig> fallbackHints;

  /// Whether to attempt L1 sharding for this op. False for reshape, permute,
  /// etc.
  bool attemptL1Sharding = true;
};

/// Get output hints for an op type. Per-op dispatch via TypeSwitch.
/// These are crossed with input candidates to form the full search space.
OutputHints getOutputHints(Operation *op,
                           const std::vector<OpConfig> &legalConfigs);

/// Whether an op should participate in reshard exploration at all.
/// Returns false for ops that must always be DRAM (e.g., reshape, permute).
bool shouldExploreReshards(Operation *op);

/// Returns true if layout has a sharded non-rectangular physical grid.
/// A grid is non-rectangular if toCoreRangeSet produces >1 core range.
bool isNonRectangularGrid(TTNNLayoutAttr layout);

/// Returns true if op requires rectangular grid on its inputs.
/// Workaround for tt-metal bug: rms_norm/layer_norm hang on non-rectangular
/// width-sharded inputs. https://github.com/tenstorrent/tt-metal/issues/38429
bool requiresRectangularGridInputs(Operation *op);

//--- Scoring ---

/// Score representing the quality of a layout candidate. Used for ranking.
struct LayoutScore {
  /// Primary: more cores = better parallelism.
  int64_t coreCount = 0;

  /// Prefer sharded over interleaved.
  bool isSharded = false;

  /// Prefer L1 over DRAM.
  bool isL1 = false;

  /// Penalize reshard.
  bool requiresReshard = false;

  /// Total bytes transferred from DRAM across all inputs. Lower is better.
  uint64_t inputDramBytes = 0;

  /// Memory footprint from ValidationResult (for tie-breaking).
  uint64_t outputL1Usage = 0;

  /// Higher score is better.
  bool operator>(const LayoutScore &other) const;
  bool operator==(const LayoutScore &other) const;
  bool operator<(const LayoutScore &other) const { return other > *this; }
  bool operator>=(const LayoutScore &other) const {
    return *this > other || *this == other;
  }
  bool operator<=(const LayoutScore &other) const {
    return other >= *this;
  }
  bool operator!=(const LayoutScore &other) const {
    return !(*this == other);
  }
};

/// A single candidate in the beam. Used for both K=1 (greedy) and K>1 (beam
/// search). Contains back pointers to support the backward pass for fork
/// consolidation.
struct BeamCandidate {
  /// Final config (actualOutputLayout from backend + opSpecificAttrs).
  OpConfig config;

  /// Quality score for this candidate.
  LayoutScore score;

  /// Backend validation result.
  op_constraint_validation::ValidationResult validationResult;

  /// Input layouts used for this candidate.
  std::vector<TTNNLayoutAttr> inputLayouts;

  /// Back pointers: per-operand, which candidate from the producer's beam was
  /// used. For func args (no producer): index into initial layout candidates.
  /// These enable the backward pass to trace which producer choices led here.
  llvm::SmallVector<size_t> producerCandidateIndices;

  /// Per-operand reshard targets (empty entry = no reshard for that operand).
  llvm::DenseMap<size_t, TTNNLayoutAttr> reshardLayouts;
};

/// Score a backend validation result. Per-op customization via TypeSwitch.
LayoutScore scoreCandidate(Operation *op, const OpConfig &config,
                           const op_constraint_validation::ValidationResult &result,
                           bool requiresReshard);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPMODELSTRATEGY_H
