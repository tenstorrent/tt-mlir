// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_OPRULEBOOK_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_OPRULEBOOK_H

#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"

#include <functional>

namespace mlir::tt::ttnn {

using LayoutFilterFn = std::function<bool(TTNNLayoutAttr)>;

/// Base class for per-op optimizer rules. Each op type (or group of ops with
/// identical rules) gets a concrete subclass. The default implementations
/// represent the behavior for ops with no special rules.
///
/// Compile-time optimization note: shouldExploreReshards() gates reshard
/// candidate generation. Returning false avoids O(K * reshardCandidates)
/// backend validation calls for ops that cannot benefit from sharded inputs.
/// The constraint validation would reject them anyway, but this saves time.
struct OpRuleBook {
  virtual ~OpRuleBook() = default;

  /// Output layout hints for this op. Default: NULL hint (backend decides)
  /// + sharded fallbacks.
  virtual OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const;

  /// Filter predicate that rejects invalid input layouts for correctness.
  /// For example, ConcatOp rejects all sharded inputs because tt-metal
  /// produces incorrect results. Returns nullptr when all layouts are
  /// accepted (default).
  ///
  /// Note: this is distinct from shouldExploreReshards(). An op may accept
  /// all input layouts (no filter) but still disable reshard exploration
  /// because resharding wouldn't improve the output — e.g., SDPA ops
  /// accept any input layout but always output DRAM-interleaved, so
  /// resharding inputs to sharded is wasted compile time with no benefit.
  virtual LayoutFilterFn getInputLayoutFilter() const { return nullptr; }

  /// Whether to generate reshard candidates for this op's inputs.
  /// This is a compile-time optimization: returning false skips
  /// O(K * reshardCandidates) backend validation calls. Use false when
  /// resharding inputs cannot improve the output layout (e.g., the op
  /// always outputs DRAM-interleaved regardless of input layout).
  virtual bool shouldExploreReshards() const { return true; }

  /// Cross-product pruning: reject input layout combinations before expensive
  /// backend validation. Default accepts all. Override for ops that require
  /// homogeneous input layouts (e.g., concat requires all inputs to share the
  /// same memory layout type).
  virtual bool isValidInputCombination(
      llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
    return true;
  }

  /// Apply op-specific attributes (Conv2dConfig, MatmulProgramConfig, etc.)
  /// from the chosen candidate to the IR op.
  virtual void applyOpSpecificAttrs(Operation *op,
                                    const BeamCandidate &candidate) const {}

  /// Tiebreaker when two candidates have equal LayoutScores.
  virtual bool preferCandidate(Operation *op, const BeamCandidate &a,
                               const BeamCandidate &b) const {
    return false;
  }
};

/// Direct lookup: maps an op to its RuleBook via OperationName.
/// Returns the default RuleBook if no specific rules are registered.
const OpRuleBook &getRuleBook(Operation *op);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_OPRULEBOOK_H
