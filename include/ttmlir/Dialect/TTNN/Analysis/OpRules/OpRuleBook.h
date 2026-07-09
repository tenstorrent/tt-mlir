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
  virtual LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const {
    return nullptr;
  }

  /// Whether to generate reshard candidates for this op's inputs.
  /// This is a compile-time optimization: returning false skips
  /// O(K * reshardCandidates) backend validation calls. Use false when
  /// resharding inputs cannot improve the output layout (e.g., the op
  /// always outputs DRAM-interleaved regardless of input layout).
  virtual bool shouldExploreReshards() const { return true; }

  /// Whether reshard candidate generation should include a constant-derived
  /// operand (a function <parameter>/<constant> arg or a const-evaled
  /// ttcore.load_cached result) at the given operand index.
  ///
  /// Default: false. Constants are assumed to already be laid out optimally
  /// (or hoisted by const-eval), so the optimizer avoids inserting a runtime
  /// reshard on them -- e.g. matmul does not reshard its weight operand.
  ///
  /// Override to true for ops whose kernel *hard-requires* a specific sharded
  /// layout on a constant operand and can only reach it via a reshard. For
  /// example rotary_embedding_llama decode requires cos/sin/trans_mat to be
  /// HeightSharded, and they arrive as <parameter> args; the reference
  /// lowering reshards all four inputs (including the constant trans_mat) via
  /// to_memory_config. Without this override the const-derived skip strands
  /// those operands on DRAM-interleaved and forces a DRAM fallback.
  virtual bool shouldReshardConstantOperand(unsigned /*operandIdx*/) const {
    return false;
  }

  /// Whether the greedy search should materialize a ROW_MAJOR sibling for each
  /// tiled input candidate of the given operand.
  ///
  /// Default: false. The candidate pool is 100% tiled (module-wide RM layout
  /// generation is off by default to bound compile time), so a RuleBook filter
  /// requiring RowMajor would reject every candidate and yield an empty set.
  /// Override to true for operands whose kernel hard-requires a RowMajor page
  /// layout (e.g. paged_update_cache's page_table / update_idxs index tensors):
  /// the greedy search then clones each tiled candidate into an RM sibling on
  /// demand. Pair with getInputLayoutFilter() -> requireRowMajor() to drop the
  /// tiled originals. No global flag, no pool growth for other ops.
  virtual bool generatesRowMajorInputSiblings(unsigned /*operandIdx*/) const {
    return false;
  }

  /// Cross-product pruning: reject input layout combinations before expensive
  /// backend validation. Default accepts all. Override for ops that require
  /// homogeneous input layouts (e.g., concat requires all inputs to share the
  /// same memory layout type).
  virtual bool
  isValidInputCombination(llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
    return true;
  }

  /// Cross-product pruning: reject output hint + input layout combinations
  /// before expensive backend validation. Default accepts all. Override for
  /// ops where certain output hints are known to be incompatible with certain
  /// input layouts (e.g., concat requires sharded output to match input
  /// memory layout type).
  virtual bool isValidOutputHintForInputs(
      const OpConfig &hint, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
    return true;
  }

  /// Apply op-specific attributes (Conv2dConfig, MatmulProgramConfig, etc.)
  /// from the chosen candidate to the IR op.
  virtual void applyOpSpecificAttrs(Operation *op,
                                    const BeamCandidate &candidate) const {}

  /// Tiebreaker when two candidates have equal LayoutScores.
  /// Default: prefer more sharded inputs (fewer DRAM/L1-interleaved reads).
  virtual bool preferCandidate(Operation *op, const BeamCandidate &a,
                               const BeamCandidate &b) const;

  /// Adjust a candidate's LayoutScore after the base scorer computed it.
  /// Receives the output config, all operand input
  /// layouts, and the reshard flag.
  virtual LayoutScore adjustScore(Operation *op, LayoutScore base,
                                  const OpConfig &config,
                                  llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                                  bool requiresReshard) const {
    return base;
  }
};

/// Direct lookup: maps an op to its RuleBook via OperationName.
/// Returns the default RuleBook if no specific rules are registered.
const OpRuleBook &getRuleBook(Operation *op);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_OPRULEBOOK_H
