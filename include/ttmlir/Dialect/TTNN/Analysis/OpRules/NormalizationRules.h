// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_NORMALIZATIONRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_NORMALIZATIONRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// Normalization ops: RMSNorm (LayerNorm has the same kernel dispatch logic
// and could share rules in a follow-up).
//
// Kernel dispatch (layernorm_device_operation.cpp:16-22):
//   input.is_sharded() ? LayerNormShardedProgramFactory
//                      : LayerNormMultiCoreProgramFactory
//
// The sharded kernel's validator enforces
// (layernorm_device_operation.cpp:148-173):
//   - input cannot be HEIGHT_SHARDED
//   - shard grid must be a full rectangular bounding box
//     (shard_spec.grid.num_cores() == bbox_num_cores)
//   - output must also be sharded (not height sharded)
//
// Without an input filter, partial-grid sharded candidates reach op-model
// validation and fail with TT_FATAL. The filter prunes them early.
//===----------------------------------------------------------------------===//

struct RmsNormRuleBook : OpRuleBook {

  /// RMSNormOp: operand 0 (activation) must be interleaved OR full-bbox
  /// block/width sharded. Height-sharded inputs and partial-bbox sharded inputs
  /// are rejected because the sharded layernorm kernel doesn't support them.
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;

  /// adjustScore reflects kernel-dispatch reality: the generic scorer derives
  /// `coreCount` from the output layout, but for rms_norm with interleaved
  /// input the kernel runs single- or few-core (parallelism = NC*Ht of input)
  /// regardless of how the output is sharded. With sharded input, the sharded
  /// kernel uses input->grid->volume cores. Without this adjustment, an
  /// interleaved-input candidate with a high-core sharded output beats a
  /// sharded-input candidate with a low-core sharded output, even though the
  /// sharded-input candidate is the strictly faster runtime path.
  ///
  /// The adjustment also intentionally lets a reshard-bearing candidate win
  /// over a no-reshard alternative when the reshard produces a full-bbox
  /// sharded input: the generic LayoutScore lexicographic order ranks
  /// `requiresReshard=false` above `coreCount`, which blocks sharded-input
  /// candidates whose only cost vs an interleaved-input peer is the inserted
  /// reshard. For rms_norm the kernel speedup (~3x+ on decode shapes)
  /// dominates the reshard cost, so `requiresReshard` is cleared on
  /// sharded-input candidates to prefer the better-parallelized path.
  LayoutScore adjustScore(Operation *op, LayoutScore base,
                          const OpConfig &config,
                          llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                          bool requiresReshard) const override;
};

//===----------------------------------------------------------------------===//
// DistributedRMSNormOp: the fused rms_allgather kernel.
//
// Same full-bbox requirement as RMSNorm, but stricter — the fused kernel only
// runs on a WIDTH-sharded L1 input (interleaved/height/block are rejected), and
// it needs a LayerNormShardedMultiCoreProgramConfig derived from that shard
// spec. This rulebook lets opt-level-2 own the sharding decision (produce the
// width-shard layout + program config) so the DistributedRMSNormWidthShardInput
// workaround can be restricted to lower opt levels.
//===----------------------------------------------------------------------===//

struct DistributedRMSNormRuleBook : OpRuleBook {
  /// operand 0 (input) and operand 2 (residual): full-bbox width-sharded L1.
  /// operand 1 (weight/gamma): ROW_MAJOR (the fused kernel requires it).
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;

  /// The weight must be ROW_MAJOR; clone tiled weight candidates into RowMajor
  /// siblings so the candidate pool can satisfy the filter above.
  bool generatesRowMajorInputSiblings(unsigned operandIdx) const override;

  /// Keep only full-bbox width-sharded L1 output candidates, and attach the
  /// LayerNormShardedMultiCoreProgramConfig computed from each candidate's
  /// shard spec (grid, block_h, block_w).
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;

  /// Write the chosen program config onto the op.
  void applyOpSpecificAttrs(Operation *op,
                            const BeamCandidate &candidate) const override;

  /// Same core-count reasoning as RMSNorm (kernel parallelizes across the
  /// input shard grid).
  LayoutScore adjustScore(Operation *op, LayoutScore base,
                          const OpConfig &config,
                          llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                          bool requiresReshard) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_NORMALIZATIONRULES_H
