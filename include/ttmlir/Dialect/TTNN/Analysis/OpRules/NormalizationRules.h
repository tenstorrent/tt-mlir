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

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_NORMALIZATIONRULES_H
