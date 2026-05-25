// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MATMULRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MATMULRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

class MatmulOp;

//===----------------------------------------------------------------------===//
// Matmul/Linear rules:
//
// Output hints:
//   Filter L1-interleaved (worst of both worlds for matmul:
//   no program config, HiFi4 fallback, L1 pressure).
//   Use partial configs (dedup by bufferType+memLayout).
//
// Op-specific attributes:
//   Set MatmulProgramConfig from candidate.
//   Remove activation attr if programConfig has fused activation
//   (tt-metal #35060: duplicate activations).
//===----------------------------------------------------------------------===//

struct MatmulRuleBook : OpRuleBook {
  /// Output hints: for DRAM-eligible matmuls, a pre-validated DRAM-sharded
  /// hint (L1 WIDTH_SHARDED) as primary; existing partial configs as fallback.
  /// For others: existing behavior (partial configs, no L1-interleaved).
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;

  /// Operand 1 (weight): reject L1-sharded. DRAM-sharded is allowed for the
  /// DRAM-sharded matmul path (injected by applyOpSpecificAttrs).
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;

  /// Apply MatmulProgramConfig + fused activation dedup.
  /// For DRAM-sharded candidates: also inserts weight/in0 reshards and
  /// strips the activation attr (inserting a separate activation op after).
  void applyOpSpecificAttrs(Operation *op,
                            const BeamCandidate &candidate) const override;

private:
  /// Build a pre-validated DRAM-sharded OpConfig for eligible matmuls.
  /// Returns nullopt if the matmul is not eligible or params don't fit L1.
  std::optional<OpConfig> buildDRAMShardingHint(Operation *op) const;

  /// Apply the full DRAM sharding IR transformation to an eligible MatmulOp.
  /// Called from applyOpSpecificAttrs when the chosen candidate carries a
  /// MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr.
  void applyDRAMShardedTransformation(MatmulOp matmulOp,
                                      const MatmulAttrs &matmulAttrs) const;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MATMULRULES_H
