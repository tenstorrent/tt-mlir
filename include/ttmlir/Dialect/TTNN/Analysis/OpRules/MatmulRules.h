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
  /// Output hints: for DS-eligible matmuls, the DS hint is included alongside
  /// the normal partial configs. adjustScore drives DS preference via
  /// isDRAMShardedCandidate. For non-eligible matmuls: normal behavior.
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;

  /// Operand 1 (weight): reject L1. DRAM layouts (interleaved and
  /// width-sharded) are accepted; the DRAM width-sharded(DS) layout is injected
  /// via getExtraInputReshardCandidates.
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;

  /// Apply MatmulProgramConfig + fused activation dedup.
  /// For DS candidates: set program/compute config and split fused activation.
  void applyOpSpecificAttrs(Operation *op,
                            const BeamCandidate &candidate) const override;

  /// Reject the DS hint for input combinations that aren't the correct DS
  /// layouts (L1 width-sharded in0, DRAM width-sharded in1). The tt-metal op model
  /// crashes (TT_FATAL) rather than returning a failure when given a DS program
  /// config with incompatible input layouts.
  bool isValidOutputHintForInputs(
      const OpConfig &hint,
      llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const override;

  /// Set isDRAMShardedCandidate / hasCanonicalDSIn0 for DS candidates.
  LayoutScore adjustScore(Operation *op, LayoutScore base,
                          const OpConfig &config,
                          llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                          bool requiresReshard) const override;

  /// Inject DS-specific input layouts into the candidate pool:
  ///   operand 0 → L1 width-sharded 1×8 (canonical DS activation layout; 8 cores empirically optimal)
  ///   operand 1 → DRAM width-sharded 1×12 (DS weight layout with padding)
  std::vector<TTNNLayoutAttr>
  getExtraInputReshardCandidates(Operation *op,
                                 unsigned operandIdx) const override;

private:
  /// Build the DS output hint (L1 1×8 width-sharded + DS program config).
  /// Returns nullopt if not eligible or params don't fit L1.
  std::optional<OpConfig> buildDRAMShardingHint(Operation *op) const;

  /// Set program/compute config and split fused activation for a DS matmul.
  void applyDRAMShardedTransformation(MatmulOp matmulOp,
                                      const MatmulAttrs &matmulAttrs) const;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MATMULRULES_H
