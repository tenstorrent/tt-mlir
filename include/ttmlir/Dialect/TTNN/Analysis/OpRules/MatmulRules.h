// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MATMULRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MATMULRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

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
  /// Output hints: partial configs without L1-interleaved.
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;

  /// Reject width-sharded inputs (accuracy issues).
  LayoutFilterFn getInputLayoutFilter() const override;

  /// Apply MatmulProgramConfig + fused activation dedup.
  void applyOpSpecificAttrs(Operation *op,
                            const BeamCandidate &candidate) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MATMULRULES_H
