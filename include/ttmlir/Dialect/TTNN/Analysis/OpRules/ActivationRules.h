// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_ACTIVATIONRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_ACTIVATIONRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

/// TTML SiLU backward:
/// The backend requires all operands to be tiled, DRAM-interleaved, and BF16.
/// It derives the result layout from the input, so only the NULL output hint
/// is valid.
struct TTMLSiluBackwardRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// TTML SwiGLU elementwise backward:
/// The backend requires all operands to be tiled and interleaved, in either L1
/// or DRAM. It derives both gradient layouts from the input, so only the NULL
/// output hint is valid.
struct TTMLSwigluElemwiseBackwardRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_ACTIVATIONRULES_H
