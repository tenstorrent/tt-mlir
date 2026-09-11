// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_LOSSRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_LOSSRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

/// TTML cross-entropy requires a tiled, DRAM-interleaved logits tensor and a
/// row-major, DRAM-interleaved target tensor. The backward operation's grad
/// operand only has to be tiled. The output layout is selected by the backend
/// so only the null output hint is meaningful.
struct CrossEntropyRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  bool generatesRowMajorInputSiblings(unsigned operandIdx) const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_LOSSRULES_H
