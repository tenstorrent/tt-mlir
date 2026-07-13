// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_REDUCTIONRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_REDUCTIONRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

/// ArgMax needs a ROW_MAJOR input to take the multicore kernel (tt-metal
/// #46340; TILE falls back to single-core). Supply it via RowMajor input
/// siblings so the optimizer owns the layout, replacing the ArgMax operand
/// workaround at opt-level >= 1.
struct ArgMaxRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool generatesRowMajorInputSiblings(unsigned operandIdx) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_REDUCTIONRULES_H
