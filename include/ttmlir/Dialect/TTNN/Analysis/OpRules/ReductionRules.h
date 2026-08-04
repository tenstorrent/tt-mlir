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

/// Reductions must not take a sharded output. Given one they return
/// reinterpreted memory rather than the reduction: a `ttnn.sum` reducing
/// 1x1023x128256 -> 1x1023x1 that the optimizer gave a height-sharded L1 result
/// (32x1 grid over an irregular core set) produced values up to +-5.8e13 plus
/// infs, where the identical op with a DRAM-interleaved result is correct. The
/// default rule book offers sharded output fallbacks, so without this the
/// analysis picks one whenever L1 pressure makes it look profitable.
///
/// Demonstrated on SumOp; applied to the reduction family, which shares the
/// same kernel path. Conservative: it gives up a sharded-output option that is
/// not known to work for any reduction.
struct ReductionRuleBook : OpRuleBook {
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_REDUCTIONRULES_H
