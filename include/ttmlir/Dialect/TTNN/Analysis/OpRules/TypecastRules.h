// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_TYPECASTRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_TYPECASTRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

/// Typecast: NULL hint only (backend decides layout).
/// Also used for WhereOp (same strategy).
struct TypecastRuleBook : OpRuleBook {
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_TYPECASTRULES_H
