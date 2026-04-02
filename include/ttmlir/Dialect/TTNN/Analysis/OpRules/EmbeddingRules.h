// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_EMBEDDINGRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_EMBEDDINGRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

/// Embedding outputs non-sharded only.
struct EmbeddingRuleBook : OpRuleBook {
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_EMBEDDINGRULES_H
