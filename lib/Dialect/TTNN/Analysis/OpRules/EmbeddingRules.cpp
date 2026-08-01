// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/EmbeddingRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"

namespace mlir::tt::ttnn {

LayoutFilterFn
EmbeddingRuleBook::getInputLayoutFilter(unsigned operandIdx) const {
  if (operandIdx <= 1) {
    return layout_filter_utils::rejectAllSharded;
  }
  return nullptr;
}

OutputHints EmbeddingRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> &legalConfigs) const {
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}

} // namespace mlir::tt::ttnn
