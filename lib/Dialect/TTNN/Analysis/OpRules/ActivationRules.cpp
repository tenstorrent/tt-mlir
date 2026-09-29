// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/ActivationRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// TTMLSwigluElemwiseBackwardRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn TTMLSwigluElemwiseBackwardRuleBook::getInputLayoutFilter(
    unsigned /*operandIdx*/) const {
  return [](TTNNLayoutAttr layout) {
    return layout_filter_utils::requireTiled(layout) &&
           layout_filter_utils::rejectAllSharded(layout);
  };
}

bool TTMLSwigluElemwiseBackwardRuleBook::shouldExploreReshards() const {
  return false;
}

OutputHints TTMLSwigluElemwiseBackwardRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> & /*legalConfigs*/) const {
  return layout_filter_utils::nullHintOnly();
}

} // namespace mlir::tt::ttnn
