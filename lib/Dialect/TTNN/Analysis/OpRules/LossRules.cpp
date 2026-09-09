// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LossRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"

namespace mlir::tt::ttnn {

LayoutFilterFn
CrossEntropyForwardRuleBook::getInputLayoutFilter(unsigned operandIdx) const {
  if (operandIdx == 0) {
    return [](TTNNLayoutAttr layout) {
      return layout_filter_utils::requireTiled(layout) &&
             layout_filter_utils::requireDRAMInterleaved(layout);
    };
  }
  if (operandIdx == 1) {
    return [](TTNNLayoutAttr layout) {
      return layout_filter_utils::requireRowMajor(layout) &&
             layout_filter_utils::requireDRAMInterleaved(layout);
    };
  }
  return nullptr;
}

bool CrossEntropyForwardRuleBook::shouldExploreReshards() const {
  return false;
}

bool CrossEntropyForwardRuleBook::generatesRowMajorInputSiblings(
    unsigned operandIdx) const {
  return operandIdx == 1;
}

OutputHints CrossEntropyForwardRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> & /*legalConfigs*/) const {
  return layout_filter_utils::nullHintOnly();
}

} // namespace mlir::tt::ttnn
