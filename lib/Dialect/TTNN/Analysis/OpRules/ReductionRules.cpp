// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/ReductionRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// ArgMaxRuleBook
//===----------------------------------------------------------------------===//

// Operand 0 (input) must be ROW_MAJOR.
LayoutFilterFn ArgMaxRuleBook::getInputLayoutFilter(unsigned operandIdx) const {
  return operandIdx == 0 ? layout_filter_utils::requireRowMajor : nullptr;
}

bool ArgMaxRuleBook::generatesRowMajorInputSiblings(unsigned operandIdx) const {
  return operandIdx == 0;
}

//===----------------------------------------------------------------------===//
// ReductionRuleBook
//===----------------------------------------------------------------------===//

OutputHints ReductionRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> &legalConfigs) const {
  // Sharded results come back as reinterpreted memory; see the header.
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}

} // namespace mlir::tt::ttnn
