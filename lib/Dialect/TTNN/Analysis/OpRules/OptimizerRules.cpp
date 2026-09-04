// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OptimizerRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"

namespace mlir::tt::ttnn {

// All operands must be tiled and DRAM-interleaved, so the filter is
// operand-independent.
LayoutFilterFn
AdamWRuleBook::getInputLayoutFilter(unsigned /*operandIdx*/) const {
  return [](TTNNLayoutAttr layout) {
    return layout_filter_utils::requireTiled(layout) &&
           layout_filter_utils::requireDRAMInterleaved(layout);
  };
}

} // namespace mlir::tt::ttnn
