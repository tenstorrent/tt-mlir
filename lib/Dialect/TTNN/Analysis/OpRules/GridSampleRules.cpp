// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/GridSampleRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"

namespace mlir::tt::ttnn {

LayoutFilterFn
GridSampleRuleBook::getInputLayoutFilter(unsigned /*operandIdx*/) const {
  // Row-major, interleaved or L1 height-sharded (the sharded page lookup of
  // the kernel is a shift; other shardings are collected to DRAM at runtime).
  return [](TTNNLayoutAttr layout) {
    if (!layout_filter_utils::requireRowMajor(layout)) {
      return false;
    }
    auto ml = layout.getMemLayout();
    if (!ml || !isShardedMemoryLayout(ml.getValue())) {
      return true;
    }
    return ml.getValue() == TensorMemoryLayout::HeightSharded &&
           layout.hasL1BufferType();
  };
}

bool GridSampleRuleBook::generatesRowMajorInputSiblings(
    unsigned /*operandIdx*/) const {
  return true;
}

bool GridSampleRuleBook::shouldExploreReshards() const { return false; }

OutputHints GridSampleRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> & /*legalConfigs*/) const {
  // The kernel chooses its own height-sharded output; a hint changes nothing.
  return layout_filter_utils::nullHintOnly();
}

} // namespace mlir::tt::ttnn
