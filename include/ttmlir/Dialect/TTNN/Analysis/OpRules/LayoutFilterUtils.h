// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_LAYOUTFILTERUTILS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_LAYOUTFILTERUTILS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"

#include <vector>

namespace mlir::tt::ttnn::layout_filter_utils {

/// Reject all sharded layouts. Returns true if the layout should be kept.
inline bool rejectAllSharded(TTNNLayoutAttr layout) {
  auto ml = layout.getMemLayout();
  return !(ml && isShardedMemoryLayout(ml.getValue()));
}

/// Reject width-sharded layouts. Returns true if the layout should be kept.
inline bool rejectWidthSharded(TTNNLayoutAttr layout) {
  auto ml = layout.getMemLayout();
  return !(ml && ml.getValue() == TensorMemoryLayout::WidthSharded);
}

/// Filter legalConfigs to only include non-sharded (DRAM or L1-interleaved).
inline std::vector<OpConfig>
filterNonSharded(const std::vector<OpConfig> &legalConfigs) {
  std::vector<OpConfig> result;
  for (const auto &config : legalConfigs) {
    if (!config.outputLayout) {
      result.push_back(config);
      continue;
    }
    auto memLayout = config.outputLayout.getMemLayout();
    if (!memLayout || !isShardedMemoryLayout(memLayout.getValue())) {
      result.push_back(config);
    }
  }
  return result;
}

/// Non-sharded output hints (common pattern for many ops).
inline OutputHints
nonShardedOutputHints(const std::vector<OpConfig> &legalConfigs) {
  return OutputHints{filterNonSharded(legalConfigs),
                     {},
                     /*attemptL1Sharding=*/false};
}

/// NULL-hint-only output (backend decides from inputs, no fallbacks).
inline OutputHints nullHintOnly() {
  return OutputHints{{OpConfig(TTNNLayoutAttr())}, {}};
}

} // namespace mlir::tt::ttnn::layout_filter_utils

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_LAYOUTFILTERUTILS_H
