// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_LAYOUTFILTERUTILS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_LAYOUTFILTERUTILS_H

#include "ttmlir/Dialect/TTCore/Utils/CoreRangeSet.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

#include <algorithm>
#include <limits>
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

/// Whether a sharded layout's core grid forms a full rectangular bounding
/// box (num_cores == bbox_num_cores). Interleaved layouts return true.
inline bool isFullBboxSharded(TTNNLayoutAttr layout) {
  auto ml = layout.getMemLayout();
  if (!isShardedMemoryLayout(ml.getValue())) {
    return true;
  }

  auto grid = layout.getGrid();
  mlir::AffineMap mapping = grid.getVirtToPhysicalMap();
  assert(mapping &&
         "sharded layout must have a grid with a virt-to-phys mapping");

  auto ranges =
      mlir::tt::ttcore::utils::toCoreRangeSet(grid.getShape(), mapping);
  if (ranges.empty()) {
    return false;
  }

  int32_t minX = std::numeric_limits<int32_t>::max();
  int32_t minY = std::numeric_limits<int32_t>::max();
  int32_t maxX = std::numeric_limits<int32_t>::min();
  int32_t maxY = std::numeric_limits<int32_t>::min();
  int64_t numCores = 0;
  for (const auto &[loc, size] : ranges) {
    numCores += static_cast<int64_t>(size[0]) * static_cast<int64_t>(size[1]);
    minX = std::min(minX, loc[0]);
    minY = std::min(minY, loc[1]);
    maxX = std::max(maxX, loc[0] + size[0] - 1);
    maxY = std::max(maxY, loc[1] + size[1] - 1);
  }
  int64_t bboxCores = static_cast<int64_t>(maxX - minX + 1) *
                      static_cast<int64_t>(maxY - minY + 1);
  return numCores == bboxCores;
}

/// Allow only a specific sharding type (plus interleaved). Returns a filter
/// function that rejects sharded layouts whose type doesn't match.
inline LayoutFilterFn
allowOnlyShardingType(TensorMemoryLayout allowedSharding) {
  return [allowedSharding](TTNNLayoutAttr layout) -> bool {
    auto ml = layout.getMemLayout();
    if (!ml || !isShardedMemoryLayout(ml.getValue())) {
      return true; // interleaved — keep
    }
    return ml.getValue() == allowedSharding;
  };
}

/// Filter legalConfigs by an output-layout predicate. Configs with a NULL
/// output layout are always kept (NULL hint means "backend picks"). Keeps any
/// config whose output layout passes `keep`.
inline std::vector<OpConfig>
filterConfigs(const std::vector<OpConfig> &legalConfigs, LayoutFilterFn keep) {
  std::vector<OpConfig> result;
  for (const auto &config : legalConfigs) {
    if (!config.outputLayout || keep(config.outputLayout)) {
      result.push_back(config);
    }
  }
  return result;
}

/// Non-sharded output hints (common pattern for many ops).
inline OutputHints
nonShardedOutputHints(const std::vector<OpConfig> &legalConfigs) {
  return OutputHints{filterConfigs(legalConfigs, rejectAllSharded), {}};
}

/// All non-width-sharded output hints (interleaved + block/height sharded).
inline OutputHints
nonWidthShardedOutputHints(const std::vector<OpConfig> &legalConfigs) {
  return OutputHints{filterConfigs(legalConfigs, rejectWidthSharded), {}};
}

/// NULL-hint-only output (backend decides from inputs, no fallbacks).
inline OutputHints nullHintOnly() {
  return OutputHints{{OpConfig(TTNNLayoutAttr())}, {}};
}

} // namespace mlir::tt::ttnn::layout_filter_utils

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_LAYOUTFILTERUTILS_H
