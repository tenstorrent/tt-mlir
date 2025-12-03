// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_DSTANALYSISGRAPHCOLORING_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_DSTANALYSISGRAPHCOLORING_H

#include "ttmlir/Dialect/D2M/Analysis/DstAnalysis.h"

#include <climits>
#include <memory>

namespace mlir::tt::d2m {

// Forward declaration
class ColoringStrategy;

/// Create a graph coloring DST analysis with a custom strategy.
/// \param strategy The graph coloring strategy to use for allocation.
/// \param maxSlices Maximum allowed DST slices. Typically obtained from
///   DstCapacityAnalysis::getMinDstCapacity(). Defaults to UINT_MAX.
std::unique_ptr<DstAnalysis>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
createGraphColoringDstAnalysis(std::unique_ptr<ColoringStrategy> strategy,
                               unsigned maxSlices = UINT_MAX);

/// Create Chaitin-Briggs graph coloring DST analysis.
/// \param maxSlices Maximum allowed DST slices. Defaults to UINT_MAX.
std::unique_ptr<DstAnalysis>
createChaitinBriggsDstAnalysis(unsigned maxSlices = UINT_MAX);

/// Create greedy graph coloring DST analysis.
/// \param maxSlices Maximum allowed DST slices. Defaults to UINT_MAX.
std::unique_ptr<DstAnalysis>
createGreedyDstAnalysis(unsigned maxSlices = UINT_MAX);

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_DSTANALYSISGRAPHCOLORING_H
