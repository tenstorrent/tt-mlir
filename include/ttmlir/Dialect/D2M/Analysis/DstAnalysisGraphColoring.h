// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_DSTANALYSISGRAPHCOLORING_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_DSTANALYSISGRAPHCOLORING_H

#include "ttmlir/Dialect/D2M/Analysis/DstAnalysis.h"

namespace mlir::tt::d2m {

// Forward declaration
class ColoringStrategy;

/// Create a graph coloring DST analysis with a custom strategy.
///
/// \param strategy The graph coloring strategy to use for allocation.
std::unique_ptr<DstAnalysis>
createGraphColoringDstAnalysis(
    std::unique_ptr<ColoringStrategy> strategy);

/// Create Chaitin-Briggs graph coloring DST analysis.
std::unique_ptr<DstAnalysis>
createChaitinBriggsDstAnalysis();

/// Create greedy graph coloring DST analysis.
std::unique_ptr<DstAnalysis> createGreedyDstAnalysis();

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_DSTANALYSISGRAPHCOLORING_H
