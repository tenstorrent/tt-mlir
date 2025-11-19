// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_TRANSFORMS_GRAPHCOLORINGSTRATEGY_H
#define TTMLIR_DIALECT_D2M_TRANSFORMS_GRAPHCOLORINGSTRATEGY_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::d2m {

/// Utility functions for building interference graphs.
namespace InterferenceGraph {
/// Build interference graph for DST operations within a region.
/// This analyzes DST memory accesses and determines which operations
/// interfere. Returns an index-based adjacency list where indices correspond
/// to dstOperations.
std::vector<std::vector<size_t>> buildIndexGraphFromDstOperations(
    mlir::Region &region,
    mlir::ArrayRef<std::pair<mlir::Operation *, int64_t>> dstAccesses);
} // namespace InterferenceGraph

/// Abstract strategy for graph coloring algorithms.
class ColoringStrategy {
public:
  virtual ~ColoringStrategy() = default;

  /// Color an index-based graph represented as adjacency list.
  /// \p coloring is filled with colors where index i contains the color
  /// assigned to node i. Returns success if coloring succeeded, failure
  /// otherwise.
  virtual LogicalResult
  colorGraph(const std::vector<std::vector<size_t>> &adjacencyList,
             unsigned numColors, std::vector<unsigned> &coloring) = 0;
};

/// Chaitin-Briggs graph coloring algorithm.
///
/// Uses simplification and selection phases:
/// 1. Simplification: Remove nodes with degree < K and push onto stack.
/// 2. Selection: Pop nodes from stack and assign colors.
///
/// Slower than the greedy algorithm but produces high-quality colorings with
/// guaranteed optimality for K-colorable graphs; best for complex interference
/// graphs.
///
/// Reference: G. J. Chaitin, "Register allocation & spilling via graph
/// coloring," SIGPLAN Not., vol. 17, no. 6, pp. 98-105, Jun. 1982.
/// P. Briggs, "Register Allocation via Graph Coloring," PhD thesis,
/// Rice University, 1992.
class ChaitinBriggsColoring : public ColoringStrategy {
public:
  LogicalResult
  colorGraph(const std::vector<std::vector<size_t>> &adjacencyList,
             unsigned numColors, std::vector<unsigned> &coloring) override;
};

/// Greedy graph coloring algorithm inspired by LLVM's RegAllocGreedy.
///
/// Main features:
/// - Priority-based allocation: processes nodes by degree (highest first)
/// - Eviction and reassignment: attempts to free up colors when needed
/// - Spill cost consideration: prefers to spill lower-priority values
///
/// Fast with good-quality colorings suitable for most real-world cases; best
/// for quick compilation with acceptable register pressure.
///
/// Reference: LLVM's RegAllocGreedy implementation:
/// https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/RegAllocGreedy.cpp
class GreedyColoring : public ColoringStrategy {
public:
  LogicalResult
  colorGraph(const std::vector<std::vector<size_t>> &adjacencyList,
             unsigned numColors, std::vector<unsigned> &coloring) override;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_TRANSFORMS_GRAPHCOLORINGSTRATEGY_H
