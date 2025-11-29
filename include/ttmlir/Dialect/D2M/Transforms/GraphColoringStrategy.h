// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_TRANSFORMS_GRAPHCOLORINGSTRATEGY_H
#define TTMLIR_DIALECT_D2M_TRANSFORMS_GRAPHCOLORINGSTRATEGY_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/EquivalenceClasses.h"

namespace mlir::tt::d2m {

/// Result of building an interference graph with coalescing constraints.
struct InterferenceGraphResult {
  /// Adjacency list representation of the interference graph.
  std::vector<std::vector<size_t>> adjacencyList;
  /// Coalescing constraints: pairs of (storeIdx, loadIdx) that must use the
  /// same color. These represent in-place operations where the result store
  /// must use the same DST index as the input load.
  std::vector<std::pair<size_t, size_t>> coalescingPairs;
  /// Equivalence classes for coalesced nodes. All nodes in an equivalence
  /// class must receive the same color.
  llvm::EquivalenceClasses<size_t> coalescedClasses;
};

/// Utility functions for building interference graphs.
namespace InterferenceGraph {
/// Build interference graph for DST operations within a region.
/// This analyzes DST memory accesses and determines which operations
/// interfere. Returns an index-based adjacency list where indices correspond
/// to dstOperations, along with coalescing constraints for in-place operations.
InterferenceGraphResult buildIndexGraphFromDstOperations(
    mlir::Region &region,
    mlir::ArrayRef<std::pair<mlir::Operation *, int64_t>> dstAccesses);

/// Compute a lower bound on the chromatic number of the graph.
/// Uses max degree + 1 heuristic.
/// \p adjacencyList is the graph represented as adjacency list.
/// Returns the estimated lower bound for the number of colors needed.
unsigned computeChromatic_Lowerbound(
    const std::vector<std::vector<size_t>> &adjacencyList);
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

/// Simple greedy graph coloring algorithm with priority-based ordering.
///
/// Main features:
/// - Priority-based allocation: processes nodes by degree (highest first)
/// - Fast single-pass coloring without eviction
///
/// Fast with good-quality colorings suitable for most real-world cases.
/// Processes nodes in order of decreasing degree and assigns the first
/// available color.
class GreedyColoring : public ColoringStrategy {
public:
  LogicalResult
  colorGraph(const std::vector<std::vector<size_t>> &adjacencyList,
             unsigned numColors, std::vector<unsigned> &coloring) override;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_TRANSFORMS_GRAPHCOLORINGSTRATEGY_H
