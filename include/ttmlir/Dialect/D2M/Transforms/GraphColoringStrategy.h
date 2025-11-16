// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_TRANSFORMS_GRAPHCOLORINGSTRATEGY_H
#define TTMLIR_DIALECT_D2M_TRANSFORMS_GRAPHCOLORINGSTRATEGY_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m {

/// Interference graph for graph coloring register allocation.
///
/// An interference graph is a graph where:
/// - Each node represents a virtual register (e.g., a DST value) or operation.
/// - Each edge connects two nodes if their values interfere (i.e., they are
///   simultaneously live and cannot use the same physical register).
///
/// The graph coloring problem assigns each node a "color" (physical register)
/// such that no two adjacent nodes have the same color. We aim to use K or
/// fewer colors, where K is the number of available physical registers.
class InterferenceGraph {
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> graph;

public:
  void addNode(mlir::Value v);
  void addEdge(mlir::Value u, mlir::Value v);

  bool hasInterference() const { return !graph.empty(); }
  unsigned degree(mlir::Value v) const;
  const llvm::SmallVector<mlir::Value> &neighbors(mlir::Value v) const;
  llvm::SmallVector<mlir::Value> getNodes() const;
  const llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> &
  getGraph() const {
    return graph;
  }
  void print(llvm::raw_ostream &os) const;

  /// Build interference graph for DST operations within a region.
  /// This analyzes DST memory accesses and determines which operations
  /// interfere. Returns an index-based adjacency list where indices correspond
  /// to dstOperations.
  static std::vector<std::vector<size_t>> buildIndexGraphFromDstOperations(
      mlir::Region &region,
      mlir::ArrayRef<std::pair<mlir::Operation *, int64_t>> dstAccesses);
};

/// Abstract strategy for graph coloring algorithms.
class ColoringStrategy {
public:
  virtual ~ColoringStrategy() = default;
  virtual llvm::DenseMap<mlir::Value, unsigned>
  colorGraph(const InterferenceGraph &graph, unsigned numColors) = 0;

  /// Color an index-based graph represented as adjacency list.
  /// \p coloring is filled with colors where index i contains the color
  /// assigned to node i. Returns success if coloring succeeded, failure
  /// otherwise.
  virtual LogicalResult
  colorIndexGraph(const std::vector<std::vector<size_t>> &adjacencyList,
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
  llvm::DenseMap<mlir::Value, unsigned>
  colorGraph(const InterferenceGraph &graph, unsigned numColors) override;
};

/// Greedy graph coloring algorithm inspired by LLVM's RegAllocGreedy.
///
/// Key features:
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
  llvm::DenseMap<mlir::Value, unsigned>
  colorGraph(const InterferenceGraph &graph, unsigned numColors) override;

  LogicalResult
  colorIndexGraph(const std::vector<std::vector<size_t>> &adjacencyList,
                  unsigned numColors, std::vector<unsigned> &coloring) override;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_TRANSFORMS_GRAPHCOLORINGSTRATEGY_H
