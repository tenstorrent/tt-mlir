// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;

namespace mlir::tt::d2m {

// ============================================================================
// InterferenceGraph Implementation
// ============================================================================

void InterferenceGraph::addNode(Value v) {
  if (graph.find(v) == graph.end()) {
    graph[v] = {};
  }
}

void InterferenceGraph::addEdge(Value u, Value v) {
  if (u == v) {
    return;
  }
  addNode(u);
  addNode(v);
  if (std::find(graph[u].begin(), graph[u].end(), v) == graph[u].end()) {
    graph[u].push_back(v);
  }
  if (std::find(graph[v].begin(), graph[v].end(), u) == graph[v].end()) {
    graph[v].push_back(u);
  }
}

unsigned InterferenceGraph::degree(Value v) const {
  auto it = graph.find(v);
  return it != graph.end() ? it->second.size() : 0;
}

const llvm::SmallVector<Value> &InterferenceGraph::neighbors(Value v) const {
  static const llvm::SmallVector<Value> empty;
  auto it = graph.find(v);
  return it != graph.end() ? it->second : empty;
}

llvm::SmallVector<Value> InterferenceGraph::getNodes() const {
  llvm::SmallVector<Value> nodes;
  for (auto &[v, _] : graph) {
    nodes.push_back(v);
  }
  return nodes;
}

void InterferenceGraph::print(llvm::raw_ostream &os) const {
  for (const auto &[v, neighbors] : graph) {
    os << "Value: ";
    v.print(os);
    os << "\n  Interferes with:\n";
    for (auto n : neighbors) {
      os << "    ";
      n.print(os);
      os << "\n";
    }
  }
}

// ============================================================================
// ChaitinBriggsColoring Implementation
// ============================================================================

llvm::DenseMap<Value, unsigned>
ChaitinBriggsColoring::colorGraph(const InterferenceGraph &graph,
                                  unsigned numColors) {
  llvm::DenseMap<Value, unsigned> coloring;
  llvm::SmallVector<Value> nodes = graph.getNodes();

  // Simplification phase: remove nodes with degree < K.
  llvm::SmallVector<Value> simplifyStack;
  llvm::DenseSet<Value> removed;

  while (true) {
    bool found = false;
    for (auto v : nodes) {
      if (removed.count(v)) {
        continue;
      }

      unsigned degree = 0;
      for (auto neighbor : graph.neighbors(v)) {
        if (!removed.count(neighbor)) {
          degree++;
        }
      }

      if (degree < numColors) {
        simplifyStack.push_back(v);
        removed.insert(v);
        found = true;
        break;
      }
    }

    if (!found) {
      break;
    }
  }

  // Selection phase: assign colors by popping from stack.
  while (!simplifyStack.empty()) {
    Value v = simplifyStack.pop_back_val();

    llvm::SmallVector<bool> usedColors(numColors, false);
    for (auto neighbor : graph.neighbors(v)) {
      if (coloring.count(neighbor)) {
        unsigned neighborColor = coloring[neighbor];
        if (neighborColor < numColors) {
          usedColors[neighborColor] = true;
        }
      }
    }

    unsigned color = 0;
    while (color < numColors && usedColors[color]) {
      color++;
    }

    coloring[v] = color;
  }

  // Handle nodes with degree >= K (spill nodes).
  unsigned nextColor = 0;
  for (auto v : nodes) {
    if (!coloring.count(v)) {
      coloring[v] = nextColor % numColors;
      nextColor++;
    }
  }

  return coloring;
}

// ============================================================================
// GreedyColoring Implementation
// ============================================================================

llvm::DenseMap<Value, unsigned>
GreedyColoring::colorGraph(const InterferenceGraph &graph, unsigned numColors) {
  llvm::DenseMap<Value, unsigned> coloring;
  llvm::SmallVector<Value> nodes = graph.getNodes();

  // Sort nodes by degree (highest first) for better allocation order.
  llvm::sort(nodes, [&](Value a, Value b) {
    return graph.degree(a) > graph.degree(b);
  });

  for (auto v : nodes) {
    llvm::SmallVector<bool> usedColors(numColors, false);
    for (auto neighbor : graph.neighbors(v)) {
      if (coloring.count(neighbor)) {
        unsigned neighborColor = coloring[neighbor];
        if (neighborColor < numColors) {
          usedColors[neighborColor] = true;
        }
      }
    }

    // Find the first available color.
    unsigned color = 0;
    while (color < numColors && usedColors[color]) {
      color++;
    }

    // If no color available, attempt eviction (simplified).
    if (color >= numColors) {
      // Find the least-used color among neighbors for spilling.
      llvm::SmallVector<unsigned> colorCounts(numColors, 0);
      for (auto neighbor : graph.neighbors(v)) {
        if (coloring.count(neighbor)) {
          unsigned neighborColor = coloring[neighbor];
          if (neighborColor < numColors) {
            colorCounts[neighborColor]++;
          }
        }
      }

      // Pick the color with minimum conflicts.
      color = std::distance(
          colorCounts.begin(),
          std::min_element(colorCounts.begin(), colorCounts.end()));
    }

    coloring[v] = color;
  }

  return coloring;
}

} // namespace mlir::tt::d2m
