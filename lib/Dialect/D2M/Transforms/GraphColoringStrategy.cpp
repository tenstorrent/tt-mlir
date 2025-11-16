// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;

namespace mlir::tt::d2m {

//===----------------------------------------------------------------------===//
//                          InterferenceGraph Implementation
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
//                          ChaitinBriggsColoring Implementation
//===----------------------------------------------------------------------===//

llvm::DenseMap<Value, unsigned>
ChaitinBriggsColoring::colorGraph(const InterferenceGraph &graph,
                                  unsigned numColors) {
  assert(numColors > 0);
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
  assert(numColors > 0);
  unsigned nextColor = 0;
  for (auto v : nodes) {
    if (!coloring.count(v)) {
      coloring[v] = nextColor % numColors;
      nextColor++;
    }
  }

  return coloring;
}

LogicalResult ChaitinBriggsColoring::colorIndexGraph(
    const std::vector<std::vector<size_t>> &adjacencyList, unsigned numColors,
    std::vector<unsigned> &coloring) {
  coloring.assign(adjacencyList.size(), UINT_MAX);
  size_t numNodes = adjacencyList.size();

  // Simplification phase: remove nodes with degree < K.
  std::vector<size_t> simplifyStack;
  std::vector<bool> removed(numNodes, false);

  while (true) {
    bool found = false;
    for (size_t node = 0; node < numNodes; ++node) {
      if (removed[node]) {
        continue;
      }

      // Count active neighbors (not removed).
      unsigned degree = 0;
      for (size_t neighbor : adjacencyList[node]) {
        if (!removed[neighbor]) {
          degree++;
        }
      }

      if (degree < numColors) {
        simplifyStack.push_back(node);
        removed[node] = true;
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
    size_t node = simplifyStack.back();
    simplifyStack.pop_back();

    std::vector<bool> usedColors(numColors, false);
    for (size_t neighbor : adjacencyList[node]) {
      if (coloring[neighbor] != UINT_MAX && coloring[neighbor] < numColors) {
        usedColors[coloring[neighbor]] = true;
      }
    }

    unsigned color = 0;
    while (color < numColors && usedColors[color]) {
      color++;
    }

    if (color >= numColors) {
      return failure(); // Spill - not enough colors.
    }

    coloring[node] = color;
  }

  // Handle nodes with degree >= K (potential spill nodes).
  // Try to color them, but if we can't, it's a spill.
  for (size_t node = 0; node < numNodes; ++node) {
    if (coloring[node] == UINT_MAX) {
      std::vector<bool> usedColors(numColors, false);
      for (size_t neighbor : adjacencyList[node]) {
        if (coloring[neighbor] != UINT_MAX && coloring[neighbor] < numColors) {
          usedColors[coloring[neighbor]] = true;
        }
      }

      unsigned color = 0;
      while (color < numColors && usedColors[color]) {
        color++;
      }

      if (color >= numColors) {
        return failure(); // Spill - not enough colors.
      }

      coloring[node] = color;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
//                          GreedyColoring Implementation
//===----------------------------------------------------------------------===//

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

LogicalResult GreedyColoring::colorIndexGraph(
    const std::vector<std::vector<size_t>> &adjacencyList, unsigned numColors,
    std::vector<unsigned> &coloring) {
  coloring.assign(adjacencyList.size(), UINT_MAX);
  std::vector<bool> usedColors(numColors, false);

  for (size_t node = 0; node < adjacencyList.size(); ++node) {
    // Reset used colors
    std::fill(usedColors.begin(), usedColors.end(), false);

    // Mark colors used by neighbors
    for (size_t neighbor : adjacencyList[node]) {
      if (coloring[neighbor] != UINT_MAX) {
        usedColors[coloring[neighbor]] = true;
      }
    }

    // Find first available color
    unsigned color = 0;
    for (; color < numColors; ++color) {
      if (!usedColors[color]) {
        break;
      }
    }

    // If no color available, this is a spill
    if (color >= numColors) {
      return failure(); // Spill - not enough colors
    }

    coloring[node] = color;
  }

  return success();
}

//===----------------------------------------------------------------------===//
//                          InterferenceGraph Implementation
//===----------------------------------------------------------------------===//

std::vector<std::vector<size_t>>
InterferenceGraph::buildIndexGraphFromDstOperations(
    mlir::Region &region,
    mlir::ArrayRef<std::pair<mlir::Operation *, int64_t>> dstAccesses) {

  size_t totalAccesses = dstAccesses.size();
  std::vector<std::vector<size_t>> interferenceGraph(totalAccesses);

  // Create mapping from operations to indices
  llvm::DenseMap<mlir::Operation *, size_t> opToIndex;
  for (size_t i = 0; i < totalAccesses; ++i) {
    opToIndex[dstAccesses[i].first] = i;
  }

  // Build interference graph using data flow analysis similar to register
  // allocation
  for (mlir::Block &block : region) {
    llvm::SmallPtrSet<mlir::Operation *, 16> currentlyLiveOps;

    // Walk backwards through operations in the block
    for (auto it = block.rbegin(); it != block.rend(); ++it) {
      mlir::Operation *op = &*it;

      // If this operation is a DST store, it interferes with all currently
      // live DST operations (since it writes to DST memory)
      if (opToIndex.count(op) && isa<affine::AffineStoreOp>(op)) {
        size_t opIndex = opToIndex[op];
        for (mlir::Operation *liveOp : currentlyLiveOps) {
          if (opToIndex.count(liveOp)) {
            size_t liveIndex = opToIndex[liveOp];
            interferenceGraph[opIndex].push_back(liveIndex);
            interferenceGraph[liveIndex].push_back(opIndex);
          }
        }
        currentlyLiveOps.erase(op);
      }

      // For DST load operations, add them to currently live set
      if (opToIndex.count(op) && isa<affine::AffineLoadOp>(op)) {
        currentlyLiveOps.insert(op);
      }
    }
  }

  return interferenceGraph;
}

} // namespace mlir::tt::d2m
