// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include <algorithm>

using namespace mlir;

namespace mlir::tt::d2m {

//===----------------------------------------------------------------------===//
//                    ChaitinBriggsColoring Implementation
//===----------------------------------------------------------------------===//

LogicalResult ChaitinBriggsColoring::colorGraph(
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

LogicalResult GreedyColoring::colorGraph(
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
//                          InterferenceGraph Utilities
//===----------------------------------------------------------------------===//

namespace InterferenceGraph {

std::vector<std::vector<size_t>> buildIndexGraphFromDstOperations(
    mlir::Region &region,
    mlir::ArrayRef<std::pair<mlir::Operation *, int64_t>> dstAccesses) {

  size_t totalAccesses = dstAccesses.size();
  std::vector<std::vector<size_t>> interferenceGraph(totalAccesses);

  // Create mapping from operations to indices.
  llvm::DenseMap<mlir::Operation *, size_t> opToIndex;
  for (size_t i = 0; i < totalAccesses; ++i) {
    opToIndex[dstAccesses[i].first] = i;
  }

  // Build interference graph using SSA value liveness analysis.
  // Two DST accesses interfere if their live ranges overlap - this means
  // they both need DST space at the same time.
  //
  // For loads: The DST space is needed from the load operation until the
  //            last use of the loaded value.
  // For stores: The DST space is needed from when the value is produced
  //             until the store operation completes.

  for (mlir::Block &block : region) {
    // Track which DST accesses are currently live at each program point.
    llvm::SmallPtrSet<mlir::Operation *, 16> currentlyLive;

    // Walk backwards through operations in the block.
    for (auto it = block.rbegin(); it != block.rend(); ++it) {
      mlir::Operation *op = &*it;

      // Process uses of values - this is where load results become live
      for (mlir::Value operand : op->getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
          if (opToIndex.count(defOp)) {
            // This DST load's result is being used - it becomes live
            currentlyLive.insert(defOp);
          }
        }
      }

      // If this operation is a DST access, it interferes with all currently
      // live DST operations.
      if (opToIndex.count(op)) {
        size_t opIndex = opToIndex[op];

        // Add interference edges with all currently live DST accesses
        for (mlir::Operation *liveOp : currentlyLive) {
          if (liveOp != op && opToIndex.count(liveOp)) {
            size_t liveIndex = opToIndex[liveOp];
            // Add bidirectional edge (avoid duplicates by checking)
            if (std::find(interferenceGraph[opIndex].begin(),
                          interferenceGraph[opIndex].end(),
                          liveIndex) == interferenceGraph[opIndex].end()) {
              interferenceGraph[opIndex].push_back(liveIndex);
              interferenceGraph[liveIndex].push_back(opIndex);
            }
          }
        }

        // If this is a load, it's now defined, so remove from live set
        // (walking backwards, definition kills the live range)
        if (isa<affine::AffineLoadOp>(op)) {
          currentlyLive.erase(op);
        }

        // If this is a store, mark it as live (it needs DST space for the
        // value being stored)
        if (isa<affine::AffineStoreOp>(op)) {
          currentlyLive.insert(op);
        }
      }
    }
  }

  return interferenceGraph;
}

} // namespace InterferenceGraph

} // namespace mlir::tt::d2m
