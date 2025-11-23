// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include <algorithm>

namespace mlir::tt::d2m {

//===----------------------------------------------------------------------===//
//                    ChaitinBriggsColoring Implementation
//===----------------------------------------------------------------------===//

mlir::LogicalResult ChaitinBriggsColoring::colorGraph(
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
      return mlir::failure(); // Spill - not enough colors.
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
  //
  // For operations inside loops or nested regions, we use a conservative
  // approach: assume all DST accesses within the same loop/region interfere
  // with each other, since loop iterations may overlap in execution.
  // TODO (bnorris): Implement precise dependence analysis

  // Collect all DST access operations in program order
  llvm::SmallVector<mlir::Operation *> dstAccessOps;
  for (const auto &[op, idx] : dstAccesses) {
    dstAccessOps.push_back(op);
  }

  // For operations in nested regions (loops), assume conservative interference
  // Build a map from each DST access to its containing loops
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
      opToLoops;
  for (mlir::Operation *op : dstAccessOps) {
    llvm::SmallVector<mlir::Operation *> loops;
    mlir::Operation *parent = op->getParentOp();
    while (parent && parent->getParentRegion() == &region) {
      if (isa<affine::AffineForOp>(parent)) {
        loops.push_back(parent);
      }
      parent = parent->getParentOp();
    }
    opToLoops[op] = loops;
  }

  // Helper to check if two operations interfere based on shared loops
  auto shareCommonLoop =
      [&](const llvm::SmallVector<mlir::Operation *> &loops1,
          const llvm::SmallVector<mlir::Operation *> &loops2) {
        for (mlir::Operation *loop1 : loops1) {
          if (llvm::is_contained(loops2, loop1)) {
            return true;
          }
        }
        return false;
      };

  // Helper to check if a value is live at an operation
  auto isLiveAt = [](mlir::Value value, mlir::Operation *op) {
    // A value is live at op if any of its uses come at or after op
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *user = use.getOwner();
      // Check if user is op itself or comes after op in the same block
      if (user == op ||
          (user->getBlock() == op->getBlock() && !user->isBeforeInBlock(op))) {
        return true;
      }
    }
    return false;
  };

  // Helper to check if two DST access operations interfere via liveness
  auto opsInterfereViaLiveness = [&](mlir::Operation *op1,
                                     mlir::Operation *op2) {
    // Loads produce values - check if the result is live at op2
    if (auto load = mlir::dyn_cast<affine::AffineLoadOp>(op1)) {
      if (isLiveAt(load.getResult(), op2)) {
        return true;
      }
    }
    return false;
  };

  // Build interference: operations interfere if they share a common loop
  // or if their SSA value live ranges overlap
  for (size_t i = 0; i < dstAccessOps.size(); ++i) {
    mlir::Operation *op1 = dstAccessOps[i];
    const auto &loops1 = opToLoops[op1];

    for (size_t j = i + 1; j < dstAccessOps.size(); ++j) {
      mlir::Operation *op2 = dstAccessOps[j];
      const auto &loops2 = opToLoops[op2];

      bool interferes = false;

      // Conservative: operations in the same loop always interfere
      if (shareCommonLoop(loops1, loops2)) {
        interferes = true;
      }
      // For operations not in loops, check SSA value liveness
      else if (loops1.empty() && loops2.empty()) {
        interferes = opsInterfereViaLiveness(op1, op2) ||
                     opsInterfereViaLiveness(op2, op1);
      }

      if (interferes) {
        size_t idx1 = opToIndex[op1];
        size_t idx2 = opToIndex[op2];
        interferenceGraph[idx1].push_back(idx2);
        interferenceGraph[idx2].push_back(idx1);
      }
    }
  }

  return interferenceGraph;
}

unsigned computeChromatic_Lowerbound(
    const std::vector<std::vector<size_t>> &adjacencyList) {
  if (adjacencyList.empty()) {
    return 0;
  }

  // Lower bound = max degree + 1.
  // This is a simple but effective lower bound based on the greedy algorithm.
  unsigned maxDegree = 0;
  for (const auto &neighbors : adjacencyList) {
    maxDegree = std::max(maxDegree, static_cast<unsigned>(neighbors.size()));
  }
  return maxDegree + 1;
}

} // namespace InterferenceGraph

} // namespace mlir::tt::d2m
