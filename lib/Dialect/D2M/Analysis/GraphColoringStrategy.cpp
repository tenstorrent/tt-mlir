// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include <algorithm>
#include <deque>

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

/// Greedy graph coloring with priority-based ordering and eviction.
/// Inspired by LLVM's RegAllocGreedy register allocator.
///
/// Key features:
/// - Priority-based ordering: processes high-degree nodes first
/// - Eviction: can evict lower-priority (lower-degree) neighbors when stuck
/// - Cascade tracking: prevents infinite eviction loops
LogicalResult GreedyColoring::colorGraph(
    const std::vector<std::vector<size_t>> &adjacencyList, unsigned numColors,
    std::vector<unsigned> &coloring) {

  size_t numNodes = adjacencyList.size();
  coloring.assign(numNodes, UINT_MAX);

  if (numNodes == 0) {
    return success();
  }

  // Step 1: Build priority queue - process high-degree nodes first.
  // Higher degree nodes are harder to color, so we prioritize them.
  std::vector<std::pair<size_t, size_t>> priorityQueue; // (degree, node)
  for (size_t i = 0; i < numNodes; ++i) {
    priorityQueue.emplace_back(adjacencyList[i].size(), i);
  }
  std::sort(priorityQueue.begin(), priorityQueue.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  // Cascade tracking to prevent infinite eviction loops.
  // Each node tracks its cascade number - if we try to evict a node that
  // was already evicted in this cascade, we skip it.
  std::vector<unsigned> cascadeNumber(numNodes, 0);
  unsigned currentCascade = 0;

  // Worklist of nodes to (re)color.
  std::deque<size_t> worklist;
  for (const auto &[degree, node] : priorityQueue) {
    worklist.push_back(node);
  }

  while (!worklist.empty()) {
    size_t node = worklist.front();
    worklist.pop_front();

    // Find colors used by neighbors.
    std::vector<bool> usedColors(numColors, false);
    for (size_t neighbor : adjacencyList[node]) {
      if (coloring[neighbor] != UINT_MAX) {
        usedColors[coloring[neighbor]] = true;
      }
    }

    // Find first available color.
    unsigned color = UINT_MAX;
    for (unsigned c = 0; c < numColors; ++c) {
      if (!usedColors[c]) {
        color = c;
        break;
      }
    }

    if (color != UINT_MAX) {
      coloring[node] = color;
      continue;
    }

    // Step 2: Try eviction - find lowest-degree neighbor to evict.
    // Only evict neighbors with lower priority (lower degree) than current
    // node.
    size_t nodeDegree = adjacencyList[node].size();
    size_t victimNode = SIZE_MAX;
    size_t victimDegree = SIZE_MAX;
    unsigned victimColor = UINT_MAX;

    for (size_t neighbor : adjacencyList[node]) {
      if (coloring[neighbor] == UINT_MAX) {
        continue;
      }
      size_t neighborDegree = adjacencyList[neighbor].size();

      // Only evict lower-priority (lower-degree) nodes.
      // Also check cascade to prevent infinite loops.
      if (neighborDegree < nodeDegree && neighborDegree < victimDegree &&
          cascadeNumber[neighbor] < currentCascade + 1) {
        victimNode = neighbor;
        victimDegree = neighborDegree;
        victimColor = coloring[neighbor];
      }
    }

    if (victimNode != SIZE_MAX) {
      // Evict the victim.
      coloring[victimNode] = UINT_MAX;
      cascadeNumber[victimNode] = ++currentCascade;

      // Assign freed color to current node.
      coloring[node] = victimColor;

      // Re-add victim to worklist for recoloring.
      worklist.push_back(victimNode);
      continue;
    }

    // No eviction possible - spill.
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
//                          InterferenceGraph Utilities
//===----------------------------------------------------------------------===//

namespace InterferenceGraph {

mlir::tt::d2m::InterferenceGraphResult buildIndexGraphFromDstOperations(
    mlir::Region &region,
    mlir::ArrayRef<std::pair<mlir::Operation *, int64_t>> dstAccesses) {

  mlir::tt::d2m::InterferenceGraphResult result;
  size_t totalAccesses = dstAccesses.size();
  result.adjacencyList.resize(totalAccesses);

  // Create mapping from operations to indices.
  llvm::DenseMap<mlir::Operation *, size_t> opToIndex;
  for (size_t i = 0; i < totalAccesses; ++i) {
    opToIndex[dstAccesses[i].first] = i;
  }

  // Create reverse mapping: for each compute op, track its input loads and
  // result store. This enables adding semantic-aware interference edges for
  // non-in-place operations.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<size_t>>
      computeOpInputLoads;
  llvm::DenseMap<mlir::Operation *, size_t> computeOpResultStore;
  // Track intermediate compute ops (those in dstAccesses that are compute ops,
  // not loads or stores). These are ops whose results feed other compute ops.
  llvm::MapVector<mlir::Operation *, size_t> intermediateComputeOpIdx;

  // Helper to link compute operations to their DST accesses.
  auto processLoad = [&](affine::AffineLoadOp loadOp, size_t loadIdx) {
    for (auto &use : loadOp.getResult().getUses()) {
      mlir::Operation *user = use.getOwner();
      if (auto computeOp =
              mlir::dyn_cast<OperandLoadStoreRegisterOpInterface>(user)) {
        computeOpInputLoads[computeOp.getOperation()].push_back(loadIdx);
      }
    }
  };

  auto processStore = [&](affine::AffineStoreOp storeOp, size_t storeIdx) {
    if (mlir::Operation *defOp = storeOp.getValueToStore().getDefiningOp()) {
      if (auto computeOp =
              mlir::dyn_cast<OperandLoadStoreRegisterOpInterface>(defOp)) {
        computeOpResultStore[defOp] = storeIdx;
      }
    }
  };

  // Helper to process intermediate compute ops in the chain.
  // These ops produce values consumed by other compute ops (not stored to L1).
  auto processIntermediateComputeOp = [&](mlir::Operation *computeOp,
                                          size_t idx) {
    intermediateComputeOpIdx[computeOp] = idx;
    // Also track the input loads for this intermediate compute op.
    for (mlir::Value operand : computeOp->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        // Check if the defining op is another compute op (chained
        // intermediate).
        if (auto *it = intermediateComputeOpIdx.find(defOp);
            it != intermediateComputeOpIdx.end()) {
          computeOpInputLoads[computeOp].push_back(it->second);
        }
      }
    }
  };

  // Process all DST accesses to link compute ops to their loads/stores.
  for (size_t i = 0; i < totalAccesses; ++i) {
    mlir::Operation *accessOp = dstAccesses[i].first;
    if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(accessOp)) {
      processLoad(loadOp, i);
    } else if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(accessOp)) {
      processStore(storeOp, i);
    } else if (mlir::isa<OperandLoadStoreRegisterOpInterface>(accessOp)) {
      // This is an intermediate compute op in the chain.
      processIntermediateComputeOp(accessOp, i);
    }
  }

  // Add interference edges between sibling operands of the same compute op.
  // All inputs to a compute operation are live simultaneously and must have
  // different DST slices.
  for (const auto &[computeOp, inputLoadIndices] : computeOpInputLoads) {
    // Add pairwise interference edges between all input loads.
    for (size_t i = 0; i < inputLoadIndices.size(); ++i) {
      for (size_t j = i + 1; j < inputLoadIndices.size(); ++j) {
        size_t idx1 = inputLoadIndices[i];
        size_t idx2 = inputLoadIndices[j];

        // Avoid duplicate edges.
        if (std::find(result.adjacencyList[idx1].begin(),
                      result.adjacencyList[idx1].end(),
                      idx2) == result.adjacencyList[idx1].end()) {
          result.adjacencyList[idx1].push_back(idx2);
          result.adjacencyList[idx2].push_back(idx1);
        }
      }
    }
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
  // This can be improved by using a more precise dependence analysis.

  // Collect all DST access operations in program order.
  llvm::SmallVector<mlir::Operation *> dstAccessOps;
  for (const auto &[op, idx] : dstAccesses) {
    dstAccessOps.push_back(op);
  }

  // For operations in nested regions (loops), assume conservative interference.
  // Build a map from each DST access to its containing loops.
  llvm::MapVector<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
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

  // Helper to check if a value is live at an operation.
  auto isLiveAt = [](mlir::Value value, mlir::Operation *op) {
    // A value is live at op if any of its uses come at or after op.
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (user == op ||
          (user->getBlock() == op->getBlock() && !user->isBeforeInBlock(op))) {
        return true;
      }
    }
    return false;
  };

  // Helper to check if two DST access operations interfere via liveness.
  auto opsInterfereViaLiveness = [&](mlir::Operation *op1,
                                     mlir::Operation *op2) {
    // Loads produce values - check if the result is live at op2.
    if (auto load = mlir::dyn_cast<affine::AffineLoadOp>(op1)) {
      if (isLiveAt(load.getResult(), op2)) {
        return true;
      }
    }
    // Intermediate compute ops also produce values - check if result is live.
    if (mlir::isa<OperandLoadStoreRegisterOpInterface>(op1)) {
      if (op1->getNumResults() > 0 && isLiveAt(op1->getResult(0), op2)) {
        return true;
      }
    }
    return false;
  };

  // Build interference based on SSA value live ranges.
  // With loop-indexed DST accesses (dst[slice, %iv1, %iv2, ...]),
  // operations in different loop iterations access different physical locations
  // and don't interfere, so we only check SSA liveness.
  for (size_t i = 0; i < dstAccessOps.size(); ++i) {
    mlir::Operation *op1 = dstAccessOps[i];

    for (size_t j = i + 1; j < dstAccessOps.size(); ++j) {
      mlir::Operation *op2 = dstAccessOps[j];

      // Check if SSA values interfere (one is live at the definition of the
      // other).
      bool interferes = opsInterfereViaLiveness(op1, op2) ||
                        opsInterfereViaLiveness(op2, op1);

      if (interferes) {
        size_t idx1 = opToIndex[op1];
        size_t idx2 = opToIndex[op2];
        result.adjacencyList[idx1].push_back(idx2);
        result.adjacencyList[idx2].push_back(idx1);
      }
    }
  }

  // Handle in-place and non-in-place operations.
  // For operations with getDstRegInPlace() = true, the result store must use
  // the same DST index as the input load - add coalescing constraint.
  // For operations with getDstRegInPlace() = false, the result store must not
  // reuse DST slices allocated to input loads - add interference edges.
  for (auto &[computeOp, resultStoreIdx] : computeOpResultStore) {
    auto iface = mlir::dyn_cast<OperandLoadStoreRegisterOpInterface>(computeOp);
    if (!iface) {
      continue;
    }

    const auto &inputLoads = computeOpInputLoads[computeOp];
    if (iface.getDstRegInPlace()) {
      // In-place operation: result store must use the same DST index as the
      // first input load. Add coalescing constraint.
      if (!inputLoads.empty()) {
        result.coalescingPairs.emplace_back(resultStoreIdx, inputLoads[0]);
      }
    } else {
      // Non-in-place operation: result store must not reuse input DST slots.
      // Add interference edges from result store to all input loads.
      for (size_t inputLoadIdx : inputLoads) {
        // Avoid duplicate edges.
        if (std::find(result.adjacencyList[resultStoreIdx].begin(),
                      result.adjacencyList[resultStoreIdx].end(),
                      inputLoadIdx) ==
            result.adjacencyList[resultStoreIdx].end()) {
          result.adjacencyList[resultStoreIdx].push_back(inputLoadIdx);
          result.adjacencyList[inputLoadIdx].push_back(resultStoreIdx);
        }
      }
    }
  }

  // Handle intermediate compute ops in chains (those whose results feed other
  // compute ops, not L1 stores).
  // - For in-place operations: coalesce with input (same DST index).
  // - For non-in-place operations: add interference edges (different DST
  // index).
  // - If the compute op also has a result store entry, coalesce them.
  for (auto &[computeOp, intermediateIdx] : intermediateComputeOpIdx) {
    auto iface = mlir::dyn_cast<OperandLoadStoreRegisterOpInterface>(computeOp);
    if (!iface) {
      continue;
    }

    // If this compute op also has a result store, coalesce the intermediate
    // entry with the store entry (they represent the same DST location).
    if (auto storeIt = computeOpResultStore.find(computeOp);
        storeIt != computeOpResultStore.end()) {
      result.coalescingPairs.emplace_back(intermediateIdx, storeIt->second);
    }

    const auto &inputLoads = computeOpInputLoads[computeOp];
    if (iface.getDstRegInPlace()) {
      // In-place intermediate op: must use same DST index as input.
      if (!inputLoads.empty()) {
        result.coalescingPairs.emplace_back(intermediateIdx, inputLoads[0]);
      }
    } else {
      // Non-in-place intermediate op: must not reuse input DST slots.
      // Add interference edges from this intermediate to all its inputs.
      for (size_t inputLoadIdx : inputLoads) {
        // Avoid duplicate edges.
        if (std::find(result.adjacencyList[intermediateIdx].begin(),
                      result.adjacencyList[intermediateIdx].end(),
                      inputLoadIdx) ==
            result.adjacencyList[intermediateIdx].end()) {
          result.adjacencyList[intermediateIdx].push_back(inputLoadIdx);
          result.adjacencyList[inputLoadIdx].push_back(intermediateIdx);
        }
      }
    }
  }

  // Build equivalence classes from coalescing pairs.
  // All nodes in an equivalence class must receive the same color.
  for (size_t i = 0; i < totalAccesses; ++i) {
    result.coalescedClasses.insert(i);
  }
  for (const auto &[idx1, idx2] : result.coalescingPairs) {
    result.coalescedClasses.unionSets(idx1, idx2);
  }

  // Remove interference edges between nodes in the same equivalence class.
  // Coalesced nodes must have the same color, so they cannot interfere.
  for (size_t node = 0; node < result.adjacencyList.size(); ++node) {
    auto &neighbors = result.adjacencyList[node];
    neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(),
                                   [&](size_t neighbor) {
                                     return result.coalescedClasses.isEquivalent(
                                         node, neighbor);
                                   }),
                    neighbors.end());
  }

  return result;
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
