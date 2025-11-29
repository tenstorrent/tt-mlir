// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::tt::d2m {

namespace {

/// Identify affine load and store operations that require DST allocation.
/// Uses getOperandsLoadFromDstRegister() to identify loads and
/// getDstRegInPlace() to determine if stores need separate DST slices.
static llvm::SmallVector<std::pair<mlir::Operation *, int64_t>>
identifyDstAccesses(mlir::Operation *op, mlir::Region &region) {
  llvm::SmallVector<std::pair<mlir::Operation *, int64_t>> dstAccesses;
  int64_t index = 0;

  // Walk the region looking for D2M compute operations
  region.walk([&](mlir::Operation *operation) {
    // Check if this is a D2M compute operation with DST interface
    auto computeOp =
        llvm::dyn_cast<OperandLoadStoreRegisterOpInterface>(operation);
    if (!computeOp) {
      return;
    }

    // Get operand indices that must be loaded from DST register
    auto operandsLoadFromDst = computeOp.getOperandsLoadFromDstRegister();

    // For each operand that loads from DST, find the corresponding load
    // operation
    for (int64_t operandIdx : operandsLoadFromDst) {
      mlir::Value operand = operation->getOperand(operandIdx);

      // Trace back through the SSA chain to find the affine.load
      if (auto *definingOp = operand.getDefiningOp()) {
        if (auto loadOp =
                llvm::dyn_cast<mlir::affine::AffineLoadOp>(definingOp)) {
          dstAccesses.push_back({loadOp.getOperation(), index++});
        }
      }
    }

    // Check if the result store needs a separate DST slice
    // If getDstRegInPlace() == false, the store needs its own slice
    if (!computeOp.getDstRegInPlace()) {
      // Find the store operation that stores this operation's result
      for (mlir::Operation *user : operation->getUsers()) {
        if (auto storeOp = llvm::dyn_cast<mlir::affine::AffineStoreOp>(user)) {
          dstAccesses.push_back({storeOp.getOperation(), index++});
        }
      }
    }
  });

  return dstAccesses;
}

/// Graph coloring strategy: uses interference analysis to minimize slices.
class DstAnalysisGraphColoring : public DstAnalysis {
public:
  explicit DstAnalysisGraphColoring(std::unique_ptr<ColoringStrategy> strategy,
                                    llvm::StringRef name,
                                    unsigned maxSlices = UINT_MAX)
      : DstAnalysis(maxSlices), coloringStrategy(std::move(strategy)),
        strategyName(name) {}

  DstAnalysisResult analyze(mlir::Operation *op) override {
    DstAnalysisResult result;
    unsigned maxSlicesNeeded = 0;

    // Walk all regions looking for d2m.generic operations
    mlir::WalkResult walkResult =
        op->walk([&](GenericOp genericOp) -> mlir::WalkResult {
          for (auto &region : genericOp.getRegions()) {
            // Skip if already has DST allocation
            if (!region.getOps<AcquireDstOp>().empty()) {
              continue;
            }

            auto dstAccesses = identifyDstAccesses(genericOp, region);
            if (dstAccesses.empty()) {
              continue;
            }

            // Store identified accesses for reuse by transformation passes
            result.dstAccesses.append(dstAccesses.begin(), dstAccesses.end());

            // Build interference graph and coalescing constraints.
            auto graphResult =
                InterferenceGraph::buildIndexGraphFromDstOperations(
                    region, dstAccesses);

            // Contract the graph: merge equivalence classes into single nodes.
            // This reduces the graph size and ensures coalesced nodes get the
            // same color.
            llvm::DenseMap<size_t, size_t> nodeToLeader;
            llvm::DenseMap<size_t, size_t> leaderToContracted;
            std::vector<size_t> contractedToLeader;

            // First pass: find the node with maximum degree in each equiv class
            // to use as the representative (leader) for that class.
            llvm::DenseMap<size_t, size_t> classToMaxDegreeNode;
            llvm::DenseMap<size_t, size_t> classMaxDegree;
            for (size_t i = 0; i < dstAccesses.size(); ++i) {
              size_t classLeader =
                  graphResult.coalescedClasses.getLeaderValue(i);
              size_t degree = graphResult.adjacencyList[i].size();
              if (classToMaxDegreeNode.find(classLeader) ==
                      classToMaxDegreeNode.end() ||
                  degree > classMaxDegree[classLeader]) {
                classToMaxDegreeNode[classLeader] = i;
                classMaxDegree[classLeader] = degree;
              }
            }

            // Build mapping from each node to its equivalence class leader
            // (the node with max degree in the class).
            for (size_t i = 0; i < dstAccesses.size(); ++i) {
              size_t classLeader =
                  graphResult.coalescedClasses.getLeaderValue(i);
              size_t leader = classToMaxDegreeNode[classLeader];
              nodeToLeader[i] = leader;
              if (leaderToContracted.find(leader) == leaderToContracted.end()) {
                size_t contractedIdx = contractedToLeader.size();
                leaderToContracted[leader] = contractedIdx;
                contractedToLeader.push_back(leader);
              }
            }

            // Build contracted adjacency list.
            size_t numContractedNodes = contractedToLeader.size();
            std::vector<std::vector<size_t>> contractedAdjList(
                numContractedNodes);
            for (size_t node = 0; node < dstAccesses.size(); ++node) {
              size_t leader = nodeToLeader[node];
              size_t contractedNode = leaderToContracted[leader];
              for (size_t neighbor : graphResult.adjacencyList[node]) {
                size_t neighborLeader = nodeToLeader[neighbor];
                if (leader == neighborLeader) {
                  continue; // Skip edges within same equivalence class.
                }
                size_t contractedNeighbor = leaderToContracted[neighborLeader];
                // Avoid duplicate edges.
                if (std::find(contractedAdjList[contractedNode].begin(),
                              contractedAdjList[contractedNode].end(),
                              contractedNeighbor) ==
                    contractedAdjList[contractedNode].end()) {
                  contractedAdjList[contractedNode].push_back(contractedNeighbor);
                }
              }
            }

            // Try coloring the contracted graph with minimum number of colors.
            // Start from 1 and increase until we find a valid coloring.
            // Respect maxSlicesAllowed constraint: fail if exceeded.
            std::vector<unsigned> contractedColoring;
            unsigned numColors = 1;
            const unsigned maxAttempts = std::min(
                static_cast<unsigned>(numContractedNodes), maxSlicesAllowed);

            bool coloringSucceeded = false;
            for (; numColors <= maxAttempts; ++numColors) {
              if (succeeded(coloringStrategy->colorGraph(
                      contractedAdjList, numColors, contractedColoring))) {
                coloringSucceeded = true;
                break;
              }
            }

            if (!coloringSucceeded) {
              result.isValid = false;
              unsigned minRequired =
                  InterferenceGraph::computeChromatic_Lowerbound(
                      contractedAdjList);
              result.numSlicesRequired = minRequired;
              result.failureReason = llvm::formatv(
                  "Graph coloring failed: requires {0} slices but only {1} "
                  "available",
                  minRequired, maxSlicesAllowed);
              return mlir::WalkResult::interrupt();
            }

            // Propagate colors from contracted graph back to original nodes.
            std::vector<unsigned> coloring(dstAccesses.size());
            for (size_t i = 0; i < dstAccesses.size(); ++i) {
              size_t leader = nodeToLeader[i];
              size_t contractedIdx = leaderToContracted[leader];
              coloring[i] = contractedColoring[contractedIdx];
            }

            // Find actual number of colors used.
            unsigned colorsUsed = 0;
            for (unsigned color : coloring) {
              colorsUsed = std::max(colorsUsed, color + 1);
            }

            maxSlicesNeeded = std::max(maxSlicesNeeded, colorsUsed);

            // Record per-operation breakdown.
            for (size_t i = 0; i < dstAccesses.size(); ++i) {
              auto &[accessOp, idx] = dstAccesses[i];
              result.operationSlices[accessOp] = coloring[i];
            }
          }
          return mlir::WalkResult::advance();
        });

    if (walkResult.wasInterrupted()) {
      return result;
    }

    result.numSlicesRequired = maxSlicesNeeded;
    result.isValid = true;
    return result;
  }

  llvm::StringRef getStrategyName() const override { return strategyName; }

private:
  std::unique_ptr<ColoringStrategy> coloringStrategy;
  std::string strategyName;
};

} // namespace

std::unique_ptr<DstAnalysis>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
createGraphColoringDstAnalysis(std::unique_ptr<ColoringStrategy> strategy,
                               unsigned maxSlices) {
  return std::make_unique<DstAnalysisGraphColoring>(
      std::move(strategy), "graph-coloring", maxSlices);
}

std::unique_ptr<DstAnalysis> createChaitinBriggsDstAnalysis() {
  return std::make_unique<DstAnalysisGraphColoring>(
      std::make_unique<ChaitinBriggsColoring>(), "graph-coloring");
}

std::unique_ptr<DstAnalysis> createGreedyDstAnalysis() {
  return std::make_unique<DstAnalysisGraphColoring>(
      std::make_unique<GreedyColoring>(), "greedy");
}

} // namespace mlir::tt::d2m
