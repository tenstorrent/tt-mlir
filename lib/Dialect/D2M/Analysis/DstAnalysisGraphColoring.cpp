// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"

using namespace mlir;

namespace mlir::tt::d2m {

namespace {

/// Identify affine load operations that require DST allocation.
static SmallVector<std::pair<Operation *, int64_t>>
identifyDstAccesses(Operation *op, Region &region) {
  SmallVector<std::pair<Operation *, int64_t>> dstAccesses;
  int64_t index = 0;

  region.walk([&](affine::AffineLoadOp loadOp) {
    // Check if this load produces a value used by DST-eligible operations.
    // For now, all loads in compute regions are considered DST candidates.
    dstAccesses.push_back({loadOp.getOperation(), index++});
  });

  return dstAccesses;
}

/// Graph coloring strategy: uses interference analysis to minimize slices.
class DstAnalysisGraphColoring : public DstAnalysis {
public:
  explicit DstAnalysisGraphColoring(std::unique_ptr<ColoringStrategy> strategy,
                                    llvm::StringRef name)
      : coloringStrategy(std::move(strategy)), strategyName(name) {}

  DstAnalysisResult analyze(Operation *op) override {
    DstAnalysisResult result;
    unsigned maxSlicesNeeded = 0;

    // Walk all regions looking for d2m.generic operations
    WalkResult walkResult = op->walk([&](GenericOp genericOp) -> WalkResult {
      for (auto &region : genericOp->getRegions()) {
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

        // Build interference graph
        auto interferenceGraph =
            InterferenceGraph::buildIndexGraphFromDstOperations(region,
                                                                dstAccesses);

        // Try coloring with minimum number of colors
        // Start from 1 and increase until we find a valid coloring
        std::vector<unsigned> coloring;
        unsigned numColors = 1;
        const unsigned maxAttempts = dstAccesses.size();

        bool coloringSucceeded = false;
        for (; numColors <= maxAttempts; ++numColors) {
          if (succeeded(coloringStrategy->colorGraph(interferenceGraph,
                                                     numColors, coloring))) {
            coloringSucceeded = true;
            break;
          }
        }

        if (!coloringSucceeded) {
          result.isValid = false;
          result.failureReason = "Graph coloring failed for operation";
          result.numSlicesRequired =
              dstAccesses.size(); // Conservative fallback
          return WalkResult::interrupt();
        }

        // Find actual number of colors used
        unsigned colorsUsed = 0;
        for (unsigned color : coloring) {
          colorsUsed = std::max(colorsUsed, color + 1);
        }

        maxSlicesNeeded = std::max(maxSlicesNeeded, colorsUsed);

        // Record per-operation breakdown
        for (size_t i = 0; i < dstAccesses.size(); ++i) {
          auto &[accessOp, idx] = dstAccesses[i];
          result.operationSlices[accessOp] = coloring[i];
        }
      }
      return WalkResult::advance();
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
createGraphColoringDstAnalysis(std::unique_ptr<ColoringStrategy> strategy) {
  return std::make_unique<DstAnalysisGraphColoring>(std::move(strategy),
                                                    "graph-coloring");
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
