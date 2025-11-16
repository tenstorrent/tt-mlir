// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstCapacityAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Utils.h"

using namespace mlir;

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERGC
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

struct D2MInsertDstRegisterGCPass
    : public impl::D2MInsertDstRegisterGCBase<D2MInsertDstRegisterGCPass> {

  D2MInsertDstRegisterGCPass() = default;
  D2MInsertDstRegisterGCPass(const D2MInsertDstRegisterGCPass &pass) = default;
  D2MInsertDstRegisterGCPass(const D2MInsertDstRegisterGCOptions &options)
      : D2MInsertDstRegisterGCBase(options) {}

  // Check if a generic region already has acquire_dst operations.
  static bool hasAcquireDstOp(Region &region) {
    return !region.getOps<AcquireDstOp>().empty();
  }

  // Check if a generic region has acquire_dst but no release_dst.
  static bool hasAcquireDstWithoutRelease(Region &region) {
    bool hasAcquire = !region.getOps<AcquireDstOp>().empty();
    bool hasRelease = !region.getOps<ReleaseDstOp>().empty();
    return hasAcquire && !hasRelease;
  }

  // Create the coloring strategy based on pass options.
  std::unique_ptr<ColoringStrategy> createColoringStrategy() {
    std::string strategy = this->coloringStrategy.getValue();
    if (strategy == "greedy") {
      return std::make_unique<GreedyColoring>();
    }
    if (strategy == "pbqp") {
      // TODO(bnorris): Implement PBQP-based coloring strategy.
      llvm::errs() << "Warning: PBQP strategy not yet implemented, falling "
                      "back to Chaitin-Briggs\n";
      return std::make_unique<ChaitinBriggsColoring>();
    }
    return std::make_unique<ChaitinBriggsColoring>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal()) {
      return;
    }

    DstCapacityAnalysis dstCapacityAnalysis(func);
    uint32_t totalDstTiles = dstCapacityAnalysis.getMinDstCapacity();
    if (maxDstPhysicalSizeTiles > 0) {
      totalDstTiles =
          std::min(totalDstTiles,
                   static_cast<uint32_t>(maxDstPhysicalSizeTiles.getValue()));
    }

    // First pass: Add release_dst to function body if it has acquire_dst but no
    // release_dst.
    if (hasAcquireDstWithoutRelease(func.getBody())) {
      IRRewriter rewriter(&getContext());
      Block &block = func.getBody().front();

      // Insert before the terminator if it exists
      if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>()) {
        rewriter.setInsertionPoint(&block.back());
      } else {
        rewriter.setInsertionPointToEnd(&block);
      }

      // Find all acquire_dst operations and add release_dst for each
      for (auto acquireDst : func.getOps<AcquireDstOp>()) {
        rewriter.create<ReleaseDstOp>(func.getLoc(), acquireDst.getResult());
      }
    }

    // Process each d2m.generic operation.
    func.walk([&](GenericOp genericOp) {
      for (auto &region : genericOp->getRegions()) {
        if (hasAcquireDstOp(region)) {
          // Skip regions that already have DST allocation.
          continue;
        }

        // Identify DST accesses that need allocation.
        auto dstAccesses = identifyDstAccesses(genericOp, region);

        if (dstAccesses.empty()) {
          continue;
        }

        // Infer CB type - find the canonical type used in DST accesses.
        MemRefType cbType = nullptr;
        for (auto &[op, idx] : dstAccesses) {
          if (cbType == nullptr) {
            cbType = mlir::cast<MemRefType>(op->getOperand(1).getType());
          }
        }

        if (!cbType) {
          genericOp.emitError("No CB type found for DST allocation");
          return signalPassFailure();
        }

        // Build interference graph from DST accesses using liveness-based
        // analysis. This follows the standard register allocation approach
        // adapted for DST operations.
        auto interferenceGraph =
            InterferenceGraph::buildIndexGraphFromDstOperations(region,
                                                                dstAccesses);

        // Calculate number of available colors (DST slices).
        const int64_t volume = ttmlir::utils::volume(cbType.getShape());
        if (volume > static_cast<int64_t>(totalDstTiles)) {
          genericOp.emitError("CB volume exceeds available DST tiles");
          return signalPassFailure();
        }
        const uint32_t numColors = totalDstTiles / volume;

        // Use the selected coloring strategy to assign DST slices.
        auto strategy = createColoringStrategy();
        std::vector<unsigned> coloring;
        if (failed(strategy->colorIndexGraph(interferenceGraph, numColors,
                                             coloring))) {
          genericOp.emitError(
              "Graph coloring failed - not enough DST slices available");
          return signalPassFailure();
        }

        // Find the maximum color used to determine DST allocation size.
        uint32_t maxColor = 0;
        for (uint32_t color : coloring) {
          maxColor = std::max(maxColor, color);
        }
        const int64_t numDstSlicesNeeded = maxColor + 1;

        // Create DST type with the required number of slices.
        SmallVector<int64_t> dstShape({numDstSlicesNeeded});
        dstShape.append(cbType.getShape().begin(), cbType.getShape().end());

        MemRefType dstType = MemRefType::get(
            dstShape, cbType.getElementType(),
            AffineMap::getMultiDimIdentityMap(dstShape.size(), &getContext()),
            ttcore::MemorySpaceAttr::get(&getContext(),
                                         ttcore::MemorySpace::RegisterDst));

        // Create rewriter and insert acquire_dst at the start of the region.
        IRRewriter rewriter(&getContext());
        rewriter.setInsertionPointToStart(&region.front());
        AcquireDstOp acquireDst =
            rewriter.create<AcquireDstOp>(genericOp.getLoc(), dstType);

        // Insert release_dst at the end of the region.
        Block &block = region.front();
        if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>()) {
          rewriter.setInsertionPoint(&block.back());
        } else {
          rewriter.setInsertionPointToEnd(&block);
        }
        rewriter.create<ReleaseDstOp>(genericOp.getLoc(),
                                      acquireDst.getResult());

        // Rewrite affine loads and stores to use assigned DST slice indices.
        for (size_t accessIndex = 0; accessIndex < dstAccesses.size();
             ++accessIndex) {
          auto &[op, origIdx] = dstAccesses[accessIndex];
          uint32_t assignedSlice = coloring[accessIndex];

          if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(op)) {
            // Replace the original slice index with the colored one.
            SmallVector<Value> indices = loadOp.getIndices();
            if (!indices.empty()) {
              indices[0] = rewriter.create<arith::ConstantIndexOp>(
                  loadOp.getLoc(), assignedSlice);
              rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
                  loadOp, acquireDst.getResult(), indices);
            }
          } else if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(op)) {
            // Replace the original slice index with the colored one.
            SmallVector<Value> indices = storeOp.getIndices();
            if (!indices.empty()) {
              indices[0] = rewriter.create<arith::ConstantIndexOp>(
                  storeOp.getLoc(), assignedSlice);
              rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
                  storeOp, storeOp.getValueToStore(), acquireDst.getResult(),
                  indices);
            }
          }
        }
      }
    });
  }

private:
  // Identify DST accesses that need coloring (loads/stores to DST memory).
  SmallVector<std::pair<Operation *, int64_t>>
  identifyDstAccesses(GenericOp genericOp, Region &region) {
    SmallVector<std::pair<Operation *, int64_t>> dstAccesses;
    int nextIndex = 0;

    auto isDstMemspace = [](auto op) {
      return op && ttcore::getMemorySpace(op.getMemRef()) ==
                       ttcore::MemorySpace::RegisterDst;
    };

    region.walk([&](Operation *op) {
      if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(op);
          isDstMemspace(loadOp)) {
        dstAccesses.emplace_back(loadOp, nextIndex++);
      } else if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(op);
                 isDstMemspace(storeOp)) {
        dstAccesses.emplace_back(storeOp, nextIndex++);
      }
    });

    return dstAccesses;
  }
};

} // namespace
} // namespace mlir::tt::d2m
