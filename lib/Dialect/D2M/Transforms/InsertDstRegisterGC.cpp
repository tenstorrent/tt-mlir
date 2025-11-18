// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstAnalysis.h"
#include "ttmlir/Dialect/D2M/Analysis/DstAnalysisGraphColoring.h"
#include "ttmlir/Dialect/D2M/Analysis/DstCapacityAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
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
    return std::make_unique<ChaitinBriggsColoring>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal()) {
      return;
    }

    // Analyze DST capacity based on fullSyncEn setting and optional override.
    DstCapacityAnalysis dstCapacityAnalysis(
        func, this->fullSyncEn,
        maxDstPhysicalSizeTiles > 0
            ? static_cast<uint32_t>(maxDstPhysicalSizeTiles.getValue())
            : 0);
    uint32_t totalDstTiles = dstCapacityAnalysis.getMinDstCapacity();

    // Pre-check: Use DstAnalysis to estimate if operations will fit.
    // This provides early feedback before expensive graph construction.
    // Use the same strategy as the actual allocation for consistency.
    std::unique_ptr<DstAnalysis> reqAnalysis;
    std::string strategy = this->coloringStrategy.getValue();
    if (strategy == "greedy") {
      reqAnalysis = createGreedyDstAnalysis();
    } else {
      reqAnalysis = createChaitinBriggsDstAnalysis();
    }

    DstAnalysisResult requirement = reqAnalysis->analyze(func);
    if (!requirement.isValid) {
      func.emitError() << "DST analysis failed: "
                       << requirement.failureReason.value_or("unknown reason");
      return signalPassFailure();
    }

    // Check if estimated requirements exceed available capacity.
    // This is a pre-check; actual allocation may still fail due to
    // interference patterns, but this catches obvious capacity issues early.
    if (requirement.numSlicesRequired > totalDstTiles) {
      func.emitError() << "Estimated DST requirement ("
                       << requirement.numSlicesRequired
                       << " slices) exceeds available capacity ("
                       << totalDstTiles
                       << " tiles). Consider enabling spilling or reducing "
                          "operation complexity.";
      return signalPassFailure();
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
        // Note: linalg operations should be converted to affine loops by
        // a separate pass before running this pass.
        auto dstAccesses = identifyDstAccesses(genericOp, region);

        if (dstAccesses.empty()) {
          continue;
        }

        // Infer CB type - find the canonical type used in DST accesses.
        MemRefType cbType = nullptr;
        for (auto &[op, idx] : dstAccesses) {
          if (cbType == nullptr) {
            if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(op)) {
              cbType = loadOp.getMemRefType();
            } else if (auto storeOp =
                           mlir::dyn_cast<affine::AffineStoreOp>(op)) {
              cbType = storeOp.getMemRefType();
            }
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
        if (failed(
                strategy->colorGraph(interferenceGraph, numColors, coloring))) {
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

        // Try linalg.generic splitting approach first for better fusion.
        // If successful, the region is fully transformed and we're done.
        // Otherwise, fall back to inline copy generation with affine loops.
        if (trySplitLinalgGeneric(genericOp, region, acquireDst, coloring,
                                  dstAccesses)) {
          continue;
        }

        // Generate data copy loops and rewrite loads/stores.
        // Process loads: generate L1→DST copy, then replace with DST load
        for (size_t accessIndex = 0; accessIndex < dstAccesses.size();
             ++accessIndex) {
          auto &[op, origIdx] = dstAccesses[accessIndex];
          uint32_t assignedSlice = coloring[accessIndex];

          if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(op)) {
            // Generate L1 → DST copy before the load
            rewriter.setInsertionPoint(loadOp);

            // Load from L1 (cb) using the original map and indices
            auto l1Value = rewriter.create<affine::AffineLoadOp>(
                loadOp.getLoc(), loadOp.getMemRef(), loadOp.getMap(),
                loadOp.getIndices());

            // Create DST affine map by inserting the slice constant at position
            // 0
            auto dstMap = loadOp.getMap().insertResult(
                getAffineConstantExpr(assignedSlice, &getContext()), 0);

            // DST indices are the same as L1 indices (slice is in the map)
            SmallVector<Value> dstIndices(loadOp.getIndices().begin(),
                                          loadOp.getIndices().end());

            // Store to DST
            rewriter.create<affine::AffineStoreOp>(
                loadOp.getLoc(), l1Value.getResult(), acquireDst.getResult(),
                dstMap, dstIndices);

            // Replace original load with load from DST
            rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
                loadOp, acquireDst.getResult(), dstMap, dstIndices);

          } else if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(op)) {
            // Generate DST → L1 copy after computing the value
            // TODO(bnorris): Some of these stores may be unnecessary if the
            // value is used immediately and should be removed by subsequent
            // liveness analysis.
            rewriter.setInsertionPoint(storeOp);

            // Create DST affine map by inserting the slice constant at position
            // 0
            auto dstMap = storeOp.getMap().insertResult(
                getAffineConstantExpr(assignedSlice, &getContext()), 0);

            // DST indices are the same as L1 indices (slice is in the map)
            SmallVector<Value> dstIndices(storeOp.getIndices().begin(),
                                          storeOp.getIndices().end());

            // Store value to DST first
            rewriter.create<affine::AffineStoreOp>(
                storeOp.getLoc(), storeOp.getValueToStore(),
                acquireDst.getResult(), dstMap, dstIndices);

            // Then load from DST and store to L1 (cb)
            auto dstValue = rewriter.create<affine::AffineLoadOp>(
                storeOp.getLoc(), acquireDst.getResult(), dstMap, dstIndices);

            rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
                storeOp, dstValue.getResult(), storeOp.getMemRef(),
                storeOp.getMap(), storeOp.getIndices());
          }
        }
      }
    });
  }

private:
  // Try to split a linalg.generic operation into three separate linalg.generic
  // operations: copy-in (L1→DST), compute, copy-out (DST→L1).
  // This enables better data movement fusion compared to inline copies.
  //
  // Returns true if the region was successfully transformed, false otherwise
  // (e.g., no linalg.generic found, or region already has affine loops).
  bool trySplitLinalgGeneric(GenericOp genericOp, Region &region,
                             AcquireDstOp acquireDst,
                             const std::vector<unsigned> &coloring,
                             const SmallVector<std::pair<Operation *, int64_t>> &dstAccesses) {
    // Find linalg.generic operation in the region
    linalg::GenericOp linalgOp = nullptr;
    region.walk([&](linalg::GenericOp op) {
      if (!linalgOp) {
        linalgOp = op;
      }
    });

    if (!linalgOp) {
      // No linalg.generic found, fall back to affine loop approach
      return false;
    }

    // TODO(bnorris): Implement linalg.generic splitting:
    // 1. Create copy-in linalg.generic: for each input, create a generic that
    //    loads from L1 input memref and stores to DST with assigned slice
    // 2. Clone compute linalg.generic: replace input block args to load from DST,
    //    replace output block arg to store to DST
    // 3. Create copy-out linalg.generic: load from DST and store to L1 output
    //
    // For now, return false to use the existing affine loop approach
    return false;
  }

  // Identify DST accesses that need coloring based on compute operations.
  // This uses the OperandLoadStoreRegisterOpInterface to find which operands
  // need to be loaded from DST registers, similar to InsertDstRegisterAccess.
  // Unlike that pass, we use graph coloring instead of linear allocation.
  //
  // Note: linalg.generic operations should be converted to affine loops before
  // calling this function.
  SmallVector<std::pair<Operation *, int64_t>>
  identifyDstAccesses(GenericOp genericOp, Region &region) {
    SmallVector<std::pair<Operation *, int64_t>> dstAccesses;
    int nextIndex = 0;

    // Helper to check if an operation loads from non-DST memory space
    auto notDstMemspace = [](auto op) {
      return op && ttcore::getMemorySpace(op.getMemRef()) !=
                       ttcore::MemorySpace::RegisterDst;
    };

    // Walk operations that implement the interface to determine DST needs
    region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
      // Collect loads for operands that need DST
      for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
        if (auto potentialLoad = computeOp->getOperand(operandIdx)
                                     .getDefiningOp<affine::AffineLoadOp>();
            notDstMemspace(potentialLoad)) {
          dstAccesses.emplace_back(potentialLoad, nextIndex++);
        }
      }

      // Collect stores from this op's result
      for (auto *user : computeOp->getUsers()) {
        if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(user);
            notDstMemspace(storeOp)) {
          dstAccesses.emplace_back(storeOp, nextIndex++);
        }
      }
    });

    return dstAccesses;
  }
};

} // namespace
} // namespace mlir::tt::d2m
