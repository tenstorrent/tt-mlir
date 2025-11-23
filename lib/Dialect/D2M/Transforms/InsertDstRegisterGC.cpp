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
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERGC
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Helper structures for organizing DST accesses by loop nest.
template <typename OpT>
using OpAndIndexOffset = std::pair<OpT, int64_t>;

// Stores dst loads/stores, organized by common loop nests.
struct CopyInfo {
  void push_back(mlir::affine::AffineLoadOp load, int64_t indexOffset) {
    loads.emplace_back(load, indexOffset);
  }

  void push_back(mlir::affine::AffineStoreOp store, int64_t indexOffset) {
    stores.emplace_back(store, indexOffset);
  }

  llvm::SmallVector<OpAndIndexOffset<mlir::affine::AffineLoadOp>> loads;
  SmallVector<OpAndIndexOffset<affine::AffineStoreOp>> stores;
};
using CopyInfoMap = DenseMap<Operation *, CopyInfo>;

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

  /// Extract the CB (Circular Buffer) MemRef type from DST accesses.
  ///
  /// This function finds the first DST access operation (AffineLoadOp or
  /// AffineStoreOp) and returns its MemRef type, which represents the CB type
  /// used for DST operations.
  ///
  /// \param dstAccesses The list of DST access operations to examine.
  /// \return The MemRefType of the first DST access, or nullptr if none found.
  static MemRefType extractCbTypeFromDstAccesses(
      const SmallVectorImpl<std::pair<Operation *, int64_t>> &dstAccesses) {
    for (auto &[op, idx] : dstAccesses) {
      if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(op)) {
        return loadOp.getMemRefType();
      }
      if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(op)) {
        return storeOp.getMemRefType();
      }
    }
    return nullptr;
  }

  /// Replace an affine load or store op to use the DST buffer instead of L1.
  template <typename OpT>
  static void replaceAffineAccessWithDst(IRRewriter &rewriter, OpT op,
                                         Value dstBuffer,
                                         uint32_t assignedSlice) {
    rewriter.setInsertionPoint(op);

    auto dstMap = op.getMap().insertResult(
        getAffineConstantExpr(assignedSlice, rewriter.getContext()), 0);
    SmallVector<Value> dstIndices(op.getIndices().begin(),
                                  op.getIndices().end());

    if constexpr (std::is_same_v<OpT, affine::AffineLoadOp>) {
      rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(op, dstBuffer, dstMap,
                                                        dstIndices);
    } else if constexpr (std::is_same_v<OpT, affine::AffineStoreOp>) {
      rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
          op, op.getValueToStore(), dstBuffer, dstMap, dstIndices);
    } else {
      static_assert(!std::is_same_v<OpT, OpT>, "Unsupported operation type");
    }
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
    if (strategy == "greedy" || strategy == "graph-coloring-greedy") {
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
    if (strategy == "greedy" || strategy == "graph-coloring-greedy") {
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

    // First pass: Add release_dst after the last use of each acquire_dst.
    if (hasAcquireDstWithoutRelease(func.getBody())) {
      IRRewriter rewriter(&getContext());
      Liveness liveness(func);

      // Find all acquire_dst operations and add release_dst after last use
      for (auto acquireDst : func.getOps<AcquireDstOp>()) {
        Value dstValue = acquireDst.getResult();
        Operation *lastUser = nullptr;

        // Find the last operation that uses this acquired DST value.
        // We iterate through all uses and find the one that appears latest
        // in the program order within its block.
        for (OpOperand &use : dstValue.getUses()) {
          Operation *user = use.getOwner();
          if (!lastUser) {
            lastUser = user;
          } else {
            // Check if user comes after lastUser in the same block
            Block *userBlock = user->getBlock();
            Block *lastUserBlock = lastUser->getBlock();

            if (userBlock == lastUserBlock) {
              // Same block: use positional comparison
              if (user->isBeforeInBlock(lastUser)) {
                // user comes before lastUser, so lastUser is still the last
              } else {
                lastUser = user;
              }
            } else {
              // Different blocks: use liveness to determine dominance
              const LivenessBlockInfo *userLiveness =
                  liveness.getLiveness(userBlock);
              if (userLiveness && userLiveness->isLiveOut(dstValue)) {
                // If the value is live-out of the user's block, then there
                // might be later uses. Keep the later block.
                lastUser = user;
              }
            }
          }
        }

        // Insert release_dst after the last user.
        if (lastUser) {
          // Find the top-level operation in the function body that contains
          // the last user. This ensures we place the release after
          // loops/regions rather than inside them.
          Operation *topLevelOp = lastUser;
          Block &funcBody = func.getBody().front();
          while (topLevelOp->getBlock() != &funcBody) {
            topLevelOp = topLevelOp->getParentOp();
            if (!topLevelOp) {
              // Shouldn't happen, but handle defensively
              topLevelOp = lastUser;
              break;
            }
          }

          rewriter.setInsertionPointAfter(topLevelOp);
          rewriter.create<ReleaseDstOp>(topLevelOp->getLoc(), dstValue);
        } else {
          // No users found - release immediately after acquire.
          // This shouldn't normally happen but handle it defensively.
          rewriter.setInsertionPointAfter(acquireDst);
          rewriter.create<ReleaseDstOp>(acquireDst.getLoc(), dstValue);
        }
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
        MemRefType cbType = extractCbTypeFromDstAccesses(dstAccesses);

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

        // Group DST accesses by their enclosing loop nest.
        CopyInfoMap copyInfoMap =
            groupAccessesByLoopNest(dstAccesses, coloring);

        // Generate hoisted prologue copy loops for each loop nest.
        for (auto &[loopOp, copyInfo] : copyInfoMap) {
          generatePrologueLoop(rewriter, acquireDst, loopOp, copyInfo);
        }

        for (auto &[loopOp, copyInfo] : copyInfoMap) {
          generateEpilogueLoop(rewriter, acquireDst.getResult(), loopOp,
                               copyInfo);
        }

        // Insert release_dst at the end, after all epilogue loops
        Block &block = region.front();
        if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>()) {
          rewriter.setInsertionPoint(&block.back());
        } else {
          rewriter.setInsertionPointToEnd(&block);
        }
        rewriter.create<ReleaseDstOp>(genericOp.getLoc(),
                                      acquireDst.getResult());

        // Rewrite original loads and stores to use DST instead of L1.
        for (size_t accessIndex = 0; accessIndex < dstAccesses.size();
             ++accessIndex) {
          auto &[op, origIdx] = dstAccesses[accessIndex];
          uint32_t assignedSlice = coloring[accessIndex];

          if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(op)) {
            replaceAffineAccessWithDst<affine::AffineLoadOp>(
                rewriter, loadOp, acquireDst.getResult(), assignedSlice);
          } else if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(op)) {
            replaceAffineAccessWithDst<affine::AffineStoreOp>(
                rewriter, storeOp, acquireDst.getResult(), assignedSlice);
          }
        }
      }
    });
  }

private:
  // Find the outermost affine.for loop that contains the given operation.
  static Operation *findOutermostLoop(Operation *op) {
    Operation *outermostLoop = nullptr;
    Operation *current = op->getParentOp();

    // Walk up the parent chain until we reach the region boundary
    while (current) {
      if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(current)) {
        outermostLoop = forOp;
      }
      // Stop if the parent is not in the same region
      if (!current->getParentOp() ||
          current->getParentRegion() != op->getParentRegion()) {
        break;
      }
      current = current->getParentOp();
    }

    return outermostLoop;
  }

  // Group DST accesses by their enclosing loop nest.
  // Returns a map from outermost loop operation to CopyInfo containing
  // the loads and stores with their assigned DST slice indices.
  static CopyInfoMap groupAccessesByLoopNest(
      const SmallVector<std::pair<Operation *, int64_t>> &dstAccesses,
      const std::vector<unsigned> &coloring) {
    CopyInfoMap copyInfoMap;

    for (size_t i = 0; i < dstAccesses.size(); ++i) {
      auto &[op, origIdx] = dstAccesses[i];
      uint32_t assignedSlice = coloring[i];

      // Find the outermost loop containing this operation
      Operation *outermostLoop = findOutermostLoop(op);

      // If no loop found, use the operation itself as the key
      Operation *key = outermostLoop ? outermostLoop : op;

      // Add to the appropriate CopyInfo
      if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(op)) {
        copyInfoMap[key].push_back(loadOp, assignedSlice);
      } else if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(op)) {
        copyInfoMap[key].push_back(storeOp, assignedSlice);
      }
    }

    return copyInfoMap;
  }

  // Clone a loop nest structure for prologue/epilogue.
  // Returns the cloned loop and the IR mapping from original to cloned values.
  static std::pair<affine::AffineForOp, IRMapping>
  cloneLoopNest(IRRewriter &rewriter, affine::AffineForOp rootLoop) {
    IRMapping mapper;
    auto *cloned = rewriter.clone(*rootLoop, mapper);
    auto clonedLoop = cast<affine::AffineForOp>(cloned);

    // Recursively clear the bodies of all nested loops, keeping only loop
    // structure and terminators
    std::function<void(affine::AffineForOp)> clearLoopBody;
    clearLoopBody = [&](affine::AffineForOp loop) {
      Block &body = loop.getRegion().front();

      // Collect nested loops first (to avoid iterator invalidation)
      SmallVector<affine::AffineForOp> nestedLoops;
      for (Operation &op : body.getOperations()) {
        if (auto nestedLoop = dyn_cast<affine::AffineForOp>(&op)) {
          nestedLoops.push_back(nestedLoop);
        }
      }

      // Recursively clear nested loops
      for (auto nestedLoop : nestedLoops) {
        clearLoopBody(nestedLoop);
      }

      // Remove all operations except nested loops and the terminator
      SmallVector<Operation *> toErase;
      for (Operation &op : body.getOperations()) {
        if (!isa<affine::AffineForOp>(&op) &&
            !op.hasTrait<OpTrait::IsTerminator>()) {
          toErase.push_back(&op);
        }
      }
      // Drop all uses before erasing to avoid use-def chain issues
      for (Operation *op : toErase) {
        op->dropAllUses();
      }
      for (Operation *op : toErase) {
        rewriter.eraseOp(op);
      }
    };

    clearLoopBody(clonedLoop);
    return {clonedLoop, std::move(mapper)};
  }

  // Find the innermost loop in a nest and collect all induction variables.
  static std::pair<affine::AffineForOp, SmallVector<Value>>
  findInnermostLoopAndIndices(affine::AffineForOp rootLoop) {
    SmallVector<Value> indices;
    affine::AffineForOp innermostLoop = rootLoop;
    affine::AffineForOp currentLoop = rootLoop;

    while (currentLoop) {
      indices.push_back(currentLoop.getInductionVar());
      innermostLoop = currentLoop;

      // Look for a nested loop in the body
      currentLoop = nullptr;
      Block &body = innermostLoop.getRegion().front();
      for (Operation &op : body.getOperations()) {
        if (auto nestedLoop = dyn_cast<affine::AffineForOp>(&op)) {
          currentLoop = nestedLoop;
          break;
        }
      }
    }

    return {innermostLoop, indices};
  }

  /// Generate copy loops for data movement between L1 and DST.
  ///
  /// This templated function handles both prologue (L1->DST) and epilogue
  /// (DST->L1) copy loop generation. The direction is controlled by the
  /// IsPrologue template parameter and the external memory reference.
  ///
  /// \tparam IsPrologue True for prologue (L1->DST), false for epilogue
  /// (DST->L1).
  /// \tparam OpT The operation type (AffineLoadOp for prologue, AffineStoreOp
  /// for epilogue).
  /// \param rewriter The rewriter to use for IR generation.
  /// \param externalMemref External memory reference (DST for both cases).
  /// \param loopTemplate The operation around which to generate copies.
  /// \param copyPairs The pairs of operations and their slice indices to
  /// process.
  template <bool IsPrologue, typename OpT, typename PairT>
  static void generateCopyLoops(IRRewriter &rewriter, Value externalMemref,
                                Operation *loopTemplate,
                                ArrayRef<PairT> copyPairs) {
    if (copyPairs.empty()) {
      return;
    }

    auto forLoop = dyn_cast<affine::AffineForOp>(loopTemplate);
    if (!forLoop) {
      // No loop nest - generate inline copies
      OpBuilder::InsertionGuard guard(rewriter);
      if constexpr (IsPrologue) {
        rewriter.setInsertionPoint(loopTemplate);
      } else {
        rewriter.setInsertionPointAfter(loopTemplate);
      }

      for (const auto &pair : copyPairs) {
        OpT op = pair.first;
        int64_t sliceIdx = pair.second;

        if constexpr (IsPrologue) {
          // L1 -> DST: load from L1 (op.getMemRef()), store to DST
          auto l1Value = rewriter.create<affine::AffineLoadOp>(
              op.getLoc(), op.getMemRef(), op.getMap(), op.getIndices());

          auto dstMap = op.getMap().insertResult(
              getAffineConstantExpr(sliceIdx, rewriter.getContext()), 0);
          SmallVector<Value> dstIndices(op.getIndices().begin(),
                                        op.getIndices().end());

          rewriter.create<affine::AffineStoreOp>(
              op.getLoc(), l1Value.getResult(), externalMemref, dstMap,
              dstIndices);
        } else {
          // DST -> L1: load from DST, store to L1 (op.getMemRef())
          auto dstMap = op.getMap().insertResult(
              getAffineConstantExpr(sliceIdx, rewriter.getContext()), 0);
          SmallVector<Value> dstIndices(op.getIndices().begin(),
                                        op.getIndices().end());

          auto dstValue = rewriter.create<affine::AffineLoadOp>(
              op.getLoc(), externalMemref, dstMap, dstIndices);

          rewriter.create<affine::AffineStoreOp>(
              op.getLoc(), dstValue.getResult(), op.getMemRef(), op.getMap(),
              op.getIndices());
        }
      }
      return;
    }

    // Handle loop case
    OpBuilder::InsertionGuard guard(rewriter);
    if constexpr (IsPrologue) {
      rewriter.setInsertionPoint(forLoop);
    } else {
      rewriter.setInsertionPointAfter(forLoop);
    }

    // Clone the loop structure and get the mapping
    auto [copyLoop, mapper] = cloneLoopNest(rewriter, forLoop);

    // Find the innermost loop and collect all induction variables
    auto [innermostLoop, newIndices] = findInnermostLoopAndIndices(copyLoop);

    // Populate with copy operations at the innermost loop level
    rewriter.setInsertionPointToStart(&innermostLoop.getRegion().front());

    for (const auto &pair : copyPairs) {
      OpT op = pair.first;
      int64_t sliceIdx = pair.second;

      // Map the original indices to the cloned loop's induction variables
      SmallVector<Value> mappedIndices;
      for (Value idx : op.getIndices()) {
        Value mappedIdx = mapper.lookupOrDefault(idx);
        mappedIndices.push_back(mappedIdx);
      }

      if constexpr (IsPrologue) {
        // Create: %val = affine.load %l1[...]
        auto l1Load = rewriter.create<affine::AffineLoadOp>(
            op.getLoc(), op.getMemRef(), op.getMap(), mappedIndices);

        // Create: affine.store %val, %dst[slice, ...]
        auto dstMap = op.getMap().insertResult(
            getAffineConstantExpr(sliceIdx, rewriter.getContext()), 0);
        rewriter.create<affine::AffineStoreOp>(op.getLoc(), l1Load.getResult(),
                                               externalMemref, dstMap,
                                               mappedIndices);
      } else {
        // Create: %val = affine.load %dst[slice, ...]
        auto dstMap = op.getMap().insertResult(
            getAffineConstantExpr(sliceIdx, rewriter.getContext()), 0);
        auto dstLoad = rewriter.create<affine::AffineLoadOp>(
            op.getLoc(), externalMemref, dstMap, mappedIndices);

        // Create: affine.store %val, %l1[...]
        rewriter.create<affine::AffineStoreOp>(op.getLoc(), dstLoad.getResult(),
                                               op.getMemRef(), op.getMap(),
                                               mappedIndices);
      }
    }
  }

  // Generate prologue copy nest (L1 -> DST) ahead of the loop/compute region
  // that uses the original loads.
  static void generatePrologueLoop(IRRewriter &rewriter,
                                   AcquireDstOp acquireDst,
                                   Operation *loopTemplate,
                                   const CopyInfo &copyInfo) {
    generateCopyLoops</*IsPrologue=*/true, affine::AffineLoadOp,
                      OpAndIndexOffset<affine::AffineLoadOp>>(
        rewriter, acquireDst.getResult(), loopTemplate, copyInfo.loads);
  }

  // Generate epilogue copy nest (DST -> L1) after the loop/compute region that
  // produced the values.
  static void generateEpilogueLoop(IRRewriter &rewriter, Value dstMemref,
                                   Operation *loopTemplate,
                                   const CopyInfo &copyInfo) {
    generateCopyLoops</*IsPrologue=*/false, affine::AffineStoreOp,
                      OpAndIndexOffset<affine::AffineStoreOp>>(
        rewriter, dstMemref, loopTemplate, copyInfo.stores);
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
