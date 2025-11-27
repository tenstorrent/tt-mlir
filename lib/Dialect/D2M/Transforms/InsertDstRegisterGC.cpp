// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstAnalysis.h"
#include "ttmlir/Dialect/D2M/Analysis/DstAnalysisGraphColoring.h"
#include "ttmlir/Dialect/D2M/Analysis/DstCapacityAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/TileMatmulUtils.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Utils.h"

#include <algorithm>
#include <numeric>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERGC
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Loop context information for DST index computation.
// Tracks the loop nest containing a DST access and its iteration space.
struct LoopContext {
  llvm::SmallVector<mlir::affine::AffineForOp, 4>
      loopNest;                             // Outermost to innermost.
  llvm::SmallVector<int64_t, 4> tripCounts; // Trip count for each loop.
  int64_t totalIterations{1};               // Product of all trip counts.

  LoopContext() = default;
};

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

  // Loop dimension indices that should guard prologue data copy.
  // Non-empty for reductions where we skip the first iteration copy.
  SmallVector<int64_t> guardIndices;
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

  /// Normalize DST indices by applying coalescing constraints and renumbering.
  ///
  /// This function:
  /// 1. Applies coalescing constraints for in-place operations (e.g., unary ops
  ///    like exp_tile where input and output must use the same DST index)
  /// 2. Renumbers colors to use consecutive indices starting from 0
  ///
  /// \param coloring The coloring result from graph coloring (modified
  /// in-place) \param coalescingPairs Pairs of (storeIdx, loadIdx) that must
  /// use same color \return The number of unique colors after normalization
  static uint32_t normalizeDstIndices(
      std::vector<unsigned> &coloring,
      const std::vector<std::pair<size_t, size_t>> &coalescingPairs) {
    // Apply coalescing constraints: for in-place ops, the store must use the
    // same color as the input load.
    for (const auto &[storeIdx, loadIdx] : coalescingPairs) {
      coloring[storeIdx] = coloring[loadIdx];
    }

    // Renumber colors to use consecutive indices starting from 0.
    llvm::DenseMap<unsigned, unsigned> colorMap;
    unsigned nextColor = 0;
    for (unsigned &color : coloring) {
      auto [it, inserted] = colorMap.insert({color, nextColor});
      if (inserted) {
        nextColor++;
      }
      color = it->second;
    }

    return nextColor;
  }

  /// Replace an affine load or store op to use the DST buffer instead of L1.
  /// Uses constant DST slice index (matches legacy behavior).
  /// Handles type mismatches (e.g., typecast ops) using DstReinterpretCastOp.
  template <typename OpT>
  static void replaceAffineAccessWithDst(IRRewriter &rewriter, OpT op,
                                         Value dstBuffer,
                                         uint32_t assignedSlice) {
    rewriter.setInsertionPoint(op);

    // Always use constant DST slice index (matches legacy behavior).
    // Loop IVs appear in CB dimensions, not in DST slice computation.
    Value dstSliceIndex =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), assignedSlice);

    // DST access pattern: %dst[dstSliceIndex, cb_indices...]
    // We need to compute the CB indices by applying the original affine map.
    SmallVector<Value> dstIndices;
    dstIndices.push_back(dstSliceIndex);

    // Apply the original operation's affine map to get the actual CB indices
    AffineMap origMap = op.getAffineMap();
    if (origMap.getNumResults() > 0) {
      // For each result dimension of the original map, create an affine.apply
      for (unsigned i = 0; i < origMap.getNumResults(); ++i) {
        AffineMap projMap =
            AffineMap::get(origMap.getNumDims(), origMap.getNumSymbols(),
                           origMap.getResult(i), rewriter.getContext());
        Value idx = rewriter.create<affine::AffineApplyOp>(op.getLoc(), projMap,
                                                           op.getMapOperands());
        dstIndices.push_back(idx);
      }
    } else {
      // No affine map results (shouldn't happen for loads/stores).
      dstIndices.append(op.getIndices().begin(), op.getIndices().end());
    }

    // Use identity map since we're passing computed index values
    auto dstMap = AffineMap::getMultiDimIdentityMap(dstIndices.size(),
                                                    rewriter.getContext());

    // Get DST element type for type comparison.
    auto dstMemRefType = mlir::cast<MemRefType>(dstBuffer.getType());
    Type dstElementType = dstMemRefType.getElementType();

    if constexpr (std::is_same_v<OpT, affine::AffineLoadOp>) {
      rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(op, dstBuffer, dstMap,
                                                        dstIndices);
    } else if constexpr (std::is_same_v<OpT, affine::AffineStoreOp>) {
      Value valueToStore = op.getValueToStore();
      Type valueType = valueToStore.getType();

      // Handle type mismatch (e.g., typecast ops produce f16 but DST is f32).
      // Insert DstReinterpretCastOp to cast the value to the DST element type.
      if (valueType != dstElementType) {
        valueToStore = rewriter.create<DstReinterpretCastOp>(
            op.getLoc(), dstElementType, valueToStore);
      }

      rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
          op, valueToStore, dstBuffer, dstMap, dstIndices);
    } else {
      static_assert(!std::is_same_v<OpT, OpT>, "Unsupported operation type");
    }
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

    // Process linalg.generic ops that were not converted by LinalgToAffine.
    // These are tile_matmul ops when useTileMatmul=false.
    // The matmul conversion creates TileMatmulBlockOp. DST allocation for
    // matmul is handled by converting the temporary affine loops and creating
    // appropriate acquire/release operations.
    if (!this->useTileMatmul) {
      IRRewriter linalgRewriter(&getContext());
      WalkResult linalgWalkResult = func.walk([&](GenericOp genericOp) {
        for (auto &region : genericOp->getRegions()) {
          if (hasAcquireDstOp(region)) {
            // Skip regions that already have DST allocation.
            continue;
          }

          // Process tile_matmul linalg ops using shared utility.
          // The callback inserts DST allocation with first-iteration guards for
          // matmul accumulation. This follows the same pattern as the legacy
          // pass.
          if (failed(utils::processTileMatmulLinalgOps(
                  linalgRewriter, genericOp, region,
                  [&](RewriterBase &rewriter, Region &loopRegion,
                      Operation *outermostLoop) -> LogicalResult {
                    return utils::insertMatmulDstAllocation(
                        rewriter, genericOp, loopRegion, outermostLoop,
                        totalDstTiles);
                  }))) {
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });

      if (linalgWalkResult.wasInterrupted()) {
        return signalPassFailure();
      }
    }

    // Process each d2m.generic operation for DST allocation.
    func.walk([&](GenericOp genericOp) {
      for (auto &region : genericOp->getRegions()) {
        if (hasAcquireDstOp(region)) {
          // Skip regions that already have DST allocation.
          continue;
        }

        // Process affine loops marked by LinalgToAffine pass.
        // This matches the legacy pass pattern where each marked loop is
        // processed separately with its own DST allocation context.
        Block &block = region.front();
        bool foundMarkedLoop = false;
        block.walk([&](affine::AffineForOp forOp) {
          if (!forOp->hasAttr("d2m.linalg_root")) {
            return;
          }

          foundMarkedLoop = true;
          // Remove the marker attribute after identifying the loop.
          forOp->removeAttr("d2m.linalg_root");

          // Process this marked loop's region.
          Region &markedLoopRegion = forOp.getRegion();
          processMarkedLoopRegion(genericOp, markedLoopRegion, forOp,
                                  totalDstTiles);
        });
      }
    });
  }

private:
  // Process a single marked loop region for DST allocation.
  // This follows the legacy pass pattern where each d2m.linalg_root loop
  // is processed independently with a common grouping key.
  void processMarkedLoopRegion(GenericOp genericOp, Region &region,
                               affine::AffineForOp markedLoop,
                               uint32_t totalDstTiles) {
    // Identify DST accesses that need allocation within this loop region.
    auto dstAccesses = identifyDstAccesses(genericOp, region);

    if (dstAccesses.empty()) {
      return;
    }

    // Analyze loop contexts for all DST accesses.
    auto loopContexts = analyzeLoopContexts(dstAccesses);

    // Check if any loop context analysis failed (dynamic bounds).
    if (llvm::any_of(loopContexts, [](const auto &entry) {
          const auto &[op, ctx] = entry;
          return !ctx.loopNest.empty() && ctx.tripCounts.empty();
        })) {
      genericOp.emitError("DST analysis failed: dynamic loop bounds");
      return;
    }

    // Find maximum loop iterations across all DST accesses.
    int64_t maxLoopIterations = 1;
    for (const auto &[op, ctx] : loopContexts) {
      maxLoopIterations = std::max(maxLoopIterations, ctx.totalIterations);
    }

    // Infer CB type - find the canonical type used in DST accesses.
    MemRefType cbType = extractCbTypeFromDstAccesses(dstAccesses);

    if (!cbType) {
      genericOp.emitError("No CB type found for DST allocation");
      return;
    }

    // Use CB element type for DST allocation.
    Type dstElementType = cbType.getElementType();

    // Build interference graph and coalescing constraints from DST accesses.
    // This follows the standard register allocation approach adapted for DST
    // operations. Coalescing constraints are used for in-place operations
    // (e.g., unary ops like exp_tile) where input and output must share a DST
    // index.
    auto graphResult = InterferenceGraph::buildIndexGraphFromDstOperations(
        region, dstAccesses);

    // Calculate number of available colors (DST slices).
    const int64_t volume = ttmlir::utils::volume(cbType.getShape());
    if (volume > static_cast<int64_t>(totalDstTiles)) {
      genericOp.emitError("CB volume exceeds available DST tiles");
      return;
    }
    const uint32_t numColors = totalDstTiles / volume;

    // Use the selected coloring strategy to assign DST slices.
    auto strategy = createColoringStrategy();
    std::vector<unsigned> coloring;
    if (failed(strategy->colorGraph(graphResult.adjacencyList, numColors,
                                    coloring))) {
      genericOp.emitError(
          "Graph coloring failed - not enough DST slices available");
      return;
    }

    TT_assertv(!coloring.empty(),
               "Graph coloring produced empty coloring result");

    // Apply coalescing constraints for in-place ops and normalize coloring
    // so that slice indices start from 0 and increment consecutively.
    const int64_t numDstSlicesNeeded =
        normalizeDstIndices(coloring, graphResult.coalescingPairs);
    SmallVector<int64_t> dstShape({numDstSlicesNeeded});
    dstShape.append(cbType.getShape().begin(), cbType.getShape().end());

    MemRefType dstType = MemRefType::get(
        dstShape, dstElementType,
        AffineMap::getMultiDimIdentityMap(dstShape.size(), &getContext()),
        ttcore::MemorySpaceAttr::get(&getContext(),
                                     ttcore::MemorySpace::RegisterDst));

    IRRewriter rewriter(&getContext());
    // Insert acquire_dst before the marked loop (like legacy pass).
    rewriter.setInsertionPoint(markedLoop);
    AcquireDstOp acquireDst =
        rewriter.create<AcquireDstOp>(genericOp.getLoc(), dstType);

    // Group DST accesses by their enclosing loop nest.
    // Use the marked loop as the common key for all accesses within it.
    CopyInfoMap copyInfoMap =
        groupAccessesByLoopNest(genericOp, dstAccesses, coloring, markedLoop);

    // Analyze each unique loop template once to get trip counts.
    // This avoids redundant analysis in prologue and epilogue loop
    // generation.
    llvm::DenseMap<Operation *, LoopContext> loopTemplateContexts;
    for (auto &[loopOp, copyInfo] : copyInfoMap) {
      loopTemplateContexts[loopOp] = analyzeLoopTemplate(loopOp);
    }

    // Generate hoisted prologue copy loops for each loop nest.
    for (auto &[loopOp, copyInfo] : copyInfoMap) {
      const LoopContext &loopTemplateCtx = loopTemplateContexts[loopOp];
      generatePrologueLoop(rewriter, acquireDst, loopOp, copyInfo,
                           loopTemplateCtx, maxLoopIterations);
    }

    for (auto &[loopOp, copyInfo] : copyInfoMap) {
      const LoopContext &loopTemplateCtx = loopTemplateContexts[loopOp];
      generateEpilogueLoop(rewriter, acquireDst.getResult(), loopOp, copyInfo,
                           loopTemplateCtx, maxLoopIterations);
    }


    // First pass: attach result_dst_index attributes to compute operations.
    // This enables the backend to retrieve DST indices without searching
    // for store operations (which may not exist with register reuse
    // optimization).
    for (size_t accessIndex = 0; accessIndex < dstAccesses.size();
         ++accessIndex) {
      auto &[op, origIdx] = dstAccesses[accessIndex];
      uint32_t assignedSlice = coloring[accessIndex];

      if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(op)) {
        // Store op: attach index to the defining compute op
        // Skip if the compute op already has the attribute (set when processing
        // the compute op entry in dstAccesses).
        if (Operation *defOp = storeOp.getValueToStore().getDefiningOp()) {
          if (mlir::isa<OperandLoadStoreRegisterOpInterface>(defOp) &&
              !defOp->hasAttr("result_dst_index")) {
            defOp->setAttr("result_dst_index",
                           rewriter.getI64IntegerAttr(assignedSlice));
          }
        }
      } else if (mlir::isa<OperandLoadStoreRegisterOpInterface>(op)) {
        // Intermediate compute op: attach index directly
        op->setAttr("result_dst_index",
                    rewriter.getI64IntegerAttr(assignedSlice));
      }
    }

    // Second pass: rewrite original loads/stores with DST accesses.
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

    // Third pass: materialize intermediate compute results to DST.
    // This ensures that compute operations whose results are consumed by other
    // compute operations (not stored to L1) have their intermediate values
    // properly materialized in DST registers via store+load pairs.
    materializeIntermediateComputeResults(rewriter, region,
                                          acquireDst.getResult());

    // Insert PackerMaskResetOp after reduce operations for correct
    // accumulation. This resets the packer mask on all but the last iteration.
    insertPackerMaskResetAfterReduce<TileReduceSumOp>(rewriter, genericOp);
    insertPackerMaskResetAfterReduce<TileReduceMaxOp>(rewriter, genericOp);
  }

  /// Materialize intermediate compute results to DST memory.
  ///
  /// When a compute operation's result is consumed by another compute operation
  /// (rather than stored to L1), the intermediate value must be explicitly
  /// materialized in DST registers. This function inserts store+load pairs to
  /// ensure intermediate results physically reside in DST hardware registers.
  ///
  /// \param rewriter The rewriter to use for IR modifications.
  /// \param region The region containing compute operations to process.
  /// \param dstBuffer The DST buffer (from acquire_dst) to store/load from.
  static void materializeIntermediateComputeResults(IRRewriter &rewriter,
                                                     Region &region,
                                                     Value dstBuffer) {
    auto dstMemRefType = llvm::cast<MemRefType>(dstBuffer.getType());
    Type dstElementType = dstMemRefType.getElementType();

    // Collect compute ops that need materialization.
    // We need to collect first, then modify, to avoid iterator invalidation.
    SmallVector<Operation *> opsToMaterialize;

    region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
      // Skip if this op has no results.
      if (computeOp->getNumResults() != 1) {
        return;
      }

      // Skip if result goes to an affine.store (already handled by DST
      // rewriting).
      bool hasStoreUser = llvm::any_of(computeOp->getUsers(), [](Operation *user) {
        return mlir::isa<affine::AffineStoreOp>(user);
      });
      if (hasStoreUser) {
        return;
      }

      // Check if consumed by another compute op (intermediate result).
      bool isIntermediate =
          llvm::any_of(computeOp->getUsers(), [](Operation *user) {
            return mlir::isa<OperandLoadStoreRegisterOpInterface>(user) &&
                   !mlir::isa<affine::AffineLoadOp, affine::AffineStoreOp>(
                       user);
          });

      if (isIntermediate) {
        opsToMaterialize.push_back(computeOp);
      }
    });

    // Now materialize each intermediate result.
    for (Operation *computeOp : opsToMaterialize) {
      // Get DST index assigned by graph coloring.
      auto dstIndexAttr =
          computeOp->getAttrOfType<IntegerAttr>("result_dst_index");

      if (!dstIndexAttr) {
        // Should not happen - intermediate ops should be identified during
        // identifyDstAccesses() and colored.
        llvm_unreachable("Intermediate compute op missing result_dst_index");
      }

      uint32_t assignedSlice = dstIndexAttr.getInt();
      Location loc = computeOp->getLoc();
      Value originalResult = computeOp->getResult(0);
      Type originalType = originalResult.getType();

      rewriter.setInsertionPointAfter(computeOp);

      // Build DST indices: [assignedSlice, loop_ivs...]
      SmallVector<Value> dstIndices;
      Value dstSliceIndex =
          rewriter.create<arith::ConstantIndexOp>(loc, assignedSlice);
      dstIndices.push_back(dstSliceIndex);

      // Find enclosing affine loops and add their induction variables.
      // collectLoopNest() already returns loops in outermost-to-innermost order.
      SmallVector<affine::AffineForOp> loops = collectLoopNest(computeOp);
      for (auto loop : loops) {
        dstIndices.push_back(loop.getInductionVar());
      }

      // Pad with zeros for remaining dimensions.
      while (dstIndices.size() < static_cast<size_t>(dstMemRefType.getRank())) {
        dstIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }

      auto dstMap = AffineMap::getMultiDimIdentityMap(dstIndices.size(),
                                                      rewriter.getContext());

      // Handle type casting if needed (e.g., f16 compute result but f32 DST).
      Value valueToStore = originalResult;
      Operation *castOp = nullptr;
      bool needsTypeCast = (originalType != dstElementType);

      if (needsTypeCast) {
        auto cast = rewriter.create<DstReinterpretCastOp>(loc, dstElementType,
                                                           valueToStore);
        valueToStore = cast.getResult();
        castOp = cast.getOperation();
      }

      // Create store to materialize the result in DST.
      auto storeOp = rewriter.create<affine::AffineStoreOp>(
          loc, valueToStore, dstBuffer, dstMap, dstIndices);

      // Create load to read the materialized result back.
      auto loadOp = rewriter.create<affine::AffineLoadOp>(loc, dstBuffer,
                                                          dstMap, dstIndices);

      Value replacementValue = loadOp.getResult();
      Operation *castBackOp = nullptr;

      // Cast back to original type if needed.
      if (needsTypeCast) {
        auto castBack = rewriter.create<DstReinterpretCastOp>(
            loc, originalType, replacementValue);
        replacementValue = castBack.getResult();
        castBackOp = castBack.getOperation();
      }

      // Replace all uses of the original result with the loaded value,
      // excluding the store and cast operations to avoid circular dependencies.
      rewriter.replaceUsesWithIf(
          originalResult, replacementValue, [&](mlir::OpOperand &operand) {
            Operation *owner = operand.getOwner();
            return owner != storeOp && owner != castOp && owner != castBackOp;
          });
    }
  }

  // Find the outermost affine.for loop that contains the given operation.
  static Operation *findOutermostLoop(Operation *op) {
    Operation *outermostLoop = nullptr;
    Operation *current = op->getParentOp();

    // Walk up the parent chain until we reach the region boundary
    while (current) {
      if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(current)) {
        outermostLoop = forOp;
      }
      // Stop if the parent is not in the same region.
      if (!current->getParentOp() ||
          current->getParentRegion() != op->getParentRegion()) {
        break;
      }
      current = current->getParentOp();
    }

    return outermostLoop;
  }

  // Look through memref.subview to find the original block argument.
  static BlockArgument lookThroughSubView(Value memref) {
    while (auto subView = mlir::dyn_cast_or_null<memref::SubViewOp>(
               memref.getDefiningOp())) {
      memref = subView.getSource();
    }
    if (auto *definingOp = memref.getDefiningOp();
        mlir::isa_and_nonnull<d2m::WaitOp, d2m::ReserveOp>(definingOp)) {
      memref = definingOp->getOperand(0);
    }
    return mlir::dyn_cast<BlockArgument>(memref);
  }

  // Group DST accesses by their enclosing loop nest.
  // Returns a map from outermost loop operation to CopyInfo containing
  // the loads and stores with their assigned DST slice indices.
  // The markedLoop parameter provides a common key for all accesses within
  // the same d2m.linalg_root loop, ensuring all prologues happen together
  // before compute and all epilogues happen together after.
  // Also computes guardIndices for reduction output loads (first-iteration
  // skip).
  static CopyInfoMap groupAccessesByLoopNest(
      GenericOp genericOp,
      const SmallVector<std::pair<Operation *, int64_t>> &dstAccesses,
      const std::vector<unsigned> &coloring, affine::AffineForOp markedLoop) {
    CopyInfoMap copyInfoMap;

    for (size_t i = 0; i < dstAccesses.size(); ++i) {
      auto &[op, origIdx] = dstAccesses[i];
      uint32_t assignedSlice = coloring[i];

      // Use the marked loop as the common key for all accesses.
      // This matches the legacy pass behavior where all accesses within
      // a d2m.linalg_root loop are grouped together.
      Operation *key = markedLoop.getOperation();

      // Add to the appropriate CopyInfo
      if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(op)) {
        copyInfoMap[key].push_back(loadOp, assignedSlice);

        // Compute guardIndices for reduction output loads.
        // For reductions, we skip copying from output CB to DST on first
        // iteration since the output CB contains uninitialized data.
        BlockArgument blockArg = lookThroughSubView(loadOp.getMemRef());
        if (blockArg && !genericOp.isExplicitDatamovementForm()) {
          SmallVector<int64_t> guardIndices =
              genericOp.getNonParticipatingLoopDims(blockArg.getArgNumber());
          if (!guardIndices.empty()) {
            // Merge guardIndices (they should be the same for all loads in
            // this group, but handle first-seen vs subsequent).
            if (copyInfoMap[key].guardIndices.empty()) {
              copyInfoMap[key].guardIndices = guardIndices;
            }
          }
        }
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
    // structure and terminators.
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

      // Recursively clear nested loops.
      for (auto nestedLoop : nestedLoops) {
        clearLoopBody(nestedLoop);
      }

      // Remove all operations except nested loops and the terminator.
      SmallVector<Operation *> toErase;
      for (Operation &op : body.getOperations()) {
        if (!isa<affine::AffineForOp>(&op) &&
            !op.hasTrait<OpTrait::IsTerminator>()) {
          toErase.push_back(&op);
        }
      }
      // Drop all uses before erasing to avoid use-def chain issues.
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

      // Look for a nested loop in the body.
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
  /// \param loopContexts Map from original operations to their loop contexts.
  /// \param maxLoopIterations Maximum loop iterations for stride computation.
  template <bool IsPrologue, typename OpT, typename PairT>
  static void generateCopyLoops(IRRewriter &rewriter, Value externalMemref,
                                Operation *loopTemplate,
                                ArrayRef<PairT> copyPairs,
                                const LoopContext &loopTemplateCtx,
                                int64_t maxLoopIterations) {
    if (copyPairs.empty()) {
      return;
    }

    auto forLoop = dyn_cast<affine::AffineForOp>(loopTemplate);
    if (!forLoop) {
      // No loop nest - generate inline copies.
      // In this case, loopTemplateCtx will be empty (no loops).
      OpBuilder::InsertionGuard guard(rewriter);
      if constexpr (IsPrologue) {
        rewriter.setInsertionPoint(loopTemplate);
      } else {
        rewriter.setInsertionPointAfter(loopTemplate);
      }

      for (const auto &pair : copyPairs) {
        OpT op = pair.first;
        int64_t sliceIdx = pair.second;

        SmallVector<Value> dstIndices(op.getIndices().begin(),
                                      op.getIndices().end());

        // Use constant DST slice index (matches legacy behavior).
        mlir::AffineExpr dstIndexExpr =
            mlir::getAffineConstantExpr(sliceIdx, rewriter.getContext());

        auto dstMap = op.getAffineMap().insertResult(dstIndexExpr, 0);

        if constexpr (IsPrologue) {
          // L1 -> DST: load from L1 (op.getMemRef()), store to DST.
          auto l1Value = rewriter.create<affine::AffineLoadOp>(
              op.getLoc(), op.getMemRef(), op.getMap(), op.getIndices());

          rewriter.create<affine::AffineStoreOp>(
              op.getLoc(), l1Value.getResult(), externalMemref, dstMap,
              dstIndices);
        } else {
          // DST -> L1: load from DST, store to L1 (op.getMemRef()).
          auto dstValue = rewriter.create<affine::AffineLoadOp>(
              op.getLoc(), externalMemref, dstMap, dstIndices);

          // Handle type mismatch: DST element type may differ from L1 type.
          Value valueToStore = dstValue.getResult();
          Type l1ElementType = op.getMemRefType().getElementType();
          if (valueToStore.getType() != l1ElementType) {
            valueToStore = rewriter.create<DstReinterpretCastOp>(
                op.getLoc(), l1ElementType, valueToStore);
          }

          rewriter.create<affine::AffineStoreOp>(op.getLoc(), valueToStore,
                                                 op.getMemRef(), op.getMap(),
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

    // Clone the loop structure and get the mapping.
    auto [copyLoop, mapper] = cloneLoopNest(rewriter, forLoop);

    // Find innermost loop.
    auto [innermostLoop, copyLoopIndices] =
        findInnermostLoopAndIndices(copyLoop);

    // Populate with copy operations at the innermost loop level.
    rewriter.setInsertionPointToStart(&innermostLoop.getRegion().front());

    for (const auto &pair : copyPairs) {
      OpT op = pair.first;
      int64_t sliceIdx = pair.second;

      // Map the original indices to the cloned loop's induction variables.
      SmallVector<Value> mappedIndices;
      for (Value idx : op.getIndices()) {
        Value mappedIdx = mapper.lookupOrDefault(idx);
        mappedIndices.push_back(mappedIdx);
      }

      // Always use constant DST slice index (matches legacy behavior).
      Value dstSliceIndex =
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), sliceIdx);

      // Build DST indices: [dstSliceIndex, mapped_cb_indices...]
      SmallVector<Value> dstIndices;
      dstIndices.push_back(dstSliceIndex);
      dstIndices.append(mappedIndices.begin(), mappedIndices.end());

      // Create identity affine map for DST access
      auto dstMap = AffineMap::getMultiDimIdentityMap(dstIndices.size(),
                                                      rewriter.getContext());

      if constexpr (IsPrologue) {
        // Create: %val = affine.load %l1[...]
        auto l1Load = rewriter.create<affine::AffineLoadOp>(
            op.getLoc(), op.getMemRef(), op.getMap(), mappedIndices);

        // Create: affine.store %val, %dst[dstSliceIndex, ...]
        rewriter.create<affine::AffineStoreOp>(op.getLoc(), l1Load.getResult(),
                                               externalMemref, dstMap,
                                               dstIndices);
      } else {
        // Create: %val = affine.load %dst[dstSliceIndex, ...]
        auto dstLoad = rewriter.create<affine::AffineLoadOp>(
            op.getLoc(), externalMemref, dstMap, dstIndices);

        // Handle type mismatch: DST element type may differ from L1 type.
        Value valueToStore = dstLoad.getResult();
        Type l1ElementType = op.getMemRefType().getElementType();
        if (valueToStore.getType() != l1ElementType) {
          valueToStore = rewriter.create<DstReinterpretCastOp>(
              op.getLoc(), l1ElementType, valueToStore);
        }

        // Create: affine.store %val, %l1[...]
        rewriter.create<affine::AffineStoreOp>(op.getLoc(), valueToStore,
                                               op.getMemRef(), op.getMap(),
                                               mappedIndices);
      }
    }
  }

  // Analyze a loop template to extract loop nest structure and trip counts.
  // Returns a LoopContext describing the template's loop nest.
  static LoopContext analyzeLoopTemplate(Operation *loopTemplate) {
    LoopContext ctx;

    // If loopTemplate is not a loop, it's a scalar case (no loops).
    auto forLoop = dyn_cast<affine::AffineForOp>(loopTemplate);
    if (!forLoop) {
      return ctx; // Empty context for scalar case.
    }

    // Traverse down from the root loop to collect the full loop nest.
    affine::AffineForOp currentLoop = forLoop;
    while (currentLoop) {
      ctx.loopNest.push_back(currentLoop);

      // Find nested loop in body.
      currentLoop = nullptr;
      Block &body = ctx.loopNest.back().getRegion().front();
      for (Operation &op : body.getOperations()) {
        if (auto nested = dyn_cast<affine::AffineForOp>(&op)) {
          currentLoop = nested;
          break;
        }
      }
    }

    // Compute trip counts from the loop nest.
    ctx.tripCounts = computeLoopTripCounts(ctx.loopNest);
    ctx.totalIterations =
        std::accumulate(ctx.tripCounts.begin(), ctx.tripCounts.end(), 1LL,
                        std::multiplies<int64_t>());

    return ctx;
  }

  /// Generate copy loops with an optional guard wrapper.
  /// If guard is non-null, the copy loops are inserted inside the guard's then
  /// region. Otherwise, they are inserted at the normal position relative to
  /// loopTemplate.
  template <bool IsPrologue, typename OpT, typename PairT>
  static void generateCopyLoopsWithOptionalGuard(
      IRRewriter &rewriter, Value externalMemref, Operation *loopTemplate,
      ArrayRef<PairT> copyPairs, const LoopContext &loopTemplateCtx,
      int64_t maxLoopIterations, scf::IfOp guard) {
    if (copyPairs.empty()) {
      return;
    }

    if (guard) {
      // Insert copy loops inside the guard's then region.
      rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
    } else {
      // No guard: set insertion point relative to loopTemplate.
      if constexpr (IsPrologue) {
        rewriter.setInsertionPoint(loopTemplate);
      } else {
        rewriter.setInsertionPointAfter(loopTemplate);
      }
    }

    auto forLoop = dyn_cast<affine::AffineForOp>(loopTemplate);
    if (!forLoop) {
      // No loop nest - generate inline copies.
      for (const auto &pair : copyPairs) {
        OpT op = pair.first;
        int64_t sliceIdx = pair.second;

        SmallVector<Value> dstIndices(op.getIndices().begin(),
                                      op.getIndices().end());

        mlir::AffineExpr dstIndexExpr =
            mlir::getAffineConstantExpr(sliceIdx, rewriter.getContext());

        auto dstMap = op.getAffineMap().insertResult(dstIndexExpr, 0);

        if constexpr (IsPrologue) {
          auto l1Value = rewriter.create<affine::AffineLoadOp>(
              op.getLoc(), op.getMemRef(), op.getMap(), op.getIndices());

          rewriter.create<affine::AffineStoreOp>(
              op.getLoc(), l1Value.getResult(), externalMemref, dstMap,
              dstIndices);
        } else {
          auto dstValue = rewriter.create<affine::AffineLoadOp>(
              op.getLoc(), externalMemref, dstMap, dstIndices);

          // Handle type mismatch: DST element type may differ from L1 type.
          Value valueToStore = dstValue.getResult();
          Type l1ElementType = op.getMemRefType().getElementType();
          if (valueToStore.getType() != l1ElementType) {
            valueToStore = rewriter.create<DstReinterpretCastOp>(
                op.getLoc(), l1ElementType, valueToStore);
          }

          rewriter.create<affine::AffineStoreOp>(op.getLoc(), valueToStore,
                                                 op.getMemRef(), op.getMap(),
                                                 op.getIndices());
        }
      }
      return;
    }

    // Clone the loop structure and get the mapping.
    auto [copyLoop, mapper] = cloneLoopNest(rewriter, forLoop);

    // Find innermost loop.
    auto [innermostLoop, copyLoopIndices] =
        findInnermostLoopAndIndices(copyLoop);

    // Populate with copy operations at the innermost loop level.
    rewriter.setInsertionPointToStart(&innermostLoop.getRegion().front());

    for (const auto &pair : copyPairs) {
      OpT op = pair.first;
      int64_t sliceIdx = pair.second;

      // Map the original indices to the cloned loop's induction variables.
      SmallVector<Value> mappedIndices;
      for (Value idx : op.getIndices()) {
        Value mappedIdx = mapper.lookupOrDefault(idx);
        mappedIndices.push_back(mappedIdx);
      }

      Value dstSliceIndex =
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), sliceIdx);

      SmallVector<Value> dstIndices;
      dstIndices.push_back(dstSliceIndex);
      dstIndices.append(mappedIndices.begin(), mappedIndices.end());

      auto dstMap = AffineMap::getMultiDimIdentityMap(dstIndices.size(),
                                                      rewriter.getContext());

      if constexpr (IsPrologue) {
        auto l1Load = rewriter.create<affine::AffineLoadOp>(
            op.getLoc(), op.getMemRef(), op.getMap(), mappedIndices);

        rewriter.create<affine::AffineStoreOp>(op.getLoc(), l1Load.getResult(),
                                               externalMemref, dstMap,
                                               dstIndices);
      } else {
        auto dstLoad = rewriter.create<affine::AffineLoadOp>(
            op.getLoc(), externalMemref, dstMap, dstIndices);

        // Handle type mismatch: DST element type may differ from L1 type.
        Value valueToStore = dstLoad.getResult();
        Type l1ElementType = op.getMemRefType().getElementType();
        if (valueToStore.getType() != l1ElementType) {
          valueToStore = rewriter.create<DstReinterpretCastOp>(
              op.getLoc(), l1ElementType, valueToStore);
        }

        rewriter.create<affine::AffineStoreOp>(op.getLoc(), valueToStore,
                                               op.getMemRef(), op.getMap(),
                                               mappedIndices);
      }
    }
  }

  // Generate prologue copy nest (L1 -> DST) ahead of the loop/compute region
  // that uses the original loads.
  // For reductions, wraps the copy in a guard that skips the first iteration.
  static void generatePrologueLoop(IRRewriter &rewriter,
                                   AcquireDstOp acquireDst,
                                   Operation *loopTemplate,
                                   const CopyInfo &copyInfo,
                                   const LoopContext &loopTemplateCtx,
                                   int64_t maxLoopIterations) {
    if (copyInfo.loads.empty()) {
      return;
    }

    // Set insertion point before the loop template for prologue operations.
    rewriter.setInsertionPoint(loopTemplate);

    // For reductions, insert a guard to skip copying on the first iteration.
    // The output CB contains uninitialized data on the first iteration.
    // We need to insert the guard first, then generate the copy loops inside
    // it.
    scf::IfOp guard = insertGuardForLoopNest(rewriter, loopTemplate->getLoc(),
                                             copyInfo.guardIndices);

    generateCopyLoopsWithOptionalGuard</*IsPrologue=*/true,
                                       affine::AffineLoadOp,
                                       OpAndIndexOffset<affine::AffineLoadOp>>(
        rewriter, acquireDst.getResult(), loopTemplate, copyInfo.loads,
        loopTemplateCtx, maxLoopIterations, guard);
  }

  // Generate epilogue copy nest (DST -> L1) after the loop/compute region that
  // produced the values.
  static void generateEpilogueLoop(IRRewriter &rewriter, Value dstMemref,
                                   Operation *loopTemplate,
                                   const CopyInfo &copyInfo,
                                   const LoopContext &loopTemplateCtx,
                                   int64_t maxLoopIterations) {
    generateCopyLoops</*IsPrologue=*/false, affine::AffineStoreOp,
                      OpAndIndexOffset<affine::AffineStoreOp>>(
        rewriter, dstMemref, loopTemplate, copyInfo.stores, loopTemplateCtx,
        maxLoopIterations);
  }

  /// Insert a guard that skips the first iteration of reduction loops.
  /// For reductions, we don't want to copy accumulated results from L1 to DST
  /// on the first iteration (when there's nothing to accumulate yet).
  /// Returns the created scf::IfOp, or nullptr if no guard is needed.
  static scf::IfOp insertGuardForLoopNest(IRRewriter &rewriter, Location loc,
                                          ArrayRef<int64_t> guardIndices) {
    if (guardIndices.empty()) {
      return nullptr;
    }
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    auto cmp = rewriter
                   .create<arith::ConstantOp>(loc, rewriter.getI1Type(),
                                              rewriter.getBoolAttr(false))
                   .getResult();
    for (int64_t index : guardIndices) {
      auto iterIndex = rewriter.create<IterIndexOp>(loc, index);
      auto ne = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                               iterIndex, zero);
      cmp = rewriter.create<arith::OrIOp>(loc, cmp, ne).getResult();
    }
    return rewriter.create<scf::IfOp>(loc, cmp);
  }

  /// Insert PackerMaskResetOp after reduce operations to reset the packer mask.
  /// This is needed for correct accumulation across reduction iterations.
  /// The reset is skipped on the last iteration.
  template <typename TileReduceOp>
  static void insertPackerMaskResetAfterReduce(IRRewriter &rewriter,
                                               GenericOp genericOp) {
    SmallVector<int64_t> loopBounds = genericOp.getLoopBounds();

    genericOp.walk([&](TileReduceOp op) {
      // Check if PackerMaskResetOp already exists in this block
      bool packerResetFound = false;
      op->getBlock()->walk([&](Operation *innerOp) {
        if (mlir::isa<PackerMaskResetOp>(innerOp)) {
          packerResetFound = true;
        }
      });
      if (packerResetFound) {
        return;
      }

      rewriter.setInsertionPointAfter(op);
      ReduceDim reduceDim = op.getReduceDim();

      scf::IfOp ifOp;
      auto index = [&](int64_t val) {
        return rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(val));
      };

      if (reduceDim == ReduceDim::R) {
        auto iterIndex =
            rewriter.create<IterIndexOp>(op.getLoc(), static_cast<int64_t>(1));
        auto condOp = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::ne, iterIndex,
            index(loopBounds[1] - 1));
        ifOp = rewriter.create<scf::IfOp>(op.getLoc(), condOp);
      } else if (reduceDim == ReduceDim::C) {
        auto iterIndex =
            rewriter.create<IterIndexOp>(op.getLoc(), static_cast<int64_t>(0));
        auto condOp = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::ne, iterIndex,
            index(loopBounds[0] - 1));
        ifOp = rewriter.create<scf::IfOp>(op.getLoc(), condOp);
      } else if (reduceDim == ReduceDim::RC) {
        auto iterIndexR =
            rewriter.create<IterIndexOp>(op.getLoc(), static_cast<int64_t>(1));
        auto iterIndexC =
            rewriter.create<IterIndexOp>(op.getLoc(), static_cast<int64_t>(0));
        auto condOp = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::ne, iterIndexR,
            index(loopBounds[1] - 1));
        auto condOp2 = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::ne, iterIndexC,
            index(loopBounds[0] - 1));
        auto finalCondOp =
            rewriter.create<arith::OrIOp>(op.getLoc(), condOp, condOp2);
        ifOp = rewriter.create<scf::IfOp>(op.getLoc(), finalCondOp);
      }
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      rewriter.create<PackerMaskResetOp>(op.getLoc());
    });
  }

  // Identify DST accesses that need coloring based on compute operations.
  // This uses the OperandLoadStoreRegisterOpInterface to find which
  // operands need to be loaded from DST registers, similar to
  // InsertDstRegisterAccess. Unlike that pass, we use graph coloring instead of
  // linear allocation.
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

      // Check if this compute op is part of a chain:
      // - It's consumed by another compute op (it's a producer in the chain), OR
      // - It consumes another compute op's result (it's a consumer in the chain)
      bool isPartOfChain = false;

      // Check if this op is consumed by another compute op.
      if (computeOp->getNumResults() == 1) {
        isPartOfChain =
            llvm::any_of(computeOp->getUsers(), [](Operation *user) {
              return mlir::isa<OperandLoadStoreRegisterOpInterface>(user) &&
                     !mlir::isa<affine::AffineLoadOp, affine::AffineStoreOp>(
                         user);
            });
      }

      // Check if this op consumes another compute op's result.
      if (!isPartOfChain) {
        for (Value operand : computeOp->getOperands()) {
          if (Operation *defOp = operand.getDefiningOp()) {
            if (mlir::isa<OperandLoadStoreRegisterOpInterface>(defOp) &&
                !mlir::isa<affine::AffineLoadOp, affine::AffineStoreOp>(defOp)) {
              isPartOfChain = true;
              break;
            }
          }
        }
      }

      // Add the compute op itself if it's part of a chain.
      // This allows chain detection to work even if the op also has a store user.
      if (isPartOfChain) {
        dstAccesses.emplace_back(computeOp, nextIndex++);
      }

      // Collect stores from this op's result.
      for (auto *user : computeOp->getUsers()) {
        if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(user);
            notDstMemspace(storeOp)) {
          dstAccesses.emplace_back(storeOp, nextIndex++);
        }
      }
    });

    return dstAccesses;
  }

  //===--------------------------------------------------------------------===//
  // Loop Context Analysis Helpers
  //===--------------------------------------------------------------------===//

  // Helper to find the position of a loop's induction variable in an array of
  // SSA values. The induction variable list comes from the affine load/store's
  // indices.
  static unsigned
  findInductionVarPosition(llvm::ArrayRef<mlir::Value> inductionVars,
                           mlir::affine::AffineForOp loop) {
    mlir::Value loopIV = loop.getInductionVar();
    for (unsigned i = 0; i < inductionVars.size(); ++i) {
      if (inductionVars[i] == loopIV) {
        return i;
      }
    }
    llvm_unreachable("Loop induction variable not found in affine map indices");
  }

  // Build affine expression for DST index with loop-dependent offset.
  // Returns: baseSlice * loopIterationStride + linearizedLoopOffset
  // Produces contiguous indices within each base slice's iteration block.
  static mlir::AffineExpr
  buildDstIndexExpr(unsigned baseSlice, int64_t loopIterationStride,
                    const LoopContext &loopCtx,
                    llvm::ArrayRef<mlir::Value> inductionVars,
                    mlir::MLIRContext *ctx) {

    if (loopCtx.loopNest.empty()) {
      // No loops: use constant base slice.
      return mlir::getAffineConstantExpr(baseSlice, ctx);
    }

    // Build linearization expression for loop offset.
    // Formula: i_n + i_{n-1}*tripCount_n + i_{n-2}*tripCount_n*tripCount_{n-1}
    // + ...
    mlir::AffineExpr offsetExpr = mlir::getAffineConstantExpr(0, ctx);
    mlir::AffineExpr strideExpr = mlir::getAffineConstantExpr(1, ctx);

    // Process from innermost to outermost
    for (int i = loopCtx.loopNest.size() - 1; i >= 0; i--) {
      // Find position of this loop's induction variable in inductionVars
      unsigned dimPos =
          findInductionVarPosition(inductionVars, loopCtx.loopNest[i]);
      mlir::AffineExpr dimExpr = mlir::getAffineDimExpr(dimPos, ctx);

      // offset += inductionVar * stride
      offsetExpr = offsetExpr + dimExpr * strideExpr;

      // stride *= tripCount (accumulates for outer dimensions).
      strideExpr =
          strideExpr * mlir::getAffineConstantExpr(loopCtx.tripCounts[i], ctx);
    }

    // Final formula: baseSlice * loopIterationStride + linearizedOffset
    // This gives contiguous blocks: [0..stride-1] for base 0,
    // [stride..2*stride-1] for base 1, etc.
    mlir::AffineExpr baseOffset =
        mlir::getAffineConstantExpr(baseSlice, ctx) *
        mlir::getAffineConstantExpr(loopIterationStride, ctx);

    return baseOffset + offsetExpr;
  }

  // Collect all containing affine.for loops from innermost to outermost,
  // then reverse to outermost-to-innermost order.
  // Walks parent chain stopping at region boundary.
  static llvm::SmallVector<mlir::affine::AffineForOp, 4>
  collectLoopNest(mlir::Operation *op) {
    llvm::SmallVector<mlir::affine::AffineForOp, 4> loops;
    mlir::Operation *current = op->getParentOp();

    // Walk up the parent chain, collecting affine.for loops.
    // Stop when we reach the d2m.generic operation.
    while (current && !mlir::isa<GenericOp>(current)) {
      if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(current)) {
        loops.push_back(forOp);
      }
      current = current->getParentOp();
    }

    // Reverse to get outermost-to-innermost order.
    std::reverse(loops.begin(), loops.end());
    return loops;
  }

  // Extract constant trip counts from loop bounds using getConstantTripCount().
  // Returns empty vector if any loop has non-constant bounds.
  // Emits error and returns empty vector for dynamic bounds.
  static llvm::SmallVector<int64_t, 4>
  computeLoopTripCounts(llvm::ArrayRef<mlir::affine::AffineForOp> loops) {
    llvm::SmallVector<int64_t, 4> tripCounts;
    tripCounts.reserve(loops.size());

    for (auto loop : loops) {
      std::optional<uint64_t> tripCount =
          mlir::affine::getConstantTripCount(loop);
      if (!tripCount.has_value()) {
        loop.emitError("DST register allocation requires constant loop bounds, "
                       "but encountered dynamic bounds");
        return {};
      }
      tripCounts.push_back(static_cast<int64_t>(tripCount.value()));
    }

    return tripCounts;
  }

  // Build map from each DST access operation to its loop context.
  // Analyzes containment and computes trip counts for all operations.
  static llvm::DenseMap<mlir::Operation *, LoopContext> analyzeLoopContexts(
      const llvm::SmallVector<std::pair<mlir::Operation *, int64_t>>
          &dstAccesses) {
    llvm::DenseMap<mlir::Operation *, LoopContext> contexts;

    for (const auto &[op, idx] : dstAccesses) {
      LoopContext ctx;
      ctx.loopNest = collectLoopNest(op);
      ctx.tripCounts = computeLoopTripCounts(ctx.loopNest);

      // Compute total iterations as product of trip counts.
      ctx.totalIterations = 1;
      for (int64_t count : ctx.tripCounts) {
        ctx.totalIterations *= count;
      }

      contexts[op] = ctx;
    }

    return contexts;
  }
};

} // namespace
} // namespace mlir::tt::d2m
