// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPILLANDSCRATCH
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Information about a loop nest with d2m.scratch_space_loop
struct ScratchLoopInfo {
  affine::AffineForOp scratchLoop; // The loop with d2m.scratch_space_loop attr
  affine::AffineForOp linalgRoot;  // The nested loop with d2m.linalg_root attr
  // All affine loops from scratch_space body to innermost (gap loops,
  // linalg_root, and inner loops), in top-down order. Does NOT include
  // scratch_space_loop itself.
  SmallVector<affine::AffineForOp> allLoops;
  SmallVector<int64_t> scratchShape; // Shape for scratch buffer
};

/// Information about an intermediate allocation that needs to become scratch
struct IntermediateAllocInfo {
  memref::AllocOp allocOp;
  ScratchLoopInfo *producer; // Loop nest that writes to this alloc
  ScratchLoopInfo *consumer; // Loop nest that reads from this alloc
  SmallVector<affine::AffineStoreOp> stores; // Stores to this alloc
  SmallVector<affine::AffineLoadOp> loads;   // Loads from this alloc
};

/// Convert an scf.for with constant bounds to affine.for in-place.
/// Returns the new affine.for, or nullptr if conversion is not possible.
static affine::AffineForOp convertScfForToAffineFor(scf::ForOp scfFor,
                                                    IRRewriter &rewriter) {
  auto lb = getConstantIntValue(scfFor.getLowerBound());
  auto ub = getConstantIntValue(scfFor.getUpperBound());
  auto step = getConstantIntValue(scfFor.getStep());
  if (!lb || !ub || !step) {
    return nullptr;
  }

  rewriter.setInsertionPoint(scfFor);
  auto affineFor =
      rewriter.create<affine::AffineForOp>(scfFor.getLoc(), *lb, *ub, *step);

  // Move the body from scf.for to affine.for.
  Block *scfBody = scfFor.getBody();
  Block *affineBody = affineFor.getBody();

  // Replace uses of scf IV with affine IV.
  scfBody->getArgument(0).replaceAllUsesWith(affineFor.getInductionVar());

  // Move operations (except terminator) from scf body to affine body.
  auto &scfOps = scfBody->getOperations();
  auto &affineOps = affineBody->getOperations();
  affineOps.splice(affineOps.begin(), scfOps, scfOps.begin(),
                   std::prev(scfOps.end()));

  // Erase the old scf.for.
  rewriter.eraseOp(scfFor);

  return affineFor;
}

/// Convert all scf.for loops inside a scratch_space_loop to affine.for.
/// This ensures all loop IVs can be used with affine.store/load operations.
static void convertGapScfLoopsToAffine(affine::AffineForOp scratchLoop,
                                       IRRewriter &rewriter) {
  SmallVector<scf::ForOp> scfLoops;
  scratchLoop.walk([&](scf::ForOp scfFor) { scfLoops.push_back(scfFor); });

  for (auto scfFor : scfLoops) {
    convertScfForToAffineFor(scfFor, rewriter);
  }
}

/// Find the innermost affine.for loop with d2m.linalg_root attribute.
static affine::AffineForOp findLinalgRootLoop(affine::AffineForOp scratchLoop) {
  affine::AffineForOp result = nullptr;
  scratchLoop.getBody()->walk([&](affine::AffineForOp forOp) {
    if (forOp->hasAttr("d2m.linalg_root")) {
      result = forOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

/// Collect ALL affine.for loops nested inside scratchLoop (excluding
/// scratchLoop itself). This captures gap loops between scratch_space_loop and
/// linalg_root, the linalg_root itself, and any inner loops, all in top-down
/// (pre-order) order. Pre-order is critical to ensure the shape dimensions
/// match the loop nesting: outermost loop -> first dimension.
static void collectAllInnerLoops(affine::AffineForOp scratchLoop,
                                 SmallVector<affine::AffineForOp> &allLoops) {
  scratchLoop.walk<WalkOrder::PreOrder>([&](affine::AffineForOp innerFor) {
    if (innerFor != scratchLoop) {
      allLoops.push_back(innerFor);
    }
  });
}

/// Compute the scratch shape from loop bounds.
/// Returns: [scratch_iters, gap_loop_iters..., root_iters, inner_iters...]
/// Each loop dimension contributes (upper - lower) / step iterations.
static bool computeScratchShape(ScratchLoopInfo &info) {
  // First dimension: scratch_space_loop iterations.
  if (!info.scratchLoop.hasConstantBounds()) {
    return false;
  }
  int64_t lower = info.scratchLoop.getConstantLowerBound();
  int64_t upper = info.scratchLoop.getConstantUpperBound();
  int64_t step = info.scratchLoop.getStepAsInt();
  info.scratchShape.push_back((upper - lower) / step);

  // Remaining dimensions: all inner loops (gap + linalg_root + inner).
  for (auto loopOp : info.allLoops) {
    if (!loopOp.hasConstantBounds()) {
      return false;
    }
    int64_t loopLower = loopOp.getConstantLowerBound();
    int64_t loopUpper = loopOp.getConstantUpperBound();
    int64_t loopStep = loopOp.getStepAsInt();
    info.scratchShape.push_back((loopUpper - loopLower) / loopStep);
  }

  return true;
}

/// Check if a memref is an intermediate allocation (not an input/output of the
/// generic op, but created inside the generic region)
static bool isIntermediateAlloc(memref::AllocOp allocOp, GenericOp genericOp) {
  // Check if alloc is inside the generic op's region
  if (!genericOp->isProperAncestor(allocOp)) {
    return false;
  }

  // Check that it's not one of the generic's operands
  Value allocResult = allocOp.getResult();
  for (Value operand : genericOp->getOperands()) {
    if (operand == allocResult) {
      return false;
    }
  }

  return true;
}

/// Check if a memref value traces back to the target allocation through
/// subviews
static bool tracesToAlloc(Value memref, Value targetAlloc) {
  Value current = memref;
  while (current) {
    if (current == targetAlloc) {
      return true;
    }
    if (auto subview = current.getDefiningOp<memref::SubViewOp>()) {
      current = subview.getSource();
    } else {
      break;
    }
  }
  return false;
}

/// Collect all affine stores to a given memref within a loop nest
static void collectStoresInLoop(Value memref, ScratchLoopInfo &loopInfo,
                                SmallVector<affine::AffineStoreOp> &stores) {
  loopInfo.scratchLoop.walk([&](affine::AffineStoreOp storeOp) {
    if (tracesToAlloc(storeOp.getMemref(), memref)) {
      stores.push_back(storeOp);
    }
  });
}

/// Collect all affine loads from a given memref within a loop nest
static void collectLoadsInLoop(Value memref, ScratchLoopInfo &loopInfo,
                               SmallVector<affine::AffineLoadOp> &loads) {
  loopInfo.scratchLoop.walk([&](affine::AffineLoadOp loadOp) {
    if (tracesToAlloc(loadOp.getMemref(), memref)) {
      loads.push_back(loadOp);
    }
  });
}

/// Build an affine map and operands for accessing the scratch buffer.
/// For loops with step > 1, the IV is normalized to give contiguous indices.
/// E.g., affine.for %i = 0 to 8 step 2 gives IV values 0,2,4,6 which are
/// normalized to 0,1,2,3 via (d) -> (d floordiv 2).
static std::pair<AffineMap, SmallVector<Value>>
getScratchAccessMapAndOperands(ScratchLoopInfo &info, MLIRContext *ctx) {
  SmallVector<AffineExpr> exprs;
  SmallVector<Value> operands;
  unsigned dimIdx = 0;

  auto addLoop = [&](affine::AffineForOp loop) {
    operands.push_back(loop.getInductionVar());
    AffineExpr dim = getAffineDimExpr(dimIdx++, ctx);
    int64_t lb = loop.getConstantLowerBound();
    int64_t step = loop.getStepAsInt();
    if (lb != 0) {
      dim = dim - lb;
    }
    if (step != 1) {
      dim = dim.floorDiv(step);
    }
    exprs.push_back(dim);
  };

  // First: scratch_space_loop.
  addLoop(info.scratchLoop);

  // Then: all inner loops (gap + linalg_root + inner).
  for (auto loop : info.allLoops) {
    addLoop(loop);
  }

  return {AffineMap::get(dimIdx, 0, exprs, ctx), operands};
}

/// Main pass implementation
class D2MSpillAndScratch
    : public impl::D2MSpillAndScratchBase<D2MSpillAndScratch> {
public:
  using impl::D2MSpillAndScratchBase<
      D2MSpillAndScratch>::D2MSpillAndScratchBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](GenericOp genericOp) {
      if (failed(processGenericOp(genericOp))) {
        // Continue processing other generics even if one fails
      }
    });
  }

private:
  LogicalResult processGenericOp(GenericOp genericOp) {
    IRRewriter rewriter(&getContext());

    // Step 0: Find all scratch_space_loops first (before any modifications).
    SmallVector<affine::AffineForOp> scratchSpaceLoops;
    genericOp.walk([&](affine::AffineForOp forOp) {
      if (forOp->hasAttr("d2m.scratch_space_loop") &&
          !forOp->hasAttr("d2m.scratch_inserted")) {
        scratchSpaceLoops.push_back(forOp);
      }
    });

    if (scratchSpaceLoops.empty()) {
      return success();
    }

    // Step 1: Convert any scf.for loops inside scratch_space_loops to
    // affine.for. This ensures all loop IVs can be used with affine
    // store/load ops for scratch access.
    for (auto scratchLoop : scratchSpaceLoops) {
      convertGapScfLoopsToAffine(scratchLoop, rewriter);
    }

    // Step 2: Now collect full loop info (after gap conversion).
    SmallVector<ScratchLoopInfo> scratchLoops;
    for (auto scratchLoop : scratchSpaceLoops) {
      ScratchLoopInfo info;
      info.scratchLoop = scratchLoop;
      info.linalgRoot = findLinalgRootLoop(scratchLoop);

      if (info.linalgRoot) {
        collectAllInnerLoops(scratchLoop, info.allLoops);
        if (computeScratchShape(info)) {
          scratchLoops.push_back(info);
        }
      }
    }

    if (scratchLoops.size() < 2) {
      // Need at least 2 loop nests to have producer/consumer relationship.
      return success();
    }

    // Step 3: Find intermediate allocations that connect the loop nests.
    SmallVector<IntermediateAllocInfo> intermediates;

    genericOp.walk([&](memref::AllocOp allocOp) {
      if (!isIntermediateAlloc(allocOp, genericOp)) {
        return;
      }

      IntermediateAllocInfo allocInfo;
      allocInfo.allocOp = allocOp;
      allocInfo.producer = nullptr;
      allocInfo.consumer = nullptr;

      Value allocResult = allocOp.getResult();

      // Find which loop nests write to and read from this allocation.
      for (auto &loopInfo : scratchLoops) {
        SmallVector<affine::AffineStoreOp> stores;
        collectStoresInLoop(allocResult, loopInfo, stores);
        if (!stores.empty()) {
          allocInfo.producer = &loopInfo;
          allocInfo.stores = stores;
        }

        SmallVector<affine::AffineLoadOp> loads;
        collectLoadsInLoop(allocResult, loopInfo, loads);
        if (!loads.empty()) {
          allocInfo.consumer = &loopInfo;
          allocInfo.loads = loads;
        }
      }

      // Only include if we found both producer and consumer.
      if (allocInfo.producer && allocInfo.consumer &&
          allocInfo.producer != allocInfo.consumer) {
        intermediates.push_back(allocInfo);
      }
    });

    if (intermediates.empty()) {
      // Mark loops as processed even if no intermediates found.
      for (auto &info : scratchLoops) {
        info.scratchLoop->setAttr("d2m.scratch_inserted",
                                  UnitAttr::get(&getContext()));
      }
      return success();
    }

    // Step 4: Replace intermediate allocations with scratch buffers.
    auto l1Attr = ttcore::MemorySpaceAttr::get(&getContext(),
                                               ttcore::MemorySpace::DeviceL1);

    for (auto &allocInfo : intermediates) {
      ScratchLoopInfo &producer = *allocInfo.producer;

      // Get element type from the original allocation.
      MemRefType origType = allocInfo.allocOp.getType();
      Type elementType = origType.getElementType();

      // Create scratch buffer type based on loop structure.
      // Shape: [scratch_iters, gap_iters..., root_iters, inner_iters...]
      auto scratchMemRefType = MemRefType::get(
          producer.scratchShape, elementType, AffineMap(), l1Attr);

      int64_t totalTiles = 1;
      for (int64_t dim : producer.scratchShape) {
        totalTiles *= dim;
      }

      // Insert scratch allocation at the same location as the original alloc.
      // This ensures proper dominance for both producer and consumer.
      rewriter.setInsertionPoint(allocInfo.allocOp);
      Location loc = allocInfo.allocOp.getLoc();

      auto scratchOp = rewriter.create<ScratchAllocateOp>(
          loc, scratchMemRefType, totalTiles);
      Value scratchBuf = scratchOp.getResult();

      // Step 5: Update stores in the producer loop.
      // Use an affine map to normalize loop IVs (handling step > 1).
      auto [producerMap, producerOperands] =
          getScratchAccessMapAndOperands(producer, &getContext());

      for (auto storeOp : allocInfo.stores) {
        rewriter.setInsertionPoint(storeOp);
        Value storedValue = storeOp.getValue();

        rewriter.create<affine::AffineStoreOp>(storeOp.getLoc(), storedValue,
                                               scratchBuf, producerMap,
                                               producerOperands);

        rewriter.eraseOp(storeOp);
      }

      // Step 6: Update loads in the consumer loop.
      ScratchLoopInfo &consumer = *allocInfo.consumer;
      auto [consumerMap, consumerOperands] =
          getScratchAccessMapAndOperands(consumer, &getContext());

      for (auto loadOp : allocInfo.loads) {
        rewriter.setInsertionPoint(loadOp);

        Value reloaded = rewriter.create<affine::AffineLoadOp>(
            loadOp.getLoc(), scratchBuf, consumerMap, consumerOperands);

        rewriter.replaceOp(loadOp, reloaded);
      }

      // Step 7: Clean up - remove subviews and old allocation if unused.
      SmallVector<memref::SubViewOp> subviews;
      for (Operation *user : allocInfo.allocOp.getResult().getUsers()) {
        if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
          subviews.push_back(subview);
        }
      }

      for (auto subview : subviews) {
        if (subview.getResult().use_empty()) {
          rewriter.eraseOp(subview);
        }
      }

      if (allocInfo.allocOp.getResult().use_empty()) {
        rewriter.eraseOp(allocInfo.allocOp);
      }
    }

    // Mark all scratch loops as processed.
    for (auto &info : scratchLoops) {
      info.scratchLoop->setAttr("d2m.scratch_inserted",
                                UnitAttr::get(&getContext()));
      info.scratchLoop->setAttr(
          "d2m.num_scratch_buffers",
          IntegerAttr::get(IntegerType::get(&getContext(), 64),
                           intermediates.size()));
    }

    return success();
  }
};

} // namespace

} // namespace mlir::tt::d2m
