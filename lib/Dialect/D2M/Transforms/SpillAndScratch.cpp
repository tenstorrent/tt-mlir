// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
  SmallVector<affine::AffineForOp> innerLoops; // All inner affine loops
  SmallVector<int64_t> scratchShape;           // Shape for scratch buffer
  int64_t scratchSpaceIterations = 0; // Iterations of scratch_space_loop
};

/// Information about an intermediate allocation that needs to become scratch
struct IntermediateAllocInfo {
  memref::AllocOp allocOp;
  ScratchLoopInfo *producer; // Loop nest that writes to this alloc
  ScratchLoopInfo *consumer; // Loop nest that reads from this alloc
  SmallVector<affine::AffineStoreOp> stores; // Stores to this alloc
  SmallVector<affine::AffineLoadOp> loads;   // Loads from this alloc
};

/// Find the innermost affine.for loop with d2m.linalg_root attribute
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

/// Collect inner affine loops from a linalg_root loop
static void collectInnerLoops(affine::AffineForOp rootLoop,
                              SmallVector<affine::AffineForOp> &innerLoops) {
  innerLoops.push_back(rootLoop);
  rootLoop.getBody()->walk(
      [&](affine::AffineForOp innerFor) { innerLoops.push_back(innerFor); });
}

/// Compute the scratch shape from loop bounds
/// Returns: [scratch_space_iterations, inner_loop_dim0, inner_loop_dim1, ...]
static bool computeScratchShape(ScratchLoopInfo &info) {
  // Get scratch_space_loop iterations
  if (!info.scratchLoop.hasConstantBounds()) {
    return false;
  }
  int64_t lower = info.scratchLoop.getConstantLowerBound();
  int64_t upper = info.scratchLoop.getConstantUpperBound();
  int64_t step = info.scratchLoop.getStepAsInt();
  info.scratchSpaceIterations = (upper - lower) / step;
  info.scratchShape.push_back(info.scratchSpaceIterations);

  // Add inner loop bounds
  for (auto loopOp : info.innerLoops) {
    if (!loopOp.hasConstantBounds()) {
      return false;
    }
    int64_t loopUpper = loopOp.getConstantUpperBound();
    int64_t loopLower = loopOp.getConstantLowerBound();
    info.scratchShape.push_back(loopUpper - loopLower);
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

/// Get the loop indices for scratch access
/// Returns: [scratch_space_loop_iv, inner_loop_iv0, inner_loop_iv1, ...]
static SmallVector<Value> getScratchIndices(ScratchLoopInfo &info) {
  SmallVector<Value> indices;
  indices.push_back(info.scratchLoop.getInductionVar());
  for (auto loopOp : info.innerLoops) {
    indices.push_back(loopOp.getInductionVar());
  }
  return indices;
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
    // Step 1: Find all scratch_space_loop nests in this generic
    SmallVector<ScratchLoopInfo> scratchLoops;

    genericOp.walk([&](affine::AffineForOp forOp) {
      if (forOp->hasAttr("d2m.scratch_space_loop") &&
          !forOp->hasAttr("d2m.scratch_inserted")) {
        ScratchLoopInfo info;
        info.scratchLoop = forOp;
        info.linalgRoot = findLinalgRootLoop(forOp);

        if (info.linalgRoot) {
          collectInnerLoops(info.linalgRoot, info.innerLoops);
          if (computeScratchShape(info)) {
            scratchLoops.push_back(info);
          }
        }
      }
    });

    if (scratchLoops.size() < 2) {
      // Need at least 2 loop nests to have producer/consumer relationship
      return success();
    }

    // Step 2: Find intermediate allocations that connect the loop nests
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

      // Find which loop nests write to and read from this allocation
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

      // Only include if we found both producer and consumer
      if (allocInfo.producer && allocInfo.consumer &&
          allocInfo.producer != allocInfo.consumer) {
        intermediates.push_back(allocInfo);
      }
    });

    if (intermediates.empty()) {
      // Mark loops as processed even if no intermediates found
      for (auto &info : scratchLoops) {
        info.scratchLoop->setAttr("d2m.scratch_inserted",
                                  UnitAttr::get(&getContext()));
      }
      return success();
    }

    // Step 3: Replace intermediate allocations with scratch buffers
    IRRewriter rewriter(&getContext());
    auto l1Attr = ttcore::MemorySpaceAttr::get(&getContext(),
                                               ttcore::MemorySpace::DeviceL1);

    for (auto &allocInfo : intermediates) {
      ScratchLoopInfo &producer = *allocInfo.producer;

      // Get element type from the original allocation
      MemRefType origType = allocInfo.allocOp.getType();
      Type elementType = origType.getElementType();

      // Create scratch buffer type based on loop structure
      // Shape: [scratch_space_iterations, inner_dim0, inner_dim1, ...]
      auto scratchMemRefType = MemRefType::get(
          producer.scratchShape, elementType, AffineMap(), l1Attr);

      int64_t totalTiles = 1;
      for (int64_t dim : producer.scratchShape) {
        totalTiles *= dim;
      }

      // Insert scratch allocation at the same location as the original alloc
      // This ensures proper dominance for both producer and consumer
      rewriter.setInsertionPoint(allocInfo.allocOp);
      Location loc = allocInfo.allocOp.getLoc();

      auto scratchOp = rewriter.create<ScratchAllocateOp>(
          loc, scratchMemRefType, totalTiles);
      Value scratchBuf = scratchOp.getResult();

      // Step 4: Update stores in the producer loop
      // Change from storing to subview of full allocation to storing to scratch
      // with scratch_space_loop iterator included
      SmallVector<Value> producerIndices = getScratchIndices(producer);

      for (auto storeOp : allocInfo.stores) {
        rewriter.setInsertionPoint(storeOp);

        // Get the value being stored
        Value storedValue = storeOp.getValue();

        // Create new store to scratch buffer with full indices
        rewriter.create<affine::AffineStoreOp>(storeOp.getLoc(), storedValue,
                                               scratchBuf, producerIndices);

        // Erase the old store
        rewriter.eraseOp(storeOp);
      }

      // Step 5: Update loads in the consumer loop
      ScratchLoopInfo &consumer = *allocInfo.consumer;
      SmallVector<Value> consumerIndices = getScratchIndices(consumer);

      for (auto loadOp : allocInfo.loads) {
        rewriter.setInsertionPoint(loadOp);

        // Create new load from scratch buffer with full indices
        Value reloaded = rewriter.create<affine::AffineLoadOp>(
            loadOp.getLoc(), scratchBuf, consumerIndices);

        // Replace all uses of the old load with the new one
        rewriter.replaceOp(loadOp, reloaded);
      }

      // Step 6: Clean up - remove subviews and old allocation if no longer used
      // First, collect all subviews of this allocation
      SmallVector<memref::SubViewOp> subviews;
      for (Operation *user : allocInfo.allocOp.getResult().getUsers()) {
        if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
          subviews.push_back(subview);
        }
      }

      // Erase subviews that have no remaining users
      for (auto subview : subviews) {
        if (subview.getResult().use_empty()) {
          rewriter.eraseOp(subview);
        }
      }

      // Erase the original allocation if it has no remaining users
      if (allocInfo.allocOp.getResult().use_empty()) {
        rewriter.eraseOp(allocInfo.allocOp);
      }
    }

    // Mark all scratch loops as processed
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
