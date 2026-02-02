// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::tt::d2m;

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSFPUTILELOOPFISSION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"
} // namespace mlir::tt::d2m

namespace {

/// Recursively search for tilize/untilize ops.
static bool containsTilizeOrUntilize(Operation *op) {
  if (isa<TileTilizeBlockOp, TileUntilizeBlockOp>(op)) {
    return true;
  }
  for (Region &region : op->getRegions()) {
    for (Operation &nestedOp : region.getOps()) {
      if (containsTilizeOrUntilize(&nestedOp)) {
        return true;
      }
    }
  }
  return false;
}

/// Recursively search for Generic Compute Ops.
static bool containsD2MGenericComputeOp(Operation *op) {
  if (op->hasTrait<D2MGenericRegionComputeOpTrait>()) {
    return true;
  }

  for (Region &region : op->getRegions()) {
    for (Operation &nestedOp : region.getOps()) {
      if (containsD2MGenericComputeOp(&nestedOp)) {
        return true;
      }
    }
  }
  return false;
}

/// Stores pattern affine.load -> tile_compute -> affine.store within a block.
struct LoadComputeStoreTriplet {
  affine::AffineLoadOp load;
  Operation *compute{nullptr};
  affine::AffineStoreOp store;
};

// Helper to find the innermost block within a cloned nest matching 'innermost'.
template <class LoopType>
static LoopType findInnermostLoop(LoopType root) {
  LoopType cur = root;
  while (true) {
    LoopType next = nullptr;
    for (Operation &op : cur.getBody()->getOperations()) {
      if (auto child = dyn_cast<LoopType>(&op)) {
        next = child;
        break;
      }
    }
    if (!next) {
      break;
    }

    cur = next;
  }
  return cur;
};

static int insertLoadOps(affine::AffineForOp outerFor, RewriterBase &rewriter) {
  // helper to track umber of inserted loads.
  int numInserted = 0;

  // Find the innermost loop.
  affine::AffineForOp innermost = findInnermostLoop(outerFor);

  for (auto &op : *innermost.getBody()) {
    if (op.hasTrait<D2MGenericRegionComputeOpTrait>()) {
      for (auto operand : op.getOperands()) {
        bool loadIsOk = false;

        Operation *loadOpRequired = operand.getDefiningOp();
        Operation *prevOp = op.getPrevNode();

        // Check to see if the required load exists between the compute op user
        // and after the previous store.
        while (prevOp && !isa<affine::AffineStoreOp>(prevOp)) {
          if (prevOp == loadOpRequired) {
            loadIsOk = true;
          }

          prevOp = prevOp->getPrevNode();
        }

        if (!loadIsOk) {
          rewriter.setInsertionPoint(&op);
          Operation *clonedOp = rewriter.clone(*operand.getDefiningOp());

          // Rewire the SSA result: replace the operand with the cloned op's
          // result
          Value clonedResult = clonedOp->getResult(0); // Assuming single result
          op.replaceUsesOfWith(operand, clonedResult);

          ++numInserted;
        }
      }
    }
  }
  return numInserted;
}

///
static bool fissionAtStore(affine::AffineForOp outerFor,
                           RewriterBase &rewriter) {
  // Find the innermost loop to search for triplets.
  affine::AffineForOp innermost = findInnermostLoop(outerFor);

  int storeIdx = -1;
  affine::AffineStoreOp store = nullptr;
  for (Operation &op : *innermost.getBody()) {
    store = dyn_cast<affine::AffineStoreOp>(&op);
    ++storeIdx;

    if (store) {
      break;
    }
  }

  if (!store || storeIdx < 0) {
    return false;
  }

  // Avoids creating an extra loop nest during the last call to this function,
  // where the loop nest is not populated with any ops as there's nothing after
  // the store op.
  if (isa<affine::AffineYieldOp>(store.getOperation()->getNextNode())) {
    return false;
  }

  OpBuilder::InsertionGuard g(rewriter);

  // Create a deep copy of the affine loop nest
  IRMapping map;
  rewriter.setInsertionPointAfter(outerFor);
  auto postNest =
      cast<affine::AffineForOp>(rewriter.clone(*outerFor.getOperation(), map));

  // Erase operations in original loop nest in a bottom up manner until
  // we reach the store op. Bottom up order guarantees we don't attempt
  // to erase ops that have results that still have existing users.
  SmallVector<Operation *> preOps;
  for (Operation &op : *innermost.getBody()) {
    preOps.push_back(&op);
  }

  int count = preOps.size() - 1;

  while (!preOps.empty()) {
    Operation *lastOp = preOps.pop_back_val();
    if (isa<affine::AffineYieldOp>(lastOp)) {
      --count;
      continue;
    }
    if (count == storeIdx) {
      break;
    }

    --count;
    rewriter.eraseOp(lastOp);
  }

  // Erase the triplet, within the cloned loop nest,
  // starting with the store and moving up.
  affine::AffineForOp postNestInnermost = findInnermostLoop(postNest);
  SmallVector<Operation *> postOps;
  for (Operation &op : *postNestInnermost.getBody()) {
    postOps.push_back(&op);
  }

  // Collect ops that will be kept (after the store).
  llvm::DenseSet<Operation *> keptOps;
  for (size_t i = storeIdx + 1; i < postOps.size(); ++i) {
    keptOps.insert(postOps[i]);
  }

  // Erase operations up to storeIdx, but skip any op that has users
  // in the kept set (e.g., scalar ops used by tile ops after the store).
  // When we keep an op, add it to keptOps so its dependencies are also kept.
  for (int i = storeIdx; i >= 0; --i) {
    Operation *op = postOps[i];

    // Check if any user of this op is in the kept set.
    bool hasKeptUser = llvm::any_of(op->getUsers(), [&](Operation *user) {
      return keptOps.contains(user);
    });

    if (hasKeptUser) {
      // This op is needed by kept ops, so keep it and mark it as kept
      // so its own dependencies will also be preserved.
      keptOps.insert(op);
    } else {
      rewriter.eraseOp(op);
    }
  }

  return true;
}

struct D2MSFPUTileLoopFission
    : public tt::d2m::impl::D2MSFPUTileLoopFissionBase<D2MSFPUTileLoopFission> {
  using D2MSFPUTileLoopFissionBase::D2MSFPUTileLoopFissionBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    bool changed = false;
    module.walk([&](GenericOp gop) {
      // Only compute-only region form.
      if (!gop.isComputeOnlyForm()) {
        return WalkResult::advance();
      }
      // Skip d2m.generic if it contains tilize/untilize
      if (containsTilizeOrUntilize(gop)) {
        return WalkResult::advance();
      }
      // Skip the book-ending load/store loops
      if (!containsD2MGenericComputeOp(gop)) {
        return WalkResult::advance();
      }

      // Find affine.for loops marked with d2m.linalg_root attribute.
      // This attribute is set by GenericTileComputeLoops and preserved by
      // InsertDstRegisterAccess.
      // Collect loops first to avoid iterator invalidation during modification.
      SmallVector<affine::AffineForOp> loopsToProcess;
      gop.walk([&](affine::AffineForOp forOp) {
        if (forOp->hasAttr("d2m.linalg_root")) {
          loopsToProcess.push_back(forOp);
        }
      });

      for (auto forOp : loopsToProcess) {
        IRRewriter rewriter(ctx);

        // Use a worklist to process all loops (original and newly created).
        // Each fission creates a new loop that may also need fissioning.
        SmallVector<affine::AffineForOp> worklist;
        worklist.push_back(forOp);

        while (!worklist.empty()) {
          auto currentLoop = worklist.pop_back_val();

          // Insert any missing loads for compute ops
          insertLoadOps(currentLoop, rewriter);

          // Fission at the first store
          if (fissionAtStore(currentLoop, rewriter)) {
            changed = true;

            // The fission creates a new loop after currentLoop.
            // Both the modified currentLoop and the new loop may need
            // further fissioning.
            if (auto nextOp = currentLoop->getNextNode()) {
              if (auto newLoop = dyn_cast<affine::AffineForOp>(nextOp)) {
                // Add the new loop to the worklist for processing
                worklist.push_back(newLoop);
              }
            }
            // Also re-add currentLoop in case it has more stores to fission
            worklist.push_back(currentLoop);
          }
        }
      }

      return WalkResult::advance();
    });

    if (!changed) {
      return;
    }
  }
};

} // namespace

// Factory is generated by TableGen.
