// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/SCFOps.h.inc"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::tt::d2m;

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLINALGLOOPFISSION
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
template<class LoopType>
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

        // Check to see if the required load exists between the compute op user and
        // after the previous store.
        while (prevOp && !isa<affine::AffineStoreOp>(prevOp)) {
          if (prevOp == loadOpRequired) {
            loadIsOk = true;
          }

          prevOp = prevOp->getPrevNode();
        }

        if (!loadIsOk) {
          rewriter.setInsertionPoint(&op);
          Operation *clonedOp = rewriter.clone(*operand.getDefiningOp());
          
          // Rewire the SSA result: replace the operand with the cloned op's result
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
static bool fissionAtStore(affine::AffineForOp outerFor, RewriterBase &rewriter) {
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
  // where the loop nest is not populated with any ops as there's nothing after the
  // store op.
  if (isa<affine::AffineYieldOp>(store.getOperation()->getNextNode())) {
    return false;
  }

  OpBuilder::InsertionGuard g(rewriter);

  // Create a deep copy of the affine loop nest
  IRMapping map;
  rewriter.setInsertionPointAfter(outerFor);
  auto postNest = cast<affine::AffineForOp>(rewriter.clone(*outerFor.getOperation(), map));

  // Erase operations in original loop nest in a bottom up manner until
  // we reach the store op. Bottom up order guarantees we don't attempt
  // to erase ops that have results that have users.
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> //
  SmallVector<Operation *> preOps ;
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
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //


  // Erase the triplet, starting with the store and moving up, within the
  // cloned loop nest.
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> //
  affine::AffineForOp postNestInnermost = findInnermostLoop(postNest);
  SmallVector<Operation *> postOps ;
  for (Operation &op : *postNestInnermost.getBody()) {
    postOps.push_back(&op);
  }
  
  count = postOps.size() - 1;

  // erase operations in a bottom up manner until
  // we reach the store op
  while (!postOps.empty()) {
    Operation *lastOp = postOps.pop_back_val();

    if (count <= storeIdx) {
      rewriter.eraseOp(lastOp);
    }

    --count;

  }
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< // 

  return true;
}

//TODO final loop creates an empty nest because it has nothing after the store to copy
struct D2MLinAlgLoopFission
    : public tt::d2m::impl::D2MLinAlgLoopFissionBase<D2MLinAlgLoopFission> {
  using D2MLinAlgLoopFissionBase::D2MLinAlgLoopFissionBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    bool changed = false;
    module.walk([&](GenericOp gop) {
      
      // Only compute-only region form.
      if (!gop.isComputeOnlyForm()) {
        return WalkResult::advance();
      }
      // Skip if region contains tilize/untilize
      if (containsTilizeOrUntilize(gop)) {
        return WalkResult::advance();
      }
      // Skip the book-ending load/store loops
      if (!containsD2MGenericComputeOp(gop)) {
        return WalkResult::advance();
      }

      // Find top-level nested affine.for in the compute region and try to fission.
      // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> //
      Block &computeBlock = gop.getRegion(0).front();

      // Get the innermost scf.for loop first
      scf::ForOp scfInnermost;

      for (Operation &op : computeBlock) {
        if (auto scfOuter = dyn_cast<scf::ForOp>(op)) {
            scfInnermost = findInnermostLoop(scfOuter);
        }
      }

      if (scfInnermost) {
        for (Operation &op : *scfInnermost.getBody()) {
            if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
              if (containsD2MGenericComputeOp(forOp)) {
                IRRewriter rewriter(ctx);
                rewriter.setInsertionPoint(forOp);

                insertLoadOps(forOp, rewriter);

                if (fissionAtStore(forOp, rewriter)) {
                    changed = true;
                }
              }
            }
        }
      }
      // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //
      return WalkResult::advance();
    });

    if (!changed) {
      return;
    }
  }
};

} // namespace

// Factory is generated by TableGen.


