// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/SCFOps.h.inc"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::tt::ttir;

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRLINALGLOOPFISSION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"
} // namespace mlir::tt::ttir

namespace {

// Recursively search for tilize/untilize ops.
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

// Identify if op is a TTIR compute op inside generic region
static bool isTileComputeOp(Operation *op) {
  return op->hasTrait<TTIRGenericRegionComputeOpTrait>();
}

// Find pattern affine.load -> tile_compute -> affine.store within a block.
struct LoadComputeStoreTriplet {
  affine::AffineLoadOp load;
  Operation *compute{nullptr};
  affine::AffineStoreOp store;
};

static std::optional<LoadComputeStoreTriplet> findTripletInBlock(Block &block) {
  affine::AffineLoadOp load = nullptr;
  Operation *compute = nullptr;
  affine::AffineStoreOp store = nullptr;

  for (Operation &op : block) {
    if (!load) {
      load = dyn_cast<affine::AffineLoadOp>(&op);
      if (load) {
        continue;
      }
    } else if (!compute) {
      // TODO
      // probably need to reset the status if what's after the load isn't a compute
      // and same if what's after compute isn't store
      if (isTileComputeOp(&op)) {
        compute = &op;
        continue;
      }
    //   // reset check, triplets need to be exactly in order
    //   load = nullptr;

    } else if (!store) {
      store = dyn_cast<affine::AffineStoreOp>(&op);
      if (store) {
        // Ensure def-use chains match: load result used by compute, compute result stored
        bool usesLoad = false;
        for (Value v : compute->getOperands()) {
          if (v == load.getResult()) {
            usesLoad = true;
            break;
          }
        }
        bool storesCompute = false;
        for (Value r : compute->getResults()) {
          if (r == store.getValue()) {
            storesCompute = true;
            break;
          }
        }
        if (usesLoad && storesCompute) {
          return LoadComputeStoreTriplet{load, compute, store};
        }
        // Reset and keep scanning for next possible triplet in order.
        load = nullptr;
        compute = nullptr;
        store = nullptr;
      }
    }
  }
  return std::nullopt;
}

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

// Given nested affine.for ... affine.for ... blocks, try triplet-based fission.
static bool fissionInnermostTriplet(affine::AffineForOp outerFor, RewriterBase &rewriter) {
  // Find the innermost loop to search for triplets.
  affine::AffineForOp innermost = findInnermostLoop(outerFor);

  llvm::errs() << "innermost " << innermost << "\n";

  auto maybeTriplet = findTripletInBlock(*innermost.getBody());
  if (!maybeTriplet) {
    return false;
  }
  auto triplet = *maybeTriplet;

  // Compute operation indices in the innermost block relative to the triplet.
  SmallVector<Operation *> opsInBlock;
  for (Operation &op : *innermost.getBody()) {
    opsInBlock.push_back(&op);
  }
  auto indexOf = [&](Operation *needle) -> int {
    for (int i = 0, e = static_cast<int>(opsInBlock.size()); i < e; ++i) {
      if (opsInBlock[i] == needle) return i;
    }
    return -1;
  };
  int loadIdx = indexOf(triplet.load.getOperation());
  int computeIdx = indexOf(triplet.compute);
  int storeIdx = indexOf(triplet.store.getOperation());
  if (loadIdx < 0 || computeIdx < 0 || storeIdx < 0 || !(loadIdx < computeIdx && computeIdx < storeIdx)) {
    return false;
  }

  llvm::errs() << "load " << triplet.load << " compute " << triplet.compute << " store " << triplet.store << "\n";

  OpBuilder::InsertionGuard g(rewriter);

  // Create three sibling loop nests: pre, middle, post.
  IRMapping map;
  rewriter.setInsertionPointAfter(outerFor);
  // auto preNest = cast<affine::AffineForOp>(rewriter.clone(*outerFor.getOperation(), map));
  // rewriter.setInsertionPointAfter(preNest);
  // map.clear();
  // auto midNest = cast<affine::AffineForOp>(rewriter.clone(*outerFor.getOperation(), map));
  // rewriter.setInsertionPointAfter(midNest);
  // map.clear();
  auto postNest = cast<affine::AffineForOp>(rewriter.clone(*outerFor.getOperation(), map));

  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> //
  SmallVector<Operation *> preOps ;
  for (Operation &op : *innermost.getBody()) {
    preOps.push_back(&op);
  }
  
  int count = preOps.size() - 1;

  // erase operations in a bottom up manner until
  // we reach the store op
  while (!preOps.empty()) {
    llvm::errs() << "PRE " << count << "\n";

    Operation *lastOp = preOps.pop_back_val();
    if (isa<affine::AffineYieldOp>(lastOp)) {
      llvm::errs() << ". . . PRE YIELD" << "\n";
      --count;
      continue;
    }
    if (count == storeIdx) {
      llvm::errs() << ". . . hit storeIdx" << "\n";
      break;
    }

    llvm::errs() << ". . . erasing op" << *lastOp << "\n";
    --count;
    rewriter.eraseOp(lastOp);
  }
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

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
    llvm::errs() << "POST " << count << "\n";

    if (count <= storeIdx) {
      llvm::errs() << ". . . erasing op" << *lastOp << "\n";
      rewriter.eraseOp(lastOp);
    }

    --count;

  }
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< // 

  return true;
}

struct TTIRLinAlgLoopFission
    : public tt::ttir::impl::TTIRLinAlgLoopFissionBase<TTIRLinAlgLoopFission> {
  using TTIRLinAlgLoopFissionBase::TTIRLinAlgLoopFissionBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    bool changed = false;
    module.walk([&](GenericOp gop) {
      llvm::errs() << "GOP " << gop << "\n";
      
      // Only compute-only region form.
      if (!gop.isComputeOnlyForm()) {
        return WalkResult::advance();
      }
      // Skip if region contains tilize/untilize
      if (containsTilizeOrUntilize(gop)) {
        return WalkResult::advance();
      }

      llvm::errs() << "    . . . checking loops . . .\n";

      // Find top-level nested affine.for in the compute region and try to fission.
      Block &computeBlock = gop.getRegion(0).front();
      llvm::errs() << "    . . . Compute Block " << computeBlock << "\n";

      scf::ForOp scfInnermost;

      for (Operation &op : computeBlock) {
        if (auto scfOuter = dyn_cast<scf::ForOp>(op)) {
            scfInnermost = findInnermostLoop(scfOuter);
        }
      }

      if (scfInnermost) {
        for (Operation &op : *scfInnermost.getBody()) {
            llvm::errs() << "    . . . . . . Op is: " << op << "\n";
            if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
                IRRewriter rewriter(ctx);
                rewriter.setInsertionPoint(forOp);
                llvm::errs() << "    . . . . . . forOp is: " << forOp << "\n";
                if (fissionInnermostTriplet(forOp, rewriter)) {
                    changed = true;
                    // Continue scanning; multiple triplets may exist.
                }
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


