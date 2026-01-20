// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MFPUSCRATCHFISSION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Check if an op is a compute or memory op we care about for fission
static bool isComputeOrMemoryOp(Operation *op) {
  return isa<affine::AffineLoadOp, affine::AffineStoreOp, TileAddOp, TileSubOp,
             TileMulOp, TileDivOp>(op);
}

/// Collect the backward slice of operations that contribute to a store.
/// Only includes ops within the same block as the store.
static void collectBackwardSlice(affine::AffineStoreOp store,
                                 DenseSet<Operation *> &slice) {
  SmallVector<Operation *> worklist;
  worklist.push_back(store);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    if (!slice.insert(op).second) {
      continue; // Already visited
    }

    for (Value operand : op->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        // Only include ops in the same block that are compute/memory ops
        if (defOp->getBlock() == store->getBlock() &&
            isComputeOrMemoryOp(defOp)) {
          worklist.push_back(defOp);
        }
      }
    }
  }
}

/// Find the innermost affine.for loop within an operation.
/// The innermost is defined as the one with no nested affine.for loops.
static affine::AffineForOp findInnermostAffineLoop(Operation *root) {
  affine::AffineForOp innermost = nullptr;
  root->walk([&](affine::AffineForOp forOp) {
    // Check if this loop has any nested affine.for loops
    bool hasNestedAffineFor = false;
    forOp.getBody()->walk([&](affine::AffineForOp nested) {
      if (nested != forOp) {
        hasNestedAffineFor = true;
      }
    });
    // If no nested affine.for, this is the innermost
    if (!hasNestedAffineFor) {
      innermost = forOp;
    }
  });
  return innermost;
}

/// Clone a loop and remove ops NOT in the keepOps set from the innermost body
static void cloneAndFilterLoop(scf::ForOp originalLoop,
                               const DenseSet<Operation *> &keepOps,
                               OpBuilder &builder) {
  // Clone the entire loop structure
  IRMapping mapping;
  Operation *clonedOp = builder.clone(*originalLoop.getOperation(), mapping);
  auto clonedLoop = cast<scf::ForOp>(clonedOp);

  // Find the innermost affine loop in both original and cloned
  affine::AffineForOp originalInnermost = findInnermostAffineLoop(originalLoop);
  affine::AffineForOp clonedInnermost = findInnermostAffineLoop(clonedLoop);

  if (!originalInnermost || !clonedInnermost) {
    return;
  }

  // Walk both bodies in parallel and mark ops for removal
  auto originalIt = originalInnermost.getBody()->begin();
  auto clonedIt = clonedInnermost.getBody()->begin();
  auto originalEnd = originalInnermost.getBody()->end();
  auto clonedEnd = clonedInnermost.getBody()->end();

  SmallVector<Operation *> toRemove;

  while (originalIt != originalEnd && clonedIt != clonedEnd) {
    Operation *originalOp = &*originalIt;
    Operation *clonedOp = &*clonedIt;

    // Only consider compute/memory ops for removal
    if (isComputeOrMemoryOp(originalOp)) {
      if (!keepOps.contains(originalOp)) {
        toRemove.push_back(clonedOp);
      }
    }

    ++originalIt;
    ++clonedIt;
  }

  // Remove ops in reverse order (to handle use-def dependencies)
  for (auto it = toRemove.rbegin(); it != toRemove.rend(); ++it) {
    Operation *op = *it;
    // Drop uses first to break any remaining references
    for (Value result : op->getResults()) {
      result.dropAllUses();
    }
    op->erase();
  }

  // Mark as fissioned
  clonedLoop->setAttr("d2m.fissioned", builder.getUnitAttr());
  clonedLoop->setAttr("d2m.scratch_space_loop", builder.getUnitAttr());
  clonedLoop->removeAttr("d2m.scratch_inserted");
  clonedLoop->removeAttr("d2m.num_scratch_buffers");
}

class D2MFPUScratchFission
    : public impl::D2MFPUScratchFissionBase<D2MFPUScratchFission> {
public:
  using impl::D2MFPUScratchFissionBase<
      D2MFPUScratchFission>::D2MFPUScratchFissionBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // Collect all scf.for loops tagged for fission
    SmallVector<scf::ForOp> loopsToFission;
    module.walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr("d2m.scratch_space_loop") &&
          forOp->hasAttr("d2m.scratch_inserted") &&
          !forOp->hasAttr("d2m.fissioned")) {
        loopsToFission.push_back(forOp);
      }
    });

    for (scf::ForOp loop : loopsToFission) {
      // Find innermost affine loop to get stores
      affine::AffineForOp innermost = findInnermostAffineLoop(loop);
      if (!innermost) {
        llvm::errs() << "FPUScratchFission: No innermost affine loop found\n";
        continue;
      }

      // Collect all stores within the innermost loop
      SmallVector<affine::AffineStoreOp> stores;
      innermost.walk(
          [&](affine::AffineStoreOp store) { stores.push_back(store); });

      if (stores.size() <= 1) {
        // Nothing to fission - mark and continue
        loop->setAttr("d2m.fissioned",
                      OpBuilder(loop.getContext()).getUnitAttr());
        continue;
      }

      // For each store, compute backward slice and create fissioned loop
      OpBuilder builder(loop);

      for (affine::AffineStoreOp store : stores) {
        DenseSet<Operation *> slice;
        collectBackwardSlice(store, slice);
        cloneAndFilterLoop(loop, slice, builder);
      }

      // Erase original loop
      loop.erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
