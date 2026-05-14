// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MPREALLOCATEMCASTSEMAPHORES
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Attribute name for storing pre-allocated semaphore indices on a triggering
// op (RemoteLoadOp with mcast, or GatherCoreOp).
constexpr StringRef kPreallocatedSemaphoresAttr = "preallocated_semaphores";

// Check if a RemoteLoadOp needs multicast semaphores.
// Returns true if the op has mcastStartIndex and mcastShape.
static bool needsMcastSemaphores(RemoteLoadOp remoteLoad) {
  return !remoteLoad.getMcastStartIndex().empty() &&
         !remoteLoad.getMcastShape().empty();
}

// Returns true if this op needs a local semaphore pair preallocated by this
// pass. Currently: multicast RemoteLoadOp and any GatherCoreOp.
static bool needsLocalSemaphorePair(Operation *op) {
  if (auto remoteLoad = mlir::dyn_cast<RemoteLoadOp>(op)) {
    return needsMcastSemaphores(remoteLoad);
  }
  if (mlir::isa<GatherCoreOp>(op)) {
    return true;
  }
  return false;
}

// Recursively collect all ops that need a local semaphore pair. RemoteLoadOps
// with mcast and GatherCoreOps are processed uniformly: each gets two local
// semaphores appended to the parent generic's additionalArgs. The exact
// meaning of those two semaphores is op-specific (sender/receiver handshake
// for mcast loads, source/collector handshake for gathers), but the storage
// and indexing scheme is identical.
static void collectSemaphoreOps(Block *block,
                                SmallVectorImpl<Operation *> &semaphoreOps) {
  for (Operation &op : block->getOperations()) {
    if (auto forOp = mlir::dyn_cast<scf::ForOp>(&op)) {
      collectSemaphoreOps(forOp.getBody(), semaphoreOps);
      continue;
    }
    if (needsLocalSemaphorePair(&op)) {
      semaphoreOps.push_back(&op);
    }
  }
}

class D2MPreallocateMcastSemaphores
    : public impl::D2MPreallocateMcastSemaphoresBase<
          D2MPreallocateMcastSemaphores> {
public:
  using impl::D2MPreallocateMcastSemaphoresBase<
      D2MPreallocateMcastSemaphores>::D2MPreallocateMcastSemaphoresBase;

  void runOnOperation() final {
    IRRewriter rewriter(&getContext());
    getOperation()->walk(
        [&](GenericOp generic) { processGenericOp(rewriter, generic); });
  }

private:
  void processGenericOp(IRRewriter &rewriter, GenericOp generic) {
    // Skip if already processed (any matching op has the attribute).
    for (Region &region : generic->getRegions()) {
      if (region.empty()) {
        continue;
      }
      SmallVector<Operation *> semaphoreOps;
      collectSemaphoreOps(&region.front(), semaphoreOps);
      for (Operation *op : semaphoreOps) {
        if (op->hasAttr(kPreallocatedSemaphoresAttr)) {
          return;
        }
      }
    }

    // Collect all ops in this generic that need a semaphore pair.
    SmallVector<Operation *> allSemaphoreOps;
    for (Region &region : generic->getRegions()) {
      if (region.empty()) {
        continue;
      }
      collectSemaphoreOps(&region.front(), allSemaphoreOps);
    }

    if (allSemaphoreOps.empty()) {
      return;
    }

    Location loc = generic.getLoc();
    LocalSemaphoreType semType = rewriter.getType<LocalSemaphoreType>();

    for (Operation *op : allSemaphoreOps) {
      unsigned argsIdx = generic.getNumOperands();
      int64_t sem0AbsIdx = static_cast<int64_t>(argsIdx);
      int64_t sem1AbsIdx = sem0AbsIdx + 1;

      // Create 2 create_local_semaphore ops before the generic op.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(generic);
      auto sem0 = rewriter.create<CreateLocalSemaphoreOp>(loc, semType, 0u);
      auto sem1 = rewriter.create<CreateLocalSemaphoreOp>(loc, semType, 0u);

      // Add the semaphores to the generic's additionalArgs only.
      // They are accessed inside regions via d2m.get_arg, not block
      // args, matching the pattern used for global semaphores.
      generic.getAdditionalArgsMutable().append(sem0.getResult());
      generic.getAdditionalArgsMutable().append(sem1.getResult());

      op->setAttr(kPreallocatedSemaphoresAttr,
                  rewriter.getI64ArrayAttr({sem0AbsIdx, sem1AbsIdx}));
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
