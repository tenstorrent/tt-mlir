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

// Attribute name for storing pre-allocated semaphore indices on RemoteLoadOp.
constexpr StringRef kPreallocatedSemaphoresAttr = "preallocated_semaphores";

// Check if a RemoteLoadOp needs multicast semaphores.
// Returns true if the op has mcastStartIndex and mcastShape.
static bool needsMcastSemaphores(RemoteLoadOp remoteLoad) {
  return !remoteLoad.getMcastStartIndex().empty() &&
         !remoteLoad.getMcastShape().empty();
}

// Recursively collect all RemoteLoadOps that need multicast semaphores.
static void collectMcastRemoteLoads(Block *block,
                                    SmallVectorImpl<RemoteLoadOp> &mcastLoads) {
  for (Operation &op : block->getOperations()) {
    if (auto forOp = mlir::dyn_cast<scf::ForOp>(&op)) {
      collectMcastRemoteLoads(forOp.getBody(), mcastLoads);
      continue;
    }

    if (auto remoteLoad = mlir::dyn_cast<RemoteLoadOp>(&op)) {
      if (needsMcastSemaphores(remoteLoad)) {
        mcastLoads.push_back(remoteLoad);
      }
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
    // Skip if already processed (any RemoteLoadOp has the attribute).
    for (Region &region : generic->getRegions()) {
      if (region.empty()) {
        continue;
      }
      SmallVector<RemoteLoadOp> mcastLoads;
      collectMcastRemoteLoads(&region.front(), mcastLoads);
      for (RemoteLoadOp load : mcastLoads) {
        if (load->hasAttr(kPreallocatedSemaphoresAttr)) {
          return;
        }
      }
    }

    // Find all RemoteLoadOps that need multicast semaphores.
    SmallVector<RemoteLoadOp> allMcastLoads;
    for (Region &region : generic->getRegions()) {
      if (region.empty()) {
        continue;
      }
      collectMcastRemoteLoads(&region.front(), allMcastLoads);
    }

    if (allMcastLoads.empty()) {
      return;
    }

    Location loc = generic.getLoc();
    LocalSemaphoreType semType = rewriter.getType<LocalSemaphoreType>();

    unsigned ioSize = generic.getInputsAndOutputs().size();

    for (RemoteLoadOp load : allMcastLoads) {
      // Compute the position of sem0 in EnqueueProgramOp.getArgs().
      // D2MToTTMetal puts inputsAndOutputs first, then only non-MemRef
      // additionalArgs (semaphores). MemRef additionalArgs (hoisted CBs) go
      // into the separate cbs list and must not be counted here.
      unsigned argsIdx = ioSize;
      for (Value additionalArg : generic.getAdditionalArgs()) {
        if (!mlir::isa<MemRefType>(additionalArg.getType())) {
          argsIdx++;
        }
      }
      int64_t sem0AbsIdx = static_cast<int64_t>(argsIdx);
      int64_t sem1AbsIdx = sem0AbsIdx + 1;

      // Create 2 create_local_semaphore ops before the generic op.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(generic);
      auto sem0 = rewriter.create<CreateLocalSemaphoreOp>(loc, semType, 0u);
      auto sem1 = rewriter.create<CreateLocalSemaphoreOp>(loc, semType, 0u);

      // Add the semaphores to the generic's additionalArgs only.
      // They are accessed inside regions via d2m.get_global_operand, not block
      // args, matching the pattern used for global semaphores.
      generic.getAdditionalArgsMutable().append(sem0.getResult());
      generic.getAdditionalArgsMutable().append(sem1.getResult());

      load->setAttr(kPreallocatedSemaphoresAttr,
                    rewriter.getI64ArrayAttr({sem0AbsIdx, sem1AbsIdx}));
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
