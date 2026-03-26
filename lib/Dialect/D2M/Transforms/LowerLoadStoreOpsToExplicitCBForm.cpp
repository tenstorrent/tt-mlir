// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERLOADSTOREOPSTOEXPLICITCBFORM
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Helper function to check if an operand is remote (i.e., implies data
// movement). This includes view ops (view_layout) and buffers with
// CBLayoutAttr (streaming circular buffers hoisted from the generic).
static bool isRemoteOperand(Value operand, Operation *op) {
  Operation *defOp = operand.getDefiningOp();
  if (!defOp) {
    return false;
  }
  if (mlir::isa<ViewOpInterface>(defOp)) {
    return true;
  }
  // A buffer with CBLayoutAttr is a streaming CB that requires real
  // data movement from an external shard.
  if (auto memrefType = mlir::dyn_cast<MemRefType>(operand.getType())) {
    if (mlir::isa<ttcore::CBLayoutAttr>(memrefType.getLayout())) {
      return true;
    }
  }
  return false;
}

// Helper function to find the ReserveOp that produces a given value,
// potentially through a chain of operations.
static ReserveOp findReserveOp(Value value) {
  if (!value) {
    return nullptr;
  }

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return nullptr;
  }

  // Direct case: value is directly produced by reserve
  if (auto reserveOp = mlir::dyn_cast<ReserveOp>(definingOp)) {
    return reserveOp;
  }

  // Trace through operations that might pass the buffer through
  for (Value operand : definingOp->getOperands()) {
    if (auto reserveOp = findReserveOp(operand)) {
      return reserveOp;
    }
  }

  return nullptr;
}

// Recognize and simplify the load-store idiom where a remote_load and
// remote_store share the same local buffer:
//   %buffer = memref.alloc()
//   %loaded = remote_load %buffer %input[indices]
//   %result = remote_store %output[indices] %buffer
// One of the operands is always a local CB, so this load-store pair can
// always be simplified to either a load or a store to a local CB, eliminating
// the other op and avoiding a redundant copy.
static void
simplifyLoadStorePairs(ModuleOp moduleOp, IRRewriter &rewriter, CBCache &cache,
                       PortCounter &portCounters) {
  SmallVector<std::pair<RemoteLoadOp, RemoteStoreOp>> loadStorePairsToSimplify;
  moduleOp->walk([&](GenericOp generic) {
    // Collect all candidate load-store pairs in the generic region
    generic->walk([&](RemoteStoreOp storeOp) {
      if (!storeOp.isImplicitForm()) {
        return;
      }
      Value localBuffer = storeOp.getLocalBuffer();
      if (!localBuffer) {
        return;
      }

      // Find a RemoteLoadOp that uses the same localBuffer
      RemoteLoadOp matchingLoadOp = nullptr;
      for (Operation *user : localBuffer.getUsers()) {
        if (auto loadOp = mlir::dyn_cast<RemoteLoadOp>(user)) {
          if (loadOp.isImplicitForm() &&
              loadOp.getLocalBuffer() == localBuffer) {
            matchingLoadOp = loadOp;
            break;
          }
        }
      }

      if (!matchingLoadOp) {
        return;
      }

      // Found a load-store pair sharing the same localBuffer
      loadStorePairsToSimplify.push_back({matchingLoadOp, storeOp});
    });
  });

  // Simplify each load-store pair
  for (auto [loadOp, storeOp] : loadStorePairsToSimplify) {
    Location loc = loadOp.getLoc();
    GenericOp generic = loadOp->getParentOfType<GenericOp>();
    if (!generic) {
      continue;
    }

    // Determine which operand is remote (has a view/stream layout)
    // In dma-only form, one operand typically has a view transformation
    Value loadMemref = loadOp.getMemref();
    Value storeMemref = storeOp.getMemref();

    // Find operand indices (verification guarantees these exist)
    auto opOperands = generic->getOpOperands();
    auto *loadOperandIt = llvm::find_if(opOperands, [&](OpOperand &opOperand) {
      return opOperand.get() == loadMemref;
    });
    auto *storeOperandIt = llvm::find_if(opOperands, [&](OpOperand &opOperand) {
      return opOperand.get() == storeMemref;
    });
    TT_assert((loadOperandIt != opOperands.end() &&
               storeOperandIt != opOperands.end()));
    TT_assert(generic.getNumRegions() == 1u);

    // Get CB values for the input and output operands.
    Value inputCB = findAssociatedCB(loadOp.getOperation(), loadMemref,
                                     rewriter, cache, portCounters);
    Value outputCB = findAssociatedCB(storeOp.getOperation(), storeMemref,
                                      rewriter, cache, portCounters);
    TT_assert((inputCB && outputCB));

    rewriter.setInsertionPoint(loadOp);

    // Case A: Load from remote (input has view layout) -> load into output CB
    // Case B: Store to remote (output has view layout) -> store from input CB
    bool isRemoteLoad = isRemoteOperand(loadMemref, loadOp.getOperation());
    bool isRemoteStore = isRemoteOperand(storeMemref, storeOp.getOperation());

    // When neither operand is remote (e.g., L1-to-L1 layout conversions in
    // TTNN mode where operands come from TTNNMetalLayoutCastOp rather than
    // ViewOpInterface), both the load and store are needed — skip
    // simplification.
    if (!isRemoteLoad && !isRemoteStore) {
      continue;
    }
    if (isRemoteLoad) {
      rewriter.create<RemoteLoadOp>(loc, loadMemref, loadOp.getIndices(),
                                    outputCB, loadOp.getMcastStartIndex(),
                                    loadOp.getMcastShape());
    } else {
      // aliased load - insert reserve and push here
      rewriter.create<ReserveOp>(loc, inputCB);
      rewriter.create<PushOp>(loc, inputCB);
    }

    if (isRemoteStore) {
      // if load is aliased local, store should use aliased input CB as local
      // buffer otherwise, store should use output CB (same as remote load)
      auto cb = (isRemoteLoad) ? outputCB : inputCB;
      rewriter.create<RemoteStoreOp>(
          loc, storeMemref, loadOp.getIndices(), cb, storeOp.getStartDevice(),
          storeOp.getDeviceMcastShape(), storeOp.getSemaphore(),
          storeOp.getSemaphoreIndices());
    } else {
      // aliased store - insert wait and pop here
      rewriter.create<WaitOp>(loc, outputCB);
      rewriter.create<PopOp>(loc, outputCB);
    }

    // Get the shared localBuffer before erasing operations
    Value localBuffer = loadOp.getLocalBuffer();
    memref::AllocOp allocToErase = nullptr;
    if (localBuffer) {
      allocToErase = mlir::dyn_cast_if_present<memref::AllocOp>(
          localBuffer.getDefiningOp());
    }

    // Erase the original load and store operations
    rewriter.eraseOp(storeOp);
    rewriter.eraseOp(loadOp);

    // Erase the shared alloc if it's now unused
    if (allocToErase && allocToErase->use_empty()) {
      rewriter.eraseOp(allocToErase);
    }
  }
}

static bool hasPopForWaitInBlock(WaitOp waitOp) {
  for (Operation &blockOp : *waitOp->getBlock()) {
    auto popOp = mlir::dyn_cast<PopOp>(blockOp);
    if (!popOp) {
      continue;
    }
    if (popOp.getCb() == waitOp.getCb()) {
      return true;
    }
  }
  return false;
}

static void ensurePopForWait(IRRewriter &rewriter, WaitOp waitOp) {
  if (hasPopForWaitInBlock(waitOp)) {
    return;
  }
  Block *waitBlock = waitOp->getBlock();
  if (waitBlock->mightHaveTerminator()) {
    rewriter.setInsertionPoint(waitBlock->getTerminator());
  } else {
    rewriter.setInsertionPointToEnd(waitBlock);
  }
  rewriter.create<PopOp>(waitOp.getLoc(), waitOp.getCb());
}

[[maybe_unused]] static LogicalResult verifyRemoteLoadLocalBufferDominance(ModuleOp moduleOp) {
  DominanceInfo dominanceInfo(moduleOp);
  LogicalResult result = success();

  moduleOp->walk([&](RemoteLoadOp remoteLoadOp) {
    if (failed(result) || !remoteLoadOp.isImplicitForm()) {
      return;
    }

    Value localBuffer = remoteLoadOp.getLocalBuffer();
    if (!localBuffer) {
      remoteLoadOp.emitOpError(
          "expected implicit remote_load to have a local buffer operand");
      result = failure();
      return;
    }

    Operation *remoteLoadOperation = remoteLoadOp.getOperation();
    for (Operation *user : localBuffer.getUsers()) {
      if (user == remoteLoadOperation) {
        continue;
      }
      // Ignore dealloc users, which are cleanup-only and do not consume the
      // loaded value semantics enforced by this dominance check.
      if (mlir::isa<memref::DeallocOp>(user)) {
        continue;
      }

      bool isDominated = false;
      if (user->getBlock() == remoteLoadOperation->getBlock()) {
        isDominated = remoteLoadOperation->isBeforeInBlock(user);
      } else {
        isDominated =
            dominanceInfo.properlyDominates(remoteLoadOperation, user);
      }

      if (isDominated) {
        continue;
      }

      remoteLoadOp.emitOpError(
          "requires local buffer users to be dominated by remote_load");
      user->emitError("user is not dominated by associated remote_load");
      result = failure();
      return;
    }
  });

  return result;
}

static LogicalResult verifyNoPreExistingExplicitRemoteOps(ModuleOp moduleOp) {
  LogicalResult result = success();

  moduleOp->walk([&](RemoteLoadOp remoteLoadOp) {
    if (!remoteLoadOp.isImplicitForm()) {
      remoteLoadOp.emitOpError("pre-existing explicit-form remote_load is "
                               "illegal for this pass input");
      result = failure();
    }
  });

  moduleOp->walk([&](RemoteStoreOp remoteStoreOp) {
    if (!remoteStoreOp.isImplicitForm()) {
      remoteStoreOp.emitOpError("pre-existing explicit-form remote_store is "
                                "illegal for this pass input");
      result = failure();
    }
  });

  return result;
}

static void rewriteImplicitRemoteLoadOpsToExplicitCBForm(
    ModuleOp moduleOp, IRRewriter &rewriter, CBCache &cache,
    PortCounter &portCounters) {
  // Transform implicit RemoteLoadOps to explicit CB form.
  SmallVector<RemoteLoadOp> remoteLoadsToConvert;
  moduleOp->walk([&](RemoteLoadOp remoteLoad) {
    if (!remoteLoad.isImplicitForm()) {
      return;
    }
    remoteLoadsToConvert.push_back(remoteLoad);
  });

  // Rewrite each collected implicit remote_load.
  for (RemoteLoadOp remoteLoad : remoteLoadsToConvert) {
    Location loc = remoteLoad.getLoc();
    Value memref = remoteLoad.getMemref();
    Value assocCb = findAssociatedCB(remoteLoad.getOperation(), memref,
                                     rewriter, cache, portCounters);

    if (!assocCb) {
      remoteLoad.emitWarning("could not find associated CB block argument, "
                             "skipping conversion");
      continue;
    }

    // Get the local buffer (destination) from implicit form remote_load
    // Downstream operations may reference this buffer directly
    Value localBuffer = remoteLoad.getLocalBuffer();
    TT_assert(localBuffer);
    // Only erase allocs that live inside the generic (local working buffers).
    // Hoisted CB allocs live outside as additionalArgs and must not be erased.
    GenericOp generic = remoteLoad->getParentOfType<GenericOp>();
    memref::AllocOp allocToErase = nullptr;
    if (auto allocOp = mlir::dyn_cast_if_present<memref::AllocOp>(
            localBuffer.getDefiningOp())) {
      if (generic && generic->isAncestor(allocOp)) {
        allocToErase = allocOp;
      }
    }

    rewriter.setInsertionPoint(remoteLoad);

    // Create explicit CB form remote_load (no localBuffer/result, has CB).
    rewriter.create<RemoteLoadOp>(loc, memref, remoteLoad.getIndices(), assocCb,
                                  remoteLoad.getMcastStartIndex(),
                                  remoteLoad.getMcastShape());

    // Create wait to produce the local memref value consumed by compute.
    auto waitOp = rewriter.create<WaitOp>(loc, assocCb);

    // Replace uses of the local buffer with the wait result.
    // Only replace uses inside the generic's regions — the local buffer may
    // be an additionalArg (hoisted CB alloc) whose operand on the generic op
    // itself must not be touched.
    rewriter.replaceUsesWithIf(
        localBuffer, waitOp.getResult(), [&](OpOperand &use) {
          return generic && generic->isProperAncestor(use.getOwner());
        });

    // Replace all uses of remote_load result with wait result
    if (remoteLoad.getResult()) {
      rewriter.replaceAllUsesWith(remoteLoad.getResult(), waitOp.getResult());
    }

    // Insert a d2m.pop at the end of the region to release the CB slot
    // acquired by the wait, signalling to producers that space is available.
    ensurePopForWait(rewriter, waitOp);

    // Erase the original remote_load operation
    rewriter.eraseOp(remoteLoad);

    if (allocToErase) {
      rewriter.eraseOp(allocToErase);
    }
  }
}

static void rewriteRemoteStoreOpsToExplicitCBForm(ModuleOp moduleOp,
                                                  IRRewriter &rewriter,
                                                  CBCache &cache,
                                                  PortCounter &portCounters) {
  // Rewrite RemoteStoreOps, converting implicit form to explicit CB form.
  SmallVector<RemoteStoreOp> remoteStoresToConvert;
  moduleOp->walk([&](RemoteStoreOp remoteStore) {
    remoteStoresToConvert.push_back(remoteStore);
  });

  // Rewrite each collected remote_store.
  for (RemoteStoreOp remoteStore : remoteStoresToConvert) {
    Location loc = remoteStore.getLoc();
    Value memref = remoteStore.getMemref();
    Value localBuffer = remoteStore.getLocalBuffer();
    Value assocCb;

    if (remoteStore.isImplicitForm()) {
      // Implicit form: find the CB by tracing back from the local buffer.
      // First try to find a ReserveOp in the chain.
      ReserveOp reserveOp = findReserveOp(localBuffer);
      if (reserveOp) {
        assocCb = reserveOp.getCb();
      } else if (auto waitOp = localBuffer.getDefiningOp<WaitOp>()) {
        // When a load-store pair was not simplified (e.g., L1-to-L1 layout
        // conversion), the load conversion replaced the shared alloc with a
        // WaitOp result. Extract the CB from the WaitOp.
        assocCb = waitOp.getCb();
      } else {
        // The localBuffer may be a hoisted additionalArg (external alloc).
        // Find the CB via the store's memref operand instead.
        assocCb = findAssociatedCB(remoteStore.getOperation(), memref, rewriter,
                                   cache, portCounters);
        if (!assocCb) {
          remoteStore.emitWarning(
              "could not find CB for local buffer, skipping conversion");
          continue;
        }
        // Convert local producer alloc to reserve so producer-side uses are
        // anchored to CB lifecycle and reserve naturally precedes first use.
        if (auto allocOp = mlir::dyn_cast_if_present<memref::AllocOp>(
                localBuffer.getDefiningOp())) {
          rewriter.setInsertionPoint(allocOp);
          auto newReserve =
              rewriter.create<ReserveOp>(allocOp.getLoc(), assocCb);
          rewriter.replaceAllUsesWith(allocOp.getResult(),
                                      newReserve.getResult());
          rewriter.eraseOp(allocOp);
          reserveOp = newReserve;
        }
      }

      rewriter.setInsertionPoint(remoteStore);

      rewriter.create<PushOp>(loc, assocCb);

      // Create the explicit CB form of remote_store (no local buffer, has CB)
      // d2m.remote_store %memref[indices] from %cb
      rewriter.create<RemoteStoreOp>(loc, memref, remoteStore.getIndices(),
                                     assocCb);

      // Erase the original remote_store operation
      rewriter.eraseOp(remoteStore);
    }
  }
}

// Rewrite remote ops to explicit CB form.
static void rewriteRemoteOpsToExplicitCB(ModuleOp moduleOp,
                                         IRRewriter &rewriter, CBCache &cache,
                                         PortCounter &portCounters) {
  simplifyLoadStorePairs(moduleOp, rewriter, cache, portCounters);

  rewriteImplicitRemoteLoadOpsToExplicitCBForm(moduleOp, rewriter, cache,
                                               portCounters);
  rewriteRemoteStoreOpsToExplicitCBForm(moduleOp, rewriter, cache,
                                        portCounters);
}

class D2MLowerLoadStoreOpsToExplicitCBForm
    : public impl::D2MLowerLoadStoreOpsToExplicitCBFormBase<
          D2MLowerLoadStoreOpsToExplicitCBForm> {
public:
  using impl::D2MLowerLoadStoreOpsToExplicitCBFormBase<
      D2MLowerLoadStoreOpsToExplicitCBForm>::
      D2MLowerLoadStoreOpsToExplicitCBFormBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());
    CBCache cbCache;
    PortCounter portCounters;

    if (failed(verifyNoPreExistingExplicitRemoteOps(moduleOp))) {
      signalPassFailure();
      return;
    }

    // TODO: disable for now; lots of false positives
    //if (failed(verifyRemoteLoadLocalBufferDominance(moduleOp))) {
    //  signalPassFailure();
    //  return;
    //}

    rewriteRemoteOpsToExplicitCB(moduleOp, rewriter, cbCache, portCounters);
  }
};
} // namespace

} // namespace mlir::tt::d2m
