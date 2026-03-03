// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERLOADSTOREOPSTOEXPLICITCBFORM
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Helper function to check if an operand is remote (i.e., comes from a view op
// such as view_layout or stream_layout).
static bool isRemoteOperand(Value operand, Operation *op) {
  Operation *defOp = operand.getDefiningOp();
  if (!defOp) {
    return false;
  }
  // Remote operands are those that come from ops implementing ViewOpInterface
  return mlir::isa<ViewOpInterface>(defOp);
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
static void simplifyLoadStorePairs(ModuleOp moduleOp, IRRewriter &rewriter,
                                   CBCache &cache, PortCounter &portCounters) {
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

    TT_assert(!(isRemoteLoad && isRemoteStore));
    // When neither operand is remote (e.g., L1-to-L1 layout conversions in
    // TTNN mode where operands come from TTNNMetalLayoutCastOp rather than
    // ViewOpInterface), both the load and store are needed — skip
    // simplification.
    if (!isRemoteLoad && !isRemoteStore) {
      continue;
    }
    if (!isRemoteStore) {
      // Create the explicit CB form of remote_load (no localBuffer, has CB
      // operand)
      RemoteLoadOp::create(rewriter, loc, loadMemref, loadOp.getIndices(),
                           outputCB, loadOp.getMcastStartIndex(),
                           loadOp.getMcastShape());
    } else {
      RemoteStoreOp::create(rewriter, loc, storeMemref, loadOp.getIndices(),
                            inputCB);
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

// Structure to hold information needed for push/pop insertion
struct PushPopInfo {
  SmallVector<std::pair<Value, Location>> cbsNeedingPop;
  SmallVector<std::pair<ReserveOp, Value>> reserveOpsNeedingPush;
};

// Pass A: Convert all remote_load and remote_store into explicit CB form.
// Returns information needed for push/pop insertion.
static PushPopInfo convertToExplicitCBForm(ModuleOp moduleOp,
                                           IRRewriter &rewriter, CBCache &cache,
                                           PortCounter &portCounters) {
  PushPopInfo info;

  // Pre-process generics with load-store idiom
  simplifyLoadStorePairs(moduleOp, rewriter, cache, portCounters);

  // Transform RemoteLoadOp (implicit form -> explicit CB form)
  SmallVector<RemoteLoadOp> remoteLoadsToConvert;
  moduleOp->walk([&](RemoteLoadOp remoteLoad) {
    if (!remoteLoad.isImplicitForm()) {
      return;
    }
    remoteLoadsToConvert.push_back(remoteLoad);
  });

  // Transform each remote_load
  for (RemoteLoadOp remoteLoad : remoteLoadsToConvert) {
    Location loc = remoteLoad.getLoc();
    Value memref = remoteLoad.getMemref();
    Value assocCb = remoteLoad.isImplicitForm()
                        ? findAssociatedCB(remoteLoad.getOperation(), memref,
                                           rewriter, cache, portCounters)
                        : remoteLoad.getCb();

    if (!assocCb) {
      remoteLoad.emitWarning("could not find associated CB block argument, "
                             "skipping conversion");
      continue;
    }

    // Get the local buffer (destination) from implicit form remote_load
    // Downstream operations may reference this buffer directly
    Value localBuffer = remoteLoad.getLocalBuffer();
    // Only erase allocs that live inside the generic (local working buffers).
    // Hoisted CB allocs live outside as additionalArgs and must not be erased.
    GenericOp generic = remoteLoad->getParentOfType<GenericOp>();
    memref::AllocOp allocToErase = nullptr;
    if (localBuffer) {
      if (auto allocOp = mlir::dyn_cast_if_present<memref::AllocOp>(
              localBuffer.getDefiningOp())) {
        if (generic && generic->isAncestor(allocOp)) {
          allocToErase = allocOp;
        }
      }
    }

    rewriter.setInsertionPoint(remoteLoad);

    // Create the explicit CB form of remote_load (no localBuffer, no result,
    // has CB operand) d2m.remote_load %memref[indices] into %cb
    RemoteLoadOp::create(rewriter, loc, memref, remoteLoad.getIndices(),
                         assocCb, remoteLoad.getMcastStartIndex(),
                         remoteLoad.getMcastShape());

    // Create wait operation to produce the result value
    // %in = d2m.wait %cb
    auto waitOp = WaitOp::create(rewriter, loc, assocCb);

    // Move any operations that use localBuffer and come before remoteLoad
    // to after waitOp. This handles cases like collapse_shape ops that were
    // inserted between the alloc and remote_load by earlier passes (e.g.,
    // D2MGenericLinearizeMemref). Without this, replacing their operand with
    // waitOp.getResult() would create a dominance violation since they would
    // use a value defined after them.
    if (localBuffer) {
      SmallVector<Operation *> opsToMove;
      for (Operation *user : localBuffer.getUsers()) {
        if (user != remoteLoad.getOperation() &&
            user->getBlock() == remoteLoad->getBlock() &&
            user->isBeforeInBlock(remoteLoad.getOperation())) {
          opsToMove.push_back(user);
        }
      }
      // Sort by original order to maintain relative ordering when moved
      llvm::sort(opsToMove, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      // Move ops to after waitOp, maintaining their relative order
      Operation *insertAfter = waitOp.getOperation();
      for (Operation *op : opsToMove) {
        op->moveAfter(insertAfter);
        insertAfter = op;
      }
    }

    // Replace uses of the local buffer with the wait result.
    // Only replace uses inside the generic's regions — the local buffer may
    // be an additionalArg (hoisted CB alloc) whose operand on the generic op
    // itself must not be touched.
    if (localBuffer) {
      rewriter.replaceUsesWithIf(
          localBuffer, waitOp.getResult(), [&](OpOperand &use) {
            return generic && generic->isProperAncestor(use.getOwner());
          });
    }

    // Replace all uses of remote_load result with wait result
    if (remoteLoad.getResult()) {
      rewriter.replaceAllUsesWith(remoteLoad.getResult(), waitOp.getResult());
    }

    // Track CB that needs pop insertion (deferred to Pass B)
    info.cbsNeedingPop.push_back({assocCb, loc});

    // Erase the original remote_load operation
    rewriter.eraseOp(remoteLoad);

    if (allocToErase) {
      rewriter.eraseOp(allocToErase);
    }
  }

  // Convert memref.alloc -> ReserveOp for remote operands
  SmallVector<memref::AllocOp> allocsToConvert;
  moduleOp->walk([&](memref::AllocOp allocOp) {
    // Check if this alloc is inside a generic op
    GenericOp generic = allocOp->getParentOfType<GenericOp>();
    if (!generic) {
      return;
    }

    // Use findAssocOperand to determine the associated generic operand
    Value assocOperand = GenericOp::findAssocOperand(allocOp);
    if (!assocOperand) {
      return;
    }

    allocsToConvert.push_back(allocOp);
  });

  // Transform each memref.alloc to reserve
  for (memref::AllocOp allocOp : allocsToConvert) {
    Location loc = allocOp.getLoc();

    // Find the associated operand
    Value assocOperand = GenericOp::findAssocOperand(allocOp);
    if (!assocOperand) {
      allocOp.emitWarning(
          "could not determine associated operand, skipping conversion");
      continue;
    }

    // Find the CB for the associated operand
    Value assocCb = findAssociatedCB(allocOp.getOperation(), assocOperand,
                                     rewriter, cache, portCounters);
    if (!assocCb) {
      allocOp.emitWarning("could not find associated CB, skipping conversion");
      continue;
    }

    rewriter.setInsertionPoint(allocOp);

    // Create reserve operation
    // %out = d2m.reserve %cb
    auto reserveOp = ReserveOp::create(rewriter, loc, assocCb);

    // Replace all uses of memref.alloc result with reserve result
    rewriter.replaceAllUsesWith(allocOp.getResult(), reserveOp.getResult());

    // Track reserve op for push insertion (deferred to Pass B)
    info.reserveOpsNeedingPush.push_back({reserveOp, assocCb});

    // Erase the original memref.alloc operation
    rewriter.eraseOp(allocOp);
  }

  // Transform RemoteStoreOp: convert implicit form to explicit CB form, and
  // ensure explicit form stores (e.g., created by simplifyLoadStorePairs) get
  // reserve/push inserted for CB synchronization.
  SmallVector<RemoteStoreOp> remoteStoresToConvert;
  moduleOp->walk([&](RemoteStoreOp remoteStore) {
    remoteStoresToConvert.push_back(remoteStore);
  });

  // Track which CBs already have reserve ops to avoid duplicates
  llvm::DenseSet<Value> cbsWithReserveOps;
  for (auto &[reserveOp, cb] : info.reserveOpsNeedingPush) {
    cbsWithReserveOps.insert(cb);
  }

  // Transform each remote_store
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
      }

      rewriter.setInsertionPoint(remoteStore);

      // Create the explicit CB form of remote_store (no local buffer, has CB)
      // d2m.remote_store %memref[indices] from %cb
      RemoteStoreOp::create(rewriter, loc, memref, remoteStore.getIndices(),
                            assocCb);

      // Track the reserve op for push insertion (avoid duplicates).
      if (reserveOp && cbsWithReserveOps.insert(assocCb).second) {
        info.reserveOpsNeedingPush.push_back({reserveOp, assocCb});
      } else if (!reserveOp && cbsWithReserveOps.insert(assocCb).second) {
        // No existing reserve (e.g. hoisted additionalArg output).
        // Create right after the get_cb so it dominates all uses.
        OpBuilder::InsertionGuard reserveGuard(rewriter);
        rewriter.setInsertionPointAfterValue(assocCb);
        auto newReserve = rewriter.create<ReserveOp>(loc, assocCb);
        info.reserveOpsNeedingPush.push_back({newReserve, assocCb});
        // Replace uses of the old localBuffer with the reserve result
        // inside the generic only.
        if (localBuffer) {
          GenericOp storeGeneric = remoteStore->getParentOfType<GenericOp>();
          rewriter.replaceUsesWithIf(
              localBuffer, newReserve.getResult(), [&](OpOperand &use) {
                return storeGeneric &&
                       storeGeneric->isProperAncestor(use.getOwner());
              });
        }
      }

      // Erase the original remote_store operation
      rewriter.eraseOp(remoteStore);
    } else {
      // Explicit form: store already uses CB directly, but we need reserve/push
      assocCb = remoteStore.getCb();
      if (!assocCb) {
        remoteStore.emitWarning(
            "explicit form remote_store has no CB operand, skipping");
        continue;
      }

      // Create reserve op before the store (avoid duplicates)
      if (cbsWithReserveOps.insert(assocCb).second) {
        rewriter.setInsertionPoint(remoteStore);
        auto reserveOp = ReserveOp::create(rewriter, loc, assocCb);
        info.reserveOpsNeedingPush.push_back({reserveOp, assocCb});
      }
    }
  }

  return info;
}

// Pass B: Insert push and pop operations
static void insertPushAndPopOps(ModuleOp moduleOp, IRRewriter &rewriter,
                                PushPopInfo &info) {
  // Insert pop ops for remote_load conversions.
  // Insert the pop in the same block as the corresponding WaitOp so that
  // D2MCBOpRewriter::hasExplicitRelease (which only checks the same block)
  // finds it and does not auto-insert a duplicate pop.
  for (auto &[assocCb, loc] : info.cbsNeedingPop) {
    // Find the WaitOp that uses this CB.
    WaitOp waitOp = nullptr;
    for (auto *user : assocCb.getUsers()) {
      if (auto w = dyn_cast<WaitOp>(user)) {
        waitOp = w;
        break;
      }
    }
    if (!waitOp) {
      continue;
    }

    Block *waitBlock = waitOp->getBlock();
    // Insert before the terminator (e.g., scf.yield) to avoid placing
    // ops after it, which would violate block structure invariants.
    if (waitBlock->mightHaveTerminator()) {
      rewriter.setInsertionPoint(waitBlock->getTerminator());
    } else {
      rewriter.setInsertionPointToEnd(waitBlock);
    }
    rewriter.create<PopOp>(loc, assocCb);
  }

  // Insert push ops for each reserve op
  for (auto &[reserveOp, assocCb] : info.reserveOpsNeedingPush) {
    Location loc = reserveOp.getLoc();

    GenericOp generic = reserveOp->getParentOfType<GenericOp>();
    Region *genericRegion = nullptr;
    if (generic.getNumRegions() == 1) {
      genericRegion = &generic.getRegion(0);
    } else {
      genericRegion = ttmlir::utils::getRegionWithParentOfType<GenericOp>(
          reserveOp.getOperation());
    }

    if (genericRegion && !genericRegion->empty()) {
      Block *topLevelBlock = &genericRegion->front();
      // Insert before the terminator (YieldOp)
      if (!topLevelBlock->empty() &&
          topLevelBlock->back().hasTrait<OpTrait::IsTerminator>()) {
        rewriter.setInsertionPoint(&topLevelBlock->back());
      } else {
        rewriter.setInsertionPointToEnd(topLevelBlock);
      }
      PushOp::create(rewriter, loc, assocCb);
    } else {
      reserveOp.emitWarning(
          "could not find top-level region block for push insertion");
    }
  }
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

    // Pass A: Convert all remote_load and remote_store into explicit CB form
    PushPopInfo info =
        convertToExplicitCBForm(moduleOp, rewriter, cbCache, portCounters);

    // Pass B: Insert push and pop operations
    insertPushAndPopOps(moduleOp, rewriter, info);
  }
};
} // namespace

} // namespace mlir::tt::d2m
