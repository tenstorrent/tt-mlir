// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERLOADSTOREOPSTOEXPLICITCBFORM
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

static LogicalResult verifyGenericOperand(Operation *op, Value memrefOperand) {
  GenericOp generic = op->getParentOfType<GenericOp>();
  if (!generic) {
    return op->emitError("must be nested in d2m.generic");
  }

  if (!llvm::is_contained(generic.getInputsAndOutputs(), memrefOperand)) {
    return op->emitError(
        "must access a d2m.generic input/output operand as its memref operand");
  }

  return success();
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

// Replace only uses owned by operations in the given block. This avoids
// rewriting enclosing d2m.generic operands to values defined inside the region.
static void replaceUsesInBlock(Value from, Value to, Block *block,
                               Operation *excludedUser = nullptr) {
  SmallVector<OpOperand *> usesToReplace;
  for (OpOperand &use : from.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == excludedUser || owner->getBlock() != block) {
      continue;
    }
    usesToReplace.push_back(&use);
  }
  for (OpOperand *use : usesToReplace) {
    use->set(to);
  }
}

// Find the first remote_store in reserveOp's block that consumes the same CB
// and appears after reserveOp.
static RemoteStoreOp getFirstFollowingRemoteStoreForCb(ReserveOp reserveOp,
                                                       Value cb) {
  if (!reserveOp || !cb) {
    return nullptr;
  }

  for (Operation *op = reserveOp->getNextNode(); op; op = op->getNextNode()) {
    auto remoteStore = mlir::dyn_cast<RemoteStoreOp>(op);
    if (remoteStore && remoteStore.getCb() == cb) {
      return remoteStore;
    }
  }

  return nullptr;
}

// Structure to hold information needed for push/pop insertion
struct PendingPopInfo {
  Value cb;
  Location loc;
  Block *insertionBlock;
};

struct PushPopInfo {
  SmallVector<PendingPopInfo> popsNeedingInsertion;
  SmallVector<std::pair<ReserveOp, Value>> reserveOpsNeedingPush;
};

// Validate shared local-buffer forwarding legality between an implicit
// remote_load and implicit remote_store users. If forwarding is legal, returns
// the unique store user that can be folded into the forward-able pair path.
static LogicalResult getLegalForwardableStore(RemoteLoadOp remoteLoad,
                                              Value localBuffer,
                                              RemoteStoreOp &forwardableStore) {
  forwardableStore = nullptr;
  if (!localBuffer) {
    return success();
  }

  forwardableStore = utils::findForwardableStore(remoteLoad);
  if (forwardableStore) {
    if (remoteLoad.isMcast()) {
      return remoteLoad.emitOpError(
          "multicast d2m.remote_load cannot be forwarded to d2m.remote_store");
    }
    return success();
  }

  // findForwardableStore returned null. If any implicit store users of the
  // local buffer exist, the pattern is illegal at this point in the pipeline.
  for (Operation *user : localBuffer.getUsers()) {
    auto storeUser = mlir::dyn_cast<RemoteStoreOp>(user);
    if (!storeUser || !storeUser.isImplicitForm() ||
        storeUser.getLocalBuffer() != localBuffer) {
      continue;
    }
    return remoteLoad.emitOpError("local buffer shared between "
                                  "d2m.remote_load and d2m.remote_store must "
                                  "form an exclusive forward-able pair");
  }

  return success();
}

// Pass A: Convert all remote_load and remote_store into explicit CB form.
// Returns information needed for push/pop insertion.
static LogicalResult convertToExplicitCBForm(ModuleOp moduleOp,
                                             IRRewriter &rewriter,
                                             PushPopInfo &info, CBCache &cache,
                                             PortCounter &portCounters) {

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

    if (failed(verifyGenericOperand(remoteLoad.getOperation(), memref))) {
      return failure();
    }

    Value assocCb = remoteLoad.isImplicitForm()
                        ? findAssociatedCB(remoteLoad.getOperation(), memref,
                                           rewriter, cache, portCounters)
                        : remoteLoad.getCb();

    if (!assocCb) {
      remoteLoad.emitError(
          "could not find associated CB block argument for memref operand");
      return failure();
    }

    // Get the local buffer (destination) from implicit form remote_load
    // Downstream operations may reference this buffer directly
    Value localBuffer = remoteLoad.getLocalBuffer();
    memref::AllocOp allocToErase = nullptr;
    RemoteStoreOp forwardableStore = nullptr;
    if (localBuffer) {
      if (auto allocOp = mlir::dyn_cast_if_present<memref::AllocOp>(
              localBuffer.getDefiningOp())) {
        allocToErase = allocOp;
      }
      if (failed(getLegalForwardableStore(remoteLoad, localBuffer,
                                          forwardableStore))) {
        return failure();
      }
    }

    Value loadTargetCb = assocCb;
    bool eraseForwardableStoreWithoutReplacement = false;
    if (forwardableStore) {
      Value storeMemref = forwardableStore.getMemref();
      bool storeMemrefIsStreamBacked =
          mlir::isa_and_nonnull<StreamLayoutOp>(storeMemref.getDefiningOp());
      if (!storeMemrefIsStreamBacked) {
        // If the forwardable store targets a local generic operand (no stream),
        // load directly into that output operand's CB and drop the store.
        loadTargetCb =
            findAssociatedCB(forwardableStore.getOperation(), storeMemref,
                             rewriter, cache, portCounters);
        if (!loadTargetCb) {
          forwardableStore.emitError(
              "could not find associated CB for forwarded memref operand");
          return failure();
        }
        eraseForwardableStoreWithoutReplacement = true;
      }
    }

    rewriter.setInsertionPoint(remoteLoad);

    // Create the explicit CB form of remote_load (no localBuffer, no result,
    // has CB operand) d2m.remote_load %memref[indices] into %cb
    auto explicitRemoteLoad = rewriter.create<RemoteLoadOp>(
        loc, memref, remoteLoad.getIndices(), loadTargetCb,
        remoteLoad.getMcastStartIndex(), remoteLoad.getMcastShape());

    // Forward-able pair:
    //   remote_load %buf %in[...]
    //   remote_store %out[...] %buf
    // Lower both to the same CB without introducing reserve/push.
    //
    // When the store remains as an explicit remote_store that consumes the same
    // CB as the load, wait/pop stay implicit in that store path and must not be
    // materialized here. If the store folds away entirely (e.g. the forwarded
    // destination is a local generic operand), the remote_load becomes the lone
    // consumer and we must still materialize wait/pop.
    if (forwardableStore) {
      rewriter.setInsertionPointAfter(explicitRemoteLoad);
      if (eraseForwardableStoreWithoutReplacement) {
        auto waitOp = rewriter.create<WaitOp>(loc, loadTargetCb);
        if (remoteLoad.getResult()) {
          replaceUsesInBlock(remoteLoad.getResult(), waitOp.getResult(),
                             remoteLoad->getBlock(), remoteLoad.getOperation());
        }
        info.popsNeedingInsertion.push_back(
            {loadTargetCb, loc, remoteLoad->getBlock()});
      } else {
        rewriter.create<RemoteStoreOp>(
            forwardableStore.getLoc(), forwardableStore.getMemref(),
            forwardableStore.getIndices(), loadTargetCb);
      }
      rewriter.eraseOp(forwardableStore);
      rewriter.eraseOp(remoteLoad);
      if (allocToErase) {
        rewriter.eraseOp(allocToErase);
      }
      continue;
    }

    // Create wait operation to produce the result value.
    // %in = d2m.wait %cb
    auto waitOp = rewriter.create<WaitOp>(loc, assocCb);

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

    // Replace all uses of the local buffer with the wait result
    // This is important because downstream operations may reference the
    // localBuffer directly (e.g., remote_store using the alloc result)
    if (localBuffer) {
      replaceUsesInBlock(localBuffer, waitOp.getResult(),
                         remoteLoad->getBlock(), remoteLoad.getOperation());
    }

    // Replace all uses of remote_load result with wait result
    if (remoteLoad.getResult()) {
      replaceUsesInBlock(remoteLoad.getResult(), waitOp.getResult(),
                         remoteLoad->getBlock(), remoteLoad.getOperation());
    }

    // Track CB that needs pop insertion (deferred to Pass B)
    info.popsNeedingInsertion.push_back({assocCb, loc, remoteLoad->getBlock()});

    // Erase the original remote_load operation
    rewriter.eraseOp(remoteLoad);

    // Erase the memref.alloc if it was the local buffer source
    if (allocToErase && allocToErase->use_empty()) {
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
    auto reserveOp = rewriter.create<ReserveOp>(loc, assocCb);

    // Replace all uses of memref.alloc result with reserve result
    rewriter.replaceAllUsesWith(allocOp.getResult(), reserveOp.getResult());

    // Track reserve op for push insertion (deferred to Pass B)
    info.reserveOpsNeedingPush.push_back({reserveOp, assocCb});

    // Erase the original memref.alloc operation
    rewriter.eraseOp(allocOp);
  }

  // Transform RemoteStoreOp (implicit form -> explicit CB form)
  SmallVector<RemoteStoreOp> remoteStoresToConvert;
  moduleOp->walk([&](RemoteStoreOp remoteStore) {
    if (!remoteStore.isImplicitForm()) {
      return;
    }
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

    if (failed(verifyGenericOperand(remoteStore.getOperation(), memref))) {
      return failure();
    }

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
      rewriter.create<RemoteStoreOp>(loc, memref, remoteStore.getIndices(),
                                     assocCb);

      // Track the reserve op for push insertion (avoid duplicates).
      // When the CB was found through a WaitOp (non-simplified load-store
      // pair), there is no ReserveOp — the load already populated the CB.
      // We only need a pop after consuming the data.
      if (reserveOp && cbsWithReserveOps.insert(assocCb).second) {
        info.reserveOpsNeedingPush.push_back({reserveOp, assocCb});
      } else if (!reserveOp && cbsWithReserveOps.insert(assocCb).second) {
        OpBuilder::InsertionGuard reserveGuard(rewriter);
        rewriter.setInsertionPointAfterValue(assocCb);
        auto newReserve = rewriter.create<ReserveOp>(loc, assocCb);
        info.reserveOpsNeedingPush.push_back({newReserve, assocCb});
        if (localBuffer) {
          replaceUsesInBlock(localBuffer, newReserve.getResult(),
                             remoteStore->getBlock(),
                             remoteStore.getOperation());
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
        auto reserveOp = rewriter.create<ReserveOp>(loc, assocCb);
        info.reserveOpsNeedingPush.push_back({reserveOp, assocCb});
      }
    }
  }

  return success();
}

// Pass B: Insert push and pop operations
static void insertPushAndPopOps(IRRewriter &rewriter, PushPopInfo &info) {
  // Insert pop ops for remote_load conversions
  for (const PendingPopInfo &pendingPop : info.popsNeedingInsertion) {
    Block *insertionBlock = pendingPop.insertionBlock;
    if (!insertionBlock) {
      continue;
    }

    if (!insertionBlock->empty() &&
        insertionBlock->back().hasTrait<OpTrait::IsTerminator>()) {
      rewriter.setInsertionPoint(&insertionBlock->back());
    } else {
      rewriter.setInsertionPointToEnd(insertionBlock);
    }
    rewriter.create<PopOp>(pendingPop.loc, pendingPop.cb);
  }

  // Insert push ops for each reserve op
  for (auto &[reserveOp, assocCb] : info.reserveOpsNeedingPush) {
    Location loc = reserveOp.getLoc();
    Block *insertionBlock = reserveOp->getBlock();
    if (!insertionBlock) {
      reserveOp.emitWarning("could not find block for push insertion");
      continue;
    }

    // Push must be in the same region as reserve/compute and before the store
    // that consumes this CB.
    if (RemoteStoreOp store =
            getFirstFollowingRemoteStoreForCb(reserveOp, assocCb)) {
      rewriter.setInsertionPoint(store);
    } else if (!insertionBlock->empty() &&
               insertionBlock->back().hasTrait<OpTrait::IsTerminator>()) {
      rewriter.setInsertionPoint(&insertionBlock->back());
    } else {
      rewriter.setInsertionPointToEnd(insertionBlock);
    }
    rewriter.create<PushOp>(loc, assocCb);
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
    PushPopInfo info;
    CBCache cbCache;
    PortCounter portCounters;

    // Pass A: Convert all remote_load and remote_store into explicit CB form
    if (failed(convertToExplicitCBForm(moduleOp, rewriter, info, cbCache,
                                       portCounters))) {
      signalPassFailure();
      return;
    }

    // Pass B: Insert push and pop operations
    insertPushAndPopOps(rewriter, info);
  }
};
} // namespace

} // namespace mlir::tt::d2m
