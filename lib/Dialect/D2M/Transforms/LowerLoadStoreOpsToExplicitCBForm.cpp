// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
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
// such as view_layout or stream_layout)
static bool isRemoteOperand(Value operand, Operation *op) {
  Operation *defOp = operand.getDefiningOp();
  if (!defOp) {
    return false;
  }
  // Remote operands are those that come from ops implementing ViewOpInterface
  return mlir::isa<ViewOpInterface>(defOp);
}

// Helper function to find the CB block argument that corresponds to a memref
// operand in a generic op. Returns the CB block argument if found, null
// otherwise. Assumes that the operand index in the generic op equals the CB
// block arg index.
//
// For DMA-only GenericOps with remote loads: returns the CB associated with
// the output operand (destination), not the input operand being loaded.
static Value findAssociatedCB(Operation *op, Value memrefOperand) {
  GenericOp generic = op->getParentOfType<GenericOp>();
  if (!generic) {
    return Value();
  }

  // Find which operand index this memref corresponds to
  unsigned operandIndex = UINT_MAX;
  for (unsigned i = 0; i < generic->getNumOperands(); ++i) {
    if (generic->getOperand(i) == memrefOperand) {
      operandIndex = i;
      break;
    }
  }

  if (operandIndex == UINT_MAX) {
    return Value();
  }

  // Find the generic op's thread region that contains this operation
  // If there's only one region, use it directly. Otherwise, use the utility
  // function
  Region *genericRegion = nullptr;
  if (generic.getNumRegions() == 1) {
    genericRegion = &generic.getRegion(0);
  } else {
    genericRegion = ttmlir::utils::getRegionWithParentOfType<GenericOp>(op);
  }

  if (!genericRegion || genericRegion->empty()) {
    return Value();
  }

  // Get the first block of the generic region (thread region block)
  Block *threadBlock = &genericRegion->front();

  // The CB block arguments are in the same order as the generic operands
  // The operand index equals the CB block arg index
  if (threadBlock->getNumArguments() > operandIndex) {
    return threadBlock->getArgument(operandIndex);
  }
  return Value();
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
static void simplifyLoadStorePairs(ModuleOp moduleOp, IRRewriter &rewriter) {
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
    unsigned loadOperandIndex = loadOperandIt->getOperandNumber();
    unsigned storeOperandIndex = storeOperandIt->getOperandNumber();

    // Get the thread region block
    TT_assert(generic.getNumRegions() == 1u);
    Region *genericRegion = &generic.getRegion(0);
    Block *threadBlock = &genericRegion->front();

    // Get CB block arguments (verification guarantees CB count aligns with
    // operand count)
    Value inputCB = threadBlock->getArgument(loadOperandIndex);
    Value outputCB = threadBlock->getArgument(storeOperandIndex);

    rewriter.setInsertionPoint(loadOp);

    // Case A: Load from remote (input has view layout) -> load into output CB
    // Case B: Store to remote (output has view layout) -> store from input CB
    bool isRemoteLoad = isRemoteOperand(loadMemref, loadOp.getOperation());
    bool isRemoteStore = isRemoteOperand(storeMemref, storeOp.getOperation());
    TT_assert(!(isRemoteLoad && isRemoteStore));
    if (isRemoteLoad) {
      // Create the explicit CB form of remote_load (no localBuffer, has CB
      // operand)
      rewriter.create<RemoteLoadOp>(loc, loadMemref, loadOp.getIndices(),
                                    outputCB, loadOp.getMcastStartIndex(),
                                    loadOp.getMcastShape());
    } else if (isRemoteStore) {
      rewriter.create<RemoteStoreOp>(loc, storeMemref, loadOp.getIndices(),
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
                                           IRRewriter &rewriter) {
  PushPopInfo info;

  // Pre-process generics with load-store idiom
  simplifyLoadStorePairs(moduleOp, rewriter);

  // Transform RemoteLoadOp (implicit form -> explicit CB form)
  SmallVector<RemoteLoadOp> remoteLoadsToConvert;
  moduleOp->walk([&](RemoteLoadOp remoteLoad) {
    Value memref = remoteLoad.getMemref();
    // Only handle remote operands (from stream_layout ops)
    if (!isRemoteOperand(memref, remoteLoad.getOperation())) {
      return;
    }
    remoteLoadsToConvert.push_back(remoteLoad);
  });

  // Transform each remote_load
  for (RemoteLoadOp remoteLoad : remoteLoadsToConvert) {
    Location loc = remoteLoad.getLoc();
    Value memref = remoteLoad.getMemref();
    Value assocCb = remoteLoad.isImplicitForm()
                        ? findAssociatedCB(remoteLoad.getOperation(), memref)
                        : remoteLoad.getCb();

    if (!assocCb) {
      remoteLoad.emitWarning("could not find associated CB block argument, "
                             "skipping conversion");
      continue;
    }

    // Get the local buffer (destination) from implicit form remote_load
    // Downstream operations may reference this buffer directly
    Value localBuffer = remoteLoad.getLocalBuffer();
    memref::AllocOp allocToErase = nullptr;
    if (localBuffer) {
      if (auto allocOp = mlir::dyn_cast_if_present<memref::AllocOp>(
              localBuffer.getDefiningOp())) {
        allocToErase = allocOp;
      }
    }

    rewriter.setInsertionPoint(remoteLoad);

    // Create the explicit CB form of remote_load (no localBuffer, no result,
    // has CB operand) d2m.remote_load %memref[indices] into %cb
    rewriter.create<RemoteLoadOp>(loc, memref, remoteLoad.getIndices(), assocCb,
                                  remoteLoad.getMcastStartIndex(),
                                  remoteLoad.getMcastShape());

    // Create wait operation to produce the result value
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
      rewriter.replaceAllUsesWith(localBuffer, waitOp.getResult());
    }

    // Replace all uses of remote_load result with wait result
    if (remoteLoad.getResult()) {
      rewriter.replaceAllUsesWith(remoteLoad.getResult(), waitOp.getResult());
    }

    // Track CB that needs pop insertion (deferred to Pass B)
    info.cbsNeedingPop.push_back({assocCb, loc});

    // Erase the original remote_load operation
    rewriter.eraseOp(remoteLoad);

    // Erase the memref.alloc if it was the local buffer source
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

    // Check if the associated operand is remote
    if (!isRemoteOperand(assocOperand, allocOp.getOperation())) {
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
    Value assocCb =
        GenericOp::findAssocCBByOperand(allocOp.getOperation(), assocOperand);
    if (!assocCb) {
      allocOp.emitWarning(
          "could not find associated CB block argument, skipping conversion");
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
    Value memref = remoteStore.getMemref();
    // Only handle remote operands (from stream_layout ops)
    if (!isRemoteOperand(memref, remoteStore.getOperation())) {
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

    if (remoteStore.isImplicitForm()) {
      // Implicit form: find the CB by tracing back from local buffer to reserve
      // op
      ReserveOp reserveOp = findReserveOp(localBuffer);
      if (!reserveOp) {
        remoteStore.emitWarning(
            "could not find reserve op for local buffer, skipping conversion");
        continue;
      }
      assocCb = reserveOp.getCb();

      rewriter.setInsertionPoint(remoteStore);

      // Create the explicit CB form of remote_store (no local buffer, has CB)
      // d2m.remote_store %memref[indices] from %cb
      rewriter.create<RemoteStoreOp>(loc, memref, remoteStore.getIndices(),
                                     assocCb);

      // Track the reserve op for push insertion (avoid duplicates)
      if (cbsWithReserveOps.insert(assocCb).second) {
        info.reserveOpsNeedingPush.push_back({reserveOp, assocCb});
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

  return info;
}

// Pass B: Insert push and pop operations
static void insertPushAndPopOps(ModuleOp moduleOp, IRRewriter &rewriter,
                                PushPopInfo &info) {
  // Insert pop ops for remote_load conversions
  for (auto &[assocCb, loc] : info.cbsNeedingPop) {
    Operation *parentOp = assocCb.getParentRegion()->getParentOp();
    GenericOp generic = mlir::dyn_cast<GenericOp>(parentOp);
    if (!generic) {
      continue;
    }

    TT_assert(generic.getNumRegions() == 1u);
    Region *genericRegion = &generic.getRegion(0);

    if (genericRegion && !genericRegion->empty()) {
      Block *topLevelBlock = &genericRegion->front();
      rewriter.setInsertionPointToEnd(topLevelBlock);
      rewriter.create<PopOp>(loc, assocCb);
    }
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
      rewriter.create<PushOp>(loc, assocCb);
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

    // Pass A: Convert all remote_load and remote_store into explicit CB form
    PushPopInfo info = convertToExplicitCBForm(moduleOp, rewriter);

    // Pass B: Insert push and pop operations
    insertPushAndPopOps(moduleOp, rewriter, info);
  }
};
} // namespace

} // namespace mlir::tt::d2m
