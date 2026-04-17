// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPLITUNIFIEDTHREAD
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Returns true if the remote_load/store requires real DMA. This is the case
// when the remote memref has a view layout, is in DRAM, or the local buffer is
// a streaming CB (CBLayoutAttr). Aliased ops do not need DMA and return false.
static bool needsDMA(Value memref, Value localBuffer) {
  // View ops need datamovement, except for reinterpret view_layout ops
  // which are just type casts.
  if (auto *defOp = memref.getDefiningOp()) {
    if (auto viewOp = mlir::dyn_cast<ViewLayoutOp>(defOp)) {
      return !viewOp.getReinterpretLayout();
    }
    if (mlir::isa<ViewOpInterface>(defOp)) {
      return true;
    }
  }
  if (auto memrefType = mlir::dyn_cast<MemRefType>(memref.getType())) {
    if (ttcore::getMemorySpace(memrefType) == ttcore::MemorySpace::DeviceDRAM) {
      return true;
    }
  }

  // Check if the local buffer is a streaming CB.
  if (localBuffer) {
    if (auto bufType = mlir::dyn_cast<MemRefType>(localBuffer.getType())) {
      if (mlir::isa<ttcore::CBLayoutAttr>(bufType.getLayout())) {
        return true;
      }
    }
  }

  return false;
}

// Walk a block and find the last operation that uses a value, including uses
// in nested regions. Tracks indirect uses through view-like operations (e.g.,
// memref.collapse_shape). Returns the top-level operation after which to
// insert cleanup ops, or null if no uses found.
static Operation *findLastUseOfAliasedValue(Value value, Block *block) {
  Operation *lastUse = nullptr;

  // Build alias set: the value itself + any view-like derivations.
  llvm::SmallPtrSet<Value, 8> aliasedValues;
  aliasedValues.insert(value);

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : *block) {
      bool takesAliasedInput = false;
      for (OpOperand &operand : op.getOpOperands()) {
        if (aliasedValues.contains(operand.get())) {
          takesAliasedInput = true;
          break;
        }
      }
      if (takesAliasedInput && mlir::isa<mlir::ViewLikeOpInterface>(op)) {
        for (Value result : op.getResults()) {
          if (aliasedValues.insert(result).second) {
            changed = true;
          }
        }
      }
    }
  }

  // Recursive check for uses in nested regions.
  std::function<bool(Region &)> isUsedInRegion = [&](Region &region) -> bool {
    for (Block &regionBlock : region) {
      for (Operation &op : regionBlock) {
        for (OpOperand &operand : op.getOpOperands()) {
          if (aliasedValues.contains(operand.get())) {
            return true;
          }
        }
        for (Region &nestedRegion : op.getRegions()) {
          if (isUsedInRegion(nestedRegion)) {
            return true;
          }
        }
      }
    }
    return false;
  };

  // Walk top-level ops to find the last one that uses any alias.
  for (Operation &op : *block) {
    bool opUsesValue = false;
    for (OpOperand &operand : op.getOpOperands()) {
      if (aliasedValues.contains(operand.get())) {
        opUsesValue = true;
        break;
      }
    }
    if (!opUsesValue) {
      for (Region &region : op.getRegions()) {
        if (isUsedInRegion(region)) {
          opUsesValue = true;
          break;
        }
      }
    }
    if (opUsesValue) {
      lastUse = &op;
    }
  }

  return lastUse;
}

// Insert a pop before the block terminator.
static void insertPopBeforeTerminator(PatternRewriter &rewriter, Location loc,
                                      Value cb, Block *block) {
  if (block->mightHaveTerminator()) {
    rewriter.setInsertionPoint(block->getTerminator());
  } else {
    rewriter.setInsertionPointToEnd(block);
  }
  rewriter.create<PopOp>(loc, cb);
}

// Find load-store pairs that share the same localBuffer in a block.
static SmallVector<std::pair<RemoteLoadOp, RemoteStoreOp>>
findSharedBufferPairs(Block *block) {
  SmallVector<std::pair<RemoteLoadOp, RemoteStoreOp>> pairs;
  block->walk([&](RemoteStoreOp storeOp) {
    if (storeOp.isExplicitCBForm()) {
      return;
    }
    Value localBuffer = storeOp.getLocalBuffer();
    if (!localBuffer) {
      return;
    }
    for (Operation *user : localBuffer.getUsers()) {
      if (auto loadOp = mlir::dyn_cast<RemoteLoadOp>(user);
          loadOp && !loadOp.isExplicitCBForm() &&
          loadOp.getLocalBuffer() == localBuffer) {
        // Read-modify-write (self read/write)pattern is not a shared-buffer
        // copy.
        if (loadOp.getMemref() == storeOp.getMemref()) {
          return;
        }
        // When types differ, they can't share a CB - each needs its own.
        auto loadElemType = mlir::cast<ShapedType>(loadOp.getMemref().getType())
                                .getElementType();
        auto storeElemType =
            mlir::cast<ShapedType>(storeOp.getMemref().getType())
                .getElementType();
        if (loadElemType == storeElemType) {
          pairs.push_back({loadOp, storeOp});
        }
        return;
      }
    }
  });
  return pairs;
}

// External allocs (e.g., hoisted CB allocs passed as additionalArgs)
// must not be erased or replaced by the splitter.
static bool isLocalAlloc(memref::AllocOp allocOp, Block *block) {
  return block->getParent()->isAncestor(allocOp->getParentRegion());
}

// Trace a value through view-like ops to a memref.alloc.
static memref::AllocOp findAllocOp(Value value) {
  return traceToDefiningOp<memref::AllocOp>(value);
}

// ---------------------------------------------------------------------------
// Compute thread: insert CB sync ops for implicit-form remote_load/store
// ---------------------------------------------------------------------------

// Handle load-store pairs that share the same local buffer (DMA-only
// generics that copy input->output with no compute in between). The shared
// buffer means one CB serves both ops.
static LogicalResult processSharedBufferPairs(Block *computeBlock,
                                              PatternRewriter &rewriter,
                                              CBCache &cache,
                                              PortCounter &portCounters,
                                              DenseSet<Operation *> &toErase) {
  auto pairs = findSharedBufferPairs(computeBlock);

  for (auto [loadOp, storeOp] : pairs) {
    Value sharedBuffer = loadOp.getLocalBuffer();
    bool loadNeedsDMA = needsDMA(loadOp.getMemref(), sharedBuffer);
    bool storeNeedsDMA = needsDMA(storeOp.getMemref(), sharedBuffer);

    // If neither side needs DMA (L1-to-L1 copy): both ops still need actual DMA
    // through a shared CB. The DM thread handles
    // reserve-read-push-wait-write-pop cycle. Erase ops from the compute.
    if (!loadNeedsDMA && !storeNeedsDMA) {
      toErase.insert(storeOp);
      toErase.insert(loadOp);
      continue;
    }

    // Insert compute-side CB ops for the aliased half of the pair.
    // The streaming half stays as a remote_load/store for DMA.
    Location loc = loadOp.getLoc();
    if (loadNeedsDMA && !storeNeedsDMA) {
      Value cb = findAssociatedCB(storeOp, storeOp.getMemref(), rewriter, cache,
                                  portCounters);
      if (!cb) {
        return storeOp.emitError(
            "could not find associated CB for shared pair");
      }
      rewriter.setInsertionPoint(storeOp);
      rewriter.create<WaitOp>(loc, cb);
      rewriter.create<PopOp>(loc, cb);
    } else if (!loadNeedsDMA && storeNeedsDMA) {
      Value cb = findAssociatedCB(loadOp, loadOp.getMemref(), rewriter, cache,
                                  portCounters);
      if (!cb) {
        return loadOp.emitError("could not find associated CB for shared pair");
      }
      rewriter.setInsertionPoint(loadOp);
      rewriter.create<ReserveOp>(loc, cb);
      rewriter.create<PushOp>(loc, cb);
    }
    // Else if both sides are streaming/need DMA, let the DM thread handle
    // everything.

    toErase.insert(storeOp);
    toErase.insert(loadOp);
  }
  return success();
}

// Process implicit-form remote_load ops in the compute thread.
static LogicalResult processComputeLoads(Block *computeBlock,
                                         PatternRewriter &rewriter,
                                         CBCache &cache,
                                         PortCounter &portCounters,
                                         DenseSet<Operation *> &toErase) {
  SmallVector<RemoteLoadOp> loads;
  computeBlock->walk([&](RemoteLoadOp op) {
    if (!op.isExplicitCBForm() && !toErase.contains(op)) {
      loads.push_back(op);
    }
  });

  for (RemoteLoadOp loadOp : loads) {
    Location loc = loadOp.getLoc();
    Value memref = loadOp.getMemref();
    Value localBuffer = loadOp.getLocalBuffer();
    Value cb = findAssociatedCB(loadOp, memref, rewriter, cache, portCounters);
    if (!cb) {
      return loadOp.emitError("could not find associated CB for load");
    }

    if (!needsDMA(memref, localBuffer)) {
      // Aliased loads should not have multicast parameters.
      if (loadOp.isMcast()) {
        return loadOp.emitError(
            "remote_load with local operand has multicast parameters");
      }
      // For aliased loads, insert reserve->push->wait at the alloc site and
      // pop after the last use of the buffer.
      memref::AllocOp allocOp = findAllocOp(localBuffer);
      if (!allocOp || !isLocalAlloc(allocOp, computeBlock)) {
        return loadOp.emitError("could not find local memref.alloc for buffer");
      }

      Block *block = allocOp->getBlock();
      rewriter.setInsertionPoint(allocOp);
      rewriter.create<ReserveOp>(loc, cb);
      rewriter.create<PushOp>(loc, cb);
      auto waitOp = rewriter.create<WaitOp>(loc, cb);

      // Replace all uses of the alloc, not just the load's operand as
      // downstream compute ops reference the alloc result directly and
      // must read from the CB. Assumes 1:1 alloc-to-load relationship.
      rewriter.replaceAllUsesWith(allocOp.getResult(), waitOp.getResult());
      if (loadOp.getResult()) {
        rewriter.replaceAllUsesWith(loadOp.getResult(), waitOp.getResult());
      }

      // Insert pop after the last use of the waited value, or before
      // the block terminator if no uses found.
      Operation *lastUse = findLastUseOfAliasedValue(waitOp.getResult(), block);
      if (lastUse && lastUse != loadOp.getOperation()) {
        rewriter.setInsertionPointAfter(lastUse);
        rewriter.create<PopOp>(loc, cb);
      } else {
        insertPopBeforeTerminator(rewriter, loc, cb, block);
      }
      toErase.insert(loadOp);
    } else {
      // Needs datamovement: wait before load, pop before terminator.
      rewriter.setInsertionPoint(loadOp);
      auto waitOp = rewriter.create<WaitOp>(loc, cb);

      Region *computeRegion = loadOp->getParentRegion();
      rewriter.replaceUsesWithIf(
          localBuffer, waitOp.getResult(), [&](OpOperand &use) {
            return computeRegion->isAncestor(use.getOwner()->getParentRegion());
          });
      if (loadOp.getResult()) {
        rewriter.replaceAllUsesWith(loadOp.getResult(), waitOp.getResult());
      }

      // Insert pop only when the waited data has compute consumers.
      // When all consumers are DM ops (e.g. local_copy), the pop is
      // managed by the DM thread via LowerLoadStoreOpsToDMA.
      bool allDMConsumers =
          !waitOp.getResult().use_empty() &&
          llvm::all_of(waitOp.getResult().getUsers(), [](Operation *user) {
            return isa<ShardDMAOpInterface>(user);
          });
      if (!allDMConsumers) {
        insertPopBeforeTerminator(rewriter, loc, cb, loadOp->getBlock());
      }
      toErase.insert(loadOp);
    }
  }
  return success();
}

// Process implicit-form remote_store ops in the compute thread.
static LogicalResult processComputeStores(Block *computeBlock,
                                          PatternRewriter &rewriter,
                                          CBCache &cache,
                                          PortCounter &portCounters,
                                          DenseSet<Operation *> &toErase) {
  SmallVector<RemoteStoreOp> stores;
  computeBlock->walk([&](RemoteStoreOp op) {
    if (!op.isExplicitCBForm() && !toErase.contains(op)) {
      stores.push_back(op);
    }
  });

  for (RemoteStoreOp storeOp : stores) {
    Location loc = storeOp.getLoc();
    Value memref = storeOp.getMemref();
    Value localBuffer = storeOp.getLocalBuffer();
    if (!localBuffer) {
      return storeOp.emitError(
          "remote_store does not have a local buffer operand");
    }

    Value cb = findAssociatedCB(storeOp, memref, rewriter, cache, portCounters);
    if (!cb) {
      return storeOp.emitError("could not find associated CB for store");
    }

    if (!needsDMA(memref, localBuffer)) {
      // For aliased stores, replace the alloc with a reserve and insert
      // push+wait+pop at the store position.
      memref::AllocOp allocOp = findAllocOp(localBuffer);
      if (allocOp && isLocalAlloc(allocOp, computeBlock)) {
        rewriter.setInsertionPoint(allocOp);
        auto reserveOp = rewriter.create<ReserveOp>(loc, cb);
        // Replace all uses as compute ops reference the alloc directly and
        // must write into the CB. Assumes 1:1 alloc-to-store relationship.
        rewriter.replaceAllUsesWith(allocOp.getResult(), reserveOp.getResult());
      } else if (auto waitOp = localBuffer.getDefiningOp<WaitOp>()) {
        // The alloc was already replaced by a WaitOp from an earlier load
        // (read-modify-write on same buffer). If it's the same CB,
        // the buffer already aliases shard memory so just erase the redundant
        // store without inserting extra sync ops.
        if (waitOp.getCb() == cb) {
          toErase.insert(storeOp);
          continue;
        }
      } else {
        return storeOp.emitError(
            "could not find memref.alloc for local buffer");
      }

      rewriter.setInsertionPoint(storeOp);
      rewriter.create<PushOp>(loc, cb);
      rewriter.create<WaitOp>(loc, cb);
      rewriter.create<PopOp>(loc, cb);
      toErase.insert(storeOp);
    } else {
      // Needs datamovement: insert reserve + push, replace in-region buffer
      // uses. Reserve must dominate all uses of localBuffer in its block. If
      // the CB def is in the same block as the store, insert after it.
      // Otherwise (eg: inside a loop), insert at the start of the store's
      // block.
      Operation *cbDefOp = cb.getDefiningOp();
      if (cbDefOp && cbDefOp->getBlock() == storeOp->getBlock()) {
        rewriter.setInsertionPointAfter(cbDefOp);
      } else {
        rewriter.setInsertionPointToStart(storeOp->getBlock());
      }
      auto reserveOp = rewriter.create<ReserveOp>(loc, cb);
      Region *computeRegion = storeOp->getParentRegion();
      rewriter.replaceUsesWithIf(
          localBuffer, reserveOp.getResult(), [&](OpOperand &use) {
            return computeRegion->isAncestor(use.getOwner()->getParentRegion());
          });
      // Push goes at the store position (after compute fills the buffer).
      rewriter.setInsertionPoint(storeOp);
      rewriter.create<PushOp>(loc, cb);
      toErase.insert(storeOp);
    }
  }
  return success();
}

// Process implicit-form local_copy ops in the compute thread.
// local_copy is a DM-only op: compute needs a wait on the destination CB
// so downstream compute ops can read the result.
static LogicalResult processComputeLocalCopies(Block *computeBlock,
                                               PatternRewriter &rewriter,
                                               CBCache &cache,
                                               PortCounter &portCounters,
                                               DenseSet<Operation *> &toErase) {
  SmallVector<LocalCopyOp> copies;
  computeBlock->walk([&](LocalCopyOp op) {
    if (op.isImplicitForm() && !toErase.contains(op)) {
      copies.push_back(op);
    }
  });

  for (LocalCopyOp copyOp : copies) {
    Location loc = copyOp.getLoc();
    Value src = copyOp.getSrc();
    Value dst = copyOp.getDst();

    // --- Source side (compute → copy) ---
    // When the source is compute-produced, insert push so the DMA thread
    // can wait on the source CB.  DMA-produced sources (from WaitOp, i.e.
    // processed by processComputeLoads) already had their push on the DMA
    // thread and don't need one here.
    if (!src.getDefiningOp<WaitOp>()) {
      Value srcCb;
      if (auto reserveOp = traceToDefiningOp<ReserveOp>(src)) {
        srcCb = reserveOp.getCb();
      } else {
        srcCb = findAssociatedCB(copyOp, src, rewriter, cache, portCounters);
      }
      if (srcCb) {
        rewriter.setInsertionPoint(copyOp);
        rewriter.create<PushOp>(loc, srcCb);
      }
    }

    // --- Destination side (copy → compute) ---
    Value dstCb = findAssociatedCB(copyOp, dst, rewriter, cache, portCounters);
    if (!dstCb) {
      return copyOp.emitError(
          "could not find associated CB for local_copy destination");
    }

    // Insert wait to produce a readable memref for downstream compute ops.
    rewriter.setInsertionPoint(copyOp);
    auto waitOp = rewriter.create<WaitOp>(loc, dstCb);

    // Replace uses of the destination buffer within the compute region only.
    // Must NOT replace the generic's own operand or uses in the DMA region.
    Region *computeRegion = copyOp->getParentRegion();
    rewriter.replaceUsesWithIf(dst, waitOp.getResult(), [&](OpOperand &use) {
      return computeRegion->isAncestor(use.getOwner()->getParentRegion());
    });

    // If the dst came from a ReserveOp that is now unused, collect it for
    // erasure (compute reads via wait, not reserve).
    if (auto reserveOp = dst.getDefiningOp<ReserveOp>()) {
      if (reserveOp.getResult().use_empty()) {
        toErase.insert(reserveOp);
      }
    }

    // Insert pop after last use of the waited value if there are
    // compute consumers.  DM-only consumers are handled on the DM thread.
    bool hasComputeConsumers =
        llvm::any_of(waitOp.getResult().getUsers(), [](Operation *user) {
          return !isa<ShardDMAOpInterface>(user);
        });
    if (hasComputeConsumers) {
      Operation *lastUse =
          findLastUseOfAliasedValue(waitOp.getResult(), copyOp->getBlock());
      if (lastUse) {
        rewriter.setInsertionPointAfter(lastUse);
        rewriter.create<PopOp>(loc, dstCb);
      } else {
        insertPopBeforeTerminator(rewriter, loc, dstCb, copyOp->getBlock());
      }
    }

    toErase.insert(copyOp);
  }
  return success();
}

// ---------------------------------------------------------------------------
// DMA thread: convert implicit-form ops to explicit CB form
// ---------------------------------------------------------------------------

// Convert remote_load/store to explicit CB form in the DMA thread.
// Aliased ops are collected for deferred erasure (no DMA needed). Shared
// buffer pairs use the output operand's CB for both ops.
static LogicalResult
convertDMAToExplicitCBForm(Block *dmBlock, PatternRewriter &rewriter,
                           CBCache &cache, PortCounter &portCounters,
                           DenseSet<Operation *> &toErase) {
  // Map shared localBuffer -> (load, store) pair. Both ops must use the same
  // CB: the aliased side's CB, or the output CB when both are streaming.
  using SharedPair = std::pair<RemoteLoadOp, RemoteStoreOp>;
  DenseMap<Value, SharedPair> sharedPairs;
  for (auto [loadOp, storeOp] : findSharedBufferPairs(dmBlock)) {
    sharedPairs[loadOp.getLocalBuffer()] = {loadOp, storeOp};
  }

  SmallVector<RemoteLoadOp> loads;
  SmallVector<RemoteStoreOp> stores;
  SmallVector<LocalCopyOp> localCopies;
  dmBlock->walk([&](RemoteLoadOp op) { loads.push_back(op); });
  dmBlock->walk([&](RemoteStoreOp op) { stores.push_back(op); });
  dmBlock->walk([&](LocalCopyOp op) {
    if (op.isImplicitForm()) {
      localCopies.push_back(op);
    }
  });

  // Map local buffers to the CB assigned during remote_load conversion.
  // Used to connect local_copy sources to the correct CB on the DMA thread.
  DenseMap<Value, Value> localBufferToCB;

  // Helper: check if an op is part of an L1-to-L1 shared pair. These need real
  // DMA even though both operands are L1. Self-read/write pairs (same memref)
  // are excluded because they are aliased ops handled entirely in compute via
  // CB sync.
  auto isL1ToL1Pair = [&](Value localBuffer) -> bool {
    auto it = sharedPairs.find(localBuffer);
    if (it == sharedPairs.end()) {
      return false;
    }
    auto &[ld, st] = it->second;
    if (ld.getMemref() == st.getMemref()) {
      return false;
    }
    return !needsDMA(ld.getMemref(), ld.getLocalBuffer()) &&
           !needsDMA(st.getMemref(), st.getLocalBuffer());
  };

  for (RemoteLoadOp loadOp : loads) {
    if (loadOp.isExplicitCBForm()) {
      continue;
    }

    Value memref = loadOp.getMemref();
    bool l1Pair = isL1ToL1Pair(loadOp.getLocalBuffer());
    if (!needsDMA(memref, loadOp.getLocalBuffer()) && !l1Pair) {
      // Aliased op does not need DMA; drop uses and mark for erasure.
      // Still record the buffer-CB mapping
      if (Value localBuffer = loadOp.getLocalBuffer()) {
        localBufferToCB[localBuffer] =
            findAssociatedCB(loadOp, memref, rewriter, cache, portCounters);
      }
      if (loadOp.getResult()) {
        loadOp.getResult().dropAllUses();
      }
      toErase.insert(loadOp);
      continue;
    }

    // If this load shares a buffer with a store (shared pair), use the
    // store's (output) memref for CB lookup so both ops share the same port.
    // L1-to-L1 pairs use the load's own (input) memref since the store side
    // also redirects to the input memref for aliased loads.
    Value cbMemref = memref;
    auto pairIt = sharedPairs.find(loadOp.getLocalBuffer());
    if (pairIt != sharedPairs.end() && !l1Pair) {
      cbMemref = pairIt->second.second.getMemref();
    }

    Value cb =
        findAssociatedCB(loadOp, cbMemref, rewriter, cache, portCounters);
    if (!cb) {
      return loadOp.emitError("could not find associated CB for DMA load");
    }

    rewriter.setInsertionPoint(loadOp);
    auto newLoad = rewriter.create<RemoteLoadOp>(
        loadOp.getLoc(), memref, loadOp.getIndices(), cb,
        loadOp.getMcastStartIndex(), loadOp.getMcastShape());
    // Preserve preallocated semaphore indices set by
    // D2MPreallocateMcastSemaphores (needed by LowerLoadStoreOpsToDMA).
    if (auto semAttr = loadOp->getAttr("preallocated_semaphores")) {
      newLoad->setAttr("preallocated_semaphores", semAttr);
    }

    // Record the buffer-to-CB mapping so local_copy ops can find
    // the CB that holds their src data.
    if (Value localBuffer = loadOp.getLocalBuffer()) {
      localBufferToCB[localBuffer] = cb;
    }

    toErase.insert(loadOp);
  }

  for (RemoteStoreOp storeOp : stores) {
    if (storeOp.isExplicitCBForm()) {
      continue;
    }

    Value memref = storeOp.getMemref();
    if (!needsDMA(memref, storeOp.getLocalBuffer()) &&
        !isL1ToL1Pair(storeOp.getLocalBuffer())) {
      // Aliased op does not need DMA; drop uses and mark for erasure.
      if (storeOp.getResult()) {
        storeOp.getResult().dropAllUses();
      }
      toErase.insert(storeOp);
      continue;
    }

    // If this store shares a buffer with a load (shared pair), use the
    // load's (input) CB when the load is aliased, to match compute.
    // If not, use store op's own output CB (default case).
    Value cbMemref = memref;
    auto storePairIt = sharedPairs.find(storeOp.getLocalBuffer());
    if (storePairIt != sharedPairs.end()) {
      RemoteLoadOp pairedLoad = storePairIt->second.first;
      if (!needsDMA(pairedLoad.getMemref(), pairedLoad.getLocalBuffer())) {
        // Load is aliased so use its (input) CB.
        cbMemref = pairedLoad.getMemref();
      }
    }

    Value cb =
        findAssociatedCB(storeOp, cbMemref, rewriter, cache, portCounters);
    if (!cb) {
      return storeOp.emitError("could not find associated CB for DMA store");
    }

    rewriter.setInsertionPoint(storeOp);
    rewriter.create<RemoteStoreOp>(
        storeOp.getLoc(), memref, storeOp.getIndices(), cb,
        storeOp.getStartDevice(), storeOp.getDeviceMcastShape(),
        storeOp.getSemaphore(), storeOp.getSemaphoreIndices());
    toErase.insert(storeOp);
  }

  // Convert implicit-form local_copy ops to explicit CB form.
  for (LocalCopyOp copyOp : localCopies) {
    Location loc = copyOp.getLoc();

    // Find the source CB.  Check localBufferToCB first (populated during
    // remote_load conversion for both streaming and aliased loads), then
    // fall back to findAssociatedCB (traces to generic operand).
    Value srcCb = localBufferToCB.lookup(copyOp.getSrc());
    if (!srcCb) {
      srcCb = findAssociatedCB(copyOp, copyOp.getSrc(), rewriter, cache,
                               portCounters);
    }
    if (!srcCb) {
      return copyOp.emitError("could not find source CB for DMA local_copy");
    }

    // Insert wait on the source CB to produce a readable memref.
    rewriter.setInsertionPoint(copyOp);
    auto srcWait = rewriter.create<WaitOp>(loc, srcCb);

    // Find the destination CB.
    Value dstCb = findAssociatedCB(copyOp, copyOp.getDst(), rewriter, cache,
                                   portCounters);
    if (!dstCb) {
      return copyOp.emitError(
          "could not find destination CB for DMA local_copy");
    }

    // Create explicit CB form: local_copy %src_memref into %dstCb.
    rewriter.create<LocalCopyOp>(loc, TypeRange{}, srcWait.getResult(),
                                 /*dst=*/Value{}, dstCb,
                                 copyOp.getIndexingMaps());
    toErase.insert(copyOp);
  }

  return success();
}

// ---------------------------------------------------------------------------
// Dead-op cleanup
// ---------------------------------------------------------------------------

// Recursively collect ops to erase from a block based on thread type.
static void collectOpsToErase(Block *block, DenseSet<Operation *> &eraseSet,
                              bool isDatamovementThread) {
  for (Operation &op : block->getOperations()) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      collectOpsToErase(forOp.getBody(), eraseSet, isDatamovementThread);
      continue;
    }

    bool isDMAOp = isa<ShardDMAOpInterface, DeviceSynchronizeOp>(&op);
    bool isReplicated = isa<SemaphoreWaitOp>(&op);

    if (isDatamovementThread && !isDMAOp && !isReplicated) {
      eraseSet.insert(&op);
    } else if (!isDatamovementThread && isDMAOp) {
      eraseSet.insert(&op);
    }
  }
}

// Iteratively erase unused ops from a block until fixpoint.
static void eraseDeadOps(PatternRewriter &rewriter, Block *block,
                         bool isDatamovementThread) {
  bool changed = true;
  while (changed) {
    changed = false;
    DenseSet<Operation *> eraseSet;
    collectOpsToErase(block, eraseSet, isDatamovementThread);

    SmallVector<Operation *> toErase;
    for (Operation *op : eraseSet) {
      if (op->use_empty()) {
        toErase.push_back(op);
        changed = true;
      }
    }
    for (Operation *op : llvm::reverse(toErase)) {
      rewriter.eraseOp(op);
    }
  }
}

// Erase collected ops. All legitimate uses must have been replaced or dropped
// before adding ops to this set, so we just drop any stale uses and erase.
static void eraseCollectedOps(PatternRewriter &rewriter,
                              DenseSet<Operation *> &ops) {
  for (Operation *op : ops) {
    op->dropAllUses();
  }
  for (Operation *op : ops) {
    rewriter.eraseOp(op);
  }
  ops.clear();
}

// ---------------------------------------------------------------------------
// Main rewriter
// ---------------------------------------------------------------------------

class D2MSplitUnifiedThreadRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getNumRegions() != 1) {
      return failure();
    }
    if (generic.getRegionThreadType(0) != ThreadType::Unified) {
      return failure();
    }

    Region &originalRegion = generic.getRegion(0);
    if (originalRegion.empty()) {
      return failure();
    }
    Block *originalBlock = &originalRegion.front();

    if (failed(utils::checkForIllegalSemaphoreOps(originalBlock))) {
      return failure();
    }

    // Create new 2-region GenericOp: datamovement + compute.
    auto newGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getAdditionalArgs(), generic.getGrid(),
        generic.getBlockFactors(), generic.getIndexingMaps(),
        generic.getIteratorTypes(),
        rewriter.getArrayAttr(
            {rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement),
             rewriter.getAttr<ThreadAttr>(ThreadType::Compute)}),
        generic.getFabricConnectionConfigAttr(),
        /*numRegions*/ 2);

    Block *dmBlock = &newGeneric.getRegion(0).emplaceBlock();
    Block *computeBlock = &newGeneric.getRegion(1).emplaceBlock();

    // Map semaphore block arguments to both new blocks.
    IRMapping dmMapping, computeMapping;
    for (unsigned i = 0; i < originalBlock->getNumArguments(); ++i) {
      BlockArgument arg = originalBlock->getArgument(i);
      assert(mlir::isa<d2m::SemaphoreType>(arg.getType()) &&
             "region block arguments must be of semaphore type");
      dmMapping.map(arg, dmBlock->addArgument(arg.getType(), generic.getLoc()));
      computeMapping.map(
          arg, computeBlock->addArgument(arg.getType(), generic.getLoc()));
    }

    // Clone all ops into both regions.
    rewriter.setInsertionPointToStart(dmBlock);
    for (Operation &op : originalBlock->without_terminator()) {
      rewriter.clone(op, dmMapping);
    }
    rewriter.setInsertionPointToStart(computeBlock);
    for (Operation &op : originalBlock->without_terminator()) {
      rewriter.clone(op, computeMapping);
    }
    if (originalBlock->mightHaveTerminator()) {
      Operation *term = originalBlock->getTerminator();
      rewriter.setInsertionPointToEnd(dmBlock);
      rewriter.clone(*term, dmMapping);
      rewriter.setInsertionPointToEnd(computeBlock);
      rewriter.clone(*term, computeMapping);
    }

    // Shared port counter ensures matching CB ports across threads.
    // Separate caches since SSA Values can't cross regions.
    PortCounter portCounters;
    CBCache computeCache, dmaCache;

    // Collect ops for deferred erasure instead of erasing inline.
    DenseSet<Operation *> toErase;

    // Compute thread: insert CB sync ops for implicit-form DMA ops.
    // local_copy must be processed before stores so that findAssociatedCB
    // can still find the destination buffer on the generic's operand list
    // (processComputeStores may replace it with a ReserveOp result).
    if (failed(processSharedBufferPairs(computeBlock, rewriter, computeCache,
                                        portCounters, toErase)) ||
        failed(processComputeLoads(computeBlock, rewriter, computeCache,
                                   portCounters, toErase)) ||
        failed(processComputeLocalCopies(computeBlock, rewriter, computeCache,
                                         portCounters, toErase)) ||
        failed(processComputeStores(computeBlock, rewriter, computeCache,
                                    portCounters, toErase))) {
      return failure();
    }

    // DMA thread: convert datamovement ops to explicit CB form.
    if (failed(convertDMAToExplicitCBForm(dmBlock, rewriter, dmaCache,
                                          portCounters, toErase))) {
      return failure();
    }

    eraseCollectedOps(rewriter, toErase);
    eraseDeadOps(rewriter, dmBlock, /*isDatamovementThread=*/true);
    eraseDeadOps(rewriter, computeBlock, /*isDatamovementThread=*/false);

    rewriter.replaceOp(generic, newGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class D2MSplitUnifiedThread
    : public impl::D2MSplitUnifiedThreadBase<D2MSplitUnifiedThread> {
public:
  using impl::D2MSplitUnifiedThreadBase<
      D2MSplitUnifiedThread>::D2MSplitUnifiedThreadBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MSplitUnifiedThreadRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
