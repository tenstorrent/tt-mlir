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

// A streaming op requires real DMA. Returns true if the remote memref
// implies data movement (view layout, DRAM) or the local buffer is a
// streaming CB (CBLayoutAttr). Aliased ops (plain L1 buffers accessing
// L1 shard operands) need no DMA.
static bool isStreamingOp(Value memref, Value localBuffer) {
  // Check if the remote memref implies data movement.
  if (auto *defOp = memref.getDefiningOp()) {
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
        pairs.push_back({loadOp, storeOp});
        return;
      }
    }
  });
  return pairs;
}

// ---------------------------------------------------------------------------
// Compute thread: insert CB sync ops for implicit-form remote_load/store
// ---------------------------------------------------------------------------

// Handle load-store pairs that share the same local buffer (DMA-only
// generics that copy input->output with no compute in between). The shared
// buffer means one CB serves both ops. Erases the paired ops from compute.
static void processSharedBufferPairs(Block *computeBlock,
                                     PatternRewriter &rewriter, CBCache &cache,
                                     PortCounter &portCounters) {
  auto pairs = findSharedBufferPairs(computeBlock);

  for (auto [loadOp, storeOp] : pairs) {
    Value sharedBuffer = loadOp.getLocalBuffer();
    bool loadIsStreaming = isStreamingOp(loadOp.getMemref(), sharedBuffer);
    bool storeIsStreaming = isStreamingOp(storeOp.getMemref(), sharedBuffer);

    // Neither streaming (L1-to-L1 layout conversion in TTNN mode): both ops
    // need actual data movement through a shared CB.
    if (!loadIsStreaming && !storeIsStreaming) {
      continue;
    }

    // Insert compute-side CB ops for the aliased half of the pair.
    // The streaming half stays as a remote_load/store for DMA.
    Location loc = loadOp.getLoc();
    if (loadIsStreaming && !storeIsStreaming) {
      Value cb = findAssociatedCB(storeOp, storeOp.getMemref(), rewriter, cache,
                                  portCounters);
      if (!cb) {
        storeOp.emitWarning("could not find associated CB for shared pair");
        continue;
      }
      rewriter.setInsertionPoint(storeOp);
      rewriter.create<WaitOp>(loc, cb);
      rewriter.create<PopOp>(loc, cb);
    } else if (!loadIsStreaming && storeIsStreaming) {
      Value cb = findAssociatedCB(loadOp, loadOp.getMemref(), rewriter, cache,
                                  portCounters);
      if (!cb) {
        loadOp.emitWarning("could not find associated CB for shared pair");
        continue;
      }
      rewriter.setInsertionPoint(loadOp);
      rewriter.create<ReserveOp>(loc, cb);
      rewriter.create<PushOp>(loc, cb);
    }
    // Both remote -> both go to DMA, nothing in compute.
    // Buffer aliasing handled in convertDMAToExplicitCBForm.

    memref::AllocOp allocToErase = findAllocOp(loadOp.getLocalBuffer());
    rewriter.eraseOp(storeOp);
    rewriter.eraseOp(loadOp);
    if (allocToErase && allocToErase->use_empty()) {
      rewriter.eraseOp(allocToErase);
    }
  }
}

// Process implicit-form remote_load ops in the compute thread.
static void processComputeLoads(Block *computeBlock, PatternRewriter &rewriter,
                                CBCache &cache, PortCounter &portCounters) {
  SmallVector<RemoteLoadOp> loads;
  computeBlock->walk([&](RemoteLoadOp op) {
    if (!op.isExplicitCBForm()) {
      loads.push_back(op);
    }
  });

  for (RemoteLoadOp loadOp : loads) {
    Location loc = loadOp.getLoc();
    Value memref = loadOp.getMemref();
    Value localBuffer = loadOp.getLocalBuffer();
    Value cb = findAssociatedCB(loadOp, memref, rewriter, cache, portCounters);
    if (!cb) {
      loadOp.emitWarning("could not find associated CB, skipping conversion");
      continue;
    }

    if (!isStreamingOp(memref, localBuffer)) {
      // Aliased loads should not have multicast parameters.
      if (loadOp.isMcast()) {
        loadOp.emitWarning(
            "remote_load with local operand has multicast parameters, "
            "skipping aliased conversion");
        continue;
      }
      // Aliased: reserve->push->wait at alloc, pop after last use.
      memref::AllocOp allocOp = findAllocOp(localBuffer);
      if (!allocOp) {
        loadOp.emitWarning(
            "could not find memref.alloc for local buffer, skipping");
        continue;
      }

      Block *block = allocOp->getBlock();
      rewriter.setInsertionPoint(allocOp);
      rewriter.create<ReserveOp>(loc, cb);
      rewriter.create<PushOp>(loc, cb);
      auto waitOp = rewriter.create<WaitOp>(loc, cb);

      rewriter.replaceAllUsesWith(allocOp.getResult(), waitOp.getResult());
      if (loadOp.getResult()) {
        rewriter.replaceAllUsesWith(loadOp.getResult(), waitOp.getResult());
      }
      rewriter.eraseOp(allocOp);

      // Insert pop after the last use of the waited value, or before
      // the block terminator if no uses found.
      Operation *lastUse = findLastUseOfAliasedValue(waitOp.getResult(), block);
      if (lastUse && lastUse != loadOp.getOperation()) {
        rewriter.setInsertionPointAfter(lastUse);
        rewriter.create<PopOp>(loc, cb);
      } else {
        insertPopBeforeTerminator(rewriter, loc, cb, block);
      }
      rewriter.eraseOp(loadOp);
    } else {
      // Streaming: wait before load, pop before terminator.
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

      insertPopBeforeTerminator(rewriter, loc, cb, loadOp->getBlock());
      rewriter.eraseOp(loadOp);
    }
  }
}

// Process implicit-form remote_store ops in the compute thread.
static void processComputeStores(Block *computeBlock, PatternRewriter &rewriter,
                                 CBCache &cache, PortCounter &portCounters) {
  SmallVector<RemoteStoreOp> stores;
  computeBlock->walk([&](RemoteStoreOp op) {
    if (!op.isExplicitCBForm()) {
      stores.push_back(op);
    }
  });

  for (RemoteStoreOp storeOp : stores) {
    Location loc = storeOp.getLoc();
    Value memref = storeOp.getMemref();
    Value localBuffer = storeOp.getLocalBuffer();
    if (!localBuffer) {
      storeOp.emitWarning(
          "remote_store does not have a local buffer operand, skipping");
      continue;
    }

    Value cb = findAssociatedCB(storeOp, memref, rewriter, cache, portCounters);
    if (!cb) {
      storeOp.emitWarning("could not find associated CB, skipping conversion");
      continue;
    }

    if (!isStreamingOp(memref, localBuffer)) {
      // Aliased: replace alloc->reserve, insert push+wait+pop at store.
      memref::AllocOp allocOp = findAllocOp(localBuffer);
      if (allocOp) {
        rewriter.setInsertionPoint(allocOp);
        auto reserveOp = rewriter.create<ReserveOp>(loc, cb);
        rewriter.replaceAllUsesWith(allocOp.getResult(), reserveOp.getResult());
        rewriter.eraseOp(allocOp);
      } else if (auto waitOp = localBuffer.getDefiningOp<WaitOp>()) {
        // The alloc was already replaced by a WaitOp from an earlier load
        // that shared this buffer (L1-to-L1 pair). The CB is already set up;
        // just insert the store-side sync ops.
      } else {
        storeOp.emitWarning(
            "could not find memref.alloc for local buffer, skipping");
        continue;
      }

      rewriter.setInsertionPoint(storeOp);
      rewriter.create<PushOp>(loc, cb);
      rewriter.create<WaitOp>(loc, cb);
      rewriter.create<PopOp>(loc, cb);
      rewriter.eraseOp(storeOp);
    } else {
      // Streaming: insert reserve + push, replace in-region buffer uses.
      // Reserve must dominate all uses of localBuffer in its block. If the
      // CB def is in the same block as the store, insert after it. Otherwise
      // (eg: inside a loop), insert at the start of the store's block.
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
      rewriter.eraseOp(storeOp);
    }
  }
}

// Replace GetScratchFromCBOp with reserve.
static void processGetScratchOps(Block *computeBlock,
                                 PatternRewriter &rewriter) {
  SmallVector<GetScratchFromCBOp> ops;
  computeBlock->walk([&](GetScratchFromCBOp op) { ops.push_back(op); });

  for (GetScratchFromCBOp op : ops) {
    rewriter.setInsertionPoint(op);
    auto reserveOp = rewriter.create<ReserveOp>(op.getLoc(), op.getCb());
    rewriter.replaceAllUsesWith(op.getResult(), reserveOp.getResult());
    rewriter.eraseOp(op);
  }
}

// ---------------------------------------------------------------------------
// DMA thread: convert implicit-form ops to explicit CB form, erase aliased
// ---------------------------------------------------------------------------

// Convert streaming remote_load/store to explicit CB form in the DMA thread.
// Aliased ops are erased (no DMA needed). Shared buffer pairs use the output
// operand's CB for both ops.
static void convertDMAToExplicitCBForm(Block *dmBlock,
                                       PatternRewriter &rewriter,
                                       CBCache &cache,
                                       PortCounter &portCounters) {
  DenseMap<Operation *, RemoteStoreOp> sharedPairMap;
  for (auto [loadOp, storeOp] : findSharedBufferPairs(dmBlock)) {
    sharedPairMap[loadOp.getOperation()] = storeOp;
  }

  SmallVector<RemoteLoadOp> loads;
  SmallVector<RemoteStoreOp> stores;
  dmBlock->walk([&](RemoteLoadOp op) { loads.push_back(op); });
  dmBlock->walk([&](RemoteStoreOp op) { stores.push_back(op); });

  for (RemoteLoadOp loadOp : loads) {
    if (loadOp.isExplicitCBForm()) {
      continue;
    }

    Value memref = loadOp.getMemref();
    if (!isStreamingOp(memref, loadOp.getLocalBuffer())) {
      // Aliased -> erase from DMA.
      if (loadOp.getResult()) {
        loadOp.getResult().dropAllUses();
      }
      rewriter.eraseOp(loadOp);
      continue;
    }

    // If this load shares a buffer with a store (DMA-only pair), use the
    // output operand's CB for both ops to avoid redundant copying.
    Value cbMemref = memref;
    auto pairIt = sharedPairMap.find(loadOp.getOperation());
    if (pairIt != sharedPairMap.end()) {
      RemoteStoreOp pairedStore = pairIt->second;
      if (isStreamingOp(pairedStore.getMemref(), loadOp.getLocalBuffer())) {
        cbMemref = pairedStore.getMemref();
      }
    }

    // Streaming -> convert to explicit CB form.
    Value cb =
        findAssociatedCB(loadOp, cbMemref, rewriter, cache, portCounters);
    if (!cb) {
      loadOp.emitWarning("could not find associated CB for DMA conversion");
      continue;
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
    if (loadOp.getResult()) {
      loadOp.getResult().dropAllUses();
    }
    rewriter.eraseOp(loadOp);
  }

  for (RemoteStoreOp storeOp : stores) {
    if (storeOp.isExplicitCBForm()) {
      continue;
    }

    Value memref = storeOp.getMemref();
    if (!isStreamingOp(memref, storeOp.getLocalBuffer())) {
      // Aliased -> erase from DMA.
      if (storeOp.getResult()) {
        storeOp.getResult().dropAllUses();
      }
      rewriter.eraseOp(storeOp);
      continue;
    }

    // Streaming -> convert to explicit CB form.
    Value cb = findAssociatedCB(storeOp, memref, rewriter, cache, portCounters);
    if (!cb) {
      storeOp.emitWarning("could not find associated CB for DMA conversion");
      continue;
    }

    rewriter.setInsertionPoint(storeOp);
    rewriter.create<RemoteStoreOp>(
        storeOp.getLoc(), memref, storeOp.getIndices(), cb,
        storeOp.getStartDevice(), storeOp.getDeviceMcastShape(),
        storeOp.getSemaphore(), storeOp.getSemaphoreIndices());
    if (storeOp.getResult()) {
      storeOp.getResult().dropAllUses();
    }
    rewriter.eraseOp(storeOp);
  }
}

// ---------------------------------------------------------------------------
// Iterative dead-op erasure
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

    bool isDMAOp = isa<RemoteLoadOp, RemoteStoreOp, DeviceSynchronizeOp>(&op);
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
        generic.getScratchInputsAttr(), generic.getFabricConnectionConfigAttr(),
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

    // Compute thread: insert CB sync ops for implicit-form remote ops.
    processSharedBufferPairs(computeBlock, rewriter, computeCache,
                             portCounters);
    processComputeLoads(computeBlock, rewriter, computeCache, portCounters);
    processComputeStores(computeBlock, rewriter, computeCache, portCounters);
    processGetScratchOps(computeBlock, rewriter);

    // DMA thread: convert streaming ops to explicit CB form, erase aliased.
    convertDMAToExplicitCBForm(dmBlock, rewriter, dmaCache, portCounters);

    // Clean up: erase non-DMA ops from DMA, non-compute ops from compute.
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
