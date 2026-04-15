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
//static Operation *findLastUseOfAliasedValue(Value value, Block *block) {
//  Operation *lastUse = nullptr;
//
//  // Build alias set: the value itself + any view-like derivations.
//  llvm::SmallPtrSet<Value, 8> aliasedValues;
//  aliasedValues.insert(value);
//
//  bool changed = true;
//  while (changed) {
//    changed = false;
//    for (Operation &op : *block) {
//      bool takesAliasedInput = false;
//      for (OpOperand &operand : op.getOpOperands()) {
//        if (aliasedValues.contains(operand.get())) {
//          takesAliasedInput = true;
//          break;
//        }
//      }
//      if (takesAliasedInput && mlir::isa<mlir::ViewLikeOpInterface>(op)) {
//        for (Value result : op.getResults()) {
//          if (aliasedValues.insert(result).second) {
//            changed = true;
//          }
//        }
//      }
//    }
//  }
//
//  // Recursive check for uses in nested regions.
//  std::function<bool(Region &)> isUsedInRegion = [&](Region &region) -> bool {
//    for (Block &regionBlock : region) {
//      for (Operation &op : regionBlock) {
//        for (OpOperand &operand : op.getOpOperands()) {
//          if (aliasedValues.contains(operand.get())) {
//            return true;
//          }
//        }
//        for (Region &nestedRegion : op.getRegions()) {
//          if (isUsedInRegion(nestedRegion)) {
//            return true;
//          }
//        }
//      }
//    }
//    return false;
//  };
//
//  // Walk top-level ops to find the last one that uses any alias.
//  for (Operation &op : *block) {
//    bool opUsesValue = false;
//    for (OpOperand &operand : op.getOpOperands()) {
//      if (aliasedValues.contains(operand.get())) {
//        opUsesValue = true;
//        break;
//      }
//    }
//    if (!opUsesValue) {
//      for (Region &region : op.getRegions()) {
//        if (isUsedInRegion(region)) {
//          opUsesValue = true;
//          break;
//        }
//      }
//    }
//    if (opUsesValue) {
//      lastUse = &op;
//    }
//  }
//
//  return lastUse;
//}

// Insert a pop before the block terminator.
//static void insertPopBeforeTerminator(PatternRewriter &rewriter, Location loc,
//                                      Value cb, Block *block) {
//  if (block->mightHaveTerminator()) {
//    rewriter.setInsertionPoint(block->getTerminator());
//  } else {
//    rewriter.setInsertionPointToEnd(block);
//  }
//  rewriter.create<PopOp>(loc, cb);
//}

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
//static bool isLocalAlloc(memref::AllocOp allocOp, Block *block) {
//  return block->getParent()->isAncestor(allocOp->getParentRegion());
//}

// ---------------------------------------------------------------------------
// Compute thread: insert CB sync ops for implicit-form remote_load/store
// ---------------------------------------------------------------------------

// Handle load-store pairs that share the same local buffer (DMA-only
// generics that copy input->output with no compute in between). The shared
// buffer means one CB serves both ops.
static LogicalResult processSharedBufferPairs(Block *computeBlock,
                                              PatternRewriter &rewriter,
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
      Value cb = getCB(storeOp, storeOp.getLocalBuffer(), rewriter);
      if (!cb) {
        return storeOp.emitError(
            "could not find associated CB for shared pair");
      }
      rewriter.setInsertionPoint(storeOp);
      rewriter.create<WaitOp>(loc, cb);
      rewriter.create<PopOp>(loc, cb);
    } else if (!loadNeedsDMA && storeNeedsDMA) {
      Value cb = getCB(loadOp, loadOp.getLocalBuffer(), rewriter);
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

// local buffer; cb ops may already have been inserted so need to handle that case
// TODO: cleanup
Value getCBGenericOperand(GenericOp genericOp, Value cbGenericOperand) {
  if (auto reserveOp = cbGenericOperand.getDefiningOp<ReserveOp>()) {
    return genericOp.getOperands()[reserveOp.getCb().getDefiningOp<GetCBOp>().getCbOperandIdx()];
  }
  else if (auto waitOp = cbGenericOperand.getDefiningOp<WaitOp>()) {
    return genericOp.getOperands()[waitOp.getCb().getDefiningOp<GetCBOp>().getCbOperandIdx()];
  }
  return cbGenericOperand;
}

// Process implicit-form remote_load ops in the compute thread.
static LogicalResult processComputeLoads(Block *computeBlock,
                                         PatternRewriter &rewriter,
                                         DenseSet<Operation *> &toErase,
                                         llvm::DenseMap<Value, CBUsageInfo> &cbUsageInfo) {
  SmallVector<RemoteLoadOp> loads;

  computeBlock->walk([&](AliasedLoadOp loadOp) {
    if (toErase.contains(loadOp)) {
      return WalkResult::advance();
    }

    llvm::errs() << "aliased load op: " << loadOp.getAliasedBuffer() << "\n";

    Location loc = loadOp.getLoc();
    Value localBuffer = getCBGenericOperand(loadOp->getParentOfType<GenericOp>(), loadOp.getAliasedBuffer());
    Value cb = getCB(loadOp, localBuffer, rewriter);
    assert(cb && "could not find associated CB for load");

    // Assumes only one consumer for this local buffer
    auto consumer = cbUsageInfo[localBuffer].consumers.front();

    llvm::errs() << "consumer with aliased load op: " << *consumer << "\n";

    rewriter.setInsertionPoint(consumer);
    rewriter.create<ReserveOp>(loc, cb);
    rewriter.create<PushOp>(loc, cb);
    auto waitOp = rewriter.create<WaitOp>(loc, cb);
    rewriter.setInsertionPointAfter(consumer);
    rewriter.create<PopOp>(loc, cb);

    // Replace uses of the local buffer in compute consumer
    localBuffer.replaceUsesWithIf(waitOp.getResult(), [&](OpOperand &use) {
      return use.getOwner() == consumer;
    });

    llvm::errs() << "inserting reserve/push/wait/pop before consumer: " << consumer->getResults().front() << "\n";

    toErase.insert(loadOp);
    return WalkResult::advance();
  });
    
  // Handle remote load
  computeBlock->walk([&](RemoteLoadOp loadOp) {
    if (loadOp.isExplicitCBForm() || toErase.contains(loadOp)) {
      return WalkResult::advance();
    }

    // Needs datamovement: wait before load, pop before terminator.
    Location loc = loadOp.getLoc();
    Value localBuffer = getCBGenericOperand(loadOp->getParentOfType<GenericOp>(), loadOp.getLocalBuffer());
    Value cb = getCB(loadOp, localBuffer, rewriter);
    assert(cb && "could not find associated CB for load");

    auto consumer = cbUsageInfo[localBuffer].consumers.front();
    rewriter.setInsertionPoint(consumer);
    auto waitOp = rewriter.create<WaitOp>(loadOp.getLoc(), cb);
    rewriter.setInsertionPointAfter(consumer);
    rewriter.create<PopOp>(loc, cb);

    rewriter.replaceUsesWithIf(
      localBuffer, waitOp.getResult(), [&](OpOperand &use) {
        return consumer == use.getOwner();
      });
    
    toErase.insert(loadOp);
    return WalkResult::advance();
  });
  
  return success();
}

static LogicalResult processComputeStores(Block *computeBlock,
                                         PatternRewriter &rewriter,
                                         DenseSet<Operation *> &toErase,
                                         llvm::DenseMap<Value, CBUsageInfo> &cbUsageInfo) {
  SmallVector<RemoteLoadOp> loads;

  computeBlock->walk([&](AliasedStoreOp storeOp) {
    if (toErase.contains(storeOp)) {
      WalkResult::advance();
    }

    Location loc = storeOp.getLoc();
    Value localBuffer = getCBGenericOperand(storeOp->getParentOfType<GenericOp>(), storeOp.getAliasedBuffer());
    Value cb = getCB(storeOp, localBuffer, rewriter);
    assert(cb && "could not find associated CB for load");

    // Assumes only one consumer for this local buffer
    auto producer = cbUsageInfo[localBuffer].producers.front();
    rewriter.setInsertionPoint(producer);
    auto reserveOp = rewriter.create<ReserveOp>(loc, cb);
    rewriter.setInsertionPointAfter(producer);
    rewriter.create<PushOp>(loc, cb);
    rewriter.create<WaitOp>(loc, cb);
    rewriter.create<PopOp>(loc, cb);

    // Replace uses of the local buffer in compute consumer
    localBuffer.replaceUsesWithIf(reserveOp.getResult(), [&](OpOperand &use) {
      return use.getOwner() == producer;
    });

    toErase.insert(storeOp);
    return WalkResult::advance();
  });
    
  // Handle remote store
  computeBlock->walk([&](RemoteStoreOp storeOp) {
    if (storeOp.isExplicitCBForm() || toErase.contains(storeOp)) {
      WalkResult::advance();
    }

    // Needs datamovement: wait before store, pop before terminator.
    Location loc = storeOp.getLoc();
    
    Value localBuffer = getCBGenericOperand(storeOp->getParentOfType<GenericOp>(), storeOp.getLocalBuffer());
    llvm::errs() << "local buffer: " << localBuffer << "\n";
    Value cb = getCB(storeOp, localBuffer, rewriter);
    assert(cb && "could not find associated CB for store");

    auto producer = cbUsageInfo[localBuffer].producers.front();
    rewriter.setInsertionPoint(producer);
    auto reserveOp = rewriter.create<ReserveOp>(storeOp.getLoc(), cb);
    rewriter.setInsertionPointAfter(producer);
    rewriter.create<PushOp>(loc, cb);

    rewriter.replaceUsesWithIf(
      localBuffer, reserveOp.getResult(), [&](OpOperand &use) {
        return producer == use.getOwner();
      });

    toErase.insert(storeOp);
    return WalkResult::advance();
  });
  
  return success();
}

// Replace GetScratchFromCBOp with reserve.
static void processGetScratchOps(Block *computeBlock, PatternRewriter &rewriter,
                                 DenseSet<Operation *> &toErase) {
  SmallVector<GetScratchFromCBOp> ops;
  computeBlock->walk([&](GetScratchFromCBOp op) { ops.push_back(op); });

  for (GetScratchFromCBOp op : ops) {
    rewriter.setInsertionPoint(op);
    auto reserveOp = rewriter.create<ReserveOp>(op.getLoc(), op.getCb());
    rewriter.replaceAllUsesWith(op.getResult(), reserveOp.getResult());
    toErase.insert(op);
  }
}

// ---------------------------------------------------------------------------
// DMA thread: convert implicit-form ops to explicit CB form
// ---------------------------------------------------------------------------

// Convert remote_load/store to explicit CB form in the DMA thread.
// Aliased ops are collected for deferred erasure (no DMA needed). Shared
// buffer pairs use the output operand's CB for both ops.
static LogicalResult
convertDMAToExplicitCBForm(Block *dmBlock, PatternRewriter &rewriter,
                           DenseSet<Operation *> &toErase) {
  SmallVector<RemoteLoadOp> loads;
  SmallVector<RemoteStoreOp> stores;
  dmBlock->walk([&](RemoteLoadOp op) { loads.push_back(op); });
  dmBlock->walk([&](RemoteStoreOp op) { stores.push_back(op); });

  for (RemoteLoadOp loadOp : loads) {
    if (loadOp.isExplicitCBForm()) {
      continue;
    }

    // removed  needsDMA and isL1ToL1Pair checks tocheck for aliased load/store with just checking aliased load/store
    // TODO: l1-to-l1 pair is equivalent to checking aliased load + aliased store, whcih we need to add a case for in allocate
    // where we are currently converting remote loads and stores to aliased load and store

    
    Value localBuffer = getCBGenericOperand(loadOp->getParentOfType<GenericOp>(), loadOp.getLocalBuffer());
    llvm::errs() << "cbMemref: " << localBuffer << "\n";
    Value cb = getCB(loadOp, localBuffer, rewriter);
    assert(cb && "could not find associated CB for load");

    rewriter.setInsertionPoint(loadOp);
    auto newLoad = rewriter.create<RemoteLoadOp>(
        loadOp.getLoc(), loadOp.getMemref(), loadOp.getIndices(), cb,
        loadOp.getMcastStartIndex(), loadOp.getMcastShape());
    // Preserve preallocated semaphore indices set by
    // D2MPreallocateMcastSemaphores (needed by LowerLoadStoreOpsToDMA).
    if (auto semAttr = loadOp->getAttr("preallocated_semaphores")) {
      newLoad->setAttr("preallocated_semaphores", semAttr);
    }
    toErase.insert(loadOp);
  }

  for (RemoteStoreOp storeOp : stores) {
    if (storeOp.isExplicitCBForm()) {
      continue;
    }

    // TODO:similar comment as for loadOp (see above)
    Value localBuffer = getCBGenericOperand(storeOp->getParentOfType<GenericOp>(), storeOp.getLocalBuffer());
    assert(localBuffer && "could not find associated local buffer for store");
    Value cb = getCB(storeOp, localBuffer, rewriter);
    assert(cb && "could not find associated CB for store");

    rewriter.setInsertionPoint(storeOp);
    rewriter.create<RemoteStoreOp>(
        storeOp.getLoc(), storeOp.getMemref(), storeOp.getIndices(), cb,
        storeOp.getStartDevice(), storeOp.getDeviceMcastShape(),
        storeOp.getSemaphore(), storeOp.getSemaphoreIndices());
    toErase.insert(storeOp);
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

    // Collect ops for deferred erasure instead of erasing inline.
    DenseSet<Operation *> toErase;

    // Compute thread: insert CB sync ops for implicit-form remote ops.
    auto cbUsageInfoCompute = getCBUsageInfo(newGeneric.getRegion(1));
    // processSharedBufferPairs no longer needed since we have explicit alias ops
    // TODO: remove processSharedBufferPairs
    if (
        failed(processSharedBufferPairs(computeBlock, rewriter, toErase)) ||
        failed(processComputeLoads(computeBlock, rewriter, toErase, cbUsageInfoCompute)) ||
        failed(processComputeStores(computeBlock, rewriter, toErase, cbUsageInfoCompute))) {
      return failure();
    }
    processGetScratchOps(computeBlock, rewriter, toErase);

    // DMA thread: convert datamovement ops to explicit CB form.
    if (failed(convertDMAToExplicitCBForm(dmBlock, rewriter, toErase))) {
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
