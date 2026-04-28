// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"
#include "ttmlir/Dialect/D2M/Utils/SynchronizableOpInterfaceUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

// TODO: move into some DMA utils place alongside load/store aliasing logic
// which should be separated out to separate function after
// inferOperandAliasing inferOperandAliasing needs to be run prior to
// assigning memory to actually benefit from reduced mem usage from aliasing
// TODO: add specific checks for load/store (mcast load, fabric store)
// can infer aliasing and do the conversion
static bool needsDMA(Value memref) {
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

  return false;
}

bool isAliasedLoad(RemoteLoadOp loadOp) { return needsDMA(loadOp.getMemref()); }

bool isAliasedStore(RemoteStoreOp storeOp) {
  return needsDMA(storeOp.getMemref());
}

Value traceComputeMemrefToCB(Value value, GenericOp genericOp) {
  llvm::errs() << "tracing value: " << value << "\n";
  while (value) {
    // check if its a cb (hoisted generic arg with cb layout attr),
    if (auto memrefType = mlir::dyn_cast<MemRefType>(value.getType())) {
      // TODO: add back later
      // if (mlir::isa<ttcore::CBLayoutAttr>(memrefType.getLayout())) {
      if (llvm::find(genericOp.getAdditionalArgs(), value) !=
          genericOp.getAdditionalArgs().end()) {
        return value;
      }
    }

    // if we are no longer inside the generic or have reached the root, stop
    // tracing and return nullptr
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || !genericOp->isProperAncestor(definingOp)) {
      llvm::errs() << "definingOp is not a proper ancestor of genericOp: "
                   << *definingOp << "\n";
      return nullptr;
    }

    // Otherwise keep tracing up the chain, if we reach an op we don't support,
    // stop tracing and return nullptr
    if (auto subviewOp = mlir::dyn_cast<memref::CollapseShapeOp>(definingOp)) {
      value = subviewOp.getSrc();
      continue;
    } else if (auto subviewOp = mlir::dyn_cast<memref::SubViewOp>(definingOp)) {
      value = subviewOp.getSource();
      continue;
    } else {
      llvm::errs() << "definingOp is not a collapse_shape or subview op: "
                   << *definingOp << "\n";
      return nullptr;
    }
  }
  llvm::errs() << "value is not a cb: " << value << "\n";
  return nullptr;
}

LogicalResult wrapComputeInSynchronizedRegion(GenericOp genericOp,
                                              PatternRewriter &rewriter) {
  // Look for a D2M_GenericRegionComputeOp, and collect the outermost ops that
  // contain them in the generic op
  // Skip ops that have the SynchronizableOpInterface::Trait,
  // such as TileTilizeBlockOp and TileUntilizeBlockOp ops since
  // they haven't been lowered yet into non-synchronized ops
  OpBuilder::InsertionGuard guard(rewriter);
  DenseSet<Operation *> outermostOps;
  genericOp.getRegion(0).walk([&](Operation *op) {
    if (!op->hasTrait<D2MGenericRegionComputeOpTrait>()) {
      return WalkResult::advance();
    }

    // Go up loops until we reach one of the two as a parent:
    // generic op or scf.for tagged as d2m.blocking_loop
    Operation *outermostOp = op;
    auto isBlockingLoop = [](Operation *op) {
      return mlir::isa<scf::ForOp>(op) && op->hasAttr("d2m.blocking_loop");
    };
    while (outermostOp->getParentOp() != genericOp.getOperation() &&
           !isBlockingLoop(outermostOp->getParentOp())) {
      outermostOp = outermostOp->getParentOp();
      if (!mlir::isa<scf::ForOp>(outermostOp) &&
          !mlir::isa<linalg::GenericOp>(outermostOp)) {
        llvm::errs() << "op " << *outermostOp << "\n";
        assert(false && "outermost loop op is not a scf.for op");
      }
    }

    if (!dyn_cast<SynchronizableOpInterface>(outermostOp)) {
      outermostOps.insert(outermostOp);
    }

    return WalkResult::advance();
  });

  // expand and merge compute regions until we hit a syncrhonizable op on both
  // ends
  SmallVector<std::pair<Block::iterator, Block::iterator>> computeRegions;
  while (!outermostOps.empty()) {
    Operation *outermostOp = *outermostOps.begin();
    outermostOps.erase(outermostOp);
    Block::iterator start = outermostOp->getIterator();
    Block::iterator end = outermostOp->getIterator();

    // expand above
    while (start != outermostOp->getBlock()->begin() &&
           !std::prev(start)->hasTrait<SynchronizableOpInterface::Trait>()) {
      start--;
      if (outermostOps.contains(&*start)) {
        outermostOps.erase(&*start);
      }
    }

    // expand below
    while (std::next(end) != outermostOp->getBlock()->end() &&
           !std::next(end)->hasTrait<SynchronizableOpInterface::Trait>()) {
      end++;
      if (outermostOps.contains(&*end)) {
        outermostOps.erase(&*end);
      }
    }

    computeRegions.push_back({start, std::next(end)});
  }

  for (auto [start, end] : computeRegions) {

    DenseSet<Value> loadedCBOperands;
    DenseSet<Value> storedCBOperands;

    // for memref load and stores, trace to cb operand to get producers and
    // consumers for syncrhonized region
    for (Operation &op : llvm::make_range(start, end)) {
      // for load trace src memref up to defining op and check if its a cb (as
      // opposed to dst)
      op.walk([&](memref::LoadOp loadOp) {
        Value cb = traceComputeMemrefToCB(loadOp.getMemref(), genericOp);
        if (cb) {
          loadedCBOperands.insert(cb);
        }
        return WalkResult::advance();
      });

      // for store trace dst memref up to defining op and check if its a cb (as
      // opposed to dst)
      op.walk([&](memref::StoreOp storeOp) {
        Value cb = traceComputeMemrefToCB(storeOp.getMemref(), genericOp);
        if (cb) {
          storedCBOperands.insert(cb);
        }
        return WalkResult::advance();
      });

      // TileMatmulBlockOp uses CBs directly without load/store
      op.walk([&](d2m::TileMatmulBlockOp tileMatmulBlockOp) {
        Value cbA = traceComputeMemrefToCB(tileMatmulBlockOp.getA(), genericOp);
        Value cbB = traceComputeMemrefToCB(tileMatmulBlockOp.getB(), genericOp);
        Value cbOutput =
            traceComputeMemrefToCB(tileMatmulBlockOp.getOutput(), genericOp);
        if (cbA) {
          loadedCBOperands.insert(cbA);
        }
        if (cbB) {
          loadedCBOperands.insert(cbB);
        }
        if (cbOutput) {
          storedCBOperands.insert(cbOutput);
        }
        return WalkResult::advance();
      });
    }

    llvm::errs() << "\n";
    llvm::errs() << "storedCBOperands: ";
    for (Value storedCBOperand : storedCBOperands) {
      llvm::errs() << storedCBOperand << " ";
    }
    llvm::errs() << "\n";

    // remove allocs in load that are in store since this is output cb reuse and
    // not an actual input
    for (Value storedCBOperand : storedCBOperands) {
      if (loadedCBOperands.contains(storedCBOperand)) {
        loadedCBOperands.erase(storedCBOperand);
      }
    }

    llvm::errs() << "loadedCBOperands after cleanup: ";
    for (Value loadedCBOperand : loadedCBOperands) {
      llvm::errs() << loadedCBOperand << " ";
    }
    llvm::errs() << "\n";
    llvm::errs() << "storedCBOperands after cleanup: ";
    for (Value storedCBOperand : storedCBOperands) {
      llvm::errs() << storedCBOperand << " ";
    }
    llvm::errs() << "\n";
    llvm::errs() << "wrapping in synchronized region\n";
    llvm::errs() << "start: " << *start << "\n";
    llvm::errs() << "end: " << *end << "\n";
    utils::wrapInSynchronizedRegion(
        rewriter, start, end,
        SmallVector<Value>(loadedCBOperands.begin(), loadedCBOperands.end()),
        SmallVector<Value>(storedCBOperands.begin(), storedCBOperands.end()));
  }

  return success();
}

// from cb usage info, check for load-store pairs and insert aliased cb ops for
// alias side
static LogicalResult processSharedBufferPairs(
    Block *computeBlock, PatternRewriter &rewriter,
    llvm::DenseMap<Value, utils::CBUsageInfo> &cbUsageInfo) {
  for (auto [localBuffer, usageInfo] : cbUsageInfo) {
    auto producer = usageInfo.producers.front();
    auto consumer = usageInfo.consumers.front();

    // Insert compute-side CB ops for the aliased half of the pair.
    // The streaming half stays as a remote_load/store for DMA.
    if (mlir::isa<RemoteLoadOp>(producer) &&
        mlir::isa<RemoteStoreOp>(consumer) &&
        isAliasedStore(mlir::cast<RemoteStoreOp>(consumer))) {
      Location loc = producer->getLoc();
      unsigned cbOperandIdx =
          producer->getParentOfType<GenericOp>().getOperandIndex(localBuffer);
      // Set insertion point before consumer so GetCBOp dominates WaitOp/PopOp
      rewriter.setInsertionPoint(consumer);
      auto cb =
          d2m::getOrCreateCB(rewriter, producer->getParentOfType<GenericOp>(),
                             computeBlock, cbOperandIdx);
      rewriter.create<WaitOp>(loc, cb);
      rewriter.create<PopOp>(loc, cb);
    } else if (mlir::isa<RemoteLoadOp>(producer) &&
               mlir::isa<RemoteStoreOp>(consumer) &&
               isAliasedLoad(mlir::cast<RemoteLoadOp>(producer))) {
      Location loc = consumer->getLoc();
      unsigned cbOperandIdx =
          consumer->getParentOfType<GenericOp>().getOperandIndex(localBuffer);
      // Set insertion point before producer so GetCBOp dominates
      // ReserveOp/PushOp
      rewriter.setInsertionPoint(producer);
      auto cb =
          d2m::getOrCreateCB(rewriter, consumer->getParentOfType<GenericOp>(),
                             computeBlock, cbOperandIdx);
      rewriter.create<ReserveOp>(loc, cb);
      rewriter.create<PushOp>(loc, cb);
    }
    // Else if both sides are streaming/need DMA, let the DM thread handle
    // everything.
  }
  return success();
}

static LogicalResult
insertCBOpsForCompute(Block *computeBlock, PatternRewriter &rewriter,
                      llvm::DenseMap<Value, utils::CBUsageInfo> &cbUsageInfo) {
  SmallVector<RemoteLoadOp> loads;

  computeBlock->walk([&](Operation *op) {
    // TODO: see if removed check for explicit CB form
    // (loadOp.isExplicitCBForm() || toErase.contains(loadOp)) causes any issues
    if (op->hasTrait<SynchronizableOpInterface::Trait>() &&
        !op->hasTrait<D2MGenericRegionDatamovementOpTrait>()) {
      auto synchronizedOp = mlir::cast<SynchronizableOpInterface>(op);
      // get consumers and insert wait+pop
      for (auto &operand : synchronizedOp->getOpOperands()) {
        if (synchronizedOp.isConsumer(operand)) {
          Location loc = synchronizedOp.getLoc();
          Value localBuffer = operand.get();
          unsigned cbOperandIdx =
              synchronizedOp->getParentOfType<GenericOp>().getOperandIndex(
                  localBuffer);

          // get the associated producer for this operand
          // Assumes only one producer for this local buffer
          auto associatedProducer = cbUsageInfo[localBuffer].producers.front();
          rewriter.setInsertionPoint(synchronizedOp);
          auto cb = d2m::getOrCreateCB(
              rewriter, synchronizedOp->getParentOfType<GenericOp>(),
              computeBlock, cbOperandIdx);

          if (mlir::isa<RemoteLoadOp>(associatedProducer) &&
              isAliasedLoad(mlir::cast<RemoteLoadOp>(associatedProducer))) {
            rewriter.create<ReserveOp>(loc, cb);
            rewriter.create<PushOp>(loc, cb);
          }
          WaitOp waitOp = rewriter.create<WaitOp>(loc, cb);
          rewriter.setInsertionPointAfter(synchronizedOp);
          rewriter.create<PopOp>(loc, cb);

          // Replace uses of the local buffer in compute consumer
          localBuffer.replaceUsesWithIf(
              waitOp.getResult(),
              [&](OpOperand &use) { return use.getOwner() == synchronizedOp; });
        }
      }

      // get producers and insert reserve+push
      for (auto &operand : synchronizedOp->getOpOperands()) {
        if (synchronizedOp.isProducer(operand)) {
          Location loc = synchronizedOp.getLoc();
          Value localBuffer = operand.get();
          unsigned cbOperandIdx =
              synchronizedOp->getParentOfType<GenericOp>().getOperandIndex(
                  localBuffer);

          // get the associated consumer for this operand
          // Assumes only one consumer for this local buffer
          auto associatedConsumer = cbUsageInfo[localBuffer].consumers.front();
          rewriter.setInsertionPoint(synchronizedOp);
          auto cb = d2m::getOrCreateCB(
              rewriter, synchronizedOp->getParentOfType<GenericOp>(),
              computeBlock, cbOperandIdx);
          auto reserveOp = rewriter.create<ReserveOp>(loc, cb);
          rewriter.setInsertionPointAfter(synchronizedOp);
          rewriter.create<PushOp>(loc, cb);
          if (mlir::isa<RemoteStoreOp>(associatedConsumer) &&
              isAliasedStore(mlir::cast<RemoteStoreOp>(associatedConsumer))) {
            rewriter.create<WaitOp>(loc, cb);
            rewriter.create<PopOp>(loc, cb);
          }

          // Replace uses of the local buffer in compute consumer
          localBuffer.replaceUsesWithIf(
              reserveOp.getResult(),
              [&](OpOperand &use) { return use.getOwner() == synchronizedOp; });
        }
      }
    }

    return WalkResult::advance();
  });

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
                           DenseSet<Operation *> &toErase) {
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

  for (RemoteLoadOp loadOp : loads) {
    if (loadOp.isExplicitCBForm()) {
      continue;
    }

    // removed  needsDMA and isL1ToL1Pair checks tocheck for aliased load/store
    // with just checking aliased load/store
    // TODO: l1-to-l1 pair is equivalent to checking aliased load + aliased
    // store, whcih we need to add a case for in allocate where we are currently
    // converting remote loads and stores to aliased load and store

    Value localBuffer = loadOp.getLocalBuffer();
    // llvm::errs() << "cbMemref: " << localBuffer << "\n";
    unsigned cbOperandIdx =
        loadOp->getParentOfType<GenericOp>().getOperandIndex(localBuffer);

    rewriter.setInsertionPoint(loadOp);
    auto cb = d2m::getOrCreateCB(rewriter, loadOp->getParentOfType<GenericOp>(),
                                 dmBlock, cbOperandIdx);
    auto newLoad = rewriter.create<RemoteLoadOp>(
        loadOp.getLoc(), loadOp.getMemref(), loadOp.getIndices(), cb,
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

    Value localBuffer = storeOp.getLocalBuffer();
    assert(localBuffer && "could not find associated local buffer for store");
    unsigned cbOperandIdx =
        storeOp->getParentOfType<GenericOp>().getOperandIndex(localBuffer);

    rewriter.setInsertionPoint(storeOp);
    auto cb = d2m::getOrCreateCB(
        rewriter, storeOp->getParentOfType<GenericOp>(), dmBlock, cbOperandIdx);
    rewriter.create<RemoteStoreOp>(
        storeOp.getLoc(), storeOp.getMemref(), storeOp.getIndices(), cb,
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
    Value dstCb = localBufferToCB.lookup(copyOp.getDst());

    // Find the destination CB.
    // Create explicit CB form: local_copy %src_memref into %dstCb.
    rewriter.create<LocalCopyOp>(loc, TypeRange{}, /*src=*/Value{},
                                 /*dst=*/Value{}, srcCb, dstCb,
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
static void eraseDMAOpsInComputeBlock(PatternRewriter &rewriter,
                                      Block *computeBlock) {
  DenseSet<Operation *> ops;
  computeBlock->walk([&](Operation *op) {
    if (isa<RemoteLoadOp, RemoteStoreOp, LocalCopyOp>(op)) {
      ops.insert(op);
    }
    return WalkResult::advance();
  });
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

    if (failed(wrapComputeInSynchronizedRegion(generic, rewriter))) {
      return failure();
    }

    llvm::errs() << "generic: " << *generic << "\n";

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
      assert(mlir::isa<d2m::LocalSemaphoreType>(arg.getType()) &&
             "region block arguments must be of local semaphore type");
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

    // Compute thread: insert CB sync ops for implicit-form remote ops.
    auto cbUsageInfoCompute = utils::getCBUsageInfo(newGeneric.getRegion(1));
    // processSharedBufferPairs no longer needed since we have explicit alias
    // ops
    // TODO: remove processSharedBufferPairs
    llvm::errs() << "compute block: " << *computeBlock << "\n";
    if (failed(processSharedBufferPairs(computeBlock, rewriter,
                                        cbUsageInfoCompute)) ||
        failed(insertCBOpsForCompute(computeBlock, rewriter,
                                     cbUsageInfoCompute))) {
      return failure();
    }

    // DMA thread: convert datamovement ops to explicit CB form.
    DenseSet<Operation *> toErase;
    if (failed(convertDMAToExplicitCBForm(dmBlock, rewriter, toErase))) {
      return failure();
    }
    // Erase the original implicit-form ops that were converted to explicit CB
    // form.
    for (Operation *op : toErase) {
      op->dropAllUses();
      rewriter.eraseOp(op);
    }
    toErase.clear();

    eraseDMAOpsInComputeBlock(rewriter, computeBlock);
    eraseDeadOps(rewriter, dmBlock, /*isDatamovementThread=*/true);
    eraseDeadOps(rewriter, computeBlock, /*isDatamovementThread=*/false);

    // Remove synchronized region ops, and move its ops to the parent level
    computeBlock->walk([&](SynchronizedRegionOp synchronizedOp) {
      if (failed(utils::unwrapSynchronizedRegion(rewriter, synchronizedOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    llvm::errs() << "newGeneric: " << *newGeneric << "\n";

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
