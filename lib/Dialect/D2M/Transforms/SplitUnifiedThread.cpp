// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"
#include "ttmlir/Dialect/D2M/Utils/SynchronizableOpInterfaceUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPLITUNIFIEDTHREAD
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void appendUnique(SmallVectorImpl<Value> &values, Value value) {
  if (!llvm::is_contained(values, value)) {
    values.push_back(value);
  }
}

bool isAliasedStore(RemoteStoreOp storeOp) {
  auto operandAliasOp =
      mlir::dyn_cast<OperandAliasOp>(storeOp.getLocalBuffer().getDefiningOp());
  return operandAliasOp && operandAliasOp.getMemref() == storeOp.getMemref();
}

bool isAliasedLoad(RemoteLoadOp loadOp) {
  auto operandAliasOp =
      mlir::dyn_cast<OperandAliasOp>(loadOp.getLocalBuffer().getDefiningOp());
  return operandAliasOp && operandAliasOp.getMemref() == loadOp.getMemref();
}

Value traceComputeMemrefToCB(Value value, GenericOp genericOp) {
  while (value) {
    // Check if its a cb (hoisted generic arg with cb layout attr).
    if (auto memrefType = mlir::dyn_cast<MemRefType>(value.getType())) {
      if (llvm::find(genericOp.getAdditionalArgs(), value) !=
          genericOp.getAdditionalArgs().end()) {
        // Skip scratch buffers.
        Operation *definingOp = value.getDefiningOp();
        if (definingOp && definingOp->getAttr("d2m.scratch_buffer")) {
          return nullptr;
        }
        if (utils::isReductionScalerBuffer(definingOp)) {
          return nullptr;
        }
        return value;
      }
    }

    // If we are no longer inside the generic or have reached the root, stop
    // tracing and return nullptr.
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || !genericOp->isProperAncestor(definingOp)) {
      return nullptr;
    }

    // Otherwise keep tracing up the chain, if we reach an op we don't support,
    // stop tracing and return nullptr.
    if (auto collapseOp = mlir::dyn_cast<memref::CollapseShapeOp>(definingOp)) {
      value = collapseOp.getSrc();
      continue;
    }
    if (auto subviewOp = mlir::dyn_cast<memref::SubViewOp>(definingOp)) {
      value = subviewOp.getSource();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

LogicalResult wrapComputeInSynchronizedRegion(GenericOp genericOp,
                                              PatternRewriter &rewriter) {
  // Look for a D2M_GenericRegionComputeOp, and collect the outermost ops that
  // contain them in the generic op.
  // Skip ops that have the SynchronizableOpInterface,
  // such as TileTilizeBlockOp and TileUntilizeBlockOp ops since
  // they haven't been lowered yet into non-synchronized ops
  OpBuilder::InsertionGuard guard(rewriter);

  // Collect the ops that directly contain SynchronizableOpInterface ops.
  // These delimit the scope where compute is synchronized: the outermost
  // compute ancestor is a direct child of such an op, alongside the
  // synchronizable ops that bound it.
  DenseSet<Operation *> opsWithSynchronizableOps;
  genericOp.getRegion(0).walk([&](Operation *op) {
    if (dyn_cast<SynchronizableOpInterface>(op)) {
      opsWithSynchronizableOps.insert(op->getParentOp());
    }
  });
  // Compute ops in distinct loop nests give an ambiguous sync scope: a single
  // per-block wait/pop pair can't balance a CB shared across nests.
  if (opsWithSynchronizableOps.size() != 1) {
    return genericOp.emitOpError()
           << "compute ops span multiple synchronization scopes (e.g. a CB "
              "consumed across distinct loop nests); cross-nest fan-out is not "
              "yet supported";
  }

  DenseSet<Operation *> outermostOps;
  bool walkFailed = false;
  genericOp.getRegion(0).walk([&](Operation *op) {
    if (!op->hasTrait<D2MGenericRegionComputeOpTrait>()) {
      return WalkResult::advance();
    }

    // Walk up loops until we reach the generic op as a parent, or a op
    // that directly contains a SynchronizableOpInterface op.
    Operation *outermostOp = op;
    while (outermostOp->getParentOp() != genericOp.getOperation() &&
           !opsWithSynchronizableOps.contains(outermostOp->getParentOp())) {
      outermostOp = outermostOp->getParentOp();
      if (!mlir::isa<scf::ForOp>(outermostOp) &&
          !mlir::isa<linalg::GenericOp>(outermostOp)) {
        outermostOp->emitOpError(
            "Parent ops containing compute ops must be scf.for or "
            "linalg.generic");
        walkFailed = true;
        return WalkResult::interrupt();
      }
    }

    // Skip ops that have the SynchronizableOpInterface,
    // such as TileTilizeBlockOp and TileUntilizeBlockOp ops since
    // they haven't been lowered yet into non-synchronized ops.
    if (!dyn_cast<SynchronizableOpInterface>(outermostOp)) {
      outermostOps.insert(outermostOp);
    }

    return WalkResult::advance();
  });

  if (walkFailed) {
    return failure();
  }

  // Expand and merge compute regions until we hit a syncrhonizable op on both
  // ends.
  SmallVector<std::pair<Block::iterator, Block::iterator>> computeRegions;
  while (!outermostOps.empty()) {
    Operation *outermostOp = *outermostOps.begin();
    outermostOps.erase(outermostOp);
    Block::iterator start = outermostOp->getIterator();
    Block::iterator end = outermostOp->getIterator();

    // Expand above.
    while (start != outermostOp->getBlock()->begin() &&
           !dyn_cast<SynchronizableOpInterface>(std::prev(start))) {
      start--;
      if (outermostOps.contains(&*start)) {
        outermostOps.erase(&*start);
      }
    }

    // Expand below.
    while (std::next(end) != outermostOp->getBlock()->end() &&
           !dyn_cast<SynchronizableOpInterface>(std::next(end))) {
      end++;
      if (outermostOps.contains(&*end)) {
        outermostOps.erase(&*end);
      }
    }

    computeRegions.push_back({start, std::next(end)});
  }

  for (auto [start, end] : computeRegions) {

    SmallVector<Value> loadedCBOperands;
    SmallVector<Value> storedCBOperands;

    // For memref load and stores, trace to cb operand to get producers and
    // consumers for syncrhonized region.
    for (Operation &op : llvm::make_range(start, end)) {
      // For load trace src memref up to defining op and check if its a cb (as
      // opposed to dst).
      op.walk([&](memref::LoadOp loadOp) {
        Value cb = traceComputeMemrefToCB(loadOp.getMemref(), genericOp);
        if (cb) {
          appendUnique(loadedCBOperands, cb);
        }
        return WalkResult::advance();
      });

      // For store trace dst memref up to defining op and check if its a cb (as
      // opposed to dst)
      op.walk([&](memref::StoreOp storeOp) {
        Value cb = traceComputeMemrefToCB(storeOp.getMemref(), genericOp);
        if (cb) {
          appendUnique(storedCBOperands, cb);
        }
        return WalkResult::advance();
      });

      // TileMatmulBlockOp uses CBs directly without load/store.
      op.walk([&](d2m::TileMatmulBlockOp tileMatmulBlockOp) {
        Value cbA = traceComputeMemrefToCB(tileMatmulBlockOp.getA(), genericOp);
        Value cbB = traceComputeMemrefToCB(tileMatmulBlockOp.getB(), genericOp);
        Value cbOutput =
            traceComputeMemrefToCB(tileMatmulBlockOp.getOutput(), genericOp);
        if (cbA) {
          appendUnique(loadedCBOperands, cbA);
        }
        if (cbB) {
          appendUnique(loadedCBOperands, cbB);
        }
        if (cbOutput) {
          appendUnique(storedCBOperands, cbOutput);
        }
        return WalkResult::advance();
      });
    }

    // Remove allocs in load that are also in store since this is output cb
    // reuse and not an actual input.
    DenseSet<Value> storedCBOperandSet(storedCBOperands.begin(),
                                       storedCBOperands.end());
    llvm::erase_if(loadedCBOperands, [&](Value loadedCBOperand) {
      return storedCBOperandSet.contains(loadedCBOperand);
    });

    utils::wrapInSynchronizedRegion(rewriter, start, end, loadedCBOperands,
                                    storedCBOperands);
  }

  return success();
}

// From cb usage info, check for load-store pairs and insert aliased cb ops for
// alias side.
static LogicalResult processSharedBufferPairs(
    Block *computeBlock, PatternRewriter &rewriter,
    llvm::DenseMap<Value, utils::CBUsageInfo> &cbUsageInfo) {
  for (auto [localBuffer, usageInfo] : cbUsageInfo) {
    // Only handles 1:1 RemoteLoad->RemoteStore alias pairs; fan-out CBs are
    // handled by insertCBOpsForCompute.
    if (usageInfo.producers.size() != 1 || usageInfo.consumers.size() != 1) {
      continue;
    }
    auto *producer = usageInfo.producers.front();
    auto *consumer = usageInfo.consumers.front();

    // Insert compute-side CB ops for the aliased half of the pair.
    // The streaming half stays as a remote_load/store for DMA.
    if (mlir::isa<RemoteLoadOp>(producer) &&
        mlir::isa<RemoteStoreOp>(consumer) &&
        isAliasedStore(mlir::cast<RemoteStoreOp>(consumer))) {
      Location loc = producer->getLoc();
      unsigned cbOperandIdx =
          producer->getParentOfType<GenericOp>().getOperandIndex(localBuffer);
      // Set insertion point before consumer so GetCBOp dominates WaitOp/PopOp.
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
      // ReserveOp/PushOp.
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

// Return the block all `ops` are direct children of, or nullptr otherwise. The
// single wait/pop pair must sit in the same block to match the producer's
// per-block push cadence; ops split across nests would deadlock.
static Block *commonParentBlock(ArrayRef<Operation *> ops) {
  Block *block = ops.front()->getBlock();
  for (Operation *op : ArrayRef<Operation *>(ops).drop_front()) {
    if (op->getBlock() != block) {
      return nullptr;
    }
  }
  return block;
}

static LogicalResult
insertCBOpsForCompute(Block *computeBlock, PatternRewriter &rewriter,
                      llvm::DenseMap<Value, utils::CBUsageInfo> &cbUsageInfo) {
  // Collect, per CB local buffer, the compute synchronizable ops that consume
  // or produce it, in program order. A CB may have multiple consumers (fan-out,
  // e.g. softmax reusing `scores`); the producer pushes once, so it's balanced
  // by one wait before the first consumer and one pop after the last.
  llvm::MapVector<Value, SmallVector<Operation *>> consumersByCB;
  llvm::MapVector<Value, SmallVector<Operation *>> producersByCB;
  computeBlock->walk([&](Operation *op) {
    auto synchronizedOp = dyn_cast<SynchronizableOpInterface>(op);
    if (!synchronizedOp ||
        op->hasTrait<D2MGenericRegionDatamovementOpTrait>()) {
      return WalkResult::advance();
    }
    for (auto &operand : op->getOpOperands()) {
      if (synchronizedOp.isConsumer(operand)) {
        consumersByCB[operand.get()].push_back(op);
      }
      if (synchronizedOp.isProducer(operand)) {
        producersByCB[operand.get()].push_back(op);
      }
    }
    return WalkResult::advance();
  });

  auto generic = cast<GenericOp>(computeBlock->getParentOp());

  // Consumers: wait once before the first consumer, pop once after the last.
  for (auto &[localBuffer, ops] : consumersByCB) {
    if (!commonParentBlock(ops)) {
      return generic.emitOpError()
             << "CB has consumers across distinct loop nests; cross-nest "
                "fan-out is not yet supported (would deadlock on a "
                "wait/pop cadence mismatch)";
    }
    unsigned cbOperandIdx = generic.getOperandIndex(localBuffer);
    Operation *first = ops.front();
    Operation *last = ops.back();
    Location loc = first->getLoc();

    rewriter.setInsertionPoint(first);
    auto cb = d2m::getOrCreateCB(rewriter, generic, computeBlock, cbOperandIdx);

    // Aliased remote_load producer has no DMA, so compute reserves+pushes.
    bool aliasedLoadProducer =
        llvm::any_of(cbUsageInfo[localBuffer].producers, [](Operation *p) {
          auto load = mlir::dyn_cast<RemoteLoadOp>(p);
          return load && isAliasedLoad(load);
        });
    if (aliasedLoadProducer) {
      rewriter.create<ReserveOp>(loc, cb);
      rewriter.create<PushOp>(loc, cb);
    }
    WaitOp waitOp = rewriter.create<WaitOp>(loc, cb);
    rewriter.setInsertionPointAfter(last);
    rewriter.create<PopOp>(last->getLoc(), cb);

    llvm::DenseSet<Operation *> consumerSet(ops.begin(), ops.end());
    localBuffer.replaceUsesWithIf(waitOp.getResult(), [&](OpOperand &use) {
      return consumerSet.contains(use.getOwner());
    });
  }

  // Producers: reserve once before the first producer, push once after last.
  for (auto &[localBuffer, ops] : producersByCB) {
    if (!commonParentBlock(ops)) {
      return generic.emitOpError()
             << "CB has producers across distinct loop nests; cross-nest "
                "fan-out is not yet supported (would deadlock on a "
                "reserve/push cadence mismatch)";
    }
    unsigned cbOperandIdx = generic.getOperandIndex(localBuffer);
    Operation *first = ops.front();
    Operation *last = ops.back();
    Location loc = first->getLoc();

    rewriter.setInsertionPoint(first);
    auto cb = d2m::getOrCreateCB(rewriter, generic, computeBlock, cbOperandIdx);
    auto reserveOp = rewriter.create<ReserveOp>(loc, cb);
    rewriter.setInsertionPointAfter(last);
    rewriter.create<PushOp>(last->getLoc(), cb);

    // Aliased remote_store consumer has no DMA, so compute waits+pops.
    bool aliasedStoreConsumer =
        llvm::any_of(cbUsageInfo[localBuffer].consumers, [](Operation *c) {
          auto store = mlir::dyn_cast<RemoteStoreOp>(c);
          return store && isAliasedStore(store);
        });
    if (aliasedStoreConsumer) {
      rewriter.create<WaitOp>(last->getLoc(), cb);
      rewriter.create<PopOp>(last->getLoc(), cb);
    }

    llvm::DenseSet<Operation *> producerSet(ops.begin(), ops.end());
    localBuffer.replaceUsesWithIf(reserveOp.getResult(), [&](OpOperand &use) {
      return producerSet.contains(use.getOwner());
    });
  }

  return success();
}

// ---------------------------------------------------------------------------
// DMA thread: convert implicit-form ops to explicit CB form
// ---------------------------------------------------------------------------

// Erase aliased load and store ops (no DMA needed).
static LogicalResult eraseAliasedLoadStoreOps(
    PatternRewriter &rewriter,
    llvm::DenseMap<Value, utils::CBUsageInfo> &cbUsageInfo) {
  for (auto [localBuffer, usageInfo] : cbUsageInfo) {
    // Only erases 1:1 aliased RemoteLoad/RemoteStore pairs; skip fan-out CBs.
    if (usageInfo.producers.size() != 1 || usageInfo.consumers.size() != 1) {
      continue;
    }
    auto *producer = usageInfo.producers.front();
    auto *consumer = usageInfo.consumers.front();

    if (mlir::isa<RemoteStoreOp>(consumer) &&
        isAliasedStore(mlir::cast<RemoteStoreOp>(consumer))) {
      rewriter.eraseOp(consumer);
    } else if (mlir::isa<RemoteLoadOp>(producer) &&
               isAliasedLoad(mlir::cast<RemoteLoadOp>(producer))) {
      rewriter.eraseOp(producer);
    }
  }
  return success();
}

// Convert remote_load/store to explicit CB form in the DMA thread.
// Aliased ops are collected for deferred erasure (no DMA needed). Shared
// buffer pairs use the output operand's CB for both ops.
static LogicalResult convertDMAToExplicitCBForm(Block *dmBlock,
                                                PatternRewriter &rewriter) {
  SmallVector<RemoteLoadOp> loads;
  SmallVector<RemoteStoreOp> stores;
  SmallVector<LocalCopyOp> localCopies;
  dmBlock->walk([&](RemoteLoadOp op) { loads.push_back(op); });
  dmBlock->walk([&](RemoteStoreOp op) { stores.push_back(op); });
  dmBlock->walk([&](LocalCopyOp op) { localCopies.push_back(op); });

  for (RemoteLoadOp loadOp : loads) {
    if (loadOp.isExplicitCBForm()) {
      continue;
    }

    Value localBuffer = loadOp.getLocalBuffer();
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

    loadOp->dropAllUses();
    rewriter.eraseOp(loadOp);
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
    storeOp->dropAllUses();
    rewriter.eraseOp(storeOp);
  }

  // Convert implicit-form local_copy ops to explicit CB form.
  for (LocalCopyOp copyOp : localCopies) {
    if (copyOp.isExplicitCBForm()) {
      continue;
    }

    Location loc = copyOp.getLoc();

    unsigned srcCbOperandIdx =
        copyOp->getParentOfType<GenericOp>().getOperandIndex(copyOp.getSrc());
    auto srcCb =
        d2m::getOrCreateCB(rewriter, copyOp->getParentOfType<GenericOp>(),
                           dmBlock, srcCbOperandIdx);
    unsigned dstCbOperandIdx =
        copyOp->getParentOfType<GenericOp>().getOperandIndex(copyOp.getDst());
    auto dstCb =
        d2m::getOrCreateCB(rewriter, copyOp->getParentOfType<GenericOp>(),
                           dmBlock, dstCbOperandIdx);

    // Create explicit CB form: local_copy %srcCb into %dstCb.
    rewriter.setInsertionPoint(copyOp);
    rewriter.create<LocalCopyOp>(loc, TypeRange{}, /*src=*/Value{},
                                 /*dst=*/Value{}, srcCb, dstCb,
                                 copyOp.getIndexingMaps());
    copyOp->dropAllUses();
    rewriter.eraseOp(copyOp);
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
    auto cbUsageInfoDm = utils::getCBUsageInfo(newGeneric.getRegion(0));
    if (failed(processSharedBufferPairs(computeBlock, rewriter,
                                        cbUsageInfoCompute)) ||
        failed(insertCBOpsForCompute(computeBlock, rewriter,
                                     cbUsageInfoCompute))) {
      return failure();
    }

    if (failed(eraseAliasedLoadStoreOps(rewriter, cbUsageInfoDm))) {
      return failure();
    }

    // DMA thread: convert datamovement ops to explicit CB form.
    if (failed(convertDMAToExplicitCBForm(dmBlock, rewriter))) {
      return failure();
    }

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
