// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MASSIGNTHREADS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// The attribute used to record a leaf op's destination thread. Ops left without
// this attribute are replicated into both threads by d2m-split-threads.
static constexpr StringRef kThreadAttrName = "d2m.thread";

static bool isInsideRouterRegion(Operation *op) {
  for (auto ifOp = op->getParentOfType<scf::IfOp>(); ifOp;
       ifOp = ifOp->getParentOfType<scf::IfOp>()) {
    if (ifOp.getCondition().getDefiningOp<IsRouterCoreOp>()) {
      return true;
    }
  }
  return false;
}

// Aliased remote ops carry no DMA transfer (the buffer is already in L1 via
// operand_alias). These predicates are null-safe so they may be called on
// explicit-CB-form ops (which have no localBuffer and are never aliased).
static bool isAliasedStore(RemoteStoreOp storeOp) {
  if (!storeOp.getLocalBuffer()) {
    return false;
  }
  auto operandAliasOp =
      mlir::dyn_cast<OperandAliasOp>(storeOp.getLocalBuffer().getDefiningOp());
  return operandAliasOp && operandAliasOp.getMemref() == storeOp.getMemref();
}

static bool isAliasedLoad(RemoteLoadOp loadOp) {
  if (!loadOp.getLocalBuffer()) {
    return false;
  }
  auto operandAliasOp =
      mlir::dyn_cast<OperandAliasOp>(loadOp.getLocalBuffer().getDefiningOp());
  return operandAliasOp && operandAliasOp.getMemref() == loadOp.getMemref();
}

// Lower implicit-form data-movement ops (remote_load/store, local_copy) to
// explicit-CB form in place. Aliased remote ops are left implicit: they have no
// DMA transfer and become compute-side CB obligations handled by
// d2m-split-threads.
static LogicalResult lowerDMAOpsToExplicitCB(GenericOp generic, Block *block,
                                             RewriterBase &rewriter) {
  SmallVector<RemoteLoadOp> loads;
  SmallVector<RemoteStoreOp> stores;
  SmallVector<LocalCopyOp> localCopies;
  SmallVector<CoreReadOp> coreReads;
  SmallVector<SemaphoreIncOp> computeSignalIncs;
  SmallVector<SemaphoreWaitOp> computeReleaseWaits;
  block->walk([&](RemoteLoadOp op) { loads.push_back(op); });
  block->walk([&](RemoteStoreOp op) { stores.push_back(op); });
  block->walk([&](LocalCopyOp op) { localCopies.push_back(op); });
  block->walk([&](CoreReadOp op) { coreReads.push_back(op); });
  block->walk([&](SemaphoreIncOp op) {
    if (op->hasAttr("d2m.compute_signal")) {
      computeSignalIncs.push_back(op);
    }
  });
  block->walk([&](SemaphoreWaitOp op) {
    if (op->hasAttr("d2m.compute_release")) {
      computeReleaseWaits.push_back(op);
    }
  });

  if ((!computeSignalIncs.empty() || !computeReleaseWaits.empty()) &&
      coreReads.empty()) {
    return generic.emitOpError(
        "compute semaphore synchronization requires a core_read");
  }

  if (!computeReleaseWaits.empty() && computeSignalIncs.empty()) {
    return generic.emitOpError(
        "compute semaphore release requires a compute producer signal");
  }

  if (!coreReads.empty() && !computeSignalIncs.empty()) {
    Value source = coreReads.front().getSrc();
    if (llvm::any_of(coreReads,
                     [&](CoreReadOp op) { return op.getSrc() != source; })) {
      return generic.emitOpError(
          "compute semaphore synchronization requires all core_read ops to "
          "gather the same source");
    }
  }

  auto markDatamovement = [&](Operation *op) {
    op->setAttr(kThreadAttrName,
                rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement));
  };

  // A producer-done signal must not fire until compute has pushed the local
  // buffer that the router will read. Acquire that buffer's CB before the
  // signal, and retarget core_read to the acquired front pointer. Deliberately
  // do not pop: the gather still needs the produced page to remain at the front
  // while the router reads every worker's uniform L1 address.
  Value fencedSrc;
  Value fenceWaited;
  if (!coreReads.empty() && !computeSignalIncs.empty()) {
    fencedSrc = coreReads.front().getSrc();
    unsigned fenceCBOperandIdx = generic.getOperandIndex(fencedSrc);
    for (SemaphoreIncOp inc : computeSignalIncs) {
      rewriter.setInsertionPoint(inc);
      Value cb =
          d2m::getOrCreateCB(rewriter, generic, block, fenceCBOperandIdx);
      auto wait = rewriter.create<WaitOp>(inc.getLoc(), cb);
      markDatamovement(wait);
      fenceWaited = wait.getResult();
      inc->removeAttr("d2m.compute_signal");
    }
  }

  // Once the router acknowledges that it has copied this worker's result,
  // release the producer CB. This balances the compute reserve/push inserted
  // for each loop iteration and lets the next matmul chunk start while the
  // router sends the gathered copy over fabric.
  if (fencedSrc) {
    unsigned fenceCBOperandIdx = generic.getOperandIndex(fencedSrc);
    for (SemaphoreWaitOp wait : computeReleaseWaits) {
      markDatamovement(wait);
      wait->removeAttr("d2m.compute_release");
      rewriter.setInsertionPointAfter(wait);
      Value cb =
          d2m::getOrCreateCB(rewriter, generic, block, fenceCBOperandIdx);
      auto pop = rewriter.create<PopOp>(wait.getLoc(), cb);
      markDatamovement(pop);
    }
  }

  for (RemoteLoadOp loadOp : loads) {
    if (loadOp.isExplicitCBForm() || isAliasedLoad(loadOp)) {
      continue;
    }
    Value localBuffer = loadOp.getLocalBuffer();
    unsigned cbOperandIdx = generic.getOperandIndex(localBuffer);

    rewriter.setInsertionPoint(loadOp);
    auto cb = d2m::getOrCreateCB(rewriter, generic, block, cbOperandIdx);
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
    if (storeOp.isExplicitCBForm() || isAliasedStore(storeOp)) {
      continue;
    }
    Value localBuffer = storeOp.getLocalBuffer();
    assert(localBuffer && "could not find associated local buffer for store");
    unsigned cbOperandIdx = generic.getOperandIndex(localBuffer);

    rewriter.setInsertionPoint(storeOp);
    auto cb = d2m::getOrCreateCB(rewriter, generic, block, cbOperandIdx);
    rewriter.create<RemoteStoreOp>(
        storeOp.getLoc(), storeOp.getMemref(), storeOp.getIndices(), cb,
        storeOp.getStartDevice(), storeOp.getDeviceMcastShape(),
        storeOp.getSemaphore(), storeOp.getSemaphoreIndices());
    storeOp->dropAllUses();
    rewriter.eraseOp(storeOp);
  }

  // core_read produces its destination CB on the datamovement thread. Wrap
  // the NoC transfer in reserve/push so a following remote_store can consume
  // the gathered buffer with its normal wait/pop protocol.
  for (CoreReadOp coreReadOp : coreReads) {
    Value dstBuffer = coreReadOp.getDst();
    unsigned cbOperandIdx = generic.getOperandIndex(dstBuffer);
    Value src = coreReadOp.getSrc();
    if (fenceWaited && src == fencedSrc) {
      src = fenceWaited;
    }

    rewriter.setInsertionPoint(coreReadOp);
    Value cb = d2m::getOrCreateCB(rewriter, generic, block, cbOperandIdx);
    auto reserve = rewriter.create<ReserveOp>(coreReadOp.getLoc(), cb);
    auto read = rewriter.create<CoreReadOp>(coreReadOp.getLoc(), TypeRange{},
                                            src, coreReadOp.getSrcCore(),
                                            reserve.getResult());
    auto push = rewriter.create<PushOp>(coreReadOp.getLoc(), cb);
    markDatamovement(reserve);
    markDatamovement(read);
    markDatamovement(push);
    coreReadOp->dropAllUses();
    rewriter.eraseOp(coreReadOp);
  }

  for (LocalCopyOp copyOp : localCopies) {
    if (copyOp.isExplicitCBForm()) {
      continue;
    }
    Location loc = copyOp.getLoc();
    unsigned srcCbOperandIdx = generic.getOperandIndex(copyOp.getSrc());
    auto srcCb = d2m::getOrCreateCB(rewriter, generic, block, srcCbOperandIdx);
    unsigned dstCbOperandIdx = generic.getOperandIndex(copyOp.getDst());
    auto dstCb = d2m::getOrCreateCB(rewriter, generic, block, dstCbOperandIdx);

    rewriter.setInsertionPoint(copyOp);
    rewriter.create<LocalCopyOp>(loc, TypeRange{}, /*src=*/Value{},
                                 /*dst=*/Value{}, srcCb, dstCb,
                                 copyOp.getIndexingMaps());
    copyOp->dropAllUses();
    rewriter.eraseOp(copyOp);
  }

  return success();
}

// Classify a leaf op in a unified region as belonging to a single thread.
// Returns std::nullopt for ops that must be replicated into both threads
// (structural/pure ops and non-mutating semaphore_wait), which are left
// untagged.
//
// All remote_load/store and local_copy are data-movement ops (the verifier
// requires remote ops to live on the datamovement thread). Aliased remote ops
// carry no DMA transfer but still belong on the datamovement thread; they
// remain in implicit form and d2m-insert-compute-cb inspects them to add the
// matching compute-side CB synchronization, then erases them.
static std::optional<ThreadType> classifyOp(Operation *op) {
  if (mlir::isa<RemoteLoadOp, RemoteStoreOp, LocalCopyOp, CoreReadOp,
                SemaphoreIncOp, SemaphoreSetOp>(op)) {
    return ThreadType::Datamovement;
  }
  // Router-only waits guard data movement after compute has already published
  // its result. Running them on compute as well adds a second waiter with no
  // compute-side work to protect and can stall fabric progress on hardware.
  // Resetting waits mutate shared state, so they also require one owner. Other
  // waits remain replicated to preserve barriers that resume both threads.
  if (auto wait = mlir::dyn_cast<SemaphoreWaitOp>(op)) {
    return isInsideRouterRegion(op) || wait.getResetValue()
               ? std::optional(ThreadType::Datamovement)
               : std::nullopt;
  }
  if (mlir::isa<ShardDMAOpInterface, DeviceSynchronizeOp>(op)) {
    return ThreadType::Datamovement;
  }
  if (op->hasTrait<D2MGenericRegionComputeOpTrait>()) {
    return ThreadType::Compute;
  }
  // A linalg.generic at this stage carries compute (tile ops in its body).
  if (mlir::isa<linalg::GenericOp>(op)) {
    return ThreadType::Compute;
  }
  return std::nullopt;
}

class D2MAssignThreads : public impl::D2MAssignThreadsBase<D2MAssignThreads> {
public:
  using impl::D2MAssignThreadsBase<D2MAssignThreads>::D2MAssignThreadsBase;

  void runOnOperation() final {
    IRRewriter rewriter(&getContext());
    SmallVector<GenericOp> generics;
    getOperation().walk([&](GenericOp generic) {
      if (generic.getNumRegions() == 1 &&
          generic.getRegionThreadType(0) == ThreadType::Unified) {
        generics.push_back(generic);
      }
    });

    for (GenericOp generic : generics) {
      Block *block = &generic.getRegion(0).front();
      // Lower data-movement ops to explicit-CB form, then record each leaf op's
      // destination thread.
      if (failed(lowerDMAOpsToExplicitCB(generic, block, rewriter))) {
        signalPassFailure();
        return;
      }
      generic.getRegion(0).walk([&](Operation *op) {
        if (std::optional<ThreadType> thread = classifyOp(op)) {
          op->setAttr(kThreadAttrName, rewriter.getAttr<ThreadAttr>(*thread));
        }
      });
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
