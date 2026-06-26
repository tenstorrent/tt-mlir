// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MASSIGNTHREADS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// The attribute used to record a leaf op's destination thread. Ops left without
// this attribute are replicated into both threads by d2m-split-threads.
static constexpr StringRef kThreadAttrName = "d2m.thread";

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
static void lowerDMAOpsToExplicitCB(GenericOp generic, Block *block,
                                    RewriterBase &rewriter) {
  SmallVector<RemoteLoadOp> loads;
  SmallVector<RemoteStoreOp> stores;
  SmallVector<LocalCopyOp> localCopies;
  block->walk([&](RemoteLoadOp op) { loads.push_back(op); });
  block->walk([&](RemoteStoreOp op) { stores.push_back(op); });
  block->walk([&](LocalCopyOp op) { localCopies.push_back(op); });

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
}

// Classify a leaf op in a unified region as belonging to a single thread.
// Returns std::nullopt for ops that must be replicated into both threads
// (structural/pure ops and semaphore_wait), which are left untagged.
//
// All remote_load/store and local_copy are data-movement ops (the verifier
// requires remote ops to live on the datamovement thread). Aliased remote ops
// carry no DMA transfer but still belong on the datamovement thread; they
// remain in implicit form and d2m-insert-compute-cb inspects them to add the
// matching compute-side CB synchronization, then erases them.
static std::optional<ThreadType> classifyOp(Operation *op) {
  if (mlir::isa<RemoteLoadOp, RemoteStoreOp, LocalCopyOp>(op)) {
    return ThreadType::Datamovement;
  }
  // semaphore_wait is replicated into both threads; leave untagged.
  if (mlir::isa<SemaphoreWaitOp>(op)) {
    return std::nullopt;
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
      lowerDMAOpsToExplicitCB(generic, block, rewriter);
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
