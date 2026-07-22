// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// d2m-split-unified-thread-v2: rewrite of d2m-split-unified-thread,
// selectable via the `use-split-unified-thread-v2` option on d2m-be-pipeline
// for A/B comparison against the legacy pass.
//
// Like the legacy pass it splits a unified-thread d2m.generic into separate
// datamovement and compute regions and inserts the circular-buffer (CB)
// synchronization ops. The difference is the compute-side model: rather than
// wrapping compute in a SynchronizedRegionOp and assuming a single
// producer/consumer per CB, V2 classifies each buffer's accesses by thread
// (see insertComputeCBOpsV2) and emits one handshake per cross-thread
// (compute<->DM) edge, anchored at the DM partner's loop level. This handles
// buffers with multiple intra-thread producers/consumers -- e.g. a
// loop-carried matmul accumulator -- that the legacy 1:1 CB model cannot.
//
// The datamovement-side conversion and dead-op cleanup are shared in spirit
// with the legacy pass (convertDMAToExplicitCBForm, eraseDeadOps).

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"
#include "ttmlir/Dialect/D2M/Utils/SynchronizableOpInterfaceUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPLITUNIFIEDTHREADV2
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
    if (auto castOp = mlir::dyn_cast<memref::CastOp>(definingOp)) {
      value = castOp.getSource();
      continue;
    }
    if (auto reinterpretOp =
            mlir::dyn_cast<memref::ReinterpretCastOp>(definingOp)) {
      value = reinterpretOp.getSource();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// DMA thread: convert implicit-form ops to explicit CB form
// ---------------------------------------------------------------------------

// Convert remote_load/store to explicit CB form in the DMA thread.
// Aliased ops are collected for deferred erasure (no DMA needed). Shared
// buffer pairs use the output operand's CB for both ops.
static LogicalResult convertDMAToExplicitCBForm(Block *dmBlock,
                                                PatternRewriter &rewriter) {
  SmallVector<RemoteLoadOp> loads;
  SmallVector<RemoteStoreOp> stores;
  SmallVector<FabricRecvOp> fabricRecvs;
  SmallVector<LocalCopyOp> localCopies;
  SmallVector<CoreReadOp> coreReads;
  SmallVector<CoreWriteOp> coreWrites;
  SmallVector<SemaphoreIncOp> computeSignalIncs;
  dmBlock->walk([&](RemoteLoadOp op) { loads.push_back(op); });
  dmBlock->walk([&](RemoteStoreOp op) { stores.push_back(op); });
  dmBlock->walk([&](FabricRecvOp op) { fabricRecvs.push_back(op); });
  dmBlock->walk([&](LocalCopyOp op) { localCopies.push_back(op); });
  dmBlock->walk([&](CoreReadOp op) { coreReads.push_back(op); });
  dmBlock->walk([&](CoreWriteOp op) { coreWrites.push_back(op); });
  dmBlock->walk([&](SemaphoreIncOp op) {
    if (op->hasAttr("d2m.compute_signal")) {
      computeSignalIncs.push_back(op);
    }
  });

  // A `d2m.compute_signal` inc is a producer-done signal: it must not fire
  // until this core's matmul output (the buffer the gather core_reads) is
  // ready. Fence it by inserting wait(outputCB) before the inc -- the
  // compute-side push that insertComputeCBOpsV2 added via the dmFence output
  // handshake fills that CB, so the wait blocks until this core's matmul
  // completes. Inputs were already pushed (below), so the compute thread can
  // run the matmul: no deadlock.
  //
  // The gather's core_read then reads the *wait result* (the acquired front
  // buffer), not the raw operand. This is load-bearing twice over: (1) it keeps
  // the wait alive (a wait with an unused result and single-use cb is
  // canonicalized away, dropping the fence), and (2) it points core_read at the
  // front read pointer -- the matmul output -- at a uniform L1 offset across
  // cores. Deliberately NO pop: cb_pop_front would advance the read pointer
  // past the just-produced data, breaking the cross-core read; the buffer is
  // single-shot (one push, read cross-core, released at block scope).
  Value fencedSrc, fenceWaited;
  if (!coreReads.empty() && !computeSignalIncs.empty()) {
    fencedSrc = coreReads.front().getSrc();
    unsigned fenceCbOperandIdx =
        coreReads.front()->getParentOfType<GenericOp>().getOperandIndex(
            fencedSrc);
    for (SemaphoreIncOp inc : computeSignalIncs) {
      GenericOp generic = inc->getParentOfType<GenericOp>();
      rewriter.setInsertionPoint(inc);
      Value cb =
          d2m::getOrCreateCB(rewriter, generic, dmBlock, fenceCbOperandIdx);
      fenceWaited = rewriter.create<WaitOp>(inc.getLoc(), cb).getResult();
      inc->removeAttr("d2m.compute_signal");
    }
  }

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

  // fabric_recv: convert to explicit-CB form using the MEMREF operand's own CB
  // (not a localBuffer -- fabric_recv has none; the operand IS the recv buffer
  // the peer fabric-wrote). LowerLoadStoreOpsToDMA then lowers this to
  // reserve+push (no dma_read / noc_async_read).
  for (FabricRecvOp recvOp : fabricRecvs) {
    if (recvOp.isExplicitCBForm()) {
      continue;
    }
    GenericOp generic = recvOp->getParentOfType<GenericOp>();
    unsigned cbOperandIdx = generic.getOperandIndex(recvOp.getMemref());
    rewriter.setInsertionPoint(recvOp);
    auto cb = d2m::getOrCreateCB(rewriter, generic, dmBlock, cbOperandIdx);
    rewriter.create<FabricRecvOp>(recvOp.getLoc(), recvOp.getMemref(),
                                  recvOp.getIndices(), cb);
    recvOp->dropAllUses();
    rewriter.eraseOp(recvOp);
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

  // core_read produces into its dst buffer; wrap it in the CB produce protocol
  // (reserve dst CB -> core_read into the reserved buffer -> push) so a
  // downstream consumer of that buffer on the DM thread (e.g. a remote_store,
  // converted to wait/pop the same CB above) gets the push and doesn't
  // deadlock. core_read's src is read cross-core by address (uniform L1 offset)
  // and needs no local CB acquire -- except a fenced src (a compute-produced
  // matmul output), which reads the fence's wait result so the read pointer
  // lands on the produced front buffer (see the fence above).
  for (CoreReadOp coreReadOp : coreReads) {
    GenericOp generic = coreReadOp->getParentOfType<GenericOp>();
    Value dstBuffer = coreReadOp.getDst();
    unsigned cbOperandIdx = generic.getOperandIndex(dstBuffer);

    Value src = coreReadOp.getSrc();
    if (fenceWaited && src == fencedSrc) {
      src = fenceWaited;
    }

    rewriter.setInsertionPoint(coreReadOp);
    Value cb = d2m::getOrCreateCB(rewriter, generic, dmBlock, cbOperandIdx);
    Value reserved =
        rewriter.create<ReserveOp>(coreReadOp.getLoc(), cb).getResult();
    rewriter.create<CoreReadOp>(coreReadOp.getLoc(), TypeRange{}, src,
                                coreReadOp.getSrcCore(), reserved);
    rewriter.create<PushOp>(coreReadOp.getLoc(), cb);
    coreReadOp->dropAllUses();
    rewriter.eraseOp(coreReadOp);
  }

  // core_write consumes its src buffer; wrap it in the CB consume protocol
  // (wait src CB -> core_write from the waited buffer -> pop) so an upstream
  // producer of that buffer on the DM thread (e.g. a remote_load, converted to
  // reserve/push) handshakes correctly. core_write's dst is written cross-core
  // by address (uniform L1 offset) and needs no local CB acquire.
  for (CoreWriteOp coreWriteOp : coreWrites) {
    GenericOp generic = coreWriteOp->getParentOfType<GenericOp>();
    Value srcBuffer = coreWriteOp.getSrc();
    unsigned cbOperandIdx = generic.getOperandIndex(srcBuffer);

    rewriter.setInsertionPoint(coreWriteOp);
    Value cb = d2m::getOrCreateCB(rewriter, generic, dmBlock, cbOperandIdx);
    Value waited =
        rewriter.create<WaitOp>(coreWriteOp.getLoc(), cb).getResult();
    rewriter.create<CoreWriteOp>(coreWriteOp.getLoc(), TypeRange{}, waited,
                                 coreWriteOp.getDst(),
                                 coreWriteOp.getDstCore());
    rewriter.create<PopOp>(coreWriteOp.getLoc(), cb);
    coreWriteOp->dropAllUses();
    rewriter.eraseOp(coreWriteOp);
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
    // Recurse into scf.if regions and classify their contents (like scf.for),
    // keeping the if op itself in both threads. Without this, an scf.if
    // guarding a fabric op (e.g. device_synchronize gated to a single core) is
    // neither isDMAOp nor isReplicated, so the whole conditional is treated as
    // compute-resident and moved off the datamovement thread -- the fabric op
    // then lands on compute (TRISC), which has no NOC/fabric access.
    if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      collectOpsToErase(ifOp.thenBlock(), eraseSet, isDatamovementThread);
      if (Block *elseBlock = ifOp.elseBlock()) {
        collectOpsToErase(elseBlock, eraseSet, isDatamovementThread);
      }
      continue;
    }

    // DM-resident ops stay on the datamovement thread and are removed from the
    // compute thread. Semaphore mutations (inc/set) belong here: compute
    // (TRISC) has no NOC -- a NOC semaphore inc cannot even compile there
    // (dataflow_api.h needs NOC_INDEX, undefined in the compute environment) --
    // and ScheduleDMA keeps a kernel that has them on a single DM thread so
    // they run exactly once. core_read is a core->core NoC read, also DM-only.
    //
    // A *producer-done* signal in a fused producer->consumer kernel (e.g. the
    // matmul -> core_read gather, where each core increments a `ready`
    // semaphore once its matmul tile is done) is marked `d2m.compute_signal`
    // and still lives here on DM, but convertDMAToExplicitCBForm fences it
    // behind this core's matmul output CB so it can't fire before the compute
    // completes (matching tt-metal's writer-signals pattern). See
    // tools/d2m-jit/unified_semaphore_design.md ("Producer-done signals").
    bool isDMAOp = isa<ShardDMAOpInterface, DeviceSynchronizeOp, SemaphoreIncOp,
                       SemaphoreSetOp, CoreReadOp, CoreWriteOp>(&op);
    bool isReplicated = isa<SemaphoreWaitOp>(&op);
    // CB-protocol ops (reserve/push/wait/pop) emitted into the DM thread during
    // the split -- e.g. the reserve/push that wrap core_read's produce into its
    // dst CB -- are result-less and would otherwise be pruned as dead, breaking
    // the producer handshake a downstream remote_store waits on.
    bool isCBProtocol = isa<ReserveOp, PushOp, WaitOp, PopOp>(&op);

    if (isDatamovementThread && !isDMAOp && !isReplicated && !isCBProtocol) {
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
// V2 compute-side CB insertion (dataflow model)
// ---------------------------------------------------------------------------
//
// Instead of wrapping compute in a SynchronizedRegionOp and assuming a single
// producer/consumer per CB, V2 drives CB op placement directly off the
// remote_load/remote_store boundaries (per @nsmith's observation that those
// program points are exactly where push/pop semantics belong) and tolerates a
// buffer with multiple intra-thread producers/consumers (e.g. a loop-carried
// accumulator) by emitting a single handshake per cross-thread edge:
//
//   input  CB (remote_load -> compute reads):  d2m.wait at the load site,
//          d2m.pop after the last compute consumer.
//   output CB (compute writes -> remote_store): d2m.reserve before the first
//          compute producer, d2m.push at the store site.
//
// Scope: handles the lowered memref/tile form the real pipeline produces --
// CB access via memref.load/store (through view ops) into the dst register,
// tile_*_block direct-CB operands, and aliased (operand_alias) load/store.

// True if `op` lies within `region` (possibly nested).
static bool isInRegion(Region &region, Operation *op) {
  for (Region *r = op->getParentRegion(); r;) {
    if (r == &region) {
      return true;
    }
    Operation *parent = r->getParentOp();
    r = parent ? parent->getParentRegion() : nullptr;
  }
  return false;
}

namespace {
// Per-CB compute-thread access summary used to decide which handshake to emit.
struct CBComputeInfo {
  bool consumed = false; // compute reads it (memref.load / tile_*_block input)
  bool produced =
      false;           // compute writes it (memref.store / tile_*_block output)
  RemoteLoadOp dmLoad; // DM partner that loads into it, if any
  RemoteStoreOp dmStore; // DM partner that stores from it, if any
  // DM partner (fabric_recv) that produces this CB from a peer's fabric write
  // (no NoC read). Acts like a dmLoad for input-CB handshaking: the compute
  // side waits/pops, the DM side reserves/pushes (the operand's own CB).
  FabricRecvOp dmRecv;
  // A `d2m.compute_signal` semaphore_inc that fences on this CB (a fused
  // producer-done signal: the inc waits for this compute output before
  // signaling a gatherer). Acts like a dmStore for output-CB handshaking, so
  // the compute side reserves/pushes the buffer; the DM-side wait/pop bracket
  // the inc (see convertDMAToExplicitCBForm).
  Operation *dmFence = nullptr;
};
} // namespace

// Compute-side CB synchronization (dataflow model, no SynchronizedRegionOp).
//
// For each CB touched by the compute thread, emit the handshake bracketing the
// compute access(es), anchored outside any wrapping loop:
//   - non-aliased input  (DM loads):   wait ... pop
//   - non-aliased output (DM stores):  reserve ... push
//   - aliased (operand_alias) buffer:  full reserve/push/wait/pop cycle, since
//     the alias makes the compute thread own the buffer's whole lifecycle.
// CB uses (including those reached through memref view ops) are rewired to the
// acquired buffer (wait/reserve result).
static LogicalResult insertComputeCBOpsV2(GenericOp generic,
                                          Block *computeBlock,
                                          PatternRewriter &rewriter) {
  Region &computeRegion = *computeBlock->getParent();

  // Pre-order index for program-order comparisons across nested blocks.
  DenseMap<Operation *, unsigned> order;
  {
    unsigned i = 0;
    computeBlock->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { order[op] = i++; });
  }

  llvm::MapVector<Value, CBComputeInfo> cbs;

  // DM partners (identify which CBs cross the thread boundary, and aliasing).
  computeBlock->walk(
      [&](RemoteLoadOp op) { cbs[op.getLocalBuffer()].dmLoad = op; });
  computeBlock->walk(
      [&](RemoteStoreOp op) { cbs[op.getLocalBuffer()].dmStore = op; });
  // fabric_recv produces the operand's own CB (keyed by the memref operand).
  computeBlock->walk(
      [&](FabricRecvOp op) { cbs[op.getMemref()].dmRecv = op; });

  // A `d2m.compute_signal` inc fences on the compute output that a gatherer
  // will read -- i.e. core_read's src. Register it as that CB's dmFence so the
  // CB gets the output handshake (compute reserve/push), turning the matmul
  // output from a raw shared buffer into a synchronized one; the DM-side
  // wait/pop bracket the inc (convertDMAToExplicitCBForm), ordering the signal
  // after this core's matmul. All core_reads in the target gather pattern share
  // one src (the matmul output), so any of them identifies the fenced CB.
  if (Value fenceCB = [&]() -> Value {
        CoreReadOp anyRead;
        computeBlock->walk([&](CoreReadOp op) { anyRead = op; });
        return anyRead ? anyRead.getSrc() : Value();
      }()) {
    computeBlock->walk([&](SemaphoreIncOp inc) {
      if (inc->hasAttr("d2m.compute_signal")) {
        cbs[fenceCB].dmFence = inc;
      }
    });
  }

  // Compute consume/produce via dst-register memref accesses (through views).
  computeBlock->walk([&](memref::LoadOp ld) {
    if (Value cb = traceComputeMemrefToCB(ld.getMemref(), generic)) {
      cbs[cb].consumed = true;
    }
  });
  computeBlock->walk([&](memref::StoreOp st) {
    if (Value cb = traceComputeMemrefToCB(st.getMemref(), generic)) {
      cbs[cb].produced = true;
    }
  });

  // Compute consume/produce via synchronizable compute ops (tile_*_block etc.).
  // RemoteLoad/RemoteStore are DM partners (handled via dmLoad/dmStore above),
  // and core_read is a DM-only op (its src/dst are DM-internal: a core->core
  // NoC read, erased from the compute thread later) -- none of them make a CB
  // compute-resident, so they must not mark produced/consumed here. Otherwise a
  // CB touched only by DM ops gets a bogus compute handshake and the DM thread
  // double-waits it (deadlock).
  for (auto &[cb, usage] : utils::getCBUsageInfo(computeRegion)) {
    for (Operation *p : usage.producers) {
      if (!isa<RemoteLoadOp, RemoteStoreOp, CoreReadOp, CoreWriteOp>(p)) {
        cbs[cb].produced = true;
      }
    }
    for (Operation *c : usage.consumers) {
      if (!isa<RemoteLoadOp, RemoteStoreOp, CoreReadOp, CoreWriteOp>(c)) {
        cbs[cb].consumed = true;
      }
    }
  }

  // tile_matmul_block reads its two input blocks (a, b) and writes its output
  // block directly -- it does NOT load/store individual tiles via memref ops
  // (the dst-register access trace above) and is deliberately not a
  // SynchronizableOpInterface op (the legacy V1 split + MarkSynchronizedBuffers
  // would then try to wrapInSynchronizedRegion its dst-register accesses and
  // crash). So neither path above sees its CB usage. Recognize it explicitly
  // here, tracing each operand's subview back to the generic's CB operand so it
  // merges with the dmLoad/dmStore entry. Without this the matmul's inputs are
  // never paired with their DMA load, the input cb_wait_front/cb_pop_front
  // handshake is silently skipped, and the matmul reads its inputs before the
  // DMA completes -- racing it.
  computeBlock->walk([&](TileMatmulBlockOp mm) {
    if (Value cb = traceComputeMemrefToCB(mm.getA(), generic)) {
      cbs[cb].consumed = true;
    }
    if (Value cb = traceComputeMemrefToCB(mm.getB(), generic)) {
      cbs[cb].consumed = true;
    }
    if (Value cb = traceComputeMemrefToCB(mm.getOutput(), generic)) {
      cbs[cb].produced = true;
    }
  });

  for (auto &[cb, info] : cbs) {
    // A buffer only needs CB synchronization if it crosses the thread
    // boundary. An output is paired with the DM store; otherwise an input is
    // paired with the DM load. Buffers with no DM partner are compute-local
    // (e.g. dst-register intermediates) and need no handshake.
    bool outputCB = info.produced && (info.dmStore || info.dmFence);
    bool inputCB = !outputCB && info.consumed && (info.dmLoad || info.dmRecv);
    if (!outputCB && !inputCB) {
      continue;
    }

    // Anchor CB ops at the DM partner's block (loop level), so a streaming
    // access inside a loop gets a per-iteration handshake matching the
    // per-iteration DMA, while an accumulator whose store sits outside the
    // reduction loop gets a single handshake around it.
    Block *partnerBlock =
        outputCB ? (info.dmStore ? info.dmStore->getBlock()
                                 : info.dmFence->getBlock())
                 : (info.dmLoad ? info.dmLoad->getBlock()
                                : info.dmRecv->getBlock());
    auto climbToPartner = [&](Operation *op) -> Operation * {
      Operation *cur = op;
      while (cur && cur->getBlock() != partnerBlock) {
        cur = cur->getParentOp();
      }
      return cur;
    };

    // Direct compute-region uses of `cb` to rewire to the acquired buffer
    // (view ops and tile_*_block; remote_load/store and core_read/core_write
    // are DM ops erased from compute later, and -- crucially -- a cross-core
    // core_read of this CB must NOT count as a compute access, else the push
    // anchors past it (e.g. after the gather's scf.if), deadlocking the fence).
    // The acquire is anchored before the earliest of these.
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : cb.getUses()) {
      Operation *owner = use.getOwner();
      if (isInRegion(computeRegion, owner) &&
          !isa<RemoteLoadOp, RemoteStoreOp, FabricRecvOp, CoreReadOp,
               CoreWriteOp>(owner)) {
        uses.push_back(&use);
      }
    }
    if (uses.empty()) {
      continue;
    }

    // Transitive accesses: follow memref view ops forward to the actual
    // load/store/tile accesses. The release (pop/push) must come after the
    // last of these, which may sit inside a loop nest below the view op.
    SmallVector<Operation *> accesses;
    {
      DenseSet<Operation *> visited;
      SmallVector<Value> worklist{cb};
      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        for (Operation *user : v.getUsers()) {
          if (!isInRegion(computeRegion, user) ||
              isa<RemoteLoadOp, RemoteStoreOp, FabricRecvOp, CoreReadOp,
                  CoreWriteOp>(user) ||
              !visited.insert(user).second) {
            continue;
          }
          if (isa<memref::CollapseShapeOp, memref::SubViewOp, memref::CastOp,
                  memref::ReinterpretCastOp>(user)) {
            worklist.push_back(user->getResult(0));
          } else {
            accesses.push_back(user);
          }
        }
      }
    }

    // Acquire anchor: earliest direct use (climbed to the partner block).
    // Release anchor: latest transitive access (climbed to the partner block).
    Operation *earliest = nullptr, *latest = nullptr;
    for (OpOperand *use : uses) {
      Operation *top = climbToPartner(use->getOwner());
      if (top && (!earliest || order.lookup(top) < order.lookup(earliest))) {
        earliest = top;
      }
    }
    for (Operation *access : accesses) {
      Operation *top = climbToPartner(access);
      if (top && (!latest || order.lookup(top) > order.lookup(latest))) {
        latest = top;
      }
    }
    if (!earliest) {
      continue;
    }
    if (!latest) {
      latest = earliest;
    }

    Location loc = generic.getLoc();
    unsigned cbIdx = generic.getOperandIndex(cb);
    bool aliasedLoad = info.dmLoad && isAliasedLoad(info.dmLoad);
    bool aliasedStore = info.dmStore && isAliasedStore(info.dmStore);

    rewriter.setInsertionPoint(earliest);
    Value cbVal = d2m::getOrCreateCB(rewriter, generic, computeBlock, cbIdx);

    Value acquired;
    if (outputCB) {
      // Output buffer (compute writes; DM stores). Compute reserves before
      // producing, pushes after; an aliased store adds the consumer half.
      acquired = rewriter.create<ReserveOp>(loc, cbVal).getResult();
      rewriter.setInsertionPointAfter(latest);
      rewriter.create<PushOp>(loc, cbVal);
      if (aliasedStore) {
        rewriter.create<WaitOp>(loc, cbVal);
        rewriter.create<PopOp>(loc, cbVal);
      }
    } else {
      // Input buffer (compute reads; DM loads). An aliased load adds the
      // producer half before the wait.
      if (aliasedLoad) {
        rewriter.create<ReserveOp>(loc, cbVal);
        rewriter.create<PushOp>(loc, cbVal);
      }
      acquired = rewriter.create<WaitOp>(loc, cbVal).getResult();
      rewriter.setInsertionPointAfter(latest);
      rewriter.create<PopOp>(loc, cbVal);
    }

    for (OpOperand *use : uses) {
      use->set(acquired);
    }
  }

  return success();
}

// ---------------------------------------------------------------------------
// Main rewriter
// ---------------------------------------------------------------------------

class D2MSplitUnifiedThreadV2Rewriter : public OpRewritePattern<GenericOp> {
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

    // V2: no SynchronizedRegionOp wrapping -- insertComputeCBOpsV2 anchors CB
    // ops directly at the remote_load/remote_store boundaries.

    Region &originalRegion = generic.getRegion(0);
    if (originalRegion.empty()) {
      return failure();
    }
    Block *originalBlock = &originalRegion.front();

    if (failed(utils::checkForIllegalSemaphoreOps(originalBlock))) {
      return failure();
    }

    // The fabric ops move to the datamovement thread; refine the fabric config
    // thread (unified -> datamovement) to match.
    auto fabricConfig = generic.getFabricConnectionConfigAttr();
    if (fabricConfig) {
      fabricConfig = fabricConfig.getWithThread(
          rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement));
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
        fabricConfig,
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

    // Compute thread: insert CB sync ops anchored at the remote_load/store
    // boundaries (dataflow model; no SynchronizedRegionOp).
    if (failed(insertComputeCBOpsV2(newGeneric, computeBlock, rewriter))) {
      return failure();
    }

    // DMA thread: convert datamovement ops to explicit CB form.
    if (failed(convertDMAToExplicitCBForm(dmBlock, rewriter))) {
      return failure();
    }

    eraseDMAOpsInComputeBlock(rewriter, computeBlock);
    eraseDeadOps(rewriter, dmBlock, /*isDatamovementThread=*/true);
    eraseDeadOps(rewriter, computeBlock, /*isDatamovementThread=*/false);

    rewriter.replaceOp(generic, newGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class D2MSplitUnifiedThreadV2
    : public impl::D2MSplitUnifiedThreadV2Base<D2MSplitUnifiedThreadV2> {
public:
  using impl::D2MSplitUnifiedThreadV2Base<
      D2MSplitUnifiedThreadV2>::D2MSplitUnifiedThreadV2Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MSplitUnifiedThreadV2Rewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
