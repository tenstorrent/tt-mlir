// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERLOADSTOREOPSTODMA
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

// Attribute name for pre-allocated semaphore indices (set by
// D2MPreallocateMcastSemaphores pass).
constexpr StringRef kPreallocatedSemaphoresAttr = "preallocated_semaphores";

// Helper to get pre-allocated semaphores for a multicast RemoteLoadOp.
// Returns the captured semaphore Values from the enclosing generic's
// additionalArgs, using the absolute arg indices stored by
// D2MPreallocateMcastSemaphores. NormalizeThreadArgs will later replace
// these direct references with d2m.get_arg ops.
static std::pair<Value, Value> getPreallocatedSemaphores(Operation *op) {
  auto arrayAttr = op->getAttrOfType<ArrayAttr>(kPreallocatedSemaphoresAttr);
  TT_assertv(arrayAttr,
             "Multicast RemoteLoadOp must have preallocated_semaphores "
             "attribute. Ensure D2MPreallocateMcastSemaphores pass runs before "
             "D2MLowerLoadStoreOpsToDMA.");
  TT_assertv(arrayAttr.size() == 2u,
             "preallocated_semaphores attribute must have exactly 2 elements");

  int64_t sem0AbsIdx = mlir::cast<IntegerAttr>(arrayAttr[0]).getInt();
  int64_t sem1AbsIdx = mlir::cast<IntegerAttr>(arrayAttr[1]).getInt();

  auto genericOp = op->getParentOfType<GenericOp>();
  TT_assertv(genericOp, "RemoteLoadOp must be inside a GenericOp");

  Value sem0 = genericOp.getOperands()[sem0AbsIdx];
  Value sem1 = genericOp.getOperands()[sem1AbsIdx];

  TT_assertv((sem0 != nullptr && mlir::isa<LocalSemaphoreType>(sem0.getType())),
             "Could not find receiversReady semaphore in generic args");
  TT_assertv((sem1 != nullptr && mlir::isa<LocalSemaphoreType>(sem1.getType())),
             "Could not find senderFinished semaphore in generic args");

  return {sem0, sem1};
}

namespace {
class D2MLowerRemoteLoadRewritePattern : public OpRewritePattern<RemoteLoadOp> {
public:
  using OpRewritePattern<RemoteLoadOp>::OpRewritePattern;

  // Handle multicast gather pattern for RemoteLoadOp.
  static LogicalResult handleMcastRemoteLoad(PatternRewriter &rewriter,
                                             Location loc,
                                             RemoteLoadOp remoteLoad) {
    Value cb = remoteLoad.getCb();
    Value remoteMemref = remoteLoad.getMemref();

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
    Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(1));

    // Calculate mcast volume dynamically by multiplying all mcastShape dims.
    Value mcastVolume = one;
    for (Value mcastDimVal : remoteLoad.getMcastShape()) {
      mcastVolume =
          rewriter.create<arith::MulIOp>(loc, mcastVolume, mcastDimVal);
    }

    // Get pre-allocated semaphores for synchronization.
    // These must have been set by D2MPreallocateMcastSemaphores pass.
    auto preallocatedSems = getPreallocatedSemaphores(remoteLoad);
    Value receiversReadySemaphore = preallocatedSems.first;
    Value senderFinishedSemaphore = preallocatedSems.second;

    // Number of receivers is mcastVolume - 1 (excluding sender itself).
    // The sender waits for this many semaphore increments before multicasting.
    Value numReceiversVal =
        rewriter.create<arith::SubIOp>(loc, mcastVolume, one);

    // Determine if this core is the sender.
    // The sender is at position mcastStartIndex[i] for each multicast
    // dimension. We need to check that ALL multicast dimensions have core_index
    // == mcastStartIndex.
    Value isSender = nullptr;
    ValueRange mcastStartIndex = remoteLoad.getMcastStartIndex();
    for (auto [i, mcastIdx] : llvm::enumerate(mcastStartIndex)) {
      Value coreIdx =
          rewriter.create<CoreIndexOp>(loc, static_cast<int64_t>(i));
      Value condition = rewriter.create<arith::CmpIOp>(
          loc, rewriter.getI1Type(), arith::CmpIPredicate::eq, coreIdx,
          mcastIdx);
      if (isSender) {
        isSender = rewriter.create<arith::AndIOp>(loc, isSender, condition)
                       .getResult();
      } else {
        isSender = condition;
      }
    }
    TT_assertv(isSender, "No multicast dimensions found in mcastShape");

    // Reserve CB unconditionally before branching - both sender and receiver
    // need to reserve to maintain proper circular buffer semantics.
    Value localMemref = rewriter.create<ReserveOp>(loc, cb).getResult();

    SmallVector<Value> gridIndices = remoteLoad.getIndices();

    rewriter.create<scf::IfOp>(
        loc, isSender,
        [&](OpBuilder &builder, Location loc) {
          // Sender: shard-level DMA read from remote.
          Value dmaTx = builder.create<DMAReadOp>(loc, remoteMemref,
                                                  gridIndices, localMemref);
          builder.create<DMAWaitOp>(loc, dmaTx);

          // Wait for all receivers to be ready (mcastVolume - 1, excluding
          // sender).
          builder.create<SemaphoreWaitOp>(loc, receiversReadySemaphore,
                                          numReceiversVal, zero);

          // Perform shard-level multicast DMA write: from local CB to local CB
          // with multicast parameters. The multicast parameters specify that
          // the data should be sent to other cores. We use localMemref (from
          // ReserveOp) as both source and destination - this is the Producer
          // buffer that was just filled by the DMA read above.
          Value mcastTx = builder.create<DMAWriteOp>(
              loc, localMemref, localMemref, mcastStartIndex,
              remoteLoad.getMcastShape());
          builder.create<DMAWaitOp>(loc, mcastTx);

          // Signal receivers that sender is finished.
          builder.create<SemaphoreSetOp>(loc, senderFinishedSemaphore, one,
                                         mcastStartIndex,
                                         remoteLoad.getMcastShape(),
                                         /*startDevice=*/ValueRange(),
                                         /*deviceMcastShape=*/ValueRange());

          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<SemaphoreIncOp>(loc, receiversReadySemaphore, one,
                                         mcastStartIndex);
          builder.create<SemaphoreWaitOp>(loc, senderFinishedSemaphore, one,
                                          zero);

          // Note: CB already reserved before the if/else, so receiver has
          // proper access to the multicast data.

          builder.create<scf::YieldOp>(loc);
        });
    rewriter.create<PushOp>(loc, cb);

    rewriter.eraseOp(remoteLoad);
    return success();
  }

  LogicalResult matchAndRewrite(RemoteLoadOp remoteLoad,
                                PatternRewriter &rewriter) const final {
    Location loc = remoteLoad.getLoc();

    // Assert that remote operand is a memref (tensors should have been
    // converted to memrefs before this pass).
    auto remoteMemrefType =
        mlir::dyn_cast<MemRefType>(remoteLoad.getShapedType());
    assert(
        remoteMemrefType &&
        "remote_load operand must be a memref at this stage in the pipeline");
    if (!remoteMemrefType) {
      return rewriter.notifyMatchFailure(
          remoteLoad, "remote operand must be a memref, not a tensor");
    }

    Value remoteMemref = remoteLoad.getMemref();
    SmallVector<Value> gridIndices = remoteLoad.getIndices();

    // Implicit (local buffer) form: DMA-read the shard straight into the local
    // buffer memref (e.g. a datamovement-thread CCL kernel's scratch buffer).
    // No CB to reserve/push; the buffer is already allocated.
    if (!remoteLoad.isExplicitCBForm()) {
      assert(remoteLoad.getLocalBuffer() &&
             "remote_load must have either a CB or a local buffer");
      if (remoteLoad.isMcast()) {
        return rewriter.notifyMatchFailure(
            remoteLoad, "multicast remote_load requires explicit CB form");
      }
      Value localMemref = remoteLoad.getLocalBuffer();
      Value dmaTx = rewriter.create<DMAReadOp>(loc, remoteMemref, gridIndices,
                                               localMemref);
      rewriter.eraseOp(remoteLoad);
      rewriter.create<DMAWaitOp>(loc, dmaTx);
      return success();
    }

    CBType cbType = remoteLoad.getCbType();
    if (!cbType.getUnderlyingAs<MemRefType>()) {
      return rewriter.notifyMatchFailure(
          remoteLoad, "circular buffer must have memref underlying type");
    }

    if (remoteLoad.isMcast()) {
      return handleMcastRemoteLoad(rewriter, loc, remoteLoad);
    }

    // Unicast path: reserve CB, emit shard-level dma_read, wait, push.
    Value cb = remoteLoad.getCb();

    Value localMemref = rewriter.create<ReserveOp>(loc, cb).getResult();
    Value dmaTx =
        rewriter.create<DMAReadOp>(loc, remoteMemref, gridIndices, localMemref);

    rewriter.eraseOp(remoteLoad);

    // Wait for DMA to complete.
    rewriter.create<DMAWaitOp>(loc, dmaTx);
    rewriter.create<PushOp>(loc, cb);
    return success();
  }
};

class D2MLowerRemoteStoreRewritePattern
    : public OpRewritePattern<RemoteStoreOp> {
public:
  using OpRewritePattern<RemoteStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RemoteStoreOp remoteStore,
                                PatternRewriter &rewriter) const final {
    Location loc = remoteStore.getLoc();

    // Assert that remote operand is a memref (tensors should have been
    // converted to memrefs before this pass).
    auto remoteMemrefType =
        mlir::dyn_cast<MemRefType>(remoteStore.getShapedType());
    assert(
        remoteMemrefType &&
        "remote_store operand must be a memref at this stage in the pipeline");
    if (!remoteMemrefType) {
      return rewriter.notifyMatchFailure(
          remoteStore, "remote operand must be a memref, not a tensor");
    }

    Value remoteMemref = remoteStore.getMemref();
    SmallVector<Value> gridIndices = remoteStore.getIndices();
    ValueRange startDevice = remoteStore.getStartDevice();
    ValueRange deviceMcastShape = remoteStore.getDeviceMcastShape();

    // The store source is either an explicit CB (wait on it, pop when done) or,
    // in implicit form, a plain local-buffer memref (e.g. a datamovement-thread
    // CCL kernel that loaded into a scratch buffer). Post-bufferization the
    // local buffer is already the memref a CB WaitOp would have yielded, so the
    // two forms share the same dma_write; only the CB form needs wait/pop.
    Value cb;
    Value localMemref;
    if (remoteStore.isExplicitCBForm()) {
      CBType cbType = remoteStore.getCbType();
      if (!cbType.getUnderlyingAs<MemRefType>()) {
        return rewriter.notifyMatchFailure(
            remoteStore, "circular buffer must have memref underlying type");
      }
      cb = remoteStore.getCb();
      localMemref = rewriter.create<WaitOp>(loc, cb).getResult();
    } else {
      localMemref = remoteStore.getLocalBuffer();
      assert(localMemref &&
             "remote_store must have either a CB or a local buffer");
    }

    // Scratch-dst mode: a cross-device store into a gridless #l1 scratch (the
    // CCL tmp-buffer). The scratch has no device layout / grid memory-map, so we
    // emit a fully-indexed write with explicit [0,0,0] dst indices (core 0,0,
    // offset 0) -- bypassing the grid memory-map expansion. buildNocEndpoint
    // then resolves the dst to castCBTypeAsAddress(scratchCB) at the peer's
    // core (0,0), i.e. the symmetric scratch (every device runs the same SPMD
    // kernel, so the scratch CB sits at the same L1 offset everywhere).
    bool isScratchDst = !ttcore::hasDeviceLayout(remoteMemref) &&
                        !startDevice.empty();
    Value dmaTx;
    if (isScratchDst) {
      Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      SmallVector<Value> srcIdx = {c0};
      SmallVector<Value> dstIdx = {c0, c0, c0};
      int64_t numElems = 1;
      for (int64_t d : remoteMemrefType.getShape()) {
        numElems *= d;
      }
      dmaTx = rewriter.create<DMAWriteOp>(
          loc, localMemref, ValueRange(srcIdx), remoteMemref, ValueRange(dstIdx),
          static_cast<size_t>(numElems), startDevice, deviceMcastShape);
    } else {
      dmaTx =
          rewriter.create<DMAWriteOp>(loc, localMemref, remoteMemref, gridIndices,
                                      startDevice, deviceMcastShape);
    }

    if (remoteStore.getSemaphore()) {
      auto incr = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      rewriter.create<SemaphoreIncOp>(loc, remoteStore.getSemaphore(), incr,
                                      remoteStore.getSemaphoreIndices(),
                                      startDevice, deviceMcastShape);
    }

    rewriter.eraseOp(remoteStore);

    // Wait for DMA to complete.
    rewriter.create<DMAWaitOp>(loc, dmaTx);
    // Pop the circular buffer to signal consumption (CB form only).
    if (cb) {
      rewriter.create<PopOp>(loc, cb);
    }
    return success();
  }
};

class D2MLowerDMACopyRewritePattern : public OpRewritePattern<LocalCopyOp> {
public:
  using OpRewritePattern<LocalCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalCopyOp dmaCopy,
                                PatternRewriter &rewriter) const final {
    if (!dmaCopy.isExplicitCBForm()) {
      return rewriter.notifyMatchFailure(
          dmaCopy, "local_copy is not in explicit CB form");
    }

    Location loc = dmaCopy.getLoc();
    Value srcCb = dmaCopy.getSrcCb();
    Value dstCb = dmaCopy.getDstCb();

    Value srcMemref = rewriter.create<WaitOp>(loc, srcCb).getResult();
    Value dstMemref = rewriter.create<ReserveOp>(loc, dstCb).getResult();

    auto memTxType = rewriter.getType<MemTxType>(DMAType::Read);
    auto newCopy = rewriter.create<LocalCopyOp>(
        loc, memTxType, srcMemref, dstMemref, dmaCopy.getIndexingMaps());
    rewriter.eraseOp(dmaCopy);

    rewriter.create<DMAWaitOp>(loc, newCopy.getResult());
    rewriter.create<PushOp>(loc, dstCb);
    rewriter.create<PopOp>(loc, srcCb);

    return success();
  }
};

// fabric_recv: the operand's CB slot was already written by a peer's
// cross-device remote_store (arrival ordered by a preceding semaphore_wait), so
// there is NO local read. Just expose the slot to the consumer: reserve the
// CB write slot and push it. This is the whole point of fabric_recv -- it
// avoids the NoC read-back (noc_async_read) that contends with the open fabric
// connection on the single DM thread.
class D2MLowerFabricRecvRewritePattern
    : public OpRewritePattern<FabricRecvOp> {
public:
  using OpRewritePattern<FabricRecvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FabricRecvOp recvOp,
                                PatternRewriter &rewriter) const final {
    if (!recvOp.isExplicitCBForm()) {
      return rewriter.notifyMatchFailure(
          recvOp, "fabric_recv must be in explicit CB form (split-v2 converts "
                  "it); the implicit form should not reach this pass");
    }
    Location loc = recvOp.getLoc();
    Value cb = recvOp.getCb();
    // Reserve the slot the peer fabric-wrote, then push it to the consumer.
    // No dma_read / noc_async_read.
    rewriter.create<ReserveOp>(loc, cb);
    rewriter.create<PushOp>(loc, cb);
    rewriter.eraseOp(recvOp);
    return success();
  }
};

class D2MLowerLoadStoreOpsToDMA
    : public impl::D2MLowerLoadStoreOpsToDMABase<D2MLowerLoadStoreOpsToDMA> {
public:
  using impl::D2MLowerLoadStoreOpsToDMABase<
      D2MLowerLoadStoreOpsToDMA>::D2MLowerLoadStoreOpsToDMABase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MLowerRemoteLoadRewritePattern>(&getContext());
    patterns.add<D2MLowerRemoteStoreRewritePattern>(&getContext());
    patterns.add<D2MLowerFabricRecvRewritePattern>(&getContext());
    patterns.add<D2MLowerDMACopyRewritePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
