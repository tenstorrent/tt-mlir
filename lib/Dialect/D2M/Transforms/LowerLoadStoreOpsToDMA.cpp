// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

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
    Value remoteMemref = remoteLoad.getMemref();
    SmallVector<Value> gridIndices = remoteLoad.getIndices();

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

    CBType cbType = remoteStore.getCbType();
    if (!cbType.getUnderlyingAs<MemRefType>()) {
      return rewriter.notifyMatchFailure(
          remoteStore, "circular buffer must have memref underlying type");
    }

    Value cb = remoteStore.getCb();
    Value remoteMemref = remoteStore.getMemref();
    SmallVector<Value> gridIndices = remoteStore.getIndices();
    ValueRange startDevice = remoteStore.getStartDevice();
    ValueRange deviceMcastShape = remoteStore.getDeviceMcastShape();

    // Wait on CB, emit shard-level dma_write, wait, pop
    Value localMemref = rewriter.create<WaitOp>(loc, cb).getResult();
    Value dmaTx =
        rewriter.create<DMAWriteOp>(loc, localMemref, remoteMemref, gridIndices,
                                    startDevice, deviceMcastShape);

    if (remoteStore.getSemaphore()) {
      auto incr = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      rewriter.create<SemaphoreIncOp>(loc, remoteStore.getSemaphore(), incr,
                                      remoteStore.getSemaphoreIndices(),
                                      startDevice, deviceMcastShape);
    }

    rewriter.eraseOp(remoteStore);

    // Wait for DMA to complete.
    rewriter.create<DMAWaitOp>(loc, dmaTx);
    // Pop the circular buffer to signal consumption.
    rewriter.create<PopOp>(loc, cb);
    return success();
  }
};

class D2MLowerGatherCoreRewritePattern : public OpRewritePattern<GatherCoreOp> {
public:
  using OpRewritePattern<GatherCoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherCoreOp gather,
                                PatternRewriter &rewriter) const final {
    Location loc = gather.getLoc();

    // Tensors should have been bufferized by the time this pass runs.
    if (!mlir::isa<MemRefType>(gather.getSrc().getType()) ||
        !mlir::isa<MemRefType>(gather.getDst().getType())) {
      return rewriter.notifyMatchFailure(
          gather, "src and dst must be memrefs at this stage in the pipeline");
    }

    auto genericOp = gather->getParentOfType<GenericOp>();
    TT_assertv(genericOp, "GatherCoreOp must be inside a GenericOp");

    Value src = gather.getSrc();
    Value dst = gather.getDst();
    ValueRange groupStart = gather.getGroupStartIndex();
    ValueRange groupShape = gather.getGroupShape();
    ValueRange collectorIdx = gather.getCollectorIndex();
    TT_assertv((groupStart.size() == 2 && groupShape.size() == 2 &&
                collectorIdx.size() == 2),
               "GatherCoreOp must have 2D group / collector indices (V1)");

    // Pre-allocated semaphores: sourceReady (sources -> collector) and
    // collectorDone (collector -> sources). The same attribute scheme is
    // used as for mcast remote loads, so getPreallocatedSemaphores works
    // verbatim.
    auto preallocatedSems = getPreallocatedSemaphores(gather);
    Value sourceReady = preallocatedSems.first;
    Value collectorDone = preallocatedSems.second;

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
    Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(1));

    // numSources = (groupShape[0] * groupShape[1]) - 1. The collector does
    // not signal sourceReady on itself; mirrors the mcast lowering where
    // the sender does not signal receiversReady on itself.
    Value groupVolume =
        rewriter.create<arith::MulIOp>(loc, groupShape[0], groupShape[1]);
    Value numSources = rewriter.create<arith::SubIOp>(loc, groupVolume, one);

    // Inner-loop bounds: end = start + size on each axis.
    Value endY =
        rewriter.create<arith::AddIOp>(loc, groupStart[0], groupShape[0]);
    Value endX =
        rewriter.create<arith::AddIOp>(loc, groupStart[1], groupShape[1]);

    // isCollector: virtual core index == collectorIdx on every axis.
    AffineMap gridMapping = genericOp.getGrid().getPhysicalToVirtMap();
    Value isCollector;
    for (auto [i, collIdx] : llvm::enumerate(collectorIdx)) {
      Value coreIdx = rewriter.create<CoreIndexOp>(loc, static_cast<int64_t>(i),
                                                   gridMapping);
      Value cond = rewriter.create<arith::CmpIOp>(loc, rewriter.getI1Type(),
                                                  arith::CmpIPredicate::eq,
                                                  coreIdx, collIdx);
      isCollector = isCollector
                        ? rewriter.create<arith::AndIOp>(loc, isCollector, cond)
                              .getResult()
                        : cond;
    }

    // Pre-compute physical coordinates we need across the branches:
    // - physGroupStart: start of the multicast region for collectorDone.
    // - physCollector: target of source-side semaphore_inc.
    SmallVector<Value> physGroupStart = mapVirtualToPhysicalCoreIndex(
        rewriter, loc, genericOp.getGrid(), groupStart);
    SmallVector<Value> physCollector = mapVirtualToPhysicalCoreIndex(
        rewriter, loc, genericOp.getGrid(), collectorIdx);

    rewriter.create<scf::IfOp>(
        loc, isCollector,
        [&](OpBuilder &builder, Location loc) {
          // Collector branch: wait for sources, pull each source's payload,
          // signal the entire group.
          builder.create<SemaphoreWaitOp>(loc, sourceReady, numSources, zero);

          // Row-major iteration over the gather group. The collector's own
          // coordinate is included; the resulting "self read" is correct
          // (just slightly inefficient) and avoids an inner-loop branch.
          builder.create<scf::ForOp>(
              loc, groupStart[0], endY, one, ValueRange(),
              [&](OpBuilder &yBuilder, Location yLoc, Value srcY, ValueRange) {
                yBuilder.create<scf::ForOp>(
                    yLoc, groupStart[1], endX, one, ValueRange(),
                    [&](OpBuilder &xBuilder, Location xLoc, Value srcX,
                        ValueRange) {
                      SmallVector<Value> physSrc =
                          mapVirtualToPhysicalCoreIndex(xBuilder, xLoc,
                                                        genericOp.getGrid(),
                                                        {srcY, srcX});
                      // Shard-level dma_read: read the entire src shard
                      // from core[physSrc] into dst on the current core.
                      // The expansion to fully indexed form (and the
                      // srcCore-aware NoC lowering) is handled by the
                      // downstream passes.
                      Value tx = xBuilder.create<DMAReadOp>(
                          xLoc, src, /*srcIndices=*/ValueRange(), dst,
                          /*srcCore=*/ValueRange(physSrc));
                      xBuilder.create<DMAWaitOp>(xLoc, tx);
                      xBuilder.create<scf::YieldOp>(xLoc);
                    });
                yBuilder.create<scf::YieldOp>(yLoc);
              });

          // Signal collector-done to every core in the group. The collector
          // itself is in this multicast region but does not wait on the
          // semaphore.
          builder.create<SemaphoreSetOp>(loc, collectorDone, one,
                                         physGroupStart, groupShape,
                                         /*startDevice=*/ValueRange(),
                                         /*deviceMcastShape=*/ValueRange());

          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          // Source branch: signal collector that we are ready, then wait
          // until the collector has finished pulling us.
          builder.create<SemaphoreIncOp>(loc, sourceReady, one, physCollector);
          builder.create<SemaphoreWaitOp>(loc, collectorDone, one, zero);
          builder.create<scf::YieldOp>(loc);
        });

    rewriter.eraseOp(gather);
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

class D2MLowerLoadStoreOpsToDMA
    : public impl::D2MLowerLoadStoreOpsToDMABase<D2MLowerLoadStoreOpsToDMA> {
public:
  using impl::D2MLowerLoadStoreOpsToDMABase<
      D2MLowerLoadStoreOpsToDMA>::D2MLowerLoadStoreOpsToDMABase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MLowerRemoteLoadRewritePattern>(&getContext());
    patterns.add<D2MLowerRemoteStoreRewritePattern>(&getContext());
    patterns.add<D2MLowerDMACopyRewritePattern>(&getContext());
    patterns.add<D2MLowerGatherCoreRewritePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
