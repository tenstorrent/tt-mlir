// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

  TT_assertv(sem0 != nullptr && mlir::isa<LocalSemaphoreType>(sem0.getType()),
             "Could not find receiversReady semaphore in generic args");
  TT_assertv(sem1 != nullptr && mlir::isa<LocalSemaphoreType>(sem1.getType()),
             "Could not find senderFinished semaphore in generic args");

  return {sem0, sem1};
}

static SmallVector<Value>
mapVirtualToPhysicalCoreIndex(OpBuilder &builder, Location loc,
                              ttcore::GridAttr grid,
                              ValueRange virtualCoreIndex) {
  SmallVector<Value> physicalCoreIndex(virtualCoreIndex.begin(),
                                       virtualCoreIndex.end());
  AffineMap map = grid.getVirtToPhysicalMap();
  if (map.isEmpty()) {
    return physicalCoreIndex;
  }

  TT_assertv(map.getNumInputs() == virtualCoreIndex.size(),
             "Expected virtual-to-physical grid map input rank to match core "
             "index rank.");
  unsigned firstCoreResult =
      map.getNumResults() == virtualCoreIndex.size() ? 0 : 1;
  TT_assertv(map.getNumResults() >= firstCoreResult + virtualCoreIndex.size(),
             "Expected virtual-to-physical grid map to have enough core "
             "coordinate results.");

  physicalCoreIndex.clear();
  physicalCoreIndex.reserve(virtualCoreIndex.size());
  for (unsigned i = 0; i < virtualCoreIndex.size(); ++i) {
    AffineMap selectedMap = AffineMap::get(
        map.getNumDims(), map.getNumSymbols(),
        {map.getResult(firstCoreResult + i)}, builder.getContext());
    physicalCoreIndex.push_back(builder.create<affine::AffineApplyOp>(
        loc, selectedMap, virtualCoreIndex));
  }
  return physicalCoreIndex;
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

    // Get parent generic op for grid mapping (needed for CoreIndexOp).
    auto genericOp = remoteLoad->getParentOfType<GenericOp>();
    TT_assertv(genericOp, "RemoteLoad must be inside a GenericOp");

    // Derive which dimensions are multicast from mcastShape.
    // A dimension is multicast if mcastShape[i] > 1.
    // Also calculate mcast volume.
    SmallVector<bool> isMcastDim;
    size_t mcastVolume = 1;
    for (Value mcastDimVal : remoteLoad.getMcastShape()) {
      int64_t dimSize = 1;
      if (auto constantOp = mcastDimVal.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constantOp.getValue())) {
          dimSize = intAttr.getInt();
        }
      }
      isMcastDim.push_back(dimSize > 1);
      mcastVolume *= dimSize;
    }

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
    Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(1));

    // Get pre-allocated semaphores for synchronization.
    // These must have been set by D2MPreallocateMcastSemaphores pass.
    auto preallocatedSems = getPreallocatedSemaphores(remoteLoad);
    Value receiversReadySemaphore = preallocatedSems.first;
    Value senderFinishedSemaphore = preallocatedSems.second;

    // Number of receivers is mcastVolume - 1 (excluding sender itself).
    // The sender waits for this many semaphore increments before multicasting.
    Value numReceiversVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(mcastVolume - 1));

    // Determine if this core is the sender.
    // The sender is at position mcastStartIndex[i] for each multicast
    // dimension. We need to check that ALL multicast dimensions have core_index
    // == mcastStartIndex. Pass grid mapping for proper virtualization support.
    Value isSender = nullptr;
    AffineMap gridMapping = genericOp.getGrid().getPhysicalToVirtMap();
    ValueRange mcastStartIndex = remoteLoad.getMcastStartIndex();
    for (size_t i = 0; i < isMcastDim.size(); ++i) {
      if (isMcastDim[i]) {
        Value coreIdx = rewriter.create<CoreIndexOp>(
            loc, static_cast<int64_t>(i), gridMapping);
        Value condition = rewriter.create<arith::CmpIOp>(
            loc, rewriter.getI1Type(), arith::CmpIPredicate::eq, coreIdx,
            mcastStartIndex[i]);
        if (isSender) {
          isSender = rewriter.create<arith::AndIOp>(loc, isSender, condition)
                         .getResult();
        } else {
          isSender = condition;
        }
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
          SmallVector<Value> physicalMcastStartIndex =
              mapVirtualToPhysicalCoreIndex(builder, loc, genericOp.getGrid(),
                                            mcastStartIndex);

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
              loc, localMemref, localMemref, physicalMcastStartIndex,
              remoteLoad.getMcastShape());
          builder.create<DMAWaitOp>(loc, mcastTx);

          // Signal receivers that sender is finished.
          builder.create<SemaphoreSetOp>(loc, senderFinishedSemaphore, one,
                                         physicalMcastStartIndex,
                                         remoteLoad.getMcastShape(),
                                         /*startDevice=*/ValueRange(),
                                         /*deviceMcastShape=*/ValueRange());

          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          // Receiver: signal ready and wait for sender to finish.
          SmallVector<Value> senderCoreIndex;
          Value zeroIdx = builder.create<arith::ConstantOp>(
              loc, builder.getIndexType(), builder.getIndexAttr(0));

          // Build sender core index by reading actual core positions.
          // For dimensions that are multicast, sender is at mcastStartIndex.
          // For non-multicast dimensions, use current core position.
          // Pass grid mapping for proper virtualization support.
          for (size_t i = 0; i < isMcastDim.size(); ++i) {
            if (isMcastDim[i]) {
              // Multicast dimension - sender is at mcastStartIndex.
              senderCoreIndex.push_back(mcastStartIndex[i]);
            } else {
              // Non-multicast dimension - use current core's position.
              Value currentCoreIdx = builder.create<CoreIndexOp>(
                  loc, static_cast<int64_t>(i), gridMapping);
              senderCoreIndex.push_back(currentCoreIdx);
            }
          }

          SmallVector<Value> physicalSenderCoreIndex =
              mapVirtualToPhysicalCoreIndex(builder, loc, genericOp.getGrid(),
                                            senderCoreIndex);
          builder.create<SemaphoreIncOp>(loc, receiversReadySemaphore, one,
                                         physicalSenderCoreIndex);
          builder.create<SemaphoreWaitOp>(loc, senderFinishedSemaphore, one,
                                          zeroIdx);

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

    Value dstMemref = rewriter.create<ReserveOp>(loc, dstCb).getResult();
    Value srcMemref = rewriter.create<WaitOp>(loc, srcCb).getResult();

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
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
