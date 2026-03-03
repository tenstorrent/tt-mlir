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
// Returns a pair of (receiversReady, senderFinished) semaphores.
// Asserts if the semaphores were not pre-allocated by
// D2MPreallocateMcastSemaphores.
static std::pair<BlockArgument, BlockArgument>
getPreallocatedSemaphores(Operation *op) {
  auto arrayAttr = op->getAttrOfType<ArrayAttr>(kPreallocatedSemaphoresAttr);
  TT_assertv(arrayAttr,
             "Multicast RemoteLoadOp must have preallocated_semaphores "
             "attribute. Ensure D2MPreallocateMcastSemaphores pass runs before "
             "D2MLowerLoadStoreOpsToDMA.");
  TT_assertv(arrayAttr.size() == 2u,
             "preallocated_semaphores attribute must have exactly 2 elements");

  auto genericOp = op->getParentOfType<GenericOp>();
  TT_assertv(genericOp, "RemoteLoadOp must be inside a GenericOp");

  // Find which region contains this operation.
  Region *parentRegion = op->getParentRegion();
  while (parentRegion && parentRegion->getParentOp() != genericOp) {
    parentRegion = parentRegion->getParentOp()->getParentRegion();
  }
  TT_assertv(parentRegion, "Failed to find parent region for RemoteLoadOp");
  TT_assertv(!parentRegion->empty(), "Parent region is empty");

  Block &block = parentRegion->front();

  unsigned receiversReadyIdx = mlir::cast<IntegerAttr>(arrayAttr[0]).getInt();
  unsigned senderFinishedIdx = mlir::cast<IntegerAttr>(arrayAttr[1]).getInt();

  TT_assertv(receiversReadyIdx < block.getNumArguments(),
             "Pre-allocated receiversReady semaphore index is out of bounds");
  TT_assertv(senderFinishedIdx < block.getNumArguments(),
             "Pre-allocated senderFinished semaphore index is out of bounds");

  return std::make_pair(block.getArgument(receiversReadyIdx),
                        block.getArgument(senderFinishedIdx));
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

    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
    Value one = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexType(), rewriter.getIndexAttr(1));

    // Get pre-allocated semaphores for synchronization.
    // These must have been set by D2MPreallocateMcastSemaphores pass.
    auto preallocatedSems = getPreallocatedSemaphores(remoteLoad);
    BlockArgument receiversReadySemaphore = preallocatedSems.first;
    BlockArgument senderFinishedSemaphore = preallocatedSems.second;

    // Number of receivers is mcastVolume - 1 (excluding sender itself).
    // The sender waits for this many semaphore increments before multicasting.
    Value numReceiversVal =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexType(),
                                  rewriter.getIndexAttr(mcastVolume - 1));

    // Determine if this core is the sender.
    // The sender is at position mcastStartIndex[i] for each multicast
    // dimension. We need to check that ALL multicast dimensions have core_index
    // == mcastStartIndex. Pass grid mapping for proper virtualization support.
    Value isSender = nullptr;
    AffineMap gridMapping = genericOp.getGrid().getMapping();
    ValueRange mcastStartIndex = remoteLoad.getMcastStartIndex();
    for (size_t i = 0; i < isMcastDim.size(); ++i) {
      if (isMcastDim[i]) {
        Value coreIdx = CoreIndexOp::create(
            rewriter, loc, static_cast<int64_t>(i), gridMapping);
        Value condition = arith::CmpIOp::create(
            rewriter, loc, rewriter.getI1Type(), arith::CmpIPredicate::eq,
            coreIdx, mcastStartIndex[i]);
        if (isSender) {
          isSender = arith::AndIOp::create(rewriter, loc, isSender, condition)
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
              loc, localMemref, localMemref, remoteLoad.getMcastStartIndex(),
              remoteLoad.getMcastShape());
          DMAWaitOp::create(builder, loc, mcastTx);

          // Signal receivers that sender is finished.
          builder.create<SemaphoreSetOp>(loc, senderFinishedSemaphore, one,
                                         remoteLoad.getMcastStartIndex(),
                                         remoteLoad.getMcastShape());

          scf::YieldOp::create(builder, loc);
        },
        [&](OpBuilder &builder, Location loc) {
          // Receiver: signal ready and wait for sender to finish.
          SmallVector<Value> senderCoreIndex;
          Value zeroIdx = arith::ConstantOp::create(
              builder, loc, builder.getIndexType(), builder.getIndexAttr(0));

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

          SemaphoreIncOp::create(builder, loc, receiversReadySemaphore, one,
                                 senderCoreIndex);
          SemaphoreWaitOp::create(builder, loc, senderFinishedSemaphore, one,
                                  zeroIdx);

          // Note: CB already reserved before the if/else, so receiver has
          // proper access to the multicast data.

          scf::YieldOp::create(builder, loc);
        });
    PushOp::create(rewriter, loc, cb);

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

    // Wait on CB, emit shard-level dma_write, wait, pop
    Value localMemref = rewriter.create<WaitOp>(loc, cb).getResult();
    Value dmaTx = rewriter.create<DMAWriteOp>(loc, localMemref, remoteMemref,
                                              gridIndices);

    rewriter.eraseOp(remoteStore);

    // Wait for DMA to complete.
    rewriter.create<DMAWaitOp>(loc, dmaTx);
    // Pop the circular buffer to signal consumption.
    rewriter.create<PopOp>(loc, cb);
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
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
