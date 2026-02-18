// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapAnalysis.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERLOADSTOREOPSTODMA
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

static std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
getLoopBounds(OpBuilder &builder, Location loc, ArrayRef<int64_t> shardShape) {
  Value zero = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                 builder.getIndexAttr(0));
  Value one = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                builder.getIndexAttr(1));
  SmallVector<Value> lbs(shardShape.size(), zero);
  SmallVector<Value> ubs(llvm::map_range(shardShape, [&](int64_t dim) {
    return builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                             builder.getIndexAttr(dim));
  }));
  SmallVector<Value> step(shardShape.size(), one);
  return std::make_tuple(lbs, ubs, step);
}

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

static size_t getElementSizeBytes(MemRefType memref) {
  mlir::Type elementType = memref.getElementType();
  auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType);
  return tileType ? tileType.getSizeBytes()
                  : elementType.getIntOrFloatBitWidth() / 8;
}

static AffineMap canonicalStridedMap(MLIRContext *context,
                                     ArrayRef<int64_t> shape, Type elementType,
                                     AffineMap map) {
  assert(map.isIdentity() && "Only identity maps are supported for now.");
  auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType);
  int64_t elementSizeBytes = tileType ? tileType.getSizeBytes()
                                      : elementType.getIntOrFloatBitWidth() / 8;
  int64_t currentStride = elementSizeBytes;
  int64_t rank = shape.size();
  mlir::AffineExpr strideExpr = mlir::getAffineConstantExpr(0, context);
  for (int64_t i = rank - 1; i >= 0; i--) {
    mlir::AffineExpr dim = mlir::getAffineDimExpr(i, context);
    mlir::AffineExpr stride =
        mlir::getAffineConstantExpr(currentStride, context);
    strideExpr = dim * stride + strideExpr;
    currentStride *= shape[i];
  }
  return mlir::AffineMap::get(shape.size(), 0, strideExpr, context);
}

static AffineMap getMemoryMap(ttcore::DeviceAttr device, Value input,
                              bool isRemote) {
  if (isRemote) {
    Operation *definingOp = input.getDefiningOp();
    if (!definingOp) {
      // If there's no defining op (e.g., block argument), use the memref type
      // directly
      MemRefType memrefType = mlir::cast<MemRefType>(input.getType());
      return device.getMemoryMap(
          std::make_pair(memrefType,
                         AffineMap::getMultiDimIdentityMap(
                             memrefType.getRank(), memrefType.getContext())),
          0 /* use default page size*/);
    }
    std::pair<MemRefType, AffineMap> underlyingMemrefAndView =
        mlir::tt::d2m::applyViews(definingOp);
    return device.getMemoryMap(underlyingMemrefAndView,
                               0 /* use default page size*/);
  }

  // For local memrefs (including CB values), get the underlying memref type
  MemRefType inputType;
  if (auto cbType = mlir::dyn_cast<CBType>(input.getType())) {
    inputType = cbType.getUnderlyingAs<MemRefType>();
  } else {
    inputType = mlir::cast<MemRefType>(input.getType());
  }
  return canonicalStridedMap(device.getContext(), inputType.getShape(),
                             inputType.getElementType(),
                             inputType.getLayout().getAffineMap());
}

template <typename Builder>
static SmallVector<Value> applyMap(Builder &builder, Location loc,
                                   AffineMap map, ValueRange index,
                                   bool isRemote) {
  auto affineApply = [&](AffineMap map, ValueRange index) {
    return builder.template create<affine::AffineApplyOp>(loc, map, index);
  };

  if (isRemote) {
    assert(map.getNumResults() == 4);
    // Break the map into respective gridY, gridX, offset "single result"
    // parts. AffineApply only supports single result affine maps.
    map = map.dropResults(0); // Drop the device index.
    auto gridY = map.dropResults({1, 2});
    auto gridX = map.dropResults({0, 2});
    auto offset = map.dropResults({0, 1});
    return {affineApply(gridY, index), affineApply(gridX, index),
            affineApply(offset, index)};
  }

  assert(map.getNumResults() == 1);
  return {affineApply(map, index)};
}

// Calculates coalescing factor using analytical method with sampling fallback.
// Mirrors the logic from GenericLowerDMAs::analyzeStream.
static size_t calculateCoalescingFactorWithFallback(
    AffineMap memoryMap, ArrayRef<int64_t> gridShape,
    ArrayRef<int64_t> shardShape, size_t elemSizeBytes,
    bool debugCoalescingInference) {

  static constexpr size_t coalescingFactorSamplingFallbackThreshold = 16;

  // Compute full shape (grid + shard)
  SmallVector<int64_t> fullShape;
  fullShape.append(gridShape.begin(), gridShape.end());
  fullShape.append(shardShape.begin(), shardShape.end());

  // Try analytical method first
  size_t coalescingFactor = ttmlir::utils::computeCoalescingFactorAnalytically(
      memoryMap, fullShape, gridShape.size(), elemSizeBytes);

  // Determine if we should fallback to sampling
  size_t analyticalChunkSize = coalescingFactor * elemSizeBytes;
  bool shouldFallbackToSampling =
      analyticalChunkSize <= coalescingFactorSamplingFallbackThreshold;

  if (shouldFallbackToSampling || debugCoalescingInference) {
    if (shouldFallbackToSampling) {
      llvm::dbgs() << "Analytical coalescing factor below threshold, "
                      "falling back to sampling based coalescing factor...\n";
    } else {
      llvm::dbgs() << "--------------------------[CoalescingFactor]------------"
                      "--------------------\n";
      llvm::dbgs() << "Computing sampling based coalescing factor...\n";
    }

    size_t sampledCoalescingFactor = ttmlir::utils::calculateCoalescingFactor(
        memoryMap, fullShape, elemSizeBytes, gridShape.size());

    if (debugCoalescingInference) {
      if (coalescingFactor == sampledCoalescingFactor) {
        llvm::dbgs() << "  [✓] Analytical and sampled coalescing "
                        "factors MATCH = "
                     << coalescingFactor << "\n";
      } else if (coalescingFactor != sampledCoalescingFactor &&
                 sampledCoalescingFactor % coalescingFactor == 0) {
        llvm::dbgs() << "  [✓] Analytical coalescing factor is valid, but "
                        "smaller than the sampled coalescing factor!\n";
        llvm::dbgs() << "    analytical = " << coalescingFactor
                     << " vs sampled = " << sampledCoalescingFactor << "\n";
        llvm::dbgs() << "    Map: " << memoryMap << "\n";
        llvm::dbgs() << "    Shape: "
                     << ttmlir::utils::formatIterable(fullShape, "x") << "\n";
        llvm::dbgs() << "  Setting coalescing factor to fallback sampled "
                        "value = "
                     << sampledCoalescingFactor << "\n";
        coalescingFactor = sampledCoalescingFactor;
      }

      if (sampledCoalescingFactor % coalescingFactor != 0) {
        llvm::dbgs() << "  [ERROR] Analytical coalescing factor is not a "
                        "divisor of sampled coalescing factor! Generated DMA "
                        "indexing is likely incorrect!\n";
        llvm::dbgs() << "    Sampled coalescing factor: "
                     << sampledCoalescingFactor << "\n";
        llvm::dbgs() << "    Analytical coalescing factor: " << coalescingFactor
                     << "\n";
        llvm::dbgs() << "    Map: " << memoryMap << "\n";
        llvm::dbgs() << "    Shape: "
                     << ttmlir::utils::formatIterable(fullShape, "x") << "\n";
        llvm::dbgs() << "  Setting coalescing factor to fallback sampled "
                        "value = "
                     << sampledCoalescingFactor << "\n";
        coalescingFactor = sampledCoalescingFactor;
      }
    } else if (shouldFallbackToSampling) {
      // Not in debug mode but we need to fallback
      coalescingFactor = sampledCoalescingFactor;
    }

    llvm::dbgs() << "--------------------------------------------------------"
                    "-------------\n";
  }

  return coalescingFactor;
}

namespace {
class D2MLowerRemoteLoadRewritePattern : public OpRewritePattern<RemoteLoadOp> {
public:
  D2MLowerRemoteLoadRewritePattern(MLIRContext *context,
                                   bool debugCoalescingInference)
      : OpRewritePattern<RemoteLoadOp>(context),
        debugCoalescingInference(debugCoalescingInference) {}

private:
  bool debugCoalescingInference;

public:
  // Helper to generate DMA read operations with proper coalescing
  // Returns the last DMA transaction value (for waiting)
  // Handles both contiguous (single DMA) and strided (loop with guarded DMAs)
  // cases
  // Note: Caller must reserve the CB and pass the resulting localMemref
  static Value generateDMAReads(OpBuilder &builder, Location loc,
                                Value remoteMemref, Value localMemref,
                                SmallVector<Value> gridIndices,
                                ArrayRef<int64_t> shardShape,
                                AffineMap remoteMemoryMap,
                                AffineMap localMemoryMap,
                                size_t coalescingFactor, size_t shardVolume) {

    if (coalescingFactor == shardVolume) {
      // Fully contiguous: single DMA operation
      SmallVector<Value> remoteIndices = gridIndices;
      SmallVector<Value> localIndices;

      Value zero = builder.create<arith::ConstantOp>(
          loc, builder.getIndexType(), builder.getIndexAttr(0));
      for (size_t i = 0; i < shardShape.size(); ++i) {
        remoteIndices.push_back(zero);
        localIndices.push_back(zero);
      }

      remoteIndices =
          applyMap(builder, loc, remoteMemoryMap, remoteIndices, true);
      localIndices =
          applyMap(builder, loc, localMemoryMap, localIndices, false);

      return builder.create<DMAReadOp>(
          loc, remoteMemref, remoteIndices, localMemref, localIndices,
          builder.getI64IntegerAttr(coalescingFactor));
    }

    // Strided/non-contiguous: generate loops with guarded DMAs
    auto [lbs, ubs, steps] = getLoopBounds(builder, loc, shardShape);
    auto nullDmaTx = builder.create<NullTxOp>(loc);

    scf::LoopNest loopNest = scf::buildLoopNest(
        builder, loc, lbs, ubs, steps, ValueRange(nullDmaTx),
        [&](OpBuilder &loopBuilder, Location innerLoc, ValueRange iters,
            ValueRange args) {
          // Build full indices: grid indices + shard iteration indices
          SmallVector<Value> remoteIndices =
              llvm::to_vector(llvm::concat<Value>(gridIndices, iters));
          SmallVector<Value> localIndices = llvm::to_vector(iters);

          // Apply memory maps
          remoteIndices = applyMap(loopBuilder, innerLoc, remoteMemoryMap,
                                   remoteIndices, true);
          localIndices = applyMap(loopBuilder, innerLoc, localMemoryMap,
                                  localIndices, false);

          // Create guarded DMA operation based on coalescing factor
          Value cfExpr = loopBuilder.create<arith::ConstantOp>(
              innerLoc, loopBuilder.getIndexType(),
              loopBuilder.getIndexAttr(coalescingFactor));
          Value zero = loopBuilder.create<arith::ConstantOp>(
              innerLoc, loopBuilder.getIndexType(),
              loopBuilder.getIntegerAttr(loopBuilder.getIndexType(), 0));

          // Construct guard function: flat_index(iters) % coalescingFactor == 0
          auto totalIterCount = zero;
          size_t currStride = 1;
          for (int i = iters.size() - 1; i >= 0; i--) {
            Value currStrideExpr = loopBuilder.create<arith::ConstantOp>(
                innerLoc, loopBuilder.getIndexType(),
                loopBuilder.getIndexAttr(currStride));
            auto scaledCount =
                loopBuilder
                    .create<arith::MulIOp>(innerLoc, currStrideExpr, iters[i])
                    .getResult();
            totalIterCount = loopBuilder
                                 .create<arith::AddIOp>(innerLoc, scaledCount,
                                                        totalIterCount)
                                 .getResult();
            currStride *= shardShape[i];
          }
          auto moduloIterCount =
              loopBuilder
                  .create<arith::RemSIOp>(innerLoc, totalIterCount, cfExpr)
                  .getResult();
          auto predicate = loopBuilder.create<arith::CmpIOp>(
              innerLoc, arith::CmpIPredicate::eq, moduloIterCount, zero);

          auto nulltx = loopBuilder.create<NullTxOp>(innerLoc);

          // Build guarded DMA
          auto ifExpr = loopBuilder.create<scf::IfOp>(
              innerLoc, TypeRange(SmallVector<Value>{nulltx}), predicate,
              true /*addThenBlock*/, true /*addElseBlock*/);

          auto thenBuilder = ifExpr.getThenBodyBuilder();
          Value dmaTx = thenBuilder.create<DMAReadOp>(
              innerLoc, remoteMemref, remoteIndices, localMemref, localIndices,
              thenBuilder.getI64IntegerAttr(coalescingFactor));
          thenBuilder.create<scf::YieldOp>(innerLoc, dmaTx);

          auto elseBuilder = ifExpr.getElseBodyBuilder();
          elseBuilder.create<scf::YieldOp>(innerLoc, args[0]);

          return SmallVector<Value>{ifExpr.getResult(0)};
        });

    return loopNest.results.front();
  }

  // Handle multicast gather pattern for RemoteLoadOp
  static LogicalResult
  handleMcastRemoteLoad(PatternRewriter &rewriter, Location loc,
                        RemoteLoadOp remoteLoad, MemRefType remoteMemrefType,
                        AffineMap remoteMemoryMap, AffineMap localMemoryMap,
                        SmallVector<Value> gridIndices,
                        ArrayRef<int64_t> shardShape, size_t coalescingFactor,
                        size_t shardVolume) {
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
      int64_t dimSize = 1; // Default: not multicast
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
    BlockArgument receiversReadySemaphore = preallocatedSems.first;
    BlockArgument senderFinishedSemaphore = preallocatedSems.second;

    // Number of receivers is mcastVolume - 1 (excluding sender itself).
    // The sender waits for this many semaphore increments before multicasting.
    Value numReceiversVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(mcastVolume - 1));

    // Determine if this core is the sender.
    // The sender is at position mcastStartIndex[i] for each multicast
    // dimension. We need to check that ALL multicast dimensions have core_index
    // == mcastStartIndex. Pass grid mapping for proper virtualization support.
    Value isSender = nullptr;
    AffineMap gridMapping = genericOp.getGrid().getMapping();
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
    // need to reserve to maintain proper circular buffer semantics
    Value localMemref = rewriter.create<ReserveOp>(loc, cb).getResult();

    rewriter.create<scf::IfOp>(
        loc, isSender,
        [&](OpBuilder &builder, Location loc) {
          // Sender: gather data from remote with proper coalescing
          // This handles both contiguous and strided memory accesses
          Value dmaTx = generateDMAReads(
              builder, loc, remoteMemref, localMemref, gridIndices, shardShape,
              remoteMemoryMap, localMemoryMap, coalescingFactor, shardVolume);
          builder.create<DMAWaitOp>(loc, dmaTx);

          // Wait for all receivers to be ready (mcastVolume - 1, excluding
          // sender)
          builder.create<SemaphoreWaitOp>(loc, receiversReadySemaphore,
                                          numReceiversVal, zero);

          // Build full indices for the local memref
          // Use the localMemref from ReserveOp (Producer) for the multicast.
          // The sender has already written data into this buffer via DMA read,
          // so we read from the same Producer buffer for multicast.
          SmallVector<Value> mcastLocalIndices;
          Value zeroLocal = builder.create<arith::ConstantOp>(
              loc, builder.getIndexType(), builder.getIndexAttr(0));
          for (size_t i = 0; i < shardShape.size(); ++i) {
            mcastLocalIndices.push_back(zeroLocal);
          }
          mcastLocalIndices =
              applyMap(builder, loc, localMemoryMap, mcastLocalIndices, false);

          // Perform multicast DMA: from local CB to local CB with multicast
          // parameters. The multicast parameters specify that the data should
          // be sent to other cores. We use localMemref (from ReserveOp) as both
          // source and destination - this is the Producer buffer that was just
          // filled by the DMA read above.
          Value mcastTx = builder.create<DMAWriteOp>(
              loc, localMemref, mcastLocalIndices, localMemref,
              mcastLocalIndices, shardVolume, remoteLoad.getMcastStartIndex(),
              remoteLoad.getMcastShape());
          builder.create<DMAWaitOp>(loc, mcastTx);

          // Signal receivers that sender is finished
          builder.create<SemaphoreSetOp>(loc, senderFinishedSemaphore, one,
                                         remoteLoad.getMcastStartIndex(),
                                         remoteLoad.getMcastShape());

          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          // Receiver: signal ready and wait for sender to finish
          SmallVector<Value> senderCoreIndex;
          Value zeroIdx = builder.create<arith::ConstantOp>(
              loc, builder.getIndexType(), builder.getIndexAttr(0));

          // Build sender core index by reading actual core positions
          // For dimensions that are multicast, sender is at mcastStartIndex
          // For non-multicast dimensions, use current core position
          // Pass grid mapping for proper virtualization support.
          for (size_t i = 0; i < isMcastDim.size(); ++i) {
            if (isMcastDim[i]) {
              // Multicast dimension - sender is at mcastStartIndex
              senderCoreIndex.push_back(mcastStartIndex[i]);
            } else {
              // Non-multicast dimension - use current core's position
              Value currentCoreIdx = builder.create<CoreIndexOp>(
                  loc, static_cast<int64_t>(i), gridMapping);
              senderCoreIndex.push_back(currentCoreIdx);
            }
          }

          builder.create<SemaphoreIncOp>(loc, receiversReadySemaphore, one,
                                         senderCoreIndex);
          builder.create<SemaphoreWaitOp>(loc, senderFinishedSemaphore, one,
                                          zeroIdx);

          // Note: CB already reserved before the if/else, so receiver has
          // proper access to the multicast data

          builder.create<scf::YieldOp>(loc);
        });
    rewriter.create<PushOp>(loc, cb);

    rewriter.eraseOp(remoteLoad);
    return success();
  }

  LogicalResult matchAndRewrite(RemoteLoadOp remoteLoad,
                                PatternRewriter &rewriter) const final {
    Location loc = remoteLoad.getLoc();
    ShapedType remoteShapedType = remoteLoad.getShapedType();

    // Assert that remote operand is a memref (tensors should have been
    // converted to memrefs before this pass)
    auto remoteMemrefType = mlir::dyn_cast<MemRefType>(remoteShapedType);
    assert(
        remoteMemrefType &&
        "remote_load operand must be a memref at this stage in the pipeline");
    if (!remoteMemrefType) {
      return rewriter.notifyMatchFailure(
          remoteLoad, "remote operand must be a memref, not a tensor");
    }

    CBType cbType = remoteLoad.getCbType();
    MemRefType localMemrefType = cbType.getUnderlyingAs<MemRefType>();

    if (!localMemrefType) {
      return rewriter.notifyMatchFailure(
          remoteLoad, "circular buffer must have memref underlying type");
    }

    // Get device layout from remote memref
    ttcore::DeviceLayoutInterface deviceLayout =
        ttcore::getDeviceLayout(remoteLoad.getMemref());
    if (!deviceLayout) {
      return rewriter.notifyMatchFailure(
          remoteLoad, "remote memref must have a device layout");
    }
    ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(remoteShapedType);
    ArrayRef<int64_t> shardShape = deviceLayout.getShardShape(remoteShapedType);

    // Get device and calculate memory map
    ttcore::DeviceAttr device = ttcore::lookupDevice(remoteLoad);
    Value remoteMemref = remoteLoad.getMemref();
    AffineMap remoteMemoryMap = getMemoryMap(device, remoteMemref, true);

    // Get local memory map - compute from the local memref type
    Value cb = remoteLoad.getCb();
    AffineMap localMemoryMap = getMemoryMap(device, cb, false);

    size_t elemSizeBytes = getElementSizeBytes(remoteMemrefType);
    size_t coalescingFactor = calculateCoalescingFactorWithFallback(
        remoteMemoryMap, gridShape, shardShape, elemSizeBytes,
        debugCoalescingInference);

    size_t shardVolume = ttmlir::utils::volume(shardShape);

    // Get grid indices from the remote_load operation
    SmallVector<Value> gridIndices = remoteLoad.getIndices();

    // Check if this is a multicast operation
    if (remoteLoad.isMcast()) {
      return handleMcastRemoteLoad(rewriter, loc, remoteLoad, remoteMemrefType,
                                   remoteMemoryMap, localMemoryMap, gridIndices,
                                   shardShape, coalescingFactor, shardVolume);
    }

    // Unicast path: reserve CB and generate DMA reads with proper coalescing
    Value localMemref = rewriter.create<ReserveOp>(loc, cb).getResult();
    Value dmaTx = generateDMAReads(
        rewriter, loc, remoteMemref, localMemref, gridIndices, shardShape,
        remoteMemoryMap, localMemoryMap, coalescingFactor, shardVolume);

    rewriter.eraseOp(remoteLoad);

    // Wait for DMA to complete
    rewriter.create<DMAWaitOp>(loc, dmaTx);
    rewriter.create<PushOp>(loc, cb);
    return success();
  }
};

class D2MLowerRemoteStoreRewritePattern
    : public OpRewritePattern<RemoteStoreOp> {
public:
  D2MLowerRemoteStoreRewritePattern(MLIRContext *context,
                                    bool debugCoalescingInference)
      : OpRewritePattern<RemoteStoreOp>(context),
        debugCoalescingInference(debugCoalescingInference) {}

private:
  bool debugCoalescingInference;

public:
  // Helper to generate DMA write operations with proper coalescing
  // Returns the last DMA transaction value (for waiting)
  // Handles both contiguous (single DMA) and strided (loop with guarded DMAs)
  // cases
  static Value generateDMAWrites(OpBuilder &builder, Location loc,
                                 Value remoteMemref,
                                 SmallVector<Value> gridIndices,
                                 ArrayRef<int64_t> shardShape,
                                 AffineMap remoteMemoryMap,
                                 AffineMap localMemoryMap, Value cb,
                                 size_t coalescingFactor, size_t shardVolume) {
    // Reserve CB to get the local memref
    Value localMemref = builder.create<WaitOp>(loc, cb).getResult();

    if (coalescingFactor == shardVolume) {
      // Fully contiguous: single DMA operation
      SmallVector<Value> remoteIndices = gridIndices;
      SmallVector<Value> localIndices;

      Value zero = builder.create<arith::ConstantOp>(
          loc, builder.getIndexType(), builder.getIndexAttr(0));
      for (size_t i = 0; i < shardShape.size(); ++i) {
        remoteIndices.push_back(zero);
        localIndices.push_back(zero);
      }

      remoteIndices =
          applyMap(builder, loc, remoteMemoryMap, remoteIndices, true);
      localIndices =
          applyMap(builder, loc, localMemoryMap, localIndices, false);

      return builder.create<DMAWriteOp>(loc, localMemref, localIndices,
                                        remoteMemref, remoteIndices,
                                        coalescingFactor);
    }

    // Strided/non-contiguous: generate loops with guarded DMAs
    auto [lbs, ubs, steps] = getLoopBounds(builder, loc, shardShape);
    auto nullDmaTx = builder.create<NullTxOp>(loc);

    scf::LoopNest loopNest = scf::buildLoopNest(
        builder, loc, lbs, ubs, steps, ValueRange(nullDmaTx),
        [&](OpBuilder &loopBuilder, Location innerLoc, ValueRange iters,
            ValueRange args) {
          // Build full indices: grid indices + shard iteration indices
          SmallVector<Value> remoteIndices =
              llvm::to_vector(llvm::concat<Value>(gridIndices, iters));
          SmallVector<Value> localIndices = llvm::to_vector(iters);

          // Apply memory maps
          remoteIndices = applyMap(loopBuilder, innerLoc, remoteMemoryMap,
                                   remoteIndices, true);
          localIndices = applyMap(loopBuilder, innerLoc, localMemoryMap,
                                  localIndices, false);

          // Create guarded DMA operation based on coalescing factor
          Value cfExpr = loopBuilder.create<arith::ConstantOp>(
              innerLoc, loopBuilder.getIndexType(),
              loopBuilder.getIndexAttr(coalescingFactor));
          Value zero = loopBuilder.create<arith::ConstantOp>(
              innerLoc, loopBuilder.getIndexType(),
              loopBuilder.getIntegerAttr(loopBuilder.getIndexType(), 0));

          // Construct guard function: flat_index(iters) % coalescingFactor == 0
          auto totalIterCount = zero;
          size_t currStride = 1;
          for (int i = iters.size() - 1; i >= 0; i--) {
            Value currStrideExpr = loopBuilder.create<arith::ConstantOp>(
                innerLoc, loopBuilder.getIndexType(),
                loopBuilder.getIndexAttr(currStride));
            auto scaledCount =
                loopBuilder
                    .create<arith::MulIOp>(innerLoc, currStrideExpr, iters[i])
                    .getResult();
            totalIterCount = loopBuilder
                                 .create<arith::AddIOp>(innerLoc, scaledCount,
                                                        totalIterCount)
                                 .getResult();
            currStride *= shardShape[i];
          }
          auto moduloIterCount =
              loopBuilder
                  .create<arith::RemSIOp>(innerLoc, totalIterCount, cfExpr)
                  .getResult();
          auto predicate = loopBuilder.create<arith::CmpIOp>(
              innerLoc, arith::CmpIPredicate::eq, moduloIterCount, zero);

          auto nulltx = loopBuilder.create<NullTxOp>(innerLoc);

          // Build guarded DMA
          auto ifExpr = loopBuilder.create<scf::IfOp>(
              innerLoc, TypeRange(SmallVector<Value>{nulltx}), predicate,
              true /*addThenBlock*/, true /*addElseBlock*/);

          auto thenBuilder = ifExpr.getThenBodyBuilder();
          Value dmaTx = thenBuilder.create<DMAWriteOp>(
              innerLoc, localMemref, localIndices, remoteMemref, remoteIndices,
              coalescingFactor);
          thenBuilder.create<scf::YieldOp>(innerLoc, dmaTx);

          auto elseBuilder = ifExpr.getElseBodyBuilder();
          elseBuilder.create<scf::YieldOp>(innerLoc, args[0]);

          return SmallVector<Value>{ifExpr.getResult(0)};
        });

    return loopNest.results.front();
  }

  LogicalResult matchAndRewrite(RemoteStoreOp remoteStore,
                                PatternRewriter &rewriter) const final {
    Location loc = remoteStore.getLoc();
    ShapedType remoteShapedType = remoteStore.getShapedType();

    // Assert that remote operand is a memref (tensors should have been
    // converted to memrefs before this pass)
    auto remoteMemrefType = mlir::dyn_cast<MemRefType>(remoteShapedType);
    assert(
        remoteMemrefType &&
        "remote_store operand must be a memref at this stage in the pipeline");
    if (!remoteMemrefType) {
      return rewriter.notifyMatchFailure(
          remoteStore, "remote operand must be a memref, not a tensor");
    }

    CBType cbType = remoteStore.getCbType();
    MemRefType localMemrefType = cbType.getUnderlyingAs<MemRefType>();

    if (!localMemrefType) {
      return rewriter.notifyMatchFailure(
          remoteStore, "circular buffer must have memref underlying type");
    }

    // Get device layout from remote memref
    ttcore::DeviceLayoutInterface deviceLayout =
        ttcore::getDeviceLayout(remoteStore.getMemref());
    if (!deviceLayout) {
      return rewriter.notifyMatchFailure(
          remoteStore, "remote memref must have a device layout");
    }
    ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(remoteShapedType);
    ArrayRef<int64_t> shardShape = deviceLayout.getShardShape(remoteShapedType);

    // Get device and calculate memory map
    ttcore::DeviceAttr device = ttcore::lookupDevice(remoteStore);
    Value remoteMemref = remoteStore.getMemref();
    AffineMap remoteMemoryMap = getMemoryMap(device, remoteMemref, true);

    // Get local memory map - compute from the CB
    Value cb = remoteStore.getCb();
    AffineMap localMemoryMap = getMemoryMap(device, cb, false);

    size_t elemSizeBytes = getElementSizeBytes(remoteMemrefType);
    size_t coalescingFactor = calculateCoalescingFactorWithFallback(
        remoteMemoryMap, gridShape, shardShape, elemSizeBytes,
        debugCoalescingInference);

    size_t shardVolume = ttmlir::utils::volume(shardShape);

    // Get grid indices from the remote_store operation
    SmallVector<Value> gridIndices = remoteStore.getIndices();

    // Generate DMA writes with proper coalescing
    Value dmaTx = generateDMAWrites(rewriter, loc, remoteMemref, gridIndices,
                                    shardShape, remoteMemoryMap, localMemoryMap,
                                    cb, coalescingFactor, shardVolume);

    rewriter.eraseOp(remoteStore);

    // Wait for DMA to complete
    rewriter.create<DMAWaitOp>(loc, dmaTx);
    // Pop the circular buffer to signal consumption
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
    patterns.add<D2MLowerRemoteLoadRewritePattern>(&getContext(),
                                                   debugCoalescingInference);
    patterns.add<D2MLowerRemoteStoreRewritePattern>(&getContext(),
                                                    debugCoalescingInference);
    populateAffineToStdConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
