// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

// Helper to create a semaphore block argument in the parent generic op
// Returns a pair of (current block's semaphore, region index)
static std::pair<BlockArgument, unsigned>
createSemaphore(OpBuilder &builder, Location loc, Operation *op) {
  // Find the parent GenericOp and add semaphore to all its regions
  auto genericOp = op->getParentOfType<GenericOp>();
  TT_assertv(genericOp, "RemoteLoad/Store must be inside a GenericOp");

  // Find which region contains the operation
  Region *parentRegion = op->getParentRegion();
  unsigned regionIndex = 0;
  for (unsigned i = 0; i < genericOp->getNumRegions(); ++i) {
    if (&genericOp->getRegion(i) == parentRegion) {
      regionIndex = i;
      break;
    }
  }

  // Add semaphore argument to all regions
  BlockArgument currentSemaphore = nullptr;
  for (unsigned i = 0; i < genericOp->getNumRegions(); ++i) {
    Region &region = genericOp->getRegion(i);
    if (!region.empty()) {
      Block &block = region.front();
      BlockArgument sem =
          block.addArgument(builder.getType<SemaphoreType>(), loc);
      if (i == regionIndex) {
        currentSemaphore = sem;
      }
    }
  }

  TT_assertv(currentSemaphore, "Failed to create semaphore for current region");
  return {currentSemaphore, regionIndex};
}

static size_t getElementSizeBytes(MemRefType memref) {
  mlir::Type elementType = memref.getElementType();
  auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType);
  return tileType ? tileType.getSizeBytes()
                  : elementType.getIntOrFloatBitWidth() / 8;
}

static size_t calculateCoalescingFactor(AffineMap memoryMap,
                                        ArrayRef<int64_t> gridShape,
                                        ArrayRef<int64_t> shardShape,
                                        size_t elemSizeBytes) {
  size_t coalescingFactor = ttmlir::utils::volume(shardShape);
  SmallVector<int64_t> memoryIndex;
  memoryIndex.resize(gridShape.size() + shardShape.size());
  ttmlir::utils::sample(gridShape, [&](ArrayRef<int64_t> gridIndex) {
    size_t currentCoalescingFactor = 0;
    SmallVector<int64_t, 4> nextAddress;
    ttmlir::utils::sample(shardShape, [&](ArrayRef<int64_t> shardIndex) {
      for (unsigned i = 0; i < gridIndex.size(); i++) {
        memoryIndex[i] = gridIndex[i];
      }
      for (unsigned i = 0; i < shardIndex.size(); i++) {
        memoryIndex[gridIndex.size() + i] = shardIndex[i];
      }
      SmallVector<int64_t, 4> address = memoryMap.compose(memoryIndex);
      if (nextAddress.empty() || nextAddress == address) {
        ++currentCoalescingFactor;
      } else {
        coalescingFactor = std::gcd(coalescingFactor, currentCoalescingFactor);
        if (coalescingFactor == 1) {
          return;
        }
        currentCoalescingFactor = 1;
      }
      nextAddress = address;
      nextAddress.back() += elemSizeBytes;
    });
  });
  return coalescingFactor;
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

namespace {
class D2MLowerRemoteLoadRewritePattern : public OpRewritePattern<RemoteLoadOp> {
public:
  using OpRewritePattern<RemoteLoadOp>::OpRewritePattern;

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

  // Calculate which dimensions are multicast dimensions based on the parent
  // GenericOp's indexing maps and iterator types. This matches the logic in
  // GenericGenerateDatamovement::calculateMcastIterators.
  // Returns a vector of bools indicating which grid dimensions are multicast.
  static SmallVector<bool> calculateMcastDimensions(Operation *op,
                                                    Value operand,
                                                    ttcore::GridAttr grid) {
    auto genericOp = op->getParentOfType<GenericOp>();
    TT_assertv(genericOp, "RemoteLoad/Store must be inside a GenericOp");

    // Find the operand index in the generic op
    unsigned operandIdx = 0;
    for (OpOperand &genericOperand : genericOp->getOpOperands()) {
      // The operand we're looking for is the remote memref, which should
      // match one of the generic op's operands
      if (genericOperand.get() == operand) {
        break;
      }
      operandIdx++;
    }

    // Get the indexing map for this operand
    ArrayAttr indexingMaps = genericOp.getIndexingMaps();
    TT_assertv(operandIdx < indexingMaps.size(),
               "Operand not found in generic op");
    AffineMap operandIndexingMap =
        mlir::cast<AffineMapAttr>(indexingMaps[operandIdx]).getValue();

    // Get iterator types
    ArrayAttr iteratorTypes = genericOp.getIteratorTypes();

    // Determine multicast dimensions based on indexing map and iterator types
    SmallVector<bool> isMcastDim;
    isMcastDim.reserve(grid.getShape().size());

    for (unsigned dim = 0; dim < grid.getShape().size(); dim++) {
      AffineExpr result = operandIndexingMap.getResult(dim);

      bool isReduction = false;
      if (mlir::isa<AffineConstantExpr>(result)) {
        // Constant expression means this operand doesn't vary with this grid
        // dimension - treat as parallel (not multicast)
        isReduction = false;
      } else if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(result)) {
        unsigned dimPosition = dimExpr.getPosition();
        auto iterType =
            mlir::cast<ttcore::IteratorTypeAttr>(iteratorTypes[dimPosition]);
        isReduction = iterType.getValue() == ttcore::IteratorType::Reduction;
      }
      isMcastDim.push_back(isReduction);
    }

    return isMcastDim;
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

    // Get parent generic op and calculate multicast dimensions
    auto genericOp = remoteLoad->getParentOfType<GenericOp>();
    TT_assertv(genericOp, "RemoteLoad must be inside a GenericOp");

    // Determine which dimensions are multicast based on iterator types
    SmallVector<bool> isMcastDim =
        calculateMcastDimensions(remoteLoad, remoteMemref, genericOp.getGrid());

    // Calculate mcast volume from mcastShape
    size_t mcastVolume = 1;
    for (Value mcastDim : remoteLoad.getMcastShape()) {
      if (auto constantOp = mcastDim.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constantOp.getValue())) {
          mcastVolume *= intAttr.getInt();
        }
      }
    }

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
    Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(1));

    // Create semaphores for synchronization
    auto semResult1 = createSemaphore(rewriter, loc, remoteLoad);
    BlockArgument receiversReadySemaphore = semResult1.first;
    auto semResult2 = createSemaphore(rewriter, loc, remoteLoad);
    BlockArgument senderFinishedSemaphore = semResult2.first;

    Value mcastVolumeVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(mcastVolume));

    // Determine if this core is the sender.
    // By convention, core 0 along each multicast dimension is the sender.
    // We need to check that ALL multicast dimensions have core_index == 0.
    Value isSender = nullptr;
    for (size_t i = 0; i < isMcastDim.size(); ++i) {
      if (isMcastDim[i]) {
        Value coreIdx = rewriter.create<CoreIndexOp>(
            loc, rewriter.getIndexType(), rewriter.getI64IntegerAttr(i));
        Value condition = rewriter.create<arith::CmpIOp>(
            loc, rewriter.getI1Type(), arith::CmpIPredicate::eq, coreIdx, zero);
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

          // Wait for all receivers to be ready
          builder.create<SemaphoreWaitOp>(loc, receiversReadySemaphore,
                                          mcastVolumeVal, zero);

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
              mcastLocalIndices, coalescingFactor,
              remoteLoad.getMcastStartIndex(), remoteLoad.getMcastShape());
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
          // For dimensions that are multicast, sender is at position 0
          // For non-multicast dimensions, use current core position
          for (size_t i = 0; i < isMcastDim.size(); ++i) {
            if (isMcastDim[i]) {
              // Multicast dimension - sender is at 0
              senderCoreIndex.push_back(zeroIdx);
            } else {
              // Non-multicast dimension - use current core's position
              Value currentCoreIdx = builder.create<CoreIndexOp>(
                  loc, builder.getIndexType(), builder.getI64IntegerAttr(i));
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
    size_t coalescingFactor = calculateCoalescingFactor(
        remoteMemoryMap, gridShape, shardShape, elemSizeBytes);

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
  using OpRewritePattern<RemoteStoreOp>::OpRewritePattern;

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
                                 size_t coalescingFactor, size_t shardVolume,
                                 ValueRange mcastStartIndex = ValueRange(),
                                 ValueRange mcastShape = ValueRange()) {
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

      return builder.create<DMAWriteOp>(
          loc, localMemref, localIndices, remoteMemref, remoteIndices,
          coalescingFactor, mcastStartIndex, mcastShape);
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
              coalescingFactor, mcastStartIndex, mcastShape);
          thenBuilder.create<scf::YieldOp>(innerLoc, dmaTx);

          auto elseBuilder = ifExpr.getElseBodyBuilder();
          elseBuilder.create<scf::YieldOp>(innerLoc, args[0]);

          return SmallVector<Value>{ifExpr.getResult(0)};
        });

    return loopNest.results.front();
  }

  // Handle multicast pattern for RemoteStoreOp
  static LogicalResult
  handleMcastRemoteStore(PatternRewriter &rewriter, Location loc,
                         RemoteStoreOp remoteStore, MemRefType remoteMemrefType,
                         AffineMap remoteMemoryMap, AffineMap localMemoryMap,
                         SmallVector<Value> gridIndices,
                         ArrayRef<int64_t> shardShape, size_t coalescingFactor,
                         size_t shardVolume) {
    Value cb = remoteStore.getCb();
    Value remoteMemref = remoteStore.getMemref();

    // For remote_store with multicast, use generateDMAWrites with multicast
    // parameters This handles both contiguous and strided memory accesses
    Value dmaTx = generateDMAWrites(
        rewriter, loc, remoteMemref, gridIndices, shardShape, remoteMemoryMap,
        localMemoryMap, cb, coalescingFactor, shardVolume,
        remoteStore.getMcastStartIndex(), remoteStore.getMcastShape());

    rewriter.eraseOp(remoteStore);

    // Wait for DMA to complete
    rewriter.create<DMAWaitOp>(loc, dmaTx);
    return success();
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
    size_t coalescingFactor = calculateCoalescingFactor(
        remoteMemoryMap, gridShape, shardShape, elemSizeBytes);

    size_t shardVolume = ttmlir::utils::volume(shardShape);

    // Get grid indices from the remote_store operation
    SmallVector<Value> gridIndices = remoteStore.getIndices();

    // Check if this is a multicast operation
    if (remoteStore.isMcast()) {
      return handleMcastRemoteStore(rewriter, loc, remoteStore,
                                    remoteMemrefType, remoteMemoryMap,
                                    localMemoryMap, gridIndices, shardShape,
                                    coalescingFactor, shardVolume);
    }

    // Unicast path: generate DMA writes with proper coalescing
    Value dmaTx = generateDMAWrites(rewriter, loc, remoteMemref, gridIndices,
                                    shardShape, remoteMemoryMap, localMemoryMap,
                                    cb, coalescingFactor, shardVolume);

    rewriter.eraseOp(remoteStore);

    // Wait for DMA to complete
    rewriter.create<DMAWaitOp>(loc, dmaTx);
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
    patterns.add<D2MLowerRemoteLoadRewritePattern,
                 D2MLowerRemoteStoreRewritePattern>(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
