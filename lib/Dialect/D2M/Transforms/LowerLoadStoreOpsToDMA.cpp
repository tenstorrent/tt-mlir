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

  LogicalResult matchAndRewrite(RemoteLoadOp remoteLoad,
                                PatternRewriter &rewriter) const final {
    Location loc = remoteLoad.getLoc();
    MemRefType remoteMemrefType = remoteLoad.getMemRefType();
    CBType cbType = remoteLoad.getCbType();
    MemRefType localMemrefType = cbType.getUnderlyingAs<MemRefType>();

    if (!localMemrefType) {
      return rewriter.notifyMatchFailure(
          remoteLoad, "circular buffer must have memref underlying type");
    }

    // Get device layout from remote memref
    ttcore::DeviceLayoutInterface deviceLayout =
        mlir::cast<ttcore::DeviceLayoutInterface>(remoteMemrefType.getLayout());
    ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(remoteMemrefType);
    ArrayRef<int64_t> shardShape = deviceLayout.getShardShape(remoteMemrefType);

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

    // Build fully indexed DMA operations
    if (coalescingFactor == shardVolume) {
      // Fully contiguous DMA; lower to a single operation
      // Build full indices: grid indices + shard indices (all zeros for
      // contiguous)
      SmallVector<Value> remoteIndices = gridIndices;
      SmallVector<Value> localIndices;

      // Add zero indices for shard dimensions
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
      for (size_t i = 0; i < shardShape.size(); ++i) {
        remoteIndices.push_back(zero);
        localIndices.push_back(zero);
      }

      // Apply memory maps
      remoteIndices =
          applyMap(rewriter, loc, remoteMemoryMap, remoteIndices, true);
      localIndices =
          applyMap(rewriter, loc, localMemoryMap, localIndices, false);

      // Create dma_read operation
      // For remote_load: src is remote, dst is local (from CB)
      // We need to reserve the CB first to get the local memref to write to
      Value localMemref = rewriter.create<ReserveOp>(loc, cb).getResult();

      rewriter.create<DMAReadOp>(loc, remoteMemref, remoteIndices, localMemref,
                                 localIndices,
                                 rewriter.getI64IntegerAttr(coalescingFactor));

      rewriter.eraseOp(remoteLoad);
      return success();
    }

    // The memory access has some stride/gaps so multiple DMA operations are
    // needed. Generate loops for shard dimensions.
    // Reserve CB once before the loop nest to get the local memref
    Value localMemref = rewriter.create<ReserveOp>(loc, cb).getResult();

    auto [lbs, ubs, steps] = getLoopBounds(rewriter, loc, shardShape);

    auto nullDmaTx = rewriter.create<NullTxOp>(loc);
    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, ValueRange(nullDmaTx),
        [&](OpBuilder &builder, Location innerLoc, ValueRange iters,
            ValueRange /*args*/) {
          // Build full indices: grid indices + shard iteration indices
          SmallVector<Value> remoteIndices =
              llvm::to_vector(llvm::concat<Value>(gridIndices, iters));
          SmallVector<Value> localIndices = llvm::to_vector(iters);

          // Apply memory maps
          remoteIndices =
              applyMap(builder, innerLoc, remoteMemoryMap, remoteIndices, true);
          localIndices =
              applyMap(builder, innerLoc, localMemoryMap, localIndices, false);

          // Create guarded DMA operation based on coalescing factor
          Value cfExpr = builder.create<arith::ConstantOp>(
              loc, builder.getIndexType(),
              builder.getIndexAttr(coalescingFactor));
          Value zero = builder.create<arith::ConstantOp>(
              innerLoc, builder.getIndexType(),
              builder.getIntegerAttr(builder.getIndexType(), 0));

          // Construct guard function
          auto totalIterCount = zero;
          size_t currStride = 1;
          for (int i = iters.size() - 1; i >= 0; i--) {
            Value currStrideExpr = builder.create<arith::ConstantOp>(
                loc, builder.getIndexType(), builder.getIndexAttr(currStride));
            auto scaledCount =
                builder
                    .create<arith::MulIOp>(innerLoc, currStrideExpr, iters[i])
                    .getResult();
            totalIterCount = builder
                                 .create<arith::AddIOp>(innerLoc, scaledCount,
                                                        totalIterCount)
                                 .getResult();
            currStride *= shardShape[i];
          }
          auto moduloIterCount =
              builder.create<arith::RemSIOp>(innerLoc, totalIterCount, cfExpr)
                  .getResult();
          auto predicate = builder.create<arith::CmpIOp>(
              innerLoc, arith::CmpIPredicate::eq, moduloIterCount, zero);

          auto nulltx = builder.create<NullTxOp>(loc);

          // Build guarded expression
          auto ifExpr = builder.create<scf::IfOp>(
              innerLoc, TypeRange(SmallVector<Value>{nulltx}), predicate,
              true /*addThenBlock*/, true /*addElseBlock*/);

          auto thenBuilder = ifExpr.getThenBodyBuilder();
          thenBuilder.create<DMAReadOp>(
              innerLoc, remoteMemref, remoteIndices, localMemref, localIndices,
              thenBuilder.getI64IntegerAttr(coalescingFactor));
          thenBuilder.create<scf::YieldOp>(innerLoc, nulltx->getResult(0));

          auto elseBuilder = ifExpr.getElseBodyBuilder();
          elseBuilder.create<scf::YieldOp>(innerLoc, nulltx->getResult(0));

          return SmallVector<Value>{ifExpr.getResult(0)};
        });

    rewriter.replaceOp(remoteLoad, loopNest.loops.front());
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
    MemRefType remoteMemrefType = remoteStore.getMemRefType();
    CBType cbType = remoteStore.getCbType();
    MemRefType localMemrefType = cbType.getUnderlyingAs<MemRefType>();

    if (!localMemrefType) {
      return rewriter.notifyMatchFailure(
          remoteStore, "circular buffer must have memref underlying type");
    }

    // Get device layout from remote memref
    ttcore::DeviceLayoutInterface deviceLayout =
        mlir::cast<ttcore::DeviceLayoutInterface>(remoteMemrefType.getLayout());
    ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(remoteMemrefType);
    ArrayRef<int64_t> shardShape = deviceLayout.getShardShape(remoteMemrefType);

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

    // Build fully indexed DMA operations
    if (coalescingFactor == shardVolume) {
      // Fully contiguous DMA; lower to a single operation
      // Build full indices: grid indices + shard indices (all zeros for
      // contiguous)
      SmallVector<Value> remoteIndices = gridIndices;
      SmallVector<Value> localIndices;

      // Add zero indices for shard dimensions
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
      for (size_t i = 0; i < shardShape.size(); ++i) {
        remoteIndices.push_back(zero);
        localIndices.push_back(zero);
      }

      // Apply memory maps
      remoteIndices =
          applyMap(rewriter, loc, remoteMemoryMap, remoteIndices, true);
      localIndices =
          applyMap(rewriter, loc, localMemoryMap, localIndices, false);

      // Create dma_write operation
      // For remote_store: src is local (from CB), dst is remote
      // We need to wait on the CB first to get the local memref to read from
      Value localMemref = rewriter.create<WaitOp>(loc, cb).getResult();

      rewriter.create<DMAWriteOp>(loc, localMemref, localIndices, remoteMemref,
                                  remoteIndices, coalescingFactor);

      rewriter.eraseOp(remoteStore);
      return success();
    }

    // The memory access has some stride/gaps so multiple DMA operations are
    // needed. Generate loops for shard dimensions.
    // Wait on CB once before the loop nest to get the local memref
    Value localMemref = rewriter.create<WaitOp>(loc, cb).getResult();

    auto [lbs, ubs, steps] = getLoopBounds(rewriter, loc, shardShape);

    auto nullDmaTx = rewriter.create<NullTxOp>(loc);
    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, ValueRange(nullDmaTx),
        [&](OpBuilder &builder, Location innerLoc, ValueRange iters,
            ValueRange /*args*/) {
          // Build full indices: grid indices + shard iteration indices
          SmallVector<Value> remoteIndices =
              llvm::to_vector(llvm::concat<Value>(gridIndices, iters));
          SmallVector<Value> localIndices = llvm::to_vector(iters);

          // Apply memory maps
          remoteIndices =
              applyMap(builder, innerLoc, remoteMemoryMap, remoteIndices, true);
          localIndices =
              applyMap(builder, innerLoc, localMemoryMap, localIndices, false);

          // Create guarded DMA operation based on coalescing factor
          Value cfExpr = builder.create<arith::ConstantOp>(
              loc, builder.getIndexType(),
              builder.getIndexAttr(coalescingFactor));
          Value zero = builder.create<arith::ConstantOp>(
              innerLoc, builder.getIndexType(),
              builder.getIntegerAttr(builder.getIndexType(), 0));

          // Construct guard function
          auto totalIterCount = zero;
          size_t currStride = 1;
          for (int i = iters.size() - 1; i >= 0; i--) {
            Value currStrideExpr = builder.create<arith::ConstantOp>(
                loc, builder.getIndexType(), builder.getIndexAttr(currStride));
            auto scaledCount =
                builder
                    .create<arith::MulIOp>(innerLoc, currStrideExpr, iters[i])
                    .getResult();
            totalIterCount = builder
                                 .create<arith::AddIOp>(innerLoc, scaledCount,
                                                        totalIterCount)
                                 .getResult();
            currStride *= shardShape[i];
          }
          auto moduloIterCount =
              builder.create<arith::RemSIOp>(innerLoc, totalIterCount, cfExpr)
                  .getResult();
          auto predicate = builder.create<arith::CmpIOp>(
              innerLoc, arith::CmpIPredicate::eq, moduloIterCount, zero);

          auto nulltx = builder.create<NullTxOp>(loc);

          // Build guarded expression
          auto ifExpr = builder.create<scf::IfOp>(
              innerLoc, TypeRange(SmallVector<Value>{nulltx}), predicate,
              true /*addThenBlock*/, true /*addElseBlock*/);

          auto thenBuilder = ifExpr.getThenBodyBuilder();
          thenBuilder.create<DMAWriteOp>(innerLoc, localMemref, localIndices,
                                         remoteMemref, remoteIndices,
                                         coalescingFactor);
          thenBuilder.create<scf::YieldOp>(innerLoc, nulltx->getResult(0));

          auto elseBuilder = ifExpr.getElseBodyBuilder();
          elseBuilder.create<scf::YieldOp>(innerLoc, nulltx->getResult(0));

          return SmallVector<Value>{ifExpr.getResult(0)};
        });

    rewriter.replaceOp(remoteStore, loopNest.loops.front());
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
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
