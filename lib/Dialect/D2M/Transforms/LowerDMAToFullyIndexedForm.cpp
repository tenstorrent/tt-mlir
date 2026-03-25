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
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
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

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERDMATOFULLYINDEXEDFORM
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
    // d2m::utils::getMemoryMap handles view tracing (applyViews) and
    // VGM lookup (getVirtualGridForwardMapping) internally.
    return d2m::utils::getMemoryMap(device, input, /*pageSize=*/0);
  }

  // For local memrefs (including CB values), get the underlying memref type.
  MemRefType inputType;
  if (auto cbType = mlir::dyn_cast<CBType>(input.getType())) {
    inputType = cbType.getUnderlyingAs<MemRefType>();
  } else {
    inputType = mlir::cast<MemRefType>(input.getType());
  }
  auto layoutMap = d2m::utils::resolveEffectiveAffineMap(input, inputType);
  return canonicalStridedMap(device.getContext(), inputType.getShape(),
                             inputType.getElementType(), layoutMap);
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

// Callback that receives the current loop iteration indices and must emit a
// single DMA operation, returning its transaction Value.
using EmitDMAForIterFn =
    llvm::function_ref<Value(OpBuilder &builder, Location loc, ValueRange iters,
                             size_t coalescingFactor, Value prevTx)>;

// Core loop generation with coalescing guard.  Handles both the fully
// contiguous case (single call at zero indices) and the strided case
// (nested scf.for loops with an `if (flat_index % CF == 0)` guard).
// The caller supplies `emitDMA` which builds the actual DMA op for a
// given set of iteration indices.
static Value generateDMAWithCoalescing(OpBuilder &builder, Location loc,
                                       ArrayRef<int64_t> iterShape,
                                       size_t coalescingFactor,
                                       EmitDMAForIterFn emitDMA) {
  size_t volume = ttmlir::utils::volume(iterShape);

  if (coalescingFactor == volume) {
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                   builder.getIndexAttr(0));
    SmallVector<Value> zeros(iterShape.size(), zero);
    return emitDMA(builder, loc, zeros, coalescingFactor, Value());
  }

  auto [lbs, ubs, steps] = getLoopBounds(builder, loc, iterShape);
  auto nullDmaTx = builder.create<NullTxOp>(loc);

  scf::LoopNest loopNest = scf::buildLoopNest(
      builder, loc, lbs, ubs, steps, ValueRange(nullDmaTx),
      [&](OpBuilder &loopBuilder, Location innerLoc, ValueRange iters,
          ValueRange args) {
        Value cfExpr = loopBuilder.create<arith::ConstantOp>(
            innerLoc, loopBuilder.getIndexType(),
            loopBuilder.getIndexAttr(coalescingFactor));
        Value zero = loopBuilder.create<arith::ConstantOp>(
            innerLoc, loopBuilder.getIndexType(),
            loopBuilder.getIntegerAttr(loopBuilder.getIndexType(), 0));

        // flat_index(iters) % coalescingFactor == 0
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
          totalIterCount =
              loopBuilder
                  .create<arith::AddIOp>(innerLoc, scaledCount, totalIterCount)
                  .getResult();
          currStride *= iterShape[i];
        }
        auto moduloIterCount =
            loopBuilder.create<arith::RemSIOp>(innerLoc, totalIterCount, cfExpr)
                .getResult();
        auto predicate = loopBuilder.create<arith::CmpIOp>(
            innerLoc, arith::CmpIPredicate::eq, moduloIterCount, zero);

        auto nulltx = loopBuilder.create<NullTxOp>(innerLoc);

        auto ifExpr = loopBuilder.create<scf::IfOp>(
            innerLoc, TypeRange(SmallVector<Value>{nulltx}), predicate,
            true /*addThenBlock*/, true /*addElseBlock*/);

        auto thenBuilder = ifExpr.getThenBodyBuilder();
        Value dmaTx =
            emitDMA(thenBuilder, innerLoc, iters, coalescingFactor, args[0]);
        thenBuilder.create<scf::YieldOp>(innerLoc, dmaTx);

        auto elseBuilder = ifExpr.getElseBodyBuilder();
        elseBuilder.create<scf::YieldOp>(innerLoc, args[0]);

        return SmallVector<Value>{ifExpr.getResult(0)};
      });

  return loopNest.results.front();
}

// Callback type for creating a single fully-indexed DMA op. Used by
// generateFullyIndexedDMAOps to abstract over DMAReadOp vs DMAWriteOp creation.
using CreateDMAOpFn = llvm::function_ref<Value(
    OpBuilder &builder, Location loc, SmallVector<Value> &remoteIndices,
    SmallVector<Value> &localIndices, size_t coalescingFactor)>;

// Generate fully-indexed DMA operations with proper coalescing for
// remote-to-local and local-to-remote transfers.  Wraps
// generateDMAWithCoalescing by prepending grid indices and applying memory maps
// before calling the DMA creation callback.
static Value generateFullyIndexedDMAOps(
    OpBuilder &builder, Location loc, SmallVector<Value> gridIndices,
    ArrayRef<int64_t> shardShape, AffineMap remoteMemoryMap,
    AffineMap localMemoryMap, size_t coalescingFactor, size_t shardVolume,
    CreateDMAOpFn createDMAOp) {

  return generateDMAWithCoalescing(
      builder, loc, shardShape, coalescingFactor,
      [&](OpBuilder &b, Location l, ValueRange iters, size_t cf,
          Value /*prevTx*/) -> Value {
        SmallVector<Value> remoteIndices =
            llvm::to_vector(llvm::concat<Value>(gridIndices, iters));
        SmallVector<Value> localIndices = llvm::to_vector(iters);

        remoteIndices = applyMap(b, l, remoteMemoryMap, remoteIndices, true);
        localIndices = applyMap(b, l, localMemoryMap, localIndices, false);

        return createDMAOp(b, l, remoteIndices, localIndices, cf);
      });
}

namespace {
class D2MLowerDMAReadToFullyIndexed : public OpRewritePattern<DMAReadOp> {
public:
  D2MLowerDMAReadToFullyIndexed(MLIRContext *context,
                                bool debugCoalescingInference)
      : OpRewritePattern<DMAReadOp>(context),
        debugCoalescingInference(debugCoalescingInference) {}

private:
  bool debugCoalescingInference;

public:
  LogicalResult matchAndRewrite(DMAReadOp op,
                                PatternRewriter &rewriter) const final {
    if (op.isFullyIndexed()) {
      return failure();
    }

    Location loc = op.getLoc();
    Value remoteMemref = op.getSrc();
    Value localMemref = op.getDst();

    MemRefType remoteMemrefType = op.getSrcMemRefType();

    ttcore::DeviceLayoutInterface deviceLayout =
        ttcore::getDeviceLayout(remoteMemref);
    if (!deviceLayout) {
      return rewriter.notifyMatchFailure(
          op, "remote memref must have a device layout");
    }

    ShapedType remoteShapedType =
        mlir::cast<ShapedType>(remoteMemref.getType());
    ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(remoteShapedType);
    ArrayRef<int64_t> shardShape = deviceLayout.getShardShape(remoteShapedType);

    ttcore::DeviceAttr device = ttcore::lookupDevice(op);
    AffineMap remoteMemoryMap = getMemoryMap(device, remoteMemref, true);
    AffineMap localMemoryMap = getMemoryMap(device, localMemref, false);

    size_t elemSizeBytes = getElementSizeBytes(remoteMemrefType);
    size_t coalescingFactor = calculateCoalescingFactorWithFallback(
        remoteMemoryMap, gridShape, shardShape, elemSizeBytes,
        debugCoalescingInference);

    size_t shardVolume = ttmlir::utils::volume(shardShape);
    SmallVector<Value> gridIndices(op.getSrcIndices());

    Value newTx = generateFullyIndexedDMAOps(
        rewriter, loc, gridIndices, shardShape, remoteMemoryMap, localMemoryMap,
        coalescingFactor, shardVolume,
        [&](OpBuilder &b, Location l, SmallVector<Value> &remoteIdx,
            SmallVector<Value> &localIdx, size_t cf) {
          return b.create<DMAReadOp>(l, remoteMemref, remoteIdx, localMemref,
                                     localIdx, b.getI64IntegerAttr(cf));
        });

    rewriter.replaceOp(op, newTx);
    return success();
  }
};

class D2MLowerDMAWriteToFullyIndexed : public OpRewritePattern<DMAWriteOp> {
public:
  D2MLowerDMAWriteToFullyIndexed(MLIRContext *context,
                                 bool debugCoalescingInference)
      : OpRewritePattern<DMAWriteOp>(context),
        debugCoalescingInference(debugCoalescingInference) {}

private:
  bool debugCoalescingInference;

public:
  LogicalResult matchAndRewrite(DMAWriteOp op,
                                PatternRewriter &rewriter) const final {
    if (op.isFullyIndexed()) {
      return failure();
    }

    Location loc = op.getLoc();
    Value localMemref = op.getSrc();
    Value dstMemref = op.getDst();

    ttcore::DeviceAttr device = ttcore::lookupDevice(op);

    if (op.isMcast()) {
      // Mcast write: local-to-local, compute local memory map and apply to
      // zero indices to get the fully-indexed form.
      AffineMap localMemoryMap = getMemoryMap(device, localMemref, false);

      MemRefType localType = op.getSrcMemRefType();
      ArrayRef<int64_t> shardShape = localType.getShape();
      size_t shardVolume = ttmlir::utils::volume(shardShape);

      SmallVector<Value> localIndices;
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
      for (size_t i = 0; i < shardShape.size(); ++i) {
        localIndices.push_back(zero);
      }
      localIndices =
          applyMap(rewriter, loc, localMemoryMap, localIndices, false);

      Value newTx = rewriter.create<DMAWriteOp>(
          loc, localMemref, localIndices, dstMemref, localIndices,
          op.getMcastStartIndex(), op.getMcastShape(), shardVolume);
      rewriter.replaceOp(op, newTx);
      return success();
    }

    // Non-mcast write: local src to remote dst.
    MemRefType remoteMemrefType = op.getDstMemRefType();

    ttcore::DeviceLayoutInterface deviceLayout =
        ttcore::getDeviceLayout(dstMemref);
    if (!deviceLayout) {
      return rewriter.notifyMatchFailure(
          op, "remote memref must have a device layout");
    }

    ShapedType remoteShapedType = mlir::cast<ShapedType>(dstMemref.getType());
    ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(remoteShapedType);
    ArrayRef<int64_t> shardShape = deviceLayout.getShardShape(remoteShapedType);

    AffineMap remoteMemoryMap = getMemoryMap(device, dstMemref, true);
    AffineMap localMemoryMap = getMemoryMap(device, localMemref, false);

    size_t elemSizeBytes = getElementSizeBytes(remoteMemrefType);
    size_t coalescingFactor = calculateCoalescingFactorWithFallback(
        remoteMemoryMap, gridShape, shardShape, elemSizeBytes,
        debugCoalescingInference);

    size_t shardVolume = ttmlir::utils::volume(shardShape);
    SmallVector<Value> gridIndices(op.getDstIndices());

    Value newTx = generateFullyIndexedDMAOps(
        rewriter, loc, gridIndices, shardShape, remoteMemoryMap, localMemoryMap,
        coalescingFactor, shardVolume,
        [&](OpBuilder &b, Location l, SmallVector<Value> &remoteIdx,
            SmallVector<Value> &localIdx, size_t cf) {
          return b.create<DMAWriteOp>(l, localMemref, localIdx, dstMemref,
                                      remoteIdx, cf);
        });

    rewriter.replaceOp(op, newTx);
    return success();
  }
};

class D2MLowerL1CopyToFullyIndexed : public OpRewritePattern<L1CopyOp> {
public:
  D2MLowerL1CopyToFullyIndexed(MLIRContext *context)
      : OpRewritePattern<L1CopyOp>(context) {}

  LogicalResult matchAndRewrite(L1CopyOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getDst();

    MemRefType srcType;
    if (auto cbType = mlir::dyn_cast<CBType>(src.getType())) {
      srcType = cbType.getUnderlyingAs<MemRefType>();
    } else {
      srcType = mlir::cast<MemRefType>(src.getType());
    }
    MemRefType dstType;
    if (auto cbType = mlir::dyn_cast<CBType>(dst.getType())) {
      dstType = cbType.getUnderlyingAs<MemRefType>();
    } else {
      dstType = mlir::cast<MemRefType>(dst.getType());
    }

    ArrayRef<int64_t> dstShape = dstType.getShape();
    size_t elemSizeBytes = getElementSizeBytes(srcType);

    ArrayRef<Attribute> maps = op.getIndexingMaps().getValue();
    AffineMap srcIndexingMap = mlir::cast<AffineMapAttr>(maps[0]).getValue();
    AffineMap dstIndexingMap = mlir::cast<AffineMapAttr>(maps[1]).getValue();

    MLIRContext *ctx = rewriter.getContext();
    AffineMap srcMemoryMap = canonicalStridedMap(
        ctx, srcType.getShape(), srcType.getElementType(),
        AffineMap::getMultiDimIdentityMap(srcType.getRank(), ctx));
    AffineMap dstMemoryMap = canonicalStridedMap(
        ctx, dstType.getShape(), dstType.getElementType(),
        AffineMap::getMultiDimIdentityMap(dstType.getRank(), ctx));

    AffineMap composedSrcMap = srcMemoryMap.compose(srcIndexingMap);
    AffineMap composedDstMap = dstMemoryMap.compose(dstIndexingMap);

    // calculateCoalescingFactor requires at least one grid dim for its
    // sampling loop.  L1Copy is local-only (no grid in the map), so we
    // prepend a single dummy grid dim of size 1 and shift the existing
    // map dimensions by one.
    unsigned dstRank = dstType.getRank();
    SmallVector<AffineExpr> dimShift;
    for (unsigned i = 0; i < dstRank; ++i) {
      dimShift.push_back(getAffineDimExpr(i + 1, ctx));
    }
    AffineMap paddedSrcMap =
        composedSrcMap.replaceDimsAndSymbols(dimShift, {}, dstRank + 1, 0);
    SmallVector<int64_t> fullShape = {1};
    fullShape.append(dstShape.begin(), dstShape.end());
    size_t coalescingFactor = ttmlir::utils::calculateCoalescingFactor(
        paddedSrcMap, fullShape, static_cast<int64_t>(elemSizeBytes),
        /*numGridDims=*/1);

    // Find enclosing blocking loop (scf.for or affine.for with
    // d2m.blocking_loop attribute).
    Value blockIV;
    int64_t blockingDim = -1;
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (auto attr =
                forOp->getAttrOfType<IntegerAttr>("d2m.blocking_loop")) {
          blockIV = forOp.getInductionVar();
          blockingDim = attr.getInt();
          break;
        }
      }
      if (auto forOp = dyn_cast<affine::AffineForOp>(parent)) {
        if (auto attr =
                forOp->getAttrOfType<IntegerAttr>("d2m.blocking_loop")) {
          blockIV = forOp.getInductionVar();
          blockingDim = attr.getInt();
          break;
        }
      }
    }

    Value newTx = generateDMAWithCoalescing(
        rewriter, loc, dstShape, coalescingFactor,
        [&](OpBuilder &b, Location l, ValueRange iters, size_t cf,
            Value /*prevTx*/) -> Value {
          // Source indices need blocking adjustment for global indexing.
          SmallVector<Value> srcEvalIters(iters.begin(), iters.end());
          if (blockIV && blockingDim >= 0) {
            Value blockSize = b.create<arith::ConstantOp>(
                l, b.getIndexType(), b.getIndexAttr(dstShape[blockingDim]));
            Value offset = b.create<arith::MulIOp>(l, blockIV, blockSize);
            srcEvalIters[blockingDim] =
                b.create<arith::AddIOp>(l, offset, iters[blockingDim]);
          }

          SmallVector<Value> srcIndices =
              applyMap(b, l, composedSrcMap, srcEvalIters, false);
          SmallVector<Value> dstIndices =
              applyMap(b, l, composedDstMap, iters, false);

          return b
              .create<DMAReadOp>(l, src, srcIndices, dst, dstIndices,
                                 b.getI64IntegerAttr(cf))
              .getResult();
        });

    rewriter.create<DMAWaitOp>(loc, newTx);
    rewriter.eraseOp(op);
    return success();
  }
};

class D2MLowerDMAToFullyIndexedForm
    : public impl::D2MLowerDMAToFullyIndexedFormBase<
          D2MLowerDMAToFullyIndexedForm> {
public:
  using impl::D2MLowerDMAToFullyIndexedFormBase<
      D2MLowerDMAToFullyIndexedForm>::D2MLowerDMAToFullyIndexedFormBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MLowerDMAReadToFullyIndexed>(&getContext(),
                                                debugCoalescingInference);
    patterns.add<D2MLowerDMAWriteToFullyIndexed>(&getContext(),
                                                 debugCoalescingInference);
    patterns.add<D2MLowerL1CopyToFullyIndexed>(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
