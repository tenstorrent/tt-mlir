// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapAnalysis.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERDMATOFULLYINDEXEDFORM
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

static size_t getElementSizeBytes(MemRefType memref) {
  mlir::Type elementType = memref.getElementType();
  auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType);
  return tileType ? tileType.getSizeBytes()
                  : elementType.getIntOrFloatBitWidth() / 8;
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

// Core loop generation with coalescing guard.  Handles both the fully
// contiguous case (single call at zero indices) and the strided case
// (nested scf.for loops with an `if (flat_index % CF == 0)` guard).
// The caller supplies `emitDMA` which builds the actual DMA op for a
// given set of iteration indices.
using EmitDMAFn =
    llvm::function_ref<Value(OpBuilder &builder, Location loc, ValueRange iters,
                             size_t coalescingFactor)>;

static Value generateDMAWithCoalescing(OpBuilder &builder, Location loc,
                                       ArrayRef<int64_t> iterShape,
                                       size_t coalescingFactor, DMAType txType,
                                       EmitDMAFn emitDMA) {
  size_t volume = ttmlir::utils::volume(iterShape);

  if (coalescingFactor == volume) {
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                   builder.getIndexAttr(0));
    SmallVector<Value> zeros(iterShape.size(), zero);
    return emitDMA(builder, loc, zeros, coalescingFactor);
  }

  auto [lbs, ubs, steps] = utils::getLoopBounds(builder, loc, iterShape);
  auto nullDmaTx = builder.create<NullTxOp>(loc, txType);

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

        auto nulltx = loopBuilder.create<NullTxOp>(innerLoc, txType);

        auto ifExpr = loopBuilder.create<scf::IfOp>(
            innerLoc, TypeRange(SmallVector<Value>{nulltx}), predicate,
            true /*addThenBlock*/, true /*addElseBlock*/);

        auto thenBuilder = ifExpr.getThenBodyBuilder();
        Value dmaTx = emitDMA(thenBuilder, innerLoc, iters, coalescingFactor);
        thenBuilder.create<scf::YieldOp>(innerLoc, dmaTx);

        auto elseBuilder = ifExpr.getElseBodyBuilder();
        elseBuilder.create<scf::YieldOp>(innerLoc, args[0]);

        return SmallVector<Value>{ifExpr.getResult(0)};
      });

  return loopNest.results.front();
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
    AffineMap remoteMemoryMap = utils::getMemoryMap(device, remoteMemref, true);
    AffineMap localMemoryMap = utils::getMemoryMap(device, localMemref, false);

    size_t elemSizeBytes = getElementSizeBytes(remoteMemrefType);
    size_t coalescingFactor = calculateCoalescingFactorWithFallback(
        remoteMemoryMap, gridShape, shardShape, elemSizeBytes,
        debugCoalescingInference);

    SmallVector<Value> gridIndices(op.getSrcIndices());

    Value newTx = generateDMAWithCoalescing(
        rewriter, loc, shardShape, coalescingFactor, DMAType::Read,
        [&](OpBuilder &b, Location l, ValueRange iters, size_t cf) -> Value {
          SmallVector<Value> remoteIndices =
              llvm::to_vector(llvm::concat<Value>(gridIndices, iters));
          SmallVector<Value> localIndices = llvm::to_vector(iters);

          remoteIndices =
              utils::applyMap(b, l, remoteMemoryMap, remoteIndices, true);
          localIndices =
              utils::applyMap(b, l, localMemoryMap, localIndices, false);

          return b.create<DMAReadOp>(l, remoteMemref, remoteIndices,
                                     localMemref, localIndices,
                                     b.getI64IntegerAttr(cf));
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
      AffineMap localMemoryMap =
          utils::getMemoryMap(device, localMemref, false);

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
          utils::applyMap(rewriter, loc, localMemoryMap, localIndices, false);

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

    AffineMap remoteMemoryMap = utils::getMemoryMap(device, dstMemref, true);
    AffineMap localMemoryMap = utils::getMemoryMap(device, localMemref, false);

    size_t elemSizeBytes = getElementSizeBytes(remoteMemrefType);
    size_t coalescingFactor = calculateCoalescingFactorWithFallback(
        remoteMemoryMap, gridShape, shardShape, elemSizeBytes,
        debugCoalescingInference);

    SmallVector<Value> gridIndices(op.getDstIndices());
    SmallVector<Value> startDevice(op.getStartDevice());
    SmallVector<Value> deviceMcastShape(op.getDeviceMcastShape());

    Value newTx = generateDMAWithCoalescing(
        rewriter, loc, shardShape, coalescingFactor, DMAType::Write,
        [&](OpBuilder &b, Location l, ValueRange iters, size_t cf) -> Value {
          SmallVector<Value> remoteIndices =
              llvm::to_vector(llvm::concat<Value>(gridIndices, iters));
          SmallVector<Value> localIndices = llvm::to_vector(iters);

          remoteIndices =
              utils::applyMap(b, l, remoteMemoryMap, remoteIndices, true);
          localIndices =
              utils::applyMap(b, l, localMemoryMap, localIndices, false);

          return b.create<DMAWriteOp>(l, localMemref, localIndices, dstMemref,
                                      remoteIndices, cf, startDevice,
                                      deviceMcastShape);
        });

    rewriter.replaceOp(op, newTx);
    return success();
  }
};

class D2MLowerLocalCopyToFullyIndexed : public OpRewritePattern<LocalCopyOp> {
public:
  D2MLowerLocalCopyToFullyIndexed(MLIRContext *context,
                                  bool debugCoalescingInference)
      : OpRewritePattern<LocalCopyOp>(context),
        debugCoalescingInference(debugCoalescingInference) {}

private:
  bool debugCoalescingInference;

public:
  LogicalResult matchAndRewrite(LocalCopyOp op,
                                PatternRewriter &rewriter) const final {
    if (!op.hasMemTxResult()) {
      return failure();
    }

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

    if (!dstIndexingMap.isIdentity()) {
      return rewriter.notifyMatchFailure(
          op, "non-identity dst indexing map not yet supported");
    }

    MLIRContext *ctx = rewriter.getContext();
    AffineMap srcMemoryMap = utils::canonicalStridedMap(
        ctx, srcType.getShape(), srcType.getElementType(),
        AffineMap::getMultiDimIdentityMap(srcType.getRank(), ctx));
    AffineMap dstMemoryMap = utils::canonicalStridedMap(
        ctx, dstType.getShape(), dstType.getElementType(),
        AffineMap::getMultiDimIdentityMap(dstType.getRank(), ctx));

    AffineMap composedSrcMap = srcMemoryMap.compose(srcIndexingMap);
    AffineMap composedDstMap = dstMemoryMap.compose(dstIndexingMap);

    // Pad the composed source map with dummy grid dims (all size 1) so that
    // the sampling loop executes at least once.
    // The function requires numGridDims > 0 for correct behaviour.
    unsigned dstRank = dstType.getRank();
    SmallVector<AffineExpr> dimShift;
    for (unsigned i = 0; i < dstRank; ++i) {
      dimShift.push_back(getAffineDimExpr(i + dstRank, ctx));
    }
    AffineMap paddedSrcMap =
        composedSrcMap.replaceDimsAndSymbols(dimShift, {}, 2 * dstRank, 0);
    SmallVector<int64_t> dummyGridShape(dstRank, 1);
    size_t coalescingFactor = calculateCoalescingFactorWithFallback(
        paddedSrcMap, dummyGridShape, dstShape, elemSizeBytes,
        debugCoalescingInference);

    // Collect all enclosing blocking loops (there may be multiple for
    // multi-dimensional blocking, e.g. block_factors = [BF0, BF1]).
    SmallVector<std::pair<Value, int64_t>> blockingLoops;
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (auto attr =
                forOp->getAttrOfType<IntegerAttr>("d2m.blocking_loop")) {
          blockingLoops.emplace_back(forOp.getInductionVar(), attr.getInt());
        }
      } else if (auto forOp = dyn_cast<affine::AffineForOp>(parent)) {
        if (auto attr =
                forOp->getAttrOfType<IntegerAttr>("d2m.blocking_loop")) {
          blockingLoops.emplace_back(forOp.getInductionVar(), attr.getInt());
        }
      }
    }

    for (auto [blockIV, blockingDim] : blockingLoops) {
      if (static_cast<size_t>(blockingDim) >=
          static_cast<size_t>(dstShape.size())) {
        return rewriter.notifyMatchFailure(
            op, "blocking_loop dimension exceeds dst rank");
      }
    }

    Value newTx = generateDMAWithCoalescing(
        rewriter, loc, dstShape, coalescingFactor, DMAType::Read,
        [&](OpBuilder &b, Location l, ValueRange iters, size_t cf) -> Value {
          SmallVector<Value> srcEvalIters(iters.begin(), iters.end());
          for (auto [blockIV, blockingDim] : blockingLoops) {
            Value blockSize = b.create<arith::ConstantOp>(
                l, b.getIndexType(), b.getIndexAttr(dstShape[blockingDim]));
            Value offset = b.create<arith::MulIOp>(l, blockIV, blockSize);
            srcEvalIters[blockingDim] =
                b.create<arith::AddIOp>(l, offset, iters[blockingDim]);
          }

          SmallVector<Value> srcIndices =
              utils::applyMap(b, l, composedSrcMap, srcEvalIters, false);
          SmallVector<Value> dstIndices =
              utils::applyMap(b, l, composedDstMap, iters, false);

          return b
              .create<DMAReadOp>(l, src, srcIndices, dst, dstIndices,
                                 b.getI64IntegerAttr(cf))
              .getResult();
        });

    rewriter.replaceOp(op, newTx);
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
    patterns.add<D2MLowerLocalCopyToFullyIndexed>(&getContext(),
                                                  debugCoalescingInference);
    populateAffineToStdConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
