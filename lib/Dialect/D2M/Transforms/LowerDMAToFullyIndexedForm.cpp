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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
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

// Callback type for creating a single fully-indexed DMA op. Used by
// generateFullyIndexedDMAOps to abstract over DMAReadOp vs DMAWriteOp creation.
using CreateDMAOpFn = llvm::function_ref<Value(
    OpBuilder &builder, Location loc, SmallVector<Value> &remoteIndices,
    SmallVector<Value> &localIndices, size_t coalescingFactor)>;

// Generate fully-indexed DMA operations with proper coalescing.
// Returns the last DMA transaction value (for waiting).
// Handles both contiguous (single DMA) and strided (loop with guarded DMAs)
// cases.
static Value generateFullyIndexedDMAOps(
    OpBuilder &builder, Location loc, SmallVector<Value> gridIndices,
    ArrayRef<int64_t> shardShape, AffineMap remoteMemoryMap,
    AffineMap localMemoryMap, size_t coalescingFactor, size_t shardVolume,
    CreateDMAOpFn createDMAOp) {

  if (coalescingFactor == shardVolume) {
    // Fully contiguous: single DMA operation.
    SmallVector<Value> remoteIndices = gridIndices;
    SmallVector<Value> localIndices;

    Value zero = arith::ConstantOp::create(builder, loc, builder.getIndexType(),
                                           builder.getIndexAttr(0));
    for (size_t i = 0; i < shardShape.size(); ++i) {
      remoteIndices.push_back(zero);
      localIndices.push_back(zero);
    }

    remoteIndices =
        utils::applyMap(builder, loc, remoteMemoryMap, remoteIndices, true);
    localIndices =
        utils::applyMap(builder, loc, localMemoryMap, localIndices, false);

    return createDMAOp(builder, loc, remoteIndices, localIndices,
                       coalescingFactor);
  }

  // Strided/non-contiguous: generate loops with guarded DMAs.
  auto [lbs, ubs, steps] = utils::getLoopBounds(builder, loc, shardShape);
  auto nullDmaTx = NullTxOp::create(builder, loc);

  scf::LoopNest loopNest = scf::buildLoopNest(
      builder, loc, lbs, ubs, steps, ValueRange(nullDmaTx),
      [&](OpBuilder &loopBuilder, Location innerLoc, ValueRange iters,
          ValueRange args) {
        // Build full indices: grid indices + shard iteration indices.
        SmallVector<Value> remoteIndices =
            llvm::to_vector(llvm::concat<Value>(gridIndices, iters));
        SmallVector<Value> localIndices = llvm::to_vector(iters);

        // Apply memory maps.
        remoteIndices = utils::applyMap(loopBuilder, innerLoc, remoteMemoryMap,
                                        remoteIndices, true);
        localIndices = utils::applyMap(loopBuilder, innerLoc, localMemoryMap,
                                       localIndices, false);

        // Create guarded DMA operation based on coalescing factor.
        Value cfExpr = arith::ConstantOp::create(
            loopBuilder, innerLoc, loopBuilder.getIndexType(),
            loopBuilder.getIndexAttr(coalescingFactor));
        Value zero = arith::ConstantOp::create(
            loopBuilder, innerLoc, loopBuilder.getIndexType(),
            loopBuilder.getIntegerAttr(loopBuilder.getIndexType(), 0));

        // Construct guard function: flat_index(iters) % coalescingFactor == 0
        auto totalIterCount = zero;
        size_t currStride = 1;
        for (int i = iters.size() - 1; i >= 0; i--) {
          Value currStrideExpr = arith::ConstantOp::create(
              loopBuilder, innerLoc, loopBuilder.getIndexType(),
              loopBuilder.getIndexAttr(currStride));
          auto scaledCount = arith::MulIOp::create(loopBuilder, innerLoc,
                                                   currStrideExpr, iters[i])
                                 .getResult();
          totalIterCount = arith::AddIOp::create(loopBuilder, innerLoc,
                                                 scaledCount, totalIterCount)
                               .getResult();
          currStride *= shardShape[i];
        }
        auto moduloIterCount = arith::RemSIOp::create(loopBuilder, innerLoc,
                                                      totalIterCount, cfExpr)
                                   .getResult();
        auto predicate = arith::CmpIOp::create(loopBuilder, innerLoc,
                                               arith::CmpIPredicate::eq,
                                               moduloIterCount, zero);

        auto nulltx = NullTxOp::create(loopBuilder, innerLoc);

        // Build guarded DMA.
        auto ifExpr = scf::IfOp::create(
            loopBuilder, innerLoc, TypeRange(SmallVector<Value>{nulltx}),
            predicate, true /*addThenBlock*/, true /*addElseBlock*/);

        auto thenBuilder = ifExpr.getThenBodyBuilder();
        Value dmaTx = createDMAOp(thenBuilder, innerLoc, remoteIndices,
                                  localIndices, coalescingFactor);
        scf::YieldOp::create(thenBuilder, innerLoc, dmaTx);

        auto elseBuilder = ifExpr.getElseBodyBuilder();
        scf::YieldOp::create(elseBuilder, innerLoc, args[0]);

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

    size_t shardVolume = ttmlir::utils::volume(shardShape);
    SmallVector<Value> gridIndices(op.getSrcIndices());

    Value newTx = generateFullyIndexedDMAOps(
        rewriter, loc, gridIndices, shardShape, remoteMemoryMap, localMemoryMap,
        coalescingFactor, shardVolume,
        [&](OpBuilder &b, Location l, SmallVector<Value> &remoteIdx,
            SmallVector<Value> &localIdx, size_t cf) {
          return DMAReadOp::create(b, l, remoteMemref, remoteIdx, localMemref,
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
      AffineMap localMemoryMap =
          utils::getMemoryMap(device, localMemref, false);

      MemRefType localType = op.getSrcMemRefType();
      ArrayRef<int64_t> shardShape = localType.getShape();
      size_t shardVolume = ttmlir::utils::volume(shardShape);

      SmallVector<Value> localIndices;
      Value zero = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
      for (size_t i = 0; i < shardShape.size(); ++i) {
        localIndices.push_back(zero);
      }
      localIndices =
          utils::applyMap(rewriter, loc, localMemoryMap, localIndices, false);

      Value newTx = DMAWriteOp::create(
          rewriter, loc, localMemref, localIndices, dstMemref, localIndices,
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

    size_t shardVolume = ttmlir::utils::volume(shardShape);
    SmallVector<Value> gridIndices(op.getDstIndices());

    Value newTx = generateFullyIndexedDMAOps(
        rewriter, loc, gridIndices, shardShape, remoteMemoryMap, localMemoryMap,
        coalescingFactor, shardVolume,
        [&](OpBuilder &b, Location l, SmallVector<Value> &remoteIdx,
            SmallVector<Value> &localIdx, size_t cf) {
          return DMAWriteOp::create(b, l, localMemref, localIdx, dstMemref,
                                    remoteIdx, cf);
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
    populateAffineToStdConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
