// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICLOWERDMAS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MGenericLowerAffineDMAsRewritePattern : public OpRewritePattern<DMAOp> {
public:
  using OpRewritePattern<DMAOp>::OpRewritePattern;

  // Returns a tuple of the stream index and the index bounds. The stream index
  // represents the position in the stream that the DMA will is currently on,
  // and is relative per core. The index bounds represent the index upper
  // bounds.
  static std::tuple<SmallVector<Value>, SmallVector<int64_t>>
  buildStreamIndex(OpBuilder &builder, Location loc,
                   ArrayRef<int64_t> gridShape, ArrayRef<int64_t> blockFactors,
                   ArrayRef<int64_t> shardShape, AffineMap dmaIndexingMap,
                   AffineMap gridIndexingMap, AffineMap coreVirtualizationMap) {
    assert(dmaIndexingMap.getNumDims() == gridIndexingMap.getNumDims());
    assert(dmaIndexingMap.getNumResults() == gridIndexingMap.getNumResults());
    assert(dmaIndexingMap.getNumResults() == shardShape.size());
    assert(gridIndexingMap.isProjectedPermutation(true) &&
           "Grid indexing map must be a permutation");

    SmallVector<Value> streamIndex;
    SmallVector<int64_t> indexBounds;
    streamIndex.reserve(dmaIndexingMap.getNumResults());
    indexBounds.reserve(dmaIndexingMap.getNumResults());

    // Compute virtualized core indices from raw physical core indices using the
    // core virtualization map. This ensures both grid and shard indices are
    // in the virtual (viewed) coord space of the generic op's output operand.
    SmallVector<Value> physicalCoreIndices(gridIndexingMap.getNumResults());
    for (unsigned gridIndex = 0; gridIndex < gridIndexingMap.getNumResults();
         gridIndex++) {
      physicalCoreIndices[gridIndex] = builder.create<CoreIndexOp>(
          loc, builder.getIndexType(), builder.getI64IntegerAttr(gridIndex));
    }
    SmallVector<Value> virtualGridIndices;
    if (!coreVirtualizationMap.isEmpty()) {
      virtualGridIndices = ttmlir::utils::fullyApplyAffineMap(
          builder, loc, coreVirtualizationMap, physicalCoreIndices);
    } else {
      virtualGridIndices = physicalCoreIndices;
    }
    TT_assertv(virtualGridIndices.size() == gridIndexingMap.getNumResults(),
               "Core virtualization map must have the same number of results "
               "as the grid indexing map");

    for (unsigned result = 0; result < dmaIndexingMap.getNumResults();
         result++) {

      AffineExpr dimOrConstant = dmaIndexingMap.getResult(result);
      assert(mlir::isa<AffineDimExpr>(dimOrConstant) ||
             mlir::isa<AffineConstantExpr>(dimOrConstant));

      Value index;
      if (AffineConstantExpr constant =
              mlir::dyn_cast<AffineConstantExpr>(dimOrConstant)) {
        assert(constant.getValue() == 0);
        assert(gridShape[result] ==
               1); // this is too conservative, I think there will be bcast
        // cases in the future where this isn't true.  Probably best
        // to have it as a canary when we hit this case tho

        index = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                  builder.getIndexAttr(0));
      } else {
        unsigned dim = mlir::cast<AffineDimExpr>(dimOrConstant).getPosition();

        std::optional<unsigned> gridResult =
            gridIndexingMap.getResultPosition(dmaIndexingMap.getResult(result));

        //
        // The following assert is pretty subtle. If this operand dimension
        // participates in the grid, for now we must assert that the relative
        // grid result is the same as the operand result. This protects against
        // permutations of the grid, ie. transposes.  For example, let's
        // consider a matmul case:
        //   dmaIndexingMap:  (m, n, k) -> (m, k)
        //   dmaIndexingMap:  (m, n, k) -> (k, n)
        //   gridIndexingMap: (m, n, k) -> (m, n)
        //                                     ^
        // This assert ensures that the m's line up. A counter example would be:
        //   dmaIndexingMap:  (m, n, k) -> (m, k)
        //   gridIndexingMap: (m, n, k) -> (n, m)
        //
        // Not currently supported.
        //
        assert(!gridResult || *gridResult == result);
        bool isGridDim = gridResult.has_value();
        Value iterIndex = builder.create<IterIndexOp>(
            loc, builder.getIndexType(), builder.getI64IntegerAttr(dim));

        if (isGridDim) {
          // The grid dimension is always 1-1 with the result position. Consider
          // the case where interchange moves k to the outermost loop. We'd have
          // output map: (k, m, n) -> (m, n)
          // In this example we want (m, n) to map to grid dims (0, 1) (not
          // their dim positions i.e. (1, 2)).
          const unsigned gridDim = result;

          // gridI * blockFactorI + iterI
          Value blockFactor = builder.create<arith::ConstantOp>(
              loc, builder.getIndexType(),
              builder.getIndexAttr(blockFactors[dim]));

          index = virtualGridIndices[gridDim];

          index = builder.create<arith::MulIOp>(loc, builder.getIndexType(),
                                                index, blockFactor);
          index = builder.create<arith::AddIOp>(loc, builder.getIndexType(),
                                                index, iterIndex);
        } else {
          index = iterIndex;
        }
      }
      streamIndex.push_back(index);
      indexBounds.push_back(gridShape[result]);
    }
    return std::make_tuple(streamIndex, indexBounds);
  }

  static size_t getElementSizeBytes(MemRefType memref) {
    mlir::Type elementType = memref.getElementType();
    auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType);
    return tileType ? tileType.getSizeBytes()
                    : elementType.getIntOrFloatBitWidth() / 8;
  }

  static size_t getMemRefShardSizeBytes(MemRefType memref) {
    ArrayRef<int64_t> memrefShardShape =
        memref.getShape().drop_front(memref.getRank() / 2);
    return std::accumulate(memrefShardShape.begin(), memrefShardShape.end(),
                           getElementSizeBytes(memref),
                           std::multiplies<int64_t>());
  }

  // For now we'll do just a simple brute force check and sample the entire map
  // to calculate the coalescing factor. In the future there are some simple
  // checks we could do for the common case that would be much faster.
  static size_t calculateCoalescingFactor(AffineMap memoryMap,
                                          ArrayRef<int64_t> gridShape,
                                          ArrayRef<int64_t> shardShape,
                                          size_t elemSizeBytes,
                                          ArrayRef<int64_t> indexBounds) {
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
          coalescingFactor =
              std::gcd(coalescingFactor, currentCoalescingFactor);
          // If coalescing factor reaches unit size, it cannot change further.
          // Early exit to save on runtime.
          if (coalescingFactor == 1) {
            return;
          }
          // current memory access can potentially be coalesced with next
          // access!
          currentCoalescingFactor = 1;
        }
        nextAddress = address;
        nextAddress.back() += elemSizeBytes;
      });
    });
    return coalescingFactor;
  }

  static std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
  getLoopBounds(OpBuilder &builder, Location loc,
                ArrayRef<int64_t> shardShape) {
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

  // Uses coalescing factor to build an optimized gather loop with uniform
  // packet size.
  static scf::LoopNest buildCoalescedGatherLoop(OpBuilder &builder,
                                                Location loc, DMAOp dma,
                                                ArrayRef<Value> streamIndex,
                                                ArrayRef<int64_t> shardShape,
                                                size_t coalescingFactor) {

    auto [lbs, ubs, steps] = getLoopBounds(builder, loc, shardShape);

    auto nullDmaTx = builder.create<d2m::NullTxOp>(dma.getLoc());
    scf::LoopNest loopNest = scf::buildLoopNest(
        builder, loc, lbs, ubs, steps, ValueRange(nullDmaTx),
        [&](OpBuilder &builder, Location loc, ValueRange iters,
            ValueRange /*args*/) {
          SmallVector<Value> remoteIndices =
              llvm::to_vector(llvm::concat<Value>(streamIndex, iters));

          Value cfExpr = builder.create<arith::ConstantOp>(
              dma.getLoc(), builder.getIndexType(),
              builder.getIndexAttr(coalescingFactor));
          Value zero = builder.create<arith::ConstantOp>(
              loc, builder.getIndexType(),
              builder.getIntegerAttr(builder.getIndexType(), 0));

          // construct guard function
          // when flat_index(iters) == coalescing_fac, the next
          // dma operation should be issued
          auto totalIterCount = zero;
          size_t currStride = 1;
          for (int i = iters.size() - 1; i >= 0; i--) {

            Value currStrideExpr = builder.create<arith::ConstantOp>(
                dma.getLoc(), builder.getIndexType(),
                builder.getIndexAttr(currStride));
            auto scaledCount =
                builder.create<arith::MulIOp>(loc, currStrideExpr, iters[i])
                    .getResult();
            totalIterCount =
                builder.create<arith::AddIOp>(loc, scaledCount, totalIterCount)
                    .getResult();

            currStride *= shardShape[i];
          }
          auto moduloIterCount =
              builder.create<arith::RemSIOp>(loc, totalIterCount, cfExpr)
                  .getResult();
          auto predicate = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, moduloIterCount, zero);

          auto nulltx = builder.create<d2m::NullTxOp>(dma.getLoc());

          // build guarded expression
          auto ifExpr = builder.create<scf::IfOp>(
              loc, TypeRange(SmallVector<Value>{nulltx}), predicate,
              true /*addThenBlock*/, true /*addElseBlock*/);

          auto thenBuilder = ifExpr.getThenBodyBuilder();

          auto srcIndices =
              dma.isSrcRemote() ? remoteIndices : llvm::to_vector(iters);
          auto dstIndices =
              dma.isDstRemote() ? remoteIndices : llvm::to_vector(iters);
          auto dmaOp = thenBuilder.create<d2m::DMAOp>(
              dma.getLoc(), dma.getSrc(), srcIndices, dma.getDst(), dstIndices,
              dma.getMcastStartIndex(), dma.getMcastShape(), coalescingFactor);
          thenBuilder.create<scf::YieldOp>(dma.getLoc(), dmaOp->getResult(0));

          auto elseBuilder = ifExpr.getElseBodyBuilder();
          elseBuilder.create<scf::YieldOp>(dma.getLoc(), nulltx->getResult(0));

          return SmallVector<Value>{ifExpr.getResult(0)};
        });

    return loopNest;
  }

  // Analyzes a DMA stream and returns a vector of stream indices, the
  // underlying shard shape, and the max coalescing factor
  static std::pair<SmallVector<Value>, size_t>
  analyzeStream(PatternRewriter &rewriter, Location loc,
                AffineMap dmaIndexingMap, MemRefType memref,
                ViewOpInterface viewInterface, GenericOp genericParent) {
    size_t elemSizeBytes = getElementSizeBytes(memref);
    ttcore::DeviceLayoutInterface layout =
        mlir::cast<ttcore::DeviceLayoutInterface>(memref.getLayout());
    ArrayRef<int64_t> memrefGridShape = layout.getGridShape(memref);
    ArrayRef<int64_t> memrefShardShape = layout.getShardShape(memref);

    unsigned outputOperandsIndex =
        genericParent.getOutputs().getBeginOperandIndex();
    // The output and the grid indexing must always be aligned.
    AffineMap gridIndexingMap =
        mlir::cast<AffineMapAttr>(
            genericParent.getIndexingMaps()[outputOperandsIndex])
            .getValue();

    // extract core virtualization map with just grid yx results
    AffineMap coreVirtualizationMap = genericParent.getGrid().getMapping();
    if (!coreVirtualizationMap.isEmpty()) {
      coreVirtualizationMap =
          genericParent.getGrid().getMapping().dropResult(0);
    }

    auto [streamIndices, indexBounds] = buildStreamIndex(
        rewriter, loc, memrefGridShape, genericParent.getBlockFactorsValue(),
        memrefShardShape, dmaIndexingMap, gridIndexingMap,
        coreVirtualizationMap);

    ttcore::DeviceAttr device = genericParent.getDevice();
    std::pair<MemRefType, AffineMap> underlyingMemrefAndView =
        viewInterface.applyViews();
    AffineMap memoryMap = device.getMemoryMap(underlyingMemrefAndView,
                                              0 /* use default page size*/);
    size_t coalescingFactor =
        calculateCoalescingFactor(memoryMap, memrefGridShape, memrefShardShape,
                                  elemSizeBytes, indexBounds);

    return {streamIndices, coalescingFactor};
  }

  LogicalResult matchAndRewrite(DMAOp dma,
                                PatternRewriter &rewriter) const final {
    if (!dma.isAffine()) {
      // Already lowered, skip.
      return failure();
    }

    auto affine_map =
        dma.isSrcRemote() ? dma.getSrcAffineMap() : dma.getDstAffineMap();
    auto memref =
        dma.isSrcRemote() ? dma.getSrcMemRefType() : dma.getDstMemRefType();
    auto defining_op = mlir::cast<d2m::ViewOpInterface>(
        dma.isSrcRemote() ? dma.getSrc().getDefiningOp()
                          : dma.getDst().getDefiningOp());

    // analyze remote stream (either src or dst) to extract a vec of stream
    // indices and a max coalescing factor
    auto [streamIndices, coalescingFactor] =
        analyzeStream(rewriter, dma.getLoc(), *affine_map, memref, defining_op,
                      dma->getParentOfType<d2m::GenericOp>());

    ArrayRef<int64_t> memrefShardShape =
        mlir::cast<ttcore::DeviceLayoutInterface>(memref.getLayout())
            .getShardShape(memref);
    size_t shardVolume = ttmlir::utils::volume(memrefShardShape);

    Operation *newDma;
    if (coalescingFactor == shardVolume) {
      // Fully contiguous DMA; lower to a single operation
      auto srcIndices =
          dma.isSrcRemote() ? streamIndices : SmallVector<Value>();
      auto dstIndices =
          dma.isDstRemote() ? streamIndices : SmallVector<Value>();
      newDma = rewriter.create<d2m::DMAOp>(
          dma.getLoc(), dma.getSrc(), srcIndices, dma.getDst(), dstIndices,
          dma.getMcastStartIndex(), dma.getMcastShape());
    } else {

      scf::LoopNest loopNest =
          buildCoalescedGatherLoop(rewriter, dma.getLoc(), dma, streamIndices,
                                   memrefShardShape, coalescingFactor);
      newDma = loopNest.loops.front();
    }

    rewriter.replaceOp(dma, newDma);
    return success();
  }
};
} // namespace

namespace {
class D2MGenericLowerToFullyIndexedDMARewritePattern
    : public OpRewritePattern<DMAOp> {
public:
  using OpRewritePattern<DMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DMAOp dma,
                                PatternRewriter &rewriter) const final {
    if (dma.isAffine()) {
      // Lower to affine first.
      // Or if it's already fully lowered, nothing to do.
      return failure();
    }

    // Fully index the memrefs.
    SmallVector<Value> srcIndices(dma.getSrcIndices());
    SmallVector<Value> dstIndices(dma.getDstIndices());
    Value zero = rewriter.create<arith::ConstantOp>(
        dma.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
    while (srcIndices.size() <
           static_cast<size_t>(dma.getSrcMemRefType().getRank())) {
      srcIndices.push_back(zero);
    }
    while (dstIndices.size() <
           static_cast<size_t>(dma.getDstMemRefType().getRank())) {
      dstIndices.push_back(zero);
    }

    ttcore::DeviceAttr device = ttcore::lookupDevice(dma);
    AffineMap srcMemoryMap =
        getMemoryMap(device, dma.getSrc(), dma.isSrcRemote());
    AffineMap dstMemoryMap =
        getMemoryMap(device, dma.getDst(), dma.isDstRemote());

    srcIndices = applyMap(rewriter, dma.getLoc(), srcMemoryMap, srcIndices,
                          dma.isSrcRemote());
    dstIndices = applyMap(rewriter, dma.getLoc(), dstMemoryMap, dstIndices,
                          dma.isDstRemote());

    bool isLoweredToWrite =
        dma.isDstRemote() || (dma.isSrcLocal() && dma.isDstLocal());

    if (isLoweredToWrite) {
      rewriter.replaceOpWithNewOp<d2m::DMAWriteOp>(
          dma, dma.getResult().getType(), dma.getSrc(), srcIndices,
          dma.getDst(), dstIndices,
          rewriter.getI64IntegerAttr(dma.getNumElems()),
          dma.getMcastStartIndex(), dma.getMcastShape());
    } else {
      // should never have multicast fields defined for reads
      assert(dma.getMcastStartIndex().empty() && dma.getMcastShape().empty());
      rewriter.replaceOpWithNewOp<d2m::DMAReadOp>(
          dma, dma.getResult().getType(), dma.getSrc(), srcIndices,
          dma.getDst(), dstIndices,
          rewriter.getI64IntegerAttr(dma.getNumElems()));
    }

    return success();
  }

  static AffineMap getMemoryMap(ttcore::DeviceAttr device, Value input,
                                bool isRemote) {
    if (isRemote) {
      std::pair<MemRefType, AffineMap> srcUnderlyingMemrefAndView =
          mlir::tt::d2m::applyViews(input.getDefiningOp());
      return device.getMemoryMap(srcUnderlyingMemrefAndView,
                                 0 /* use default page size*/);
    }

    MemRefType inputType = mlir::cast<MemRefType>(input.getType());
    return canonicalStridedMap(device.getContext(), inputType.getShape(),
                               inputType.getElementType(),
                               inputType.getLayout().getAffineMap());
  }

  static SmallVector<Value> applyMap(PatternRewriter &rewriter, Location loc,
                                     AffineMap map, ValueRange index,
                                     bool isRemote) {
    auto affineApply = [&](AffineMap map, ValueRange index) {
      return rewriter.create<affine::AffineApplyOp>(loc, map, index);
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

  static AffineMap canonicalStridedMap(MLIRContext *context,
                                       ArrayRef<int64_t> shape,
                                       Type elementType, AffineMap map) {
    assert(map.isIdentity() && "Only identity maps are supported for now.");
    auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType);
    int64_t elementSizeBytes = tileType
                                   ? tileType.getSizeBytes()
                                   : elementType.getIntOrFloatBitWidth() / 8;
    int64_t currentStride = elementSizeBytes;
    int64_t rank = shape.size();
    mlir::AffineExpr strideExpr = getAffineConstantExpr(0, context);
    for (int64_t i = rank - 1; i >= 0; i--) {
      mlir::AffineExpr dim = getAffineDimExpr(i, context);
      mlir::AffineExpr stride = getAffineConstantExpr(currentStride, context);
      strideExpr = dim * stride + strideExpr;
      currentStride *= shape[i];
    }
    return mlir::AffineMap::get(shape.size(), 0, strideExpr, context);
  }
};
} // namespace

namespace {
class D2MGenericLowerDMAs
    : public impl::D2MGenericLowerDMAsBase<D2MGenericLowerDMAs> {
public:
  using impl::D2MGenericLowerDMAsBase<
      D2MGenericLowerDMAs>::D2MGenericLowerDMAsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MGenericLowerAffineDMAsRewritePattern,
                 D2MGenericLowerToFullyIndexedDMARewritePattern>(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
