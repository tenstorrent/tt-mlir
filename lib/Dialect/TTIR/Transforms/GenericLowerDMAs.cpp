// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
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

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICLOWERDMAS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRGenericLowerAffineDMAsRewritePattern
    : public OpRewritePattern<DMAOp> {
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
                   AffineMap gridIndexingMap) {
    assert(dmaIndexingMap.getNumDims() == gridIndexingMap.getNumDims());
    assert(dmaIndexingMap.getNumResults() == gridIndexingMap.getNumResults());
    assert(dmaIndexingMap.getNumResults() == shardShape.size());
    assert(gridIndexingMap.isProjectedPermutation() &&
           "Grid indexing map must be a permutation");

    SmallVector<Value> streamIndex;
    SmallVector<int64_t> indexBounds;
    streamIndex.reserve(dmaIndexingMap.getNumResults());
    indexBounds.reserve(dmaIndexingMap.getNumResults());
    for (unsigned result = 0; result < dmaIndexingMap.getNumResults();
         result++) {
      unsigned dim = dmaIndexingMap.getDimPosition(result);
      std::optional<unsigned> gridResult =
          gridIndexingMap.getResultPosition(dmaIndexingMap.getResult(result));
      //
      // The following assert is pretty subtle. If this operand dimension
      // participates in the grid, for now we must assert that the relative grid
      // result is the same as the operand result. This protects against
      // permutations of the grid, ie. transposes.  For example, let's consider
      // a matmul case:
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
      Value index;
      if (isGridDim) {
        // The grid dimension is always 1-1 with the result position.  Consider
        // the case where interchange moves k to the outermost loop. We'd have
        // output map: (k, m, n) -> (m, n)
        // In this example we want (m, n) to map to grid dims (0, 1) (not their
        // dim positions i.e. (1, 2)).
        const unsigned gridDim = result;

        // gridI * blockFactorI + iterI
        Value blockFactor = builder.create<arith::ConstantOp>(
            loc, builder.getIndexType(),
            builder.getIndexAttr(blockFactors[dim]));
        index = builder.create<CoreIndexOp>(loc, builder.getIndexType(),
                                            builder.getI64IntegerAttr(gridDim));
        index = builder.create<arith::MulIOp>(loc, builder.getIndexType(),
                                              index, blockFactor);
        index = builder.create<arith::AddIOp>(loc, builder.getIndexType(),
                                              index, iterIndex);
      } else {
        index = iterIndex;
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

  static scf::LoopNest
  fallbackSingleTileGatherLoop(OpBuilder &builder, Location loc, DMAOp dma,
                               ArrayRef<Value> streamIndex,
                               ArrayRef<int64_t> shardShape) {
    auto [lbs, ubs, steps] = getLoopBounds(builder, loc, shardShape);

    auto initTx = builder.create<ttir::NullTxOp>(dma.getLoc());
    scf::LoopNest loopNest = scf::buildLoopNest(
        builder, loc, lbs, ubs, steps, ValueRange(initTx),
        [&](OpBuilder &builder, Location loc, ValueRange iters,
            ValueRange /*args*/) {
          SmallVector<Value> srcIndex =
              llvm::to_vector(llvm::concat<Value>(streamIndex, iters));
          return SmallVector<Value>{builder.create<ttir::DMAOp>(
              dma.getLoc(), dma.getSrc(), srcIndex, dma.getDst(), iters,
              dma.getMcastStartIndex(), dma.getMcastShape())};
        });
    return loopNest;
  }

  LogicalResult matchAndRewrite(DMAOp dma,
                                PatternRewriter &rewriter) const final {
    if (!dma.isAffine()) {
      // Already lowered, skip.
      return failure();
    }
    assert(!dma.getDstAffineMap() && "DMA dst affine map not supported yet");
    assert(dma.getSrcAffineMap() && "DMA src affine map expected");

    AffineMap dmaIndexingMap = *dma.getSrcAffineMap();
    MemRefType memref = dma.getSrcMemRefType();
    ttcore::DeviceLayoutInterface layout =
        mlir::cast<ttcore::DeviceLayoutInterface>(memref.getLayout());
    ArrayRef<int64_t> memrefGridShape = layout.getGridShape(memref);
    ArrayRef<int64_t> memrefShardShape = layout.getShardShape(memref);

    GenericOp genericParent = dma->getParentOfType<ttir::GenericOp>();
    unsigned outputOperandsIndex =
        genericParent.getOutputs().getBeginOperandIndex();
    // The output and the grid indexing must always be aligned.
    AffineMap gridIndexingMap =
        mlir::cast<AffineMapAttr>(
            genericParent.getIndexingMaps()[outputOperandsIndex])
            .getValue();

    auto [streamIndex, indexBounds] =
        buildStreamIndex(rewriter, dma.getLoc(), memrefGridShape,
                         genericParent.getBlockFactorsValue(), memrefShardShape,
                         dmaIndexingMap, gridIndexingMap);

    ttcore::DeviceAttr device = genericParent.getDevice();
    std::pair<MemRefType, AffineMap> underlyingMemrefAndView =
        mlir::cast<ttir::ViewOpInterface>(dma.getSrc().getDefiningOp())
            .applyViews();
    size_t size = device.getMemrefSizeBytes(underlyingMemrefAndView.first);
    AffineMap memoryMap = device.getMemoryMap(underlyingMemrefAndView, size);
    size_t elemSizeBytes = getElementSizeBytes(memref);
    size_t coalescingFactor =
        calculateCoalescingFactor(memoryMap, memrefGridShape, memrefShardShape,
                                  elemSizeBytes, indexBounds);

    Operation *newDma;
    if (coalescingFactor ==
        static_cast<size_t>(ttmlir::utils::volume(memrefShardShape))) {
      // Fully coalesced, we can trivially lower.
      newDma = rewriter.create<ttir::DMAOp>(
          dma.getLoc(), dma.getSrc(), streamIndex, dma.getDst(),
          dma.getMcastStartIndex(), dma.getMcastShape());
    } else {
      // Fallback to single tile gather for now, in the future we can chage this
      // to support more sophisticated gathering.
      scf::LoopNest loopNest = fallbackSingleTileGatherLoop(
          rewriter, dma.getLoc(), dma, streamIndex, memrefShardShape);
      assert(loopNest.loops.size() == memrefShardShape.size());
      newDma = loopNest.loops.front();
    }

    rewriter.replaceOp(dma, newDma);
    return success();
  }
};
} // namespace

namespace {
class TTIRGenericLowerToFullyIndexedDMARewritePattern
    : public OpRewritePattern<DMAOp> {
public:
  using OpRewritePattern<DMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DMAOp dma,
                                PatternRewriter &rewriter) const final {
    if (dma.isAffine() || dma.isLowered()) {
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

    rewriter.replaceOpWithNewOp<ttir::DMAOp>(
        dma, dma.getResult().getType(), dma.getSrc(), nullptr, srcIndices,
        dma.getDst(), nullptr, dstIndices,
        rewriter.getI64IntegerAttr(dma.getNumElems()), dma.getMcastStartIndex(),
        dma.getMcastShape());

    return success();
  }

  static AffineMap getMemoryMap(ttcore::DeviceAttr device, Value input,
                                bool isRemote) {
    if (isRemote) {
      std::pair<MemRefType, AffineMap> srcUnderlyingMemrefAndView =
          mlir::tt::ttir::applyViews(input.getDefiningOp());
      size_t srcSize =
          device.getMemrefSizeBytes(srcUnderlyingMemrefAndView.first);
      return device.getMemoryMap(srcUnderlyingMemrefAndView, srcSize);
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
class TTIRGenericLowerDMAs
    : public impl::TTIRGenericLowerDMAsBase<TTIRGenericLowerDMAs> {
public:
  using impl::TTIRGenericLowerDMAsBase<
      TTIRGenericLowerDMAs>::TTIRGenericLowerDMAsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericLowerAffineDMAsRewritePattern,
                 TTIRGenericLowerToFullyIndexedDMARewritePattern>(
        &getContext());
    populateAffineToStdConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
