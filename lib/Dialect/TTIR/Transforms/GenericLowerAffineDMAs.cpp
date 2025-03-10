// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Memref/IR/Memref.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICLOWERAFFINEDMAS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRGenericLowerAffineDMAsRewritePattern
    : public OpRewritePattern<DMAOp> {
public:
  using OpRewritePattern<DMAOp>::OpRewritePattern;

  static std::tuple<SmallVector<Value>, SmallVector<int64_t>>
  buildStreamIndex(OpBuilder &builder, Location loc,
                   ArrayRef<int64_t> gridShape, ArrayRef<int64_t> shardShape,
                   AffineMap dmaIndexingMap, AffineMap gridIndexingMap) {
    assert(dmaIndexingMap.getNumDims() == gridIndexingMap.getNumDims());
    assert(dmaIndexingMap.getNumResults() ==
           gridIndexingMap.getNumResults());
    assert(dmaIndexingMap.getNumResults() == shardShape.size());

    SmallVector<Value> streamIndex;
    SmallVector<int64_t> indexBounds;
    streamIndex.reserve(dmaIndexingMap.getNumResults());
    indexBounds.reserve(dmaIndexingMap.getNumResults());
    for (unsigned result = 0; result < dmaIndexingMap.getNumResults();
         result++) {
      unsigned dim = dmaIndexingMap.getDimPosition(result);
      std::optional<unsigned> gridResult = gridIndexingMap.getResultPosition(
          dmaIndexingMap.getResult(result));
      //
      // The following assert is pretty subtle. If this operand dimension
      // participates in the grid, for now we must assert that the relative grid
      // result is the same as the operand result. This protects against
      // permutations of the grid, ie. transposes.  For example, let's consider
      // a matmul case:
      //   dmaIndexingMap: (m, n, k) -> (m, k)
      //   dmaIndexingMap: (m, n, k) -> (k, n)
      //   gridIndexingMap:    (m, n, k) -> (m, n)
      //                                     ^
      // This assert ensures that the m's line up. A counter example would be:
      //   dmaIndexingMap: (m, n, k) -> (m, k)
      //   gridIndexingMap:    (m, n, k) -> (n, m)
      //
      // Not currently supported.
      //
      assert(!gridResult || *gridResult == result);
      bool isGridDim = gridResult.has_value();
      Value iterIndex = builder.create<IterIndexOp>(
          loc, builder.getIndexType(), builder.getI64IntegerAttr(dim));
      Value index;
      if (isGridDim) {
        // gridI * dimI + iterI
        Value dimConstant = builder.create<arith::ConstantOp>(
            loc, builder.getIndexType(),
            builder.getIndexAttr(shardShape[result]));
        index = builder.create<CoreIndexOp>(loc, builder.getIndexType(),
                                            builder.getI64IntegerAttr(dim));
        index = builder.create<arith::MulIOp>(loc, builder.getIndexType(),
                                              index, dimConstant);
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
    if (mlir::isa<TileType>(elementType)) {
      auto tileType = mlir::cast<TileType>(elementType);
      return tileType.getSizeBytes();
    }
    return elementType.getIntOrFloatBitWidth() / 8;
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

    auto initTx = builder.create<ttir::NullTxOp>(dma.getLoc(),
                                                 builder.getType<MemTxType>());
    scf::LoopNest loopNest = scf::buildLoopNest(
        builder, loc, lbs, ubs, steps, ValueRange(initTx),
        [&](OpBuilder &builder, Location loc, ValueRange iters,
            ValueRange /*args*/) {
          SmallVector<Value> srcIndex =
              llvm::to_vector(llvm::concat<Value>(streamIndex, iters));
          return SmallVector<Value>{builder.create<ttir::DMAOp>(
              dma.getLoc(), builder.getType<MemTxType>(), dma.getSrc(), nullptr,
              srcIndex, dma.getDst(), nullptr, iters, nullptr,
              dma.getDstCoreIndex(), dma.getMcastShape())};
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
    assert(memref.getRank() % 2 == 0 && "Only even rank memrefs are supported");
    ArrayRef<int64_t> memrefShape = memref.getShape();
    ArrayRef<int64_t> memrefGridShape =
        memrefShape.take_front(memrefShape.size() / 2);
    ArrayRef<int64_t> memrefShardShape =
        memrefShape.drop_front(memrefShape.size() / 2);

    GenericOp genericParent = dma->getParentOfType<ttir::GenericOp>();
    auto [outputOperandsIndex, outputOperandsLength] =
        genericParent.getODSOperandIndexAndLength(1);
    // The output and the grid indexing must always be aligned.
    AffineMap gridIndexingMap =
        mlir::cast<AffineMapAttr>(
            genericParent.getIndexingMaps()[outputOperandsIndex])
            .getValue();

    auto [streamIndex, indexBounds] =
        buildStreamIndex(rewriter, dma.getLoc(), memrefGridShape,
                         memrefShardShape, dmaIndexingMap, gridIndexingMap);

    DeviceAttr device = genericParent.getDevice();
    // TODO(#1909) Once we have an allocation pass, we need to lookup the page
    // size instead of calculating it here
    size_t pageSize = getMemRefShardSizeBytes(memref);
    AffineMap memoryMap = device.getMemoryMap(memref, pageSize);
    size_t elemSizeBytes = getElementSizeBytes(memref);
    size_t coalescingFactor =
        calculateCoalescingFactor(memoryMap, memrefGridShape, memrefShardShape,
                                  elemSizeBytes, indexBounds);

    Operation *newDma;
    if (coalescingFactor ==
        static_cast<size_t>(ttmlir::utils::volume(memrefShardShape))) {
      // Fully coalesced, we can trivially lower
      newDma = rewriter.create<ttir::DMAOp>(
          dma.getLoc(), rewriter.getType<MemTxType>(), dma.getSrc(), nullptr,
          streamIndex, dma.getDst(), nullptr, ValueRange(), nullptr,
          dma.getDstCoreIndex(), dma.getMcastShape());
    } else {
      scf::LoopNest loopNest = fallbackSingleTileGatherLoop(
          rewriter, dma.getLoc(), dma, streamIndex, memrefShardShape);
      assert(loopNest.loops.size() == memrefShardShape.size());
      loopNest.loops.front().dump();
      newDma = loopNest.loops.front();
    }

    rewriter.replaceOp(dma, newDma);
    return success();
  }
};
} // namespace

namespace {
class TTIRGenericLowerAffineDMAs
    : public impl::TTIRGenericLowerAffineDMAsBase<TTIRGenericLowerAffineDMAs> {
public:
  using impl::TTIRGenericLowerAffineDMAsBase<
      TTIRGenericLowerAffineDMAs>::TTIRGenericLowerAffineDMAsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericLowerAffineDMAsRewritePattern>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
