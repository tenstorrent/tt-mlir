// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include <functional>
#include <limits>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSEMASKING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kTileHeight = 32;
constexpr int64_t kTileWidth = 32;

struct BoundsInterval {
  Value start;
  Value end;
};

// Build an element-typed OOB fill attribute. For integer types, Inf/NegInf
// are saturated to the representable min/max of the type.
static TypedAttr getFillValueAttr(Builder &builder, Type elemType,
                                  ttcore::OOBVal oobVal) {
  if (auto floatTy = dyn_cast<FloatType>(elemType)) {
    double v = 0.0;
    switch (oobVal) {
    case ttcore::OOBVal::Undef:
    case ttcore::OOBVal::Zero:
      v = 0.0;
      break;
    case ttcore::OOBVal::One:
      v = 1.0;
      break;
    case ttcore::OOBVal::Inf:
      v = std::numeric_limits<double>::infinity();
      break;
    case ttcore::OOBVal::NegInf:
      v = -std::numeric_limits<double>::infinity();
      break;
    }
    return builder.getFloatAttr(floatTy, v);
  }
  if (auto intTy = dyn_cast<IntegerType>(elemType)) {
    unsigned w = intTy.getWidth();
    bool isUnsigned = intTy.isUnsigned();
    APInt v(w, 0);
    switch (oobVal) {
    case ttcore::OOBVal::Undef:
    case ttcore::OOBVal::Zero:
      break;
    case ttcore::OOBVal::One:
      v = APInt(w, 1);
      break;
    case ttcore::OOBVal::Inf:
      v = isUnsigned ? APInt::getMaxValue(w) : APInt::getSignedMaxValue(w);
      break;
    case ttcore::OOBVal::NegInf:
      v = isUnsigned ? APInt(w, 0) : APInt::getSignedMinValue(w);
      break;
    }
    // arith.constant requires signless int types; TileFillOp accepts any
    // integer so this is safe for the downstream use.
    return builder.getIntegerAttr(IntegerType::get(builder.getContext(), w), v);
  }
  llvm_unreachable("unsupported element type for OOB fill");
}

static ttcore::GridAttr getMaskGridAttr(OpBuilder &builder, Value output,
                                        ArrayRef<int64_t> gridShape) {
  if (auto maps = utils::getGridMapsFromVirtualGridMapping(output, gridShape)) {
    return ttcore::GridAttr::get(builder.getContext(), gridShape, maps->first,
                                 maps->second);
  }
  return ttcore::GridAttr::get(builder.getContext(), gridShape);
}

/// Decompose MaskOp with multi-core support.
///
/// The key change from single-core: loop bounds become dynamic based on
/// which portion of the global tile space this core is responsible for.
///
/// For each loop:
///   start = stride * coreIndex.
///   end = min(regionEnd, stride * (coreIndex + 1)).
///   localIdx = globalIdx - (stride * coreIndex).
///
/// If start >= end, the loop doesn't run--we only pad rightmost + downmost
/// regions, so this should be correct w/o modifying start with max().
struct DecomposeMaskPattern : OpRewritePattern<MaskOp> {
  DecomposeMaskPattern(MLIRContext *ctx, unsigned numStreamBuffers)
      : OpRewritePattern<MaskOp>(ctx), numStreamBuffers(numStreamBuffers) {}

  unsigned numStreamBuffers;

  // Compute local loop bounds for a core given a global region
  // [globalRegionStart, globalRegionEnd). Returns (localStart, localEnd) such
  // that iterating [localStart, localEnd) in local coordinates covers exactly
  // the tiles that fall within both the global region and this core's shard.
  static std::pair<Value, Value>
  computeLocalBounds(PatternRewriter &rewriter, Location loc,
                     int64_t globalRegionStart, int64_t globalRegionEnd,
                     Value coreIdx, int64_t shardSize) {

    Value shardSizeVal =
        rewriter.create<arith::ConstantIndexOp>(loc, shardSize);
    Value globalCoreStart =
        rewriter.create<arith::MulIOp>(loc, coreIdx, shardSizeVal);

    Value globalRegionStartVal =
        rewriter.create<arith::ConstantIndexOp>(loc, globalRegionStart);
    Value globalRegionEndVal =
        rewriter.create<arith::ConstantIndexOp>(loc, globalRegionEnd);

    // We define localStart = max(globalRegionStart - globalCoreStart, 0); in
    // turn this can be rewritten as localStart = globalRegionStart -
    // min(globalRegionStart, globalCoreStart).
    Value clampedStart = rewriter.create<arith::MinUIOp>(
        loc, globalRegionStartVal, globalCoreStart);
    Value localStart =
        rewriter.create<arith::SubIOp>(loc, globalRegionStartVal, clampedStart);

    // Similarly, we define localEnd = min(globalRegionEnd - globalCoreStart,
    // shardSize). However, to avoid underflow on unsigned, we re-express it as
    // clampedEnd = max(min(globalRegionEnd, globalCoreEnd), globalCoreStart),
    // and localEnd = clampedEnd - globalCoreStart, which is equivalent.
    Value globalCoreEnd =
        rewriter.create<arith::AddIOp>(loc, globalCoreStart, shardSizeVal);
    Value clampedEnd =
        rewriter.create<arith::MinUIOp>(loc, globalRegionEndVal, globalCoreEnd);
    clampedEnd =
        rewriter.create<arith::MaxUIOp>(loc, clampedEnd, globalCoreStart);
    Value localEnd =
        rewriter.create<arith::SubIOp>(loc, clampedEnd, globalCoreStart);

    return {localStart, localEnd};
  }

  LogicalResult matchAndRewrite(MaskOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value globalInput = op.getInput();
    Value globalOutput = op.getOutput();
    ttcore::OOBVal fillOOBVal = op.getFillValue();

    if (isa<RankedTensorType>(globalInput.getType())) {
      return rewriter.notifyMatchFailure(op, "tensor semantics not supported");
    }

    auto globalInputType = cast<MemRefType>(globalInput.getType());
    auto globalOutputType = cast<MemRefType>(globalOutput.getType());
    if (!ttcore::hasDeviceLayout(globalInput) ||
        !ttcore::hasDeviceLayout(globalOutput)) {
      return rewriter.notifyMatchFailure(op, "mask operands need layouts");
    }

    ArrayRef<int64_t> gridShape = ttcore::getGridShape(globalOutput);
    if (gridShape.size() < 2) {
      return rewriter.notifyMatchFailure(
          op, "grid must have at least 2 dimensions");
    }

    SmallVector<int64_t> shardShape =
        llvm::to_vector(ttcore::getShardShape(globalInput));
    if (shardShape.size() < 2) {
      return rewriter.notifyMatchFailure(op, "input must have at least 2 dims");
    }

    ArrayRef<int64_t> logicalShape = op.getLogicalShape();
    int64_t logicalRows = logicalShape[logicalShape.size() - 2];
    int64_t logicalCols = logicalShape[logicalShape.size() - 1];

    auto gridAttr = getMaskGridAttr(rewriter, globalOutput, gridShape);
    ArrayAttr emptyArray = rewriter.getArrayAttr({});
    ArrayAttr threads = rewriter.getArrayAttr(
        rewriter.getAttr<ThreadAttr>(ThreadType::Unified));
    auto genericOp = rewriter.create<GenericOp>(
        loc, TypeRange{}, ValueRange{globalInput}, ValueRange{globalOutput},
        ValueRange{}, gridAttr, emptyArray, emptyArray, emptyArray, threads,
        /*fabricConnectionConfig=*/nullptr, /*regionsCount=*/1);

    Region &region = genericOp.getRegion(0);
    rewriter.createBlock(&region);
    rewriter.setInsertionPointToStart(&region.front());

    Attribute memorySpace = globalOutputType.getMemorySpace();
    Type tileElementType = globalInputType.getElementType();
    auto inputType = MemRefType::get(shardShape, tileElementType,
                                     MemRefLayoutAttrInterface{}, memorySpace);
    auto outputType = MemRefType::get(shardShape, tileElementType,
                                      MemRefLayoutAttrInterface{}, memorySpace);
    auto maskType = MemRefType::get({1, 1}, tileElementType,
                                    MemRefLayoutAttrInterface{}, memorySpace);

    // The synchronized buffer attribute is set by MarkSynchronizedBuffers pass
    auto inputOp = rewriter.create<memref::AllocOp>(loc, inputType);
    Value input = inputOp.getResult();
    auto outputOp = rewriter.create<memref::AllocOp>(loc, outputType);
    Value output = outputOp.getResult();
    auto rowMaskCBOp = rewriter.create<memref::AllocOp>(loc, maskType);
    Value rowMaskCB = rowMaskCBOp.getResult();
    auto colMaskCBOp = rewriter.create<memref::AllocOp>(loc, maskType);
    Value colMaskCB = colMaskCBOp.getResult();

    SmallVector<Value> remoteIndices;
    remoteIndices.reserve(gridShape.size());
    for (size_t dim = 0; dim < gridShape.size(); ++dim) {
      remoteIndices.push_back(
          rewriter.create<CoreIndexOp>(loc, static_cast<int64_t>(dim)));
    }

    rewriter.create<RemoteLoadOp>(loc, input, globalInput, remoteIndices);

    ArrayRef<int64_t> inputShape = inputType.getShape();
    auto tileType = cast<ttcore::TileType>(inputType.getElementType());
    Type elemType = tileType.getElementType();

    int64_t shardTileRows = inputShape[inputShape.size() - 2];
    int64_t shardTileCols = inputShape[inputShape.size() - 1];

    // Compute tile-level boundaries (compile-time constants).
    // lastValidRow: the last tile row that contains any valid data (may be
    // partial).
    // lastValidCol: the last tile col that contains any valid data
    // (may be partial).
    int64_t lastValidRow = (logicalRows - 1) / kTileHeight;
    int64_t lastValidCol = (logicalCols - 1) / kTileWidth;

    // Compute how many elements are valid in the last partial tile to generate
    // bitmask.
    int64_t validRowsInLastTile = logicalRows % kTileHeight;
    if (validRowsInLastTile == 0) {
      validRowsInLastTile = kTileHeight;
    }
    int64_t validColsInLastTile = logicalCols % kTileWidth;
    if (validColsInLastTile == 0) {
      validColsInLastTile = kTileWidth;
    }
    bool hasPartialRow = validRowsInLastTile != kTileHeight;
    bool hasPartialCol = validColsInLastTile != kTileWidth;

    // Total tiles in the padded shape.
    int64_t totalTileRows = shardTileRows * gridShape[gridShape.size() - 2];
    int64_t totalTileCols = shardTileCols * gridShape[gridShape.size() - 1];
    int64_t validTileRows = lastValidRow + 1;
    int64_t validTileCols = lastValidCol + 1;

    // Fully valid regions can include the final tile row/col when the logical
    // size is tile-aligned along that dimension.
    int64_t interiorRowEnd = hasPartialRow ? lastValidRow : validTileRows;
    int64_t interiorColEnd = hasPartialCol ? lastValidCol : validTileCols;

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    TypedAttr fillAttr = getFillValueAttr(rewriter, elemType, fillOOBVal);
    Value fillScalar =
        rewriter.create<arith::ConstantOp>(loc, fillAttr.getType(), fillAttr);

    // Get this core's coordinates for the two tiled dimensions that masking
    // applies to. Higher-rank shards may have leading dimensions before the
    // tiled row/col dimensions.
    int64_t rowGridDim = static_cast<int64_t>(gridShape.size()) - 2;
    int64_t colGridDim = static_cast<int64_t>(gridShape.size()) - 1;
    Value coreY = rewriter.create<CoreIndexOp>(loc, rowGridDim);
    Value coreX = rewriter.create<CoreIndexOp>(loc, colGridDim);

    // Write the mask tiles.
    Value validRowsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, validRowsInLastTile);
    Value validColsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, validColsInLastTile);

    TT_assert(rowMaskCB);
    if (hasPartialRow) {
      rewriter.create<WriteRowMaskTileOp>(loc, validRowsVal, rowMaskCB);
    }
    TT_assert(colMaskCB);
    if (hasPartialCol) {
      rewriter.create<WriteColMaskTileOp>(loc, validColsVal, colMaskCB);
    }

    // === Tile operation helpers ===
    auto createTileFill = [&]() {
      return rewriter.create<TileFillOp>(loc, tileType, fillScalar).getResult();
    };

    auto buildLocalIndices = [&](ArrayRef<Value> leadingIndices,
                                 Value localRowIdx, Value localColIdx) {
      SmallVector<Value> indices;
      indices.reserve(inputType.getRank());
      indices.append(leadingIndices.begin(), leadingIndices.end());
      indices.push_back(localRowIdx);
      indices.push_back(localColIdx);
      return indices;
    };

    auto emitPassthrough = [&](ArrayRef<Value> leadingIndices,
                               Value localRowIdx, Value localColIdx) {
      SmallVector<Value> indices =
          buildLocalIndices(leadingIndices, localRowIdx, localColIdx);
      auto inputTile = rewriter.create<memref::LoadOp>(loc, input, indices);
      rewriter.create<memref::StoreOp>(loc, inputTile.getResult(), output,
                                       indices);
    };

    auto emitRowMasked = [&](ArrayRef<Value> leadingIndices, Value localRowIdx,
                             Value localColIdx) {
      SmallVector<Value> indices =
          buildLocalIndices(leadingIndices, localRowIdx, localColIdx);
      auto inputTile = rewriter.create<memref::LoadOp>(loc, input, indices);
      auto tileFill = createTileFill();
      auto rowMaskTile = rewriter.create<memref::LoadOp>(
          loc, rowMaskCB, ValueRange{zeroIdx, zeroIdx});
      auto result =
          rewriter.create<TileWhereOp>(loc, tileType, rowMaskTile.getResult(),
                                       inputTile.getResult(), tileFill);
      rewriter.create<memref::StoreOp>(loc, result.getResult(), output,
                                       indices);
    };

    auto emitColMasked = [&](ArrayRef<Value> leadingIndices, Value localRowIdx,
                             Value localColIdx) {
      SmallVector<Value> indices =
          buildLocalIndices(leadingIndices, localRowIdx, localColIdx);
      auto inputTile = rewriter.create<memref::LoadOp>(loc, input, indices);
      auto tileFill = createTileFill();
      auto colMaskTile = rewriter.create<memref::LoadOp>(
          loc, colMaskCB, ValueRange{zeroIdx, zeroIdx});
      auto result =
          rewriter.create<TileWhereOp>(loc, tileType, colMaskTile.getResult(),
                                       inputTile.getResult(), tileFill);
      rewriter.create<memref::StoreOp>(loc, result.getResult(), output,
                                       indices);
    };

    auto emitCornerMasked = [&](ArrayRef<Value> leadingIndices,
                                Value localRowIdx, Value localColIdx) {
      SmallVector<Value> indices =
          buildLocalIndices(leadingIndices, localRowIdx, localColIdx);
      auto inputTile = rewriter.create<memref::LoadOp>(loc, input, indices);
      auto tileFill1 = createTileFill();
      auto rowMaskTile = rewriter.create<memref::LoadOp>(
          loc, rowMaskCB, ValueRange{zeroIdx, zeroIdx});
      auto rowMaskedResult =
          rewriter.create<TileWhereOp>(loc, tileType, rowMaskTile.getResult(),
                                       inputTile.getResult(), tileFill1);
      auto tileFill2 = createTileFill();
      auto colMaskTile = rewriter.create<memref::LoadOp>(
          loc, colMaskCB, ValueRange{zeroIdx, zeroIdx});
      auto finalResult =
          rewriter.create<TileWhereOp>(loc, tileType, colMaskTile.getResult(),
                                       rowMaskedResult.getResult(), tileFill2);
      rewriter.create<memref::StoreOp>(loc, finalResult.getResult(), output,
                                       indices);
    };

    auto emitFill = [&](ArrayRef<Value> leadingIndices, Value localRowIdx,
                        Value localColIdx) {
      SmallVector<Value> indices =
          buildLocalIndices(leadingIndices, localRowIdx, localColIdx);
      auto tileFill = createTileFill();
      rewriter.create<memref::StoreOp>(loc, tileFill, output, indices);
    };

    // Helper to create a nested loop over local coordinates.
    auto createLocalLoop =
        [&](Value rowStart, Value rowEnd, Value colStart, Value colEnd,
            std::function<void(ArrayRef<Value>, Value, Value)> emitBody) {
          SmallVector<Value> leadingIndices;
          Operation *outermostLoop = nullptr;

          std::function<Operation *(unsigned)> buildLoopNest =
              [&](unsigned dim) -> Operation * {
            if (dim == inputType.getRank() - 2) {
              auto rowLoop =
                  rewriter.create<scf::ForOp>(loc, rowStart, rowEnd, oneIdx);
              if (!outermostLoop) {
                outermostLoop = rowLoop;
              }
              rewriter.setInsertionPointToStart(rowLoop.getBody());

              auto colLoop =
                  rewriter.create<scf::ForOp>(loc, colStart, colEnd, oneIdx);
              // Mark the INNER loop as the compute root, since that's where
              // the actual compute operations are emitted. This ensures DST
              // syncs are placed inside the inner loop body, not the outer.
              // Since we emit an scf.for directly, we must tag this here
              // since linalg-to-affine and d2m-op-scheduler won't process this.
              colLoop->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
              colLoop->setAttr("d2m.scheduled", rewriter.getUnitAttr());
              rewriter.setInsertionPointToStart(colLoop.getBody());

              emitBody(leadingIndices, rowLoop.getInductionVar(),
                       colLoop.getInductionVar());
              return rowLoop;
            }

            Value loopEnd =
                rewriter.create<arith::ConstantIndexOp>(loc, inputShape[dim]);
            auto loop =
                rewriter.create<scf::ForOp>(loc, zeroIdx, loopEnd, oneIdx);
            if (!outermostLoop) {
              outermostLoop = loop;
            }
            leadingIndices.push_back(loop.getInductionVar());
            rewriter.setInsertionPointToStart(loop.getBody());
            buildLoopNest(dim + 1);
            leadingIndices.pop_back();
            return loop;
          };

          buildLoopNest(/*dim=*/0);
          return outermostLoop;
        };

    auto createGlobalRegionLoop =
        [&](int64_t rowGlobalStart, int64_t rowGlobalEnd,
            int64_t colGlobalStart, int64_t colGlobalEnd,
            std::function<void(ArrayRef<Value>, Value, Value)> emitBody) {
          if (rowGlobalStart >= rowGlobalEnd ||
              colGlobalStart >= colGlobalEnd) {
            return static_cast<Operation *>(nullptr);
          }
          auto [rowStart, rowEnd] =
              computeLocalBounds(rewriter, loc, rowGlobalStart, rowGlobalEnd,
                                 coreY, shardTileRows);
          auto [colStart, colEnd] =
              computeLocalBounds(rewriter, loc, colGlobalStart, colGlobalEnd,
                                 coreX, shardTileCols);
          return createLocalLoop(rowStart, rowEnd, colStart, colEnd, emitBody);
        };

    OpBuilder::InsertionGuard guard(rewriter);
    Operation *insertionPoint = &region.front().back();

    // =========================================================================
    // LOOP 0: Interior tiles - fully valid.
    // Global region: [0, interiorRowEnd) x [0, interiorColEnd).
    // Tile-aligned dimensions include their final valid tile here.
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto *loop = createGlobalRegionLoop(0, interiorRowEnd, 0, interiorColEnd,
                                          emitPassthrough);
      if (loop) {
        insertionPoint = loop;
      }
    }

    // =========================================================================
    // LOOP 1: Last valid row - needs row masking.
    // Global region: [lastValidRow, lastValidRow+1) x [0, interiorColEnd).
    // Emitted only when the row dimension has a partial final tile.
    // =========================================================================
    if (hasPartialRow) {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto *loop = createGlobalRegionLoop(lastValidRow, lastValidRow + 1, 0,
                                          interiorColEnd, emitRowMasked);
      if (loop) {
        insertionPoint = loop;
      }
    }

    // =========================================================================
    // LOOP 2: Last valid col - needs col masking.
    // Global region: [0, interiorRowEnd) x [lastValidCol, lastValidCol+1).
    // Emitted only when the column dimension has a partial final tile.
    // =========================================================================
    if (hasPartialCol) {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto *loop = createGlobalRegionLoop(0, interiorRowEnd, lastValidCol,
                                          lastValidCol + 1, emitColMasked);
      if (loop) {
        insertionPoint = loop;
      }
    }

    // =========================================================================
    // LOOP 3: Corner tile - needs both row and col masking.
    // Global region: [lastValidRow, lastValidRow+1) x [lastValidCol,
    // lastValidCol+1).
    // =========================================================================
    if (hasPartialRow && hasPartialCol) {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto *loop =
          createGlobalRegionLoop(lastValidRow, lastValidRow + 1, lastValidCol,
                                 lastValidCol + 1, emitCornerMasked);
      if (loop) {
        insertionPoint = loop;
      }
    }

    // =========================================================================
    // LOOP 4: OOB rows - fill entire rows beyond valid region.
    // Global region: [lastValidRow+1, totalTileRows) x [0, totalTileCols).
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto *loop = createGlobalRegionLoop(validTileRows, totalTileRows, 0,
                                          totalTileCols, emitFill);
      if (loop) {
        insertionPoint = loop;
      }
    }

    // =========================================================================
    // LOOP 5: OOB cols - fill columns beyond valid region (for valid rows
    // only). Global region: [0, validTileRows) x [validTileCols,
    // totalTileCols).
    // =========================================================================
    {
      rewriter.setInsertionPointAfter(insertionPoint);
      auto *loop = createGlobalRegionLoop(0, validTileRows, validTileCols,
                                          totalTileCols, emitFill);
      if (loop) {
        insertionPoint = loop;
      }
    }

    rewriter.setInsertionPointAfter(insertionPoint);
    rewriter.create<RemoteStoreOp>(
        loc, /*resultTypes=*/TypeRange{}, globalOutput, remoteIndices, output,
        /*cb=*/Value{}, /*startDevice=*/ValueRange{},
        /*deviceMcastShape=*/ValueRange{}, /*semaphore=*/Value{},
        /*semaphoreIndices=*/ValueRange{});

    rewriter.replaceOp(op, op.getOutput());
    return success();
  }
};

struct D2MDecomposeMasking
    : public impl::D2MDecomposeMaskingBase<D2MDecomposeMasking> {
  using impl::D2MDecomposeMaskingBase<
      D2MDecomposeMasking>::D2MDecomposeMaskingBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeMaskPattern>(ctx, numStreamBuffers);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
} // namespace mlir::tt::d2m
