// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <array>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSEMASKING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Get float value for OOBVal enum.
static double getOOBValAsFloat(ttcore::OOBVal oobVal) {
  switch (oobVal) {
  case ttcore::OOBVal::Zero:
    return 0.0;
  case ttcore::OOBVal::One:
    return 1.0;
  case ttcore::OOBVal::Inf:
    return std::numeric_limits<double>::infinity();
  case ttcore::OOBVal::NegInf:
    return -std::numeric_limits<double>::infinity();
  case ttcore::OOBVal::Undef:
    llvm_unreachable("Undef OOBVal should not reach masking decomposition");
  }
  llvm_unreachable("Unknown OOBVal");
}

// Create a constant scalar value for use with tile ops.
static Value createScalarConstant(OpBuilder &builder, Location loc, double val,
                                  Type elementType) {
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    return builder.create<arith::ConstantOp>(
        loc, builder.getFloatAttr(floatType, val));
  }
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    return builder.create<arith::ConstantOp>(
        loc, builder.getIntegerAttr(intType, static_cast<int64_t>(val)));
  }
  llvm_unreachable("Unsupported element type for constant creation");
}

// Get the scalar element type from a tile type.
static Type getScalarElementType(Type type) {
  if (auto tile = dyn_cast<ttcore::TileType>(type)) {
    return tile.getElementType();
  }
  if (auto memref = dyn_cast<MemRefType>(type)) {
    return getScalarElementType(memref.getElementType());
  }
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    return getScalarElementType(tensor.getElementType());
  }
  return type;
}

/// Create a global memref containing index tile data as scalars.
/// We store the data as memref<[tileH, tileW]xscalarType> and will convert
/// it to a tile when loading. This avoids the DenseElementsAttr tile type
/// issue.
static memref::GlobalOp
getOrCreateIndexTileGlobal(ModuleOp module, OpBuilder &builder, Type scalarType,
                           bool isRowIndex, int64_t tileH, int64_t tileW) {
  // Create unique name for the global based on type and direction.
  std::string globalName =
      isRowIndex ? "__d2m_row_index_tile" : "__d2m_col_index_tile";
  if (isa<Float32Type>(scalarType)) {
    globalName += "_f32";
  } else if (isa<BFloat16Type>(scalarType)) {
    globalName += "_bf16";
  } else if (isa<Float16Type>(scalarType)) {
    globalName += "_f16";
  } else {
    globalName += "_unknown";
  }

  // Check if global already exists.
  if (auto existingGlobal = module.lookupSymbol<memref::GlobalOp>(globalName)) {
    return existingGlobal;
  }

  // Generate the index tile data.
  // For row index: element[i,j] = i
  // For col index: element[i,j] = j
  SmallVector<Attribute> values;
  values.reserve(tileH * tileW);
  for (int64_t i = 0; i < tileH; ++i) {
    for (int64_t j = 0; j < tileW; ++j) {
      double val = isRowIndex ? static_cast<double>(i) : static_cast<double>(j);
      if (auto floatType = dyn_cast<FloatType>(scalarType)) {
        values.push_back(builder.getFloatAttr(floatType, val));
      } else {
        llvm_unreachable("Unsupported scalar type for index tile");
      }
    }
  }

  // Create memref type: memref<[tileH, tileW]xscalarType>
  // We'll load this and convert it to a tile using tile_tilize_block
  auto memrefType =
      MemRefType::get({tileH, tileW}, scalarType, AffineMap(),
                      ttcore::MemorySpaceAttr::get(
                          builder.getContext(), ttcore::MemorySpace::DeviceL1));

  // Create dense attribute for the scalar data
  auto tensorType = RankedTensorType::get({tileH, tileW}, scalarType);
  auto denseAttr = DenseElementsAttr::get(tensorType, values);

  // Insert global at module level.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  return builder.create<memref::GlobalOp>(
      module.getLoc(), globalName,
      /*sym_visibility=*/builder.getStringAttr("private"),
      /*type=*/memrefType,
      /*initial_value=*/denseAttr,
      /*constant=*/true,
      /*alignment=*/nullptr);
}

/// Decompose BlockMaskOp into linalg.generic with per-tile masking.
///
/// This pattern handles both complete and partial tile out-of-bounds masking:
///
/// 1. Full tile in-bounds: pass through unchanged
/// 2. Full tile OOB: replace entire tile with fill value
/// 3. Partial tile (straddles boundary): per-element masking using index tiles
///
/// For multicore execution, each core processes a shard of tiles. We use
/// CoreIndexOp to get the grid position and compute global tile indices as:
///   globalTileRow = coreRowIdx * shardRows + localTileRow
///   globalTileCol = coreColIdx * shardCols + localTileCol
///
/// Partial tile masking uses pre-computed index tiles where element[i,j] = i
/// (row index tile) or j (col index tile). We compare these against the valid
/// bounds to create per-element masks, then use TileWhereOp to blend.
///
/// Since we only pad down and to the right:
/// - validRows = clamp(logicalRows - globalRowStart, 0, tileH)
/// - validCols = clamp(logicalCols - globalColStart, 0, tileW)
/// - rowMask[i,j] = (i < validRows)
/// - colMask[i,j] = (j < validCols)
/// - combinedMask = rowMask AND colMask
class DecomposeBlockMaskPattern : public OpRewritePattern<BlockMaskOp> {
public:
  using OpRewritePattern<BlockMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockMaskOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    Value logicalRows = op.getLogicalRows();
    Value logicalCols = op.getLogicalCols();
    ttcore::OOBVal fillValue = op.getFillValue();

    auto inputType = dyn_cast<ShapedType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "input must be a shaped type");
    }

    // Get the element type (should be a tile type).
    Type elementType = inputType.getElementType();
    auto tileType = dyn_cast<ttcore::TileType>(elementType);
    if (!tileType) {
      return rewriter.notifyMatchFailure(op,
                                         "element type must be a tile type");
    }

    Type scalarType = getScalarElementType(elementType);
    ArrayRef<int64_t> blockShape = inputType.getShape();
    size_t blockRank = blockShape.size();

    if (blockRank < 2) {
      return rewriter.notifyMatchFailure(
          op, "block shape must have at least 2 dimensions for tile masking");
    }

    // Get shard shape (tiles per core) from the last two dimensions.
    // This assumes the standard D2M layout where tile row/col are the trailing
    // dimensions (i.e., default collapse dims). Any batch or other dimensions
    // are collapsed into leading dimensions.
    int64_t shardRows = blockShape[blockRank - 2];
    int64_t shardCols = blockShape[blockRank - 1];

    // Build identity indexing maps for linalg.generic (input and output).
    SmallVector<AffineMap> linalgIndexingMaps(
        2, rewriter.getMultiDimIdentityMap(blockRank));
    SmallVector<mlir::utils::IteratorType> linalgIteratorTypes(
        blockRank, mlir::utils::IteratorType::parallel);

    // Tile dimensions (assume 32x32).
    constexpr int64_t tileH = 32;
    constexpr int64_t tileW = 32;

    // BlockMaskOp decomposition only supports memref semantics.
    // For tensor semantics, let the op survive through bufferization first.
    if (isa<RankedTensorType>(input.getType())) {
      return rewriter.notifyMatchFailure(
          op, "decomposition requires memref types; run after bufferization");
    }

    // Get index tile globals and create tilized versions.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    memref::GlobalOp rowIndexGlobal = getOrCreateIndexTileGlobal(
        module, rewriter, scalarType, /*isRowIndex=*/true, tileH, tileW);
    memref::GlobalOp colIndexGlobal = getOrCreateIndexTileGlobal(
        module, rewriter, scalarType, /*isRowIndex=*/false, tileH, tileW);

    // Get scalar memrefs from globals.
    Value rowIndexScalarMemref = rewriter.create<memref::GetGlobalOp>(
        loc, rowIndexGlobal.getType(), rowIndexGlobal.getName());
    Value colIndexScalarMemref = rewriter.create<memref::GetGlobalOp>(
        loc, colIndexGlobal.getType(), colIndexGlobal.getName());

    // Allocate tile memrefs for tilized index tiles (outside generic to avoid
    // dominance issues).
    auto tileMemRefType = MemRefType::get(
        {1, 1}, tileType, AffineMap(),
        ttcore::MemorySpaceAttr::get(rewriter.getContext(),
                                     ttcore::MemorySpace::DeviceL1));
    Value rowTileMemref = rewriter.create<memref::AllocOp>(loc, tileMemRefType);
    Value colTileMemref = rewriter.create<memref::AllocOp>(loc, tileMemRefType);

    // Tilize: convert scalar memrefs to tile memrefs (2 operations total!).
    rewriter.create<TileTilizeBlockOp>(loc, rowIndexScalarMemref,
                                       rowTileMemref);
    rewriter.create<TileTilizeBlockOp>(loc, colIndexScalarMemref,
                                       colTileMemref);

    // Create linalg.generic that iterates over tiles in the block.
    // For memref semantics, linalg.generic has no results (side-effecting).
    rewriter.create<mlir::linalg::GenericOp>(
        loc,
        /* result types */ TypeRange{},
        /* inputs */ input,
        /* outputs */ output, linalgIndexingMaps, linalgIteratorTypes,
        [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
            mlir::ValueRange bbArgs) {
          // bbArgs[0] is the input tile, bbArgs[1] is the output tile.
          Value inputTile = bbArgs[0];
          Type resultType = bbArgs[1].getType();

          // Get core indices from the surrounding d2m.generic grid.
          // CoreIndexOp(0) = row, CoreIndexOp(1) = col in the grid.
          Value coreRowIdx = bbBuilder.create<CoreIndexOp>(bbLoc, int64_t{0});
          Value coreColIdx = bbBuilder.create<CoreIndexOp>(bbLoc, int64_t{1});

          // Get local tile indices within the shard using linalg.index.
          // For a 1x1 shard this is always (0, 0), but for larger shards
          // we iterate over multiple tiles per core. Use the last two dims.
          Value localTileRowIdx =
              bbBuilder.create<linalg::IndexOp>(bbLoc, blockRank - 2);
          Value localTileColIdx =
              bbBuilder.create<linalg::IndexOp>(bbLoc, blockRank - 1);

          // Compute global tile indices:
          // globalTileRow = coreRowIdx * shardRows + localTileRow
          Value shardRowsConst =
              bbBuilder.create<arith::ConstantIndexOp>(bbLoc, shardRows);
          Value shardColsConst =
              bbBuilder.create<arith::ConstantIndexOp>(bbLoc, shardCols);
          Value coreRowOffset = bbBuilder.create<arith::MulIOp>(
              bbLoc, coreRowIdx, shardRowsConst);
          Value coreColOffset = bbBuilder.create<arith::MulIOp>(
              bbLoc, coreColIdx, shardColsConst);
          Value globalTileRowIdx = bbBuilder.create<arith::AddIOp>(
              bbLoc, coreRowOffset, localTileRowIdx);
          Value globalTileColIdx = bbBuilder.create<arith::AddIOp>(
              bbLoc, coreColOffset, localTileColIdx);

          // Compute global element offsets for this tile.
          Value tileHConst =
              bbBuilder.create<arith::ConstantIndexOp>(bbLoc, tileH);
          Value tileWConst =
              bbBuilder.create<arith::ConstantIndexOp>(bbLoc, tileW);
          Value globalRowStart = bbBuilder.create<arith::MulIOp>(
              bbLoc, globalTileRowIdx, tileHConst);
          Value globalColStart = bbBuilder.create<arith::MulIOp>(
              bbLoc, globalTileColIdx, tileWConst);

          // Compute the end of this tile's row/col range.
          //   Value globalRowEnd =
          //       bbBuilder.create<arith::AddIOp>(bbLoc, globalRowStart,
          //       tileHConst);
          //   Value globalColEnd =
          //       bbBuilder.create<arith::AddIOp>(bbLoc, globalColStart,
          //       tileWConst);

          // Compute the number of valid rows/cols within this tile.
          // validRows = clamp(logicalRows - globalRowStart, 0, tileH)
          // validCols = clamp(logicalCols - globalColStart, 0, tileW)
          Value zeroIdx = bbBuilder.create<arith::ConstantIndexOp>(bbLoc, 0);
          Value validRowsRaw = bbBuilder.create<arith::SubIOp>(
              bbLoc, logicalRows, globalRowStart);
          Value validColsRaw = bbBuilder.create<arith::SubIOp>(
              bbLoc, logicalCols, globalColStart);
          // Clamp to [0, tileH] using select: max(0, min(raw, tileH))
          Value validRowsRawLtZero = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::slt, validRowsRaw, zeroIdx);
          Value validRowsClampedLower = bbBuilder.create<arith::SelectOp>(
              bbLoc, validRowsRawLtZero, zeroIdx, validRowsRaw);
          Value validRowsClampedGtTileH = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::sgt, validRowsClampedLower,
              tileHConst);
          Value validRows = bbBuilder.create<arith::SelectOp>(
              bbLoc, validRowsClampedGtTileH, tileHConst,
              validRowsClampedLower);
          // Clamp to [0, tileW] using select: max(0, min(raw, tileW))
          Value validColsRawLtZero = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::slt, validColsRaw, zeroIdx);
          Value validColsClampedLower = bbBuilder.create<arith::SelectOp>(
              bbLoc, validColsRawLtZero, zeroIdx, validColsRaw);
          Value validColsClampedGtTileW = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::sgt, validColsClampedLower,
              tileWConst);
          Value validCols = bbBuilder.create<arith::SelectOp>(
              bbLoc, validColsClampedGtTileW, tileWConst,
              validColsClampedLower);

          // ================================================================
          // Generate mask and apply using TileWhereOp.
          //
          // Strategy:
          // 1. Generate index tiles using pure tile arithmetic (no memory)
          // 2. Create masks using: mask = TileLtzOp(indexTile - validBound)
          //    This gives 1.0 where index < validBound (valid data), 0.0
          //    elsewhere
          // 3. Combined mask = rowMask * colMask
          // 4. Use TileWhereOp(mask, input, fill) for efficient masking
          //
          // TileWhereOp handles all cases naturally:
          // - Full in-bounds (mask all 1s): returns input everywhere
          // - Full OOB (mask all 0s): returns fill everywhere
          // - Partial: element-wise selection
          // ================================================================

          // Create fill constant and zero tile.
          double fillVal = getOOBValAsFloat(fillValue);
          Value fillConst =
              createScalarConstant(bbBuilder, bbLoc, fillVal, scalarType);
          Value zeroTileScalar =
              createScalarConstant(bbBuilder, bbLoc, 0.0, scalarType);
          Value zeroTile = bbBuilder.create<TileMulOp>(
              bbLoc, resultType, inputTile, zeroTileScalar);

          // Load the pre-computed index tiles from the tile memrefs.
          // These are allocated outside the generic, so they should be
          // accessible.
          Value tileIdx0 = bbBuilder.create<arith::ConstantIndexOp>(bbLoc, 0);
          Value rowIndexTile = bbBuilder.create<memref::LoadOp>(
              bbLoc, rowTileMemref, ValueRange{tileIdx0, tileIdx0});
          Value colIndexTile = bbBuilder.create<memref::LoadOp>(
              bbLoc, colTileMemref, ValueRange{tileIdx0, tileIdx0});

          // Convert validRows/validCols from index to scalar type.
          Value validRowsI64 = bbBuilder.create<arith::IndexCastOp>(
              bbLoc, bbBuilder.getI64Type(), validRows);
          Value validColsI64 = bbBuilder.create<arith::IndexCastOp>(
              bbLoc, bbBuilder.getI64Type(), validCols);
          Value validRowsScalar = bbBuilder.create<arith::SIToFPOp>(
              bbLoc, scalarType, validRowsI64);
          Value validColsScalar = bbBuilder.create<arith::SIToFPOp>(
              bbLoc, scalarType, validColsI64);

          // Create row mask: mask[i,j] = (i < validRows ? 1.0 : 0.0)
          // Computed as: TileLtzOp(rowIndexTile - validRows)
          Value rowDiff = bbBuilder.create<TileSubOp>(
              bbLoc, resultType, rowIndexTile, validRowsScalar);
          Value rowMask =
              bbBuilder.create<TileLtzOp>(bbLoc, resultType, rowDiff);

          // Create col mask: mask[i,j] = (j < validCols ? 1.0 : 0.0)
          Value colDiff = bbBuilder.create<TileSubOp>(
              bbLoc, resultType, colIndexTile, validColsScalar);
          Value colMask =
              bbBuilder.create<TileLtzOp>(bbLoc, resultType, colDiff);

          // Combined mask = rowMask AND colMask (element-wise multiply).
          // Mask has 1s for valid data, 0s for padding.
          Value combinedMask =
              bbBuilder.create<TileMulOp>(bbLoc, resultType, rowMask, colMask);

          // Create fill tile: a tile filled with the OOB fill value.
          // (zeroTile was already created above)
          Value fillTile = bbBuilder.create<TileAddOp>(bbLoc, resultType,
                                                       zeroTile, fillConst);

          // Use TileWhereOp for efficient element-wise selection.
          // result[i,j] = mask[i,j] ? input[i,j] : fill[i,j]
          // This handles all cases (full in-bounds, full OOB, partial)
          // naturally.
          Value result = bbBuilder.create<TileWhereOp>(
              bbLoc, resultType, combinedMask, inputTile, fillTile);

          bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, result);
        });

    rewriter.replaceOp(op, op.getOutput());
    return success();
  }
};

class D2MDecomposeMasking
    : public impl::D2MDecomposeMaskingBase<D2MDecomposeMasking> {
public:
  using D2MDecomposeMaskingBase::D2MDecomposeMaskingBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeBlockMaskPattern>(ctx);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
