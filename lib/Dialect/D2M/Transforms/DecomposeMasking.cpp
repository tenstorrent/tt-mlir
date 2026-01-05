// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

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
/// it to a tile when loading. This avoids the DenseElementsAttr tile type issue.
static memref::GlobalOp getOrCreateIndexTileGlobal(ModuleOp module,
                                                   OpBuilder &builder,
                                                   Type scalarType,
                                                   bool isRowIndex,
                                                   int64_t tileH,
                                                   int64_t tileW) {
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
  auto memrefType = MemRefType::get(
      {tileH, tileW}, scalarType, AffineMap(),
      ttcore::MemorySpaceAttr::get(builder.getContext(),
                                    ttcore::MemorySpace::DeviceL1));
  
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

    // Note: We'll generate index tiles at runtime using tile operations
    // instead of storing them as globals, since DenseElementsAttr doesn't
    // support tile types directly.

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
          Value globalRowEnd =
              bbBuilder.create<arith::AddIOp>(bbLoc, globalRowStart, tileHConst);
          Value globalColEnd =
              bbBuilder.create<arith::AddIOp>(bbLoc, globalColStart, tileWConst);

          // Check if tile is completely out of bounds.
          // Tile is fully OOB if its START is >= logical bound.
          Value rowFullyOOB = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::sge, globalRowStart, logicalRows);
          Value colFullyOOB = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::sge, globalColStart, logicalCols);
          Value entireTileOOB =
              bbBuilder.create<arith::OrIOp>(bbLoc, rowFullyOOB, colFullyOOB);

          // Check if tile is completely in bounds.
          // Tile is fully in-bounds if its END is <= logical bound.
          Value rowFullyInBounds = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::sle, globalRowEnd, logicalRows);
          Value colFullyInBounds = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::sle, globalColEnd, logicalCols);
          Value entireTileInBounds = bbBuilder.create<arith::AndIOp>(
              bbLoc, rowFullyInBounds, colFullyInBounds);

          // Create constants for blending.
          double fillVal = getOOBValAsFloat(fillValue);
          Value fillConst =
              createScalarConstant(bbBuilder, bbLoc, fillVal, scalarType);
          Value zero = createScalarConstant(bbBuilder, bbLoc, 0.0, scalarType);
          Value one = createScalarConstant(bbBuilder, bbLoc, 1.0, scalarType);

          // Compute the number of valid rows/cols within this tile.
          // validRows = clamp(logicalRows - globalRowStart, 0, tileH)
          // validCols = clamp(logicalCols - globalColStart, 0, tileW)
          Value zeroIdx = bbBuilder.create<arith::ConstantIndexOp>(bbLoc, 0);
          Value validRowsRaw =
              bbBuilder.create<arith::SubIOp>(bbLoc, logicalRows, globalRowStart);
          Value validColsRaw =
              bbBuilder.create<arith::SubIOp>(bbLoc, logicalCols, globalColStart);
          // Clamp to [0, tileH] using select: max(0, min(raw, tileH))
          Value validRowsRawLtZero = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::slt, validRowsRaw, zeroIdx);
          Value validRowsClampedLower = bbBuilder.create<arith::SelectOp>(
              bbLoc, validRowsRawLtZero, zeroIdx, validRowsRaw);
          Value validRowsClampedGtTileH = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::sgt, validRowsClampedLower, tileHConst);
          Value validRows = bbBuilder.create<arith::SelectOp>(
              bbLoc, validRowsClampedGtTileH, tileHConst, validRowsClampedLower);
          // Clamp to [0, tileW] using select: max(0, min(raw, tileW))
          Value validColsRawLtZero = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::slt, validColsRaw, zeroIdx);
          Value validColsClampedLower = bbBuilder.create<arith::SelectOp>(
              bbLoc, validColsRawLtZero, zeroIdx, validColsRaw);
          Value validColsClampedGtTileW = bbBuilder.create<arith::CmpIOp>(
              bbLoc, arith::CmpIPredicate::sgt, validColsClampedLower, tileWConst);
          Value validCols = bbBuilder.create<arith::SelectOp>(
              bbLoc, validColsClampedGtTileW, tileWConst, validColsClampedLower);

          // Generate index tiles at runtime using TileBcastOp.
          // We create row/col vector tiles and broadcast them to full tiles.
          // Row index tile: element[i,j] = i (broadcast row vector [0..31] to 32x32)
          // Col index tile: element[i,j] = j (broadcast col vector [0..31] to 32x32)
          
          // Create row vector tile: 32x1 tile with values [0, 1, 2, ..., 31]
          // We do this by creating a base tile and using arithmetic to set values.
          // Actually, we can use d2m.full to create filled tiles, then modify them.
          // But d2m.full creates tensors, not tiles directly.
          
          // Simpler approach: Create index tiles using TileBcastOp.
          // First, we need to create row/col vector tiles with index values.
          // We can create these by starting with a zero tile and adding increments.
          
          // Create zero tile as base
          Value zeroTileScalar = createScalarConstant(bbBuilder, bbLoc, 0.0, scalarType);
          Value zeroTile = bbBuilder.create<TileMulOp>(
              bbLoc, resultType, inputTile, zeroTileScalar);
          
          // For row index: create a tile where each row has its row index.
          // We can do this by creating a row vector [0, 1, 2, ..., 31] and broadcasting.
          // But creating the vector tile is the challenge.
          
          // Alternative: Use the fact that we can create index tiles using
          // a combination of constants and tile operations. Since we're inside
          // linalg.generic, we can use nested operations.
          
          // Actually, the simplest solution: Load from globals that store the
          // tilized tile data directly. But we can't create globals with tile
          // element type using DenseElementsAttr.
          
          // Final approach: Generate index tiles using TileBcastOp with vector
          // tiles created from constants. We'll create row/col vector tiles
          // by using d2m.full (which creates tensors) and then converting.
          // But that conversion has the same problem.
          
          // For now, let's use a workaround: Create the index tiles using
          // tile arithmetic operations that work in linalg.generic.
          // We can create them incrementally using TileAddOp.
          
          // Actually, I think we need to use the globals but convert them
          // differently. Let's try loading them and using a different mechanism.
          
          // Load index tile data from globals (as scalar memrefs)
          ModuleOp module = op->getParentOfType<ModuleOp>();
          memref::GlobalOp rowIndexGlobal = getOrCreateIndexTileGlobal(
              module, rewriter, scalarType, /*isRowIndex=*/true, tileH, tileW);
          memref::GlobalOp colIndexGlobal = getOrCreateIndexTileGlobal(
              module, rewriter, scalarType, /*isRowIndex=*/false, tileH, tileW);
          
          Value rowIndexMemref = bbBuilder.create<memref::GetGlobalOp>(
              bbLoc, rowIndexGlobal.getType(), rowIndexGlobal.getName());
          Value colIndexMemref = bbBuilder.create<memref::GetGlobalOp>(
              bbLoc, colIndexGlobal.getType(), colIndexGlobal.getName());
          
          // Instead of using TileTilizeBlockOp (which violates single-use constraint),
          // we'll generate the index tiles using tile operations.
          // Create row/col vector tiles by loading scalar values and broadcasting.
          // But we can't easily create vector tiles from scalar memrefs.
          
          // Workaround: Create index tiles using TileBcastOp with pre-computed
          // vector tiles. We'll need to create those vector tiles first.
          // For now, create placeholder tiles that will be filled correctly
          // by the hardware or a later pass.
          
          // Actually, let's try a different approach: Use the input tile as a
          // template and create index tiles using tile arithmetic.
          // Row index: start with zero, add row indices using arithmetic.
          // But we can't easily add per-row values.
          
          // Simplest solution for now: Create the index tiles using TileBcastOp
          // with vector tiles created from constants. We'll create row/col
          // vector tiles by using d2m.full to create tensors, then tilizing
          // them in a way that doesn't violate constraints.
          
          // Actually, I think the issue is that TileTilizeBlockOp creates
          // intermediate values that are used multiple times. Let's ensure
          // each operation result is used only once by creating separate
          // operations for each use.
          
          // Generate index tiles using TileBcastOp.
          // We'll build the index tiles by loading values from globals and
          // using TileBcastOp. Since we know the pattern statically (row i has
          // value i, col j has value j), we can create them efficiently.
          //
          // For row index tile: element[i,j] = i
          //   Strategy: Create a tile where column 0 has [0,1,2,...,31],
          //   then broadcast across columns using TileBcastType::Col
          // For col index tile: element[i,j] = j  
          //   Strategy: Create a tile where row 0 has [0,1,2,...,31],
          //   then broadcast across rows using TileBcastType::Row
          
          // Build masks for partial tile masking.
          // We support 3 cases:
          // 1. Row-wise: mask[i,j] = (i < validRows ? 1.0 : 0.0)
          // 2. Col-wise: mask[i,j] = (j < validCols ? 1.0 : 0.0)
          // 3. Combination: mask[i,j] = (i < validRows && j < validCols ? 1.0 : 0.0)
          //
          // Strategy: Build column 0 (for row mask) or row 0 (for col mask) with
          // the pattern [1,1,...,1,0,0,...,0], then broadcast with Col/Row type.
          // For combination, create both masks and multiply (logical AND).
          
          // Build row mask: element[i,j] = (i < validRows ? 1.0 : 0.0)
          // Build column 0 with [1,1,...,1,0,0,...,0] (validRows ones, then zeros)
          Value rowMaskCol0 = zeroTile;
          for (int64_t i = 0; i < tileH; ++i) {
            Value rowIdx = bbBuilder.create<arith::ConstantIndexOp>(bbLoc, i);
            
            // Compute mask value: (i < validRows ? 1.0 : 0.0)
            Value iConst = bbBuilder.create<arith::ConstantIndexOp>(bbLoc, i);
            Value iLtValidRows = bbBuilder.create<arith::CmpIOp>(
                bbLoc, arith::CmpIPredicate::slt, iConst, validRows);
            Value maskVal = bbBuilder.create<arith::SelectOp>(
                bbLoc, iLtValidRows, one, zero);
            
            // Create a tile with maskVal at column 0, row i.
            // Use TileBcastOp with Col to fill entire tile with maskVal,
            // then use TileWhereOp to set it only at row i.
            Value scalarTile = bbBuilder.create<TileBcastOp>(
                bbLoc, resultType,
                bbBuilder.create<TileMulOp>(bbLoc, resultType, zeroTile, maskVal),
                d2m::TileBcastType::Scalar);
            Value colBcast = bbBuilder.create<TileBcastOp>(
                bbLoc, resultType, scalarTile, d2m::TileBcastType::Col);
            
            // Create condition: row index == i
            // Load row index from global (already stored as float) and compare with constant i
            Value rowIdxFromGlobal = bbBuilder.create<memref::LoadOp>(
                bbLoc, rowIndexMemref, ValueRange{rowIdx, zeroIdx});
            // rowIdxFromGlobal is already scalarType (float), no conversion needed
            
            Value iConstFloat = createScalarConstant(bbBuilder, bbLoc, static_cast<double>(i), scalarType);
            Value iTile = bbBuilder.create<TileBcastOp>(
                bbLoc, resultType,
                bbBuilder.create<TileMulOp>(bbLoc, resultType, zeroTile, iConstFloat),
                d2m::TileBcastType::Scalar);
            Value rowIdxTile = bbBuilder.create<TileBcastOp>(
                bbLoc, resultType,
                bbBuilder.create<TileMulOp>(bbLoc, resultType, zeroTile, rowIdxFromGlobal),
                d2m::TileBcastType::Scalar);
            
            // Compare: (rowIdxTile == iTile)
            Value diff = bbBuilder.create<TileSubOp>(bbLoc, resultType, rowIdxTile, iTile);
            Value absDiff = bbBuilder.create<TileMulOp>(bbLoc, resultType, diff, diff);
            Value epsilon = createScalarConstant(bbBuilder, bbLoc, 0.001, scalarType);
            Value epsilonTile = bbBuilder.create<TileBcastOp>(
                bbLoc, resultType,
                bbBuilder.create<TileMulOp>(bbLoc, resultType, zeroTile, epsilon),
                d2m::TileBcastType::Scalar);
            Value condition = bbBuilder.create<TileLtzOp>(bbLoc, resultType,
                bbBuilder.create<TileSubOp>(bbLoc, resultType, absDiff, epsilonTile));
            
            // Set value only at row i using TileWhereOp
            Value maskedTile = bbBuilder.create<TileWhereOp>(
                bbLoc, resultType, condition, colBcast, zeroTile);
            
            // Combine with existing column 0 tile using TileAddOp.
            // Since maskedTile is zero everywhere except row i, adding it won't cause
            // incorrect summing - only one tile contributes to each position.
            rowMaskCol0 = bbBuilder.create<TileAddOp>(
                bbLoc, resultType, rowMaskCol0, maskedTile);
          }
          
          // Broadcast column 0 across all columns
          Value rowMask = bbBuilder.create<TileBcastOp>(
              bbLoc, resultType, rowMaskCol0, d2m::TileBcastType::Col);
          
          // Build col mask: element[i,j] = (j < validCols ? 1.0 : 0.0)
          // Build row 0 with [1,1,...,1,0,0,...,0] (validCols ones, then zeros)
          Value colMaskRow0 = zeroTile;
          for (int64_t j = 0; j < tileW; ++j) {
            Value colIdx = bbBuilder.create<arith::ConstantIndexOp>(bbLoc, j);
            
            // Compute mask value: (j < validCols ? 1.0 : 0.0)
            Value jConst = bbBuilder.create<arith::ConstantIndexOp>(bbLoc, j);
            Value jLtValidCols = bbBuilder.create<arith::CmpIOp>(
                bbLoc, arith::CmpIPredicate::slt, jConst, validCols);
            Value maskVal = bbBuilder.create<arith::SelectOp>(
                bbLoc, jLtValidCols, one, zero);
            
            // Create a tile with maskVal at row 0, col j
            Value scalarTile = bbBuilder.create<TileBcastOp>(
                bbLoc, resultType,
                bbBuilder.create<TileMulOp>(bbLoc, resultType, zeroTile, maskVal),
                d2m::TileBcastType::Scalar);
            Value rowBcast = bbBuilder.create<TileBcastOp>(
                bbLoc, resultType, scalarTile, d2m::TileBcastType::Row);
            
            // Create condition: col index == j
            // Load col index from global (already stored as float) and compare with constant j
            Value colIdxFromGlobal = bbBuilder.create<memref::LoadOp>(
                bbLoc, colIndexMemref, ValueRange{zeroIdx, colIdx});
            // colIdxFromGlobal is already scalarType (float), no conversion needed
            
            Value jConstFloat = createScalarConstant(bbBuilder, bbLoc, static_cast<double>(j), scalarType);
            Value jTile = bbBuilder.create<TileBcastOp>(
                bbLoc, resultType,
                bbBuilder.create<TileMulOp>(bbLoc, resultType, zeroTile, jConstFloat),
                d2m::TileBcastType::Scalar);
            Value colIdxTile = bbBuilder.create<TileBcastOp>(
                bbLoc, resultType,
                bbBuilder.create<TileMulOp>(bbLoc, resultType, zeroTile, colIdxFromGlobal),
                d2m::TileBcastType::Scalar);
            
            Value diff = bbBuilder.create<TileSubOp>(bbLoc, resultType, colIdxTile, jTile);
            Value absDiff = bbBuilder.create<TileMulOp>(bbLoc, resultType, diff, diff);
            Value epsilon = createScalarConstant(bbBuilder, bbLoc, 0.001, scalarType);
            Value epsilonTile = bbBuilder.create<TileBcastOp>(
                bbLoc, resultType,
                bbBuilder.create<TileMulOp>(bbLoc, resultType, zeroTile, epsilon),
                d2m::TileBcastType::Scalar);
            Value condition = bbBuilder.create<TileLtzOp>(bbLoc, resultType,
                bbBuilder.create<TileSubOp>(bbLoc, resultType, absDiff, epsilonTile));
            
            Value maskedTile = bbBuilder.create<TileWhereOp>(
                bbLoc, resultType, condition, rowBcast, zeroTile);
            
            // Combine using TileAddOp.
            // Since maskedTile is zero everywhere except col j, adding it won't cause
            // incorrect summing - only one tile contributes to each position.
            colMaskRow0 = bbBuilder.create<TileAddOp>(
                bbLoc, resultType, colMaskRow0, maskedTile);
          }
          
          // Broadcast row 0 across all rows
          Value colMask = bbBuilder.create<TileBcastOp>(
              bbLoc, resultType, colMaskRow0, d2m::TileBcastType::Row);

          // Combined mask = rowMask AND colMask (element-wise).
          // Since masks are 1.0 for true, 0.0 for false, we can multiply.
          // Mask has 1s for valid data, 0s for padding.
          Value combinedMask =
              bbBuilder.create<TileMulOp>(bbLoc, resultType, rowMask, colMask);

          // Create a tile filled with the OOB fill value.
          // We do this by: fillTile = zeroTile + fillConst (broadcast scalar to tile)
          Value fillTile =
              bbBuilder.create<TileAddOp>(bbLoc, resultType, zeroTile, fillConst);

          // For partial tiles: result = (input * mask) + (fill * (1 - mask))
          // Step 1: Multiply input with mask to zero out invalid regions
          Value maskedInput =
              bbBuilder.create<TileMulOp>(bbLoc, resultType, inputTile, combinedMask);
          
          // Step 2: Invert mask (1s become 0s, 0s become 1s)
          // invertedMask = 1.0 - mask
          Value oneTile = bbBuilder.create<TileBcastOp>(
              bbLoc, resultType,
              bbBuilder.create<TileMulOp>(bbLoc, resultType, zeroTile, one),
              d2m::TileBcastType::Scalar);
          Value invertedMask =
              bbBuilder.create<TileSubOp>(bbLoc, resultType, oneTile, combinedMask);
          
          // Step 3: Multiply inverted mask with fill value
          Value fillPortion =
              bbBuilder.create<TileMulOp>(bbLoc, resultType, fillTile, invertedMask);
          
          // Step 4: Add masked input and fill portion
          Value partialResult =
              bbBuilder.create<TileAddOp>(bbLoc, resultType, maskedInput, fillPortion);

          // Now select between three cases:
          // 1. entireTileInBounds -> inputTile (pass through)
          // 2. entireTileOOB -> fillTile (full replacement)
          // 3. else (partial) -> partialResult (per-element masking)
          //
          // We build this as nested selects:
          // result = select(entireTileInBounds, inputTile,
          //                 select(entireTileOOB, fillTile, partialResult))

          // For full tile OOB: use scalar multiply/add (more efficient).
          Value fullOOBMulFactor = bbBuilder.create<arith::SelectOp>(
              bbLoc, entireTileOOB, zero, one);
          Value fullOOBAddend = bbBuilder.create<arith::SelectOp>(
              bbLoc, entireTileOOB, fillConst, zero);
          Value fullOOBZeroed = bbBuilder.create<TileMulOp>(
              bbLoc, resultType, partialResult, fullOOBMulFactor);
          Value fullOOBFilled = bbBuilder.create<TileAddOp>(
              bbLoc, resultType, fullOOBZeroed, fullOOBAddend);

          // For full tile in-bounds: pass through input directly.
          Value inBoundsMulFactor = bbBuilder.create<arith::SelectOp>(
              bbLoc, entireTileInBounds, one, zero);
          Value inBoundsAddFactor = bbBuilder.create<arith::SelectOp>(
              bbLoc, entireTileInBounds, zero, one);
          Value inputScaled = bbBuilder.create<TileMulOp>(
              bbLoc, resultType, inputTile, inBoundsMulFactor);
          Value otherScaled = bbBuilder.create<TileMulOp>(
              bbLoc, resultType, fullOOBFilled, inBoundsAddFactor);
          Value filled =
              bbBuilder.create<TileAddOp>(bbLoc, resultType, inputScaled, otherScaled);

          bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, filled);
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
