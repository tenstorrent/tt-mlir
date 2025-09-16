// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Utils/CoreRangeSet.h"
#include "ttmlir/Dialect/TTNN/Utils/Conv2dConfigParams.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include <cassert>
#include <cstdint>
#include <numeric>
#include <optional>

using namespace mlir::tt::ttnn;

// Check if the tensor is tiled
bool TTNNLayoutAttr::isTiled() const {
  return ::mlir::isa<::mlir::tt::ttcore::TileType>(getElementType());
}

// Get layout of the tensor (RowMajor/Tile)
Layout TTNNLayoutAttr::getLayout() const {
  return isTiled() ? Layout::Tile : Layout::RowMajor;
}

// Get optional memory layout
std::optional<TensorMemoryLayout> TTNNLayoutAttr::getMemLayoutOpt() const {
  return getMemLayout() ? std::make_optional(getMemLayout().getValue())
                        : std::nullopt;
}

// Check if the tensor memory buffer type is L1
bool TTNNLayoutAttr::hasL1BufferType() const {
  return isL1BufferType(getBufferType());
}

// Check if the tensor memory buffer type is DRAM
bool TTNNLayoutAttr::hasDRAMBufferType() const {
  return isDRAMBufferType(getBufferType());
}

std::pair<std::int64_t, std::int64_t>
TTNNLayoutAttr::getDefaultCollapseIntervals() const {
  return {0, -1};
}

// Check if the tensor memory layout is sharded
bool TTNNLayoutAttr::hasShardedTensorMemoryLayout() const {
  return isDeviceBufferType() &&
         (getMemLayout().getValue() == TensorMemoryLayout::HeightSharded ||
          getMemLayout().getValue() == TensorMemoryLayout::WidthSharded ||
          getMemLayout().getValue() == TensorMemoryLayout::BlockSharded);
}

// Check if the tensor memory layout is sharded in L1 memory
bool TTNNLayoutAttr::hasShardedL1TensorMemoryLayout() const {
  return hasL1BufferType() &&
         (getMemLayout().getValue() == TensorMemoryLayout::HeightSharded ||
          getMemLayout().getValue() == TensorMemoryLayout::WidthSharded ||
          getMemLayout().getValue() == TensorMemoryLayout::BlockSharded);
}

// Check if the tensor memory layout is interleaved and in L1 memory
bool TTNNLayoutAttr::hasInterleavedL1TensorMemoryLayout() const {
  return hasL1BufferType() &&
         (getMemLayout().getValue() == TensorMemoryLayout::Interleaved);
}

// Check if the tensor memory layout is interleaved and in DRAM memory
bool TTNNLayoutAttr::hasInterleavedDRAMTensorMemoryLayout() const {
  return hasDRAMBufferType() &&
         (getMemLayout().getValue() == TensorMemoryLayout::Interleaved);
}

// Checks:
// 1. If buffer type is L1, then any grid shape is allowed.
// 2. Otherwise, unit grid is expected.
llvm::LogicalResult
verifyGridShape(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                mlir::tt::ttcore::GridAttr gridAttr, BufferType bufferType) {
  if (isL1BufferType(bufferType)) {
    return llvm::success();
  }

  llvm::SmallVector<int64_t> expectedGridShape({1, 1});
  if (llvm::equal(gridAttr.getShape(), expectedGridShape)) {
    return llvm::success();
  }
  return emitError() << "expected (" << expectedGridShape
                     << ") grid shape for non-L1 buffer type, got ("
                     << gridAttr.getShape() << ") for "
                     << stringifyBufferType(bufferType) << " buffer type";
}

// Checks:
// 1. If memory layout is present then:
//   - System memory buffer type is not allowed
//   - DRAM buffer type must have Interleaved memory layout
// 2. If memory layout is not present then:
//   - Buffer type must be SystemMemory
llvm::LogicalResult verifyBufferAndMemoryLayout(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    BufferType bufferType, TensorMemoryLayoutAttr memLayoutAttr) {
  if (memLayoutAttr) {
    if (bufferType == BufferType::SystemMemory) {
      return emitError()
             << "Memory layout is not allowed for SystemMemory buffer type.";
    }

    if (bufferType == BufferType::DRAM &&
        memLayoutAttr.getValue() != TensorMemoryLayout::Interleaved) {
      return emitError()
             << "DRAM buffer type must have Interleaved memory layout.";
    }
  } else if (bufferType != BufferType::SystemMemory) {
    return emitError()
           << "Memory layout is required for non-SystemMemory buffer type.";
  }

  return ::llvm::success();
}

// Checks:
// 1. If shard spec is present then:
//   - Buffer type must be L1
//   - Tensor memory layout must be sharded: HeightSharded, WidthSharded,
//   BlockSharded
llvm::LogicalResult
verifySharding(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
               BufferType bufferType, TensorMemoryLayoutAttr memLayoutAttr,
               std::optional<ShardSpecAttr> shardSpec) {
  if (shardSpec && *shardSpec) {
    if (bufferType != BufferType::L1) {
      return emitError() << "Sharding is only valid for L1 buffer type";
    }

    if (!memLayoutAttr) {
      return emitError() << "Tensor memory layout is required for sharding";
    }

    if (memLayoutAttr.getValue() != TensorMemoryLayout::BlockSharded &&
        memLayoutAttr.getValue() != TensorMemoryLayout::HeightSharded &&
        memLayoutAttr.getValue() != TensorMemoryLayout::WidthSharded) {
      return emitError() << "Sharding is only valid for block sharded, height "
                            "sharded, or width sharded tensor memory layout";
    }
  }
  return ::llvm::success();
}

// Calculate the logical shape of the shard.
//
// Shard is defined as a piece of the tensor that is mapped to a single grid
// core. This function returns the shard shape for tensors with BLOCK SHARDED
// tensor memory layout.
//
// All examples assume that the tensor is mapped to a 8x8 grid.
// Example: tensor<32x32xbf16> -> {4, 4}
// Example: tensor<65x65xbf16> -> {9, 9}
//
// return The logical shard shape in case of block sharded tensor memory layout.
llvm::SmallVector<int64_t>
TTNNLayoutAttr::calculateLogicalShardShapeForSharding(
    ArrayRef<int64_t> tensorShape, mlir::AffineMap linear,
    mlir::tt::ttcore::GridAttr grid) {
  assert(linear.getNumResults() == grid.getShape().size());
  mlir::SmallVector<std::int64_t> physicalShape =
      ttmlir::utils::evalShape(linear, tensorShape);
  mlir::SmallVector<std::int64_t> shardShape(linear.getNumResults());
  for (size_t i = 0; i < linear.getNumResults(); ++i) {
    shardShape[i] =
        (physicalShape[i] + grid.getShape()[i] - 1) / grid.getShape()[i];
  }
  return shardShape;
}

// Calculate the logical shape of the shard.
//
// Shard is defined as a piece of the tensor that is mapped to a single grid
// core. This function returns the shard shape for tensors with INTERLEAVED
// tensor memory layout. Returned shard shape is in scalar elements.
//
// All examples assume that the tensor is mapped to a 8x8 grid.
// Examples for TileType:
// Example: tensor<1x1024xbf16> ( -> 32 tiles ) -> {32, 32}
// Example: tensor<512x512xbf16> ( -> 256 tiles ) -> {32, 128}
// Example: tensor<32x2049xbf16> ( -> 65 tiles ) -> {32, 64}
//
// Examples for RowMajor:
// Example: tensor<1x1024xbf16> -> {1, 16}
// Example: tensor<512x512xbf16> -> {1, 4096}
// Example: tensor<32x2049xbf16> -> {1, 1025}
//
// return The logical shard shape in case of interleaved tensor memory layout.
llvm::SmallVector<int64_t>
TTNNLayoutAttr::calculateLogicalShardShapeForL1Interleaved(
    ArrayRef<int64_t> tensorShape, mlir::Type elementType,
    mlir::AffineMap linear, mlir::tt::ttcore::GridAttr grid) {
  assert(linear.getNumResults() == grid.getShape().size());

  mlir::SmallVector<std::int64_t> physicalShape =
      ttmlir::utils::evalShape(linear, tensorShape);

  uint64_t numOfGridUnits =
      std::accumulate(grid.getShape().begin(), grid.getShape().end(), 1,
                      std::multiplies<std::int64_t>());

  // Create shard shape with all dims set to 1 except the last one which is
  // set to shardVolume.
  mlir::SmallVector<std::int64_t> shardShape;
  shardShape.resize(grid.getShape().size() - 1, 1);

  if (!mlir::isa<mlir::tt::ttcore::TileType>(elementType)) {
    // If RowMajor, single shard should be TensorVolume / NumCores rounded up.
    // So in case of tensor<5120x1024xbf16> on 8x8 grid we have
    // 5120*1024/64 = 81920 -> shard shape is <1x81920xbf16>
    int64_t tensorVolume =
        std::accumulate(physicalShape.begin(), physicalShape.end(), 1,
                        std::multiplies<int64_t>());
    int64_t shardVolume = (tensorVolume + numOfGridUnits - 1) / numOfGridUnits;
    shardShape.push_back(shardVolume);
    return shardShape;
  }

  // TileType case
  mlir::SmallVector<std::int64_t> physicalTiledShape =
      mlir::cast<mlir::tt::ttcore::TileType>(elementType)
          .getTiledShape(physicalShape);
  uint64_t numOfTiles =
      std::accumulate(physicalTiledShape.begin(), physicalTiledShape.end(), 1,
                      std::multiplies<std::int64_t>());
  shardShape.push_back((numOfTiles + numOfGridUnits - 1) / numOfGridUnits);
  return mlir::cast<mlir::tt::ttcore::TileType>(elementType)
      .getScalarShape(shardShape);
}

// Get the buffer type (DRAM/L1/SystemMemory)
BufferType TTNNLayoutAttr::getBufferType() const {
  return mlir::cast<BufferTypeAttr>(getMemref().getMemorySpace()).getValue();
}

// Get element type i.e FloatType/IntegerType/TileType
mlir::Type TTNNLayoutAttr::getElementType() const {
  return getMemref().getElementType();
}

// If the element type is TileType, return the nested element type i.e
// FloatType/IntegerType
mlir::Type TTNNLayoutAttr::getScalarElementType() const {
  Type elementType = getElementType();
  if (mlir::isa<mlir::tt::ttcore::TileType>(elementType)) {
    return mlir::cast<mlir::tt::ttcore::TileType>(elementType).getElementType();
  }
  return elementType;
}

// Get scalar element type.
// Example: memref<2x2xf32> -> f32
// Example: memref<2x2x!ttcore.tile<32x32xf32>> -> f32
//
// return The scalar element type.
mlir::tt::ttcore::DataType TTNNLayoutAttr::getDataType() const {
  Type elementType = getElementType();
  if (isTiled()) {
    mlir::tt::ttcore::TileType tileType =
        mlir::cast<mlir::tt::ttcore::TileType>(elementType);
    return tileType.getDataType();
  }

  return mlir::tt::ttcore::elementTypeToDataType(elementType);
}

// Get the size of the element in bytes
//
// This function returns the size of a single tensor element in bytes.
// Distinction is made between scalar types and TileType.
//
// return The size of the element in bytes.
uint64_t TTNNLayoutAttr::getElementSizeBytes() const {
  mlir::Type elementType = getElementType();
  if (isTiled()) {
    mlir::tt::ttcore::TileType tileType =
        mlir::cast<mlir::tt::ttcore::TileType>(elementType);
    return tileType.getSizeBytes();
  }
  return elementType.getIntOrFloatBitWidth() / 8;
}

// Get shard shape
//
// Return the shape of the shard.
// Example: memref<2x2x!ttcore.tile<32x32xf32>> -> { 2, 2 }
// Example: memref<128x128xf32> -> { 128, 128 }
// Example: memref<2x3x!ttcore.tile<32x32xf32>> -> { 2, 3 }
//
// return The shape of the shard.
llvm::SmallVector<int64_t> TTNNLayoutAttr::getShardShape() const {
  return SmallVector<int64_t>(getMemref().getShape());
}

// Get scalar shard shape
//
// If the element type is TileType, this function returns the scalar shape of
// the shard.
// Example: memref<2x2x!ttcore.tile<32x32xf32>> -> { 64, 64 }
// Example: memref<128x128xf32> -> { 128, 128 }
// Example: memref<2x3!ttcore.tile<32x32xf32>> -> { 64, 96 }
//
// return The scalar shape of the shard.
llvm::SmallVector<int64_t> TTNNLayoutAttr::getScalarShardShape() const {
  SmallVector<int64_t> shardShape(getMemref().getShape());
  if (isTiled()) {
    return mlir::cast<mlir::tt::ttcore::TileType>(getElementType())
        .getScalarShape(shardShape);
  }

  return shardShape;
}

// Get size of tensor in tiles
//
// This function returns the size of the tensor in tiles.
// Size is calculated by rounding up the last two dims of the tensor to tile
// size and then plugging the tensor shape into the linear map. This result is
// then divided by the tile shape. Example: tensor shape (6, 15, 10), linear map
// (d0, d1, d2) -> (d0 * 32 + d1, d2) and tile shape (32, 32).
//
// The result is calculated: (6, 15, 10) -> (6, 32, 32) -> (6 * 32, 32) -> (192,
// 32) which is then divided by tile shape (32, 32) -> (6, 1)
//
// param tensorShape The shape of the tensor
// return The size of the tensor in tiles.
llvm::SmallVector<int64_t>
TTNNLayoutAttr::getTiledShape(llvm::ArrayRef<int64_t> tensorShape) const {
  assert(isTiled() && "Expected a tiled layout");

  // Affine map in form of (d0, d1, d2) -> (d0 * 15 + d1, d2)
  mlir::AffineMap linear = getLinear();
  uint32_t rank = linear.getNumResults();
  assert(rank >= 2 && "Expected at least two results in linear map");
  mlir::AffineExpr y = linear.getResult(rank - 2);
  mlir::AffineExpr x = linear.getResult(rank - 1);

  mlir::tt::ttcore::TileType tileType =
      mlir::cast<mlir::tt::ttcore::TileType>(getElementType());
  int64_t tileH = tileType.getHeight();
  int64_t tileW = tileType.getWidth();

  // Construct new affine map with where x and y are divided by tile width and
  // height respectively. Example:
  // (d0, d1, d2) -> (d0 * 15 + d1) / 32, d2 / 32
  // Note: even though we floorDiv the result, eval will return ceiled...
  mlir::AffineMap tiled =
      linear.replace(mlir::DenseMap<mlir::AffineExpr, mlir::AffineExpr>{
          {y, y.floorDiv(tileH)}, {x, x.floorDiv(tileW)}});

  // Get tiled shape by evaluating the affine map with tensor shape.
  return ttmlir::utils::evalShape(tiled,
                                  utils::getTilePaddedShape(tensorShape));
}

// Get the size of shard in bytes
//
// This function returns the size of the shard in bytes.
// Size is calculated by multiplying shard shape with element size.
// Element size for TileType is tile width * tile height * sizeof(element).
// For scalar types, element size is sizeof(element).
//
// return The size of the shard in bytes.
uint64_t TTNNLayoutAttr::getShardSizeInBytes() const {
  SmallVector<int64_t> shape = getShardShape();
  uint64_t size = getElementSizeBytes();
  return std::accumulate(shape.begin(), shape.end(), size,
                         std::multiplies<uint64_t>());
}

// Construct a new TTNNLayoutAttr
//
// This function creates a new TTNNLayoutAttr with the given parameters.
// The element type, buffer type and memory layout are preserved.
//
// param context The MLIR context.
// param tensorShape The shape of the tensor (i.e 6x10x10)
// param grid The grid where the tensor will be placed (i.e 2x3)
// param collapseIntervals The intervals to collapse (i.e. {{0, -1}})
// return The constructed TTNNLayoutAttr
TTNNLayoutAttr TTNNLayoutAttr::withGrid(
    ArrayRef<int64_t> tensorShape, mlir::tt::ttcore::GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  return get(getContext(), tensorShape, getElementType(), getBufferType(), grid,
             getMemLayout(), getTensorMesh(), collapseIntervals,
             getIgnorePhysicalLayout());
}

// Construct a new TTNNLayoutAttr
//
// This function creates a new TTNNLayoutAttr with the given parameters.
// The shape of the tensor, buffer type, element type and memory layout are
// preserved.
//
// param context The MLIR context.
// param grid The grid where the tensor will be placed.
// param collapseIntervals The intervals to collapse (i.e. {{0, -1}})
// return The constructed TTNNLayoutAttr
TTNNLayoutAttr TTNNLayoutAttr::withGrid(
    RankedTensorType ty, mlir::tt::ttcore::GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  assert(ty);
  return TTNNLayoutAttr::withGrid(ty.getShape(), grid, collapseIntervals);
}

// Construct a new TTNNLayoutAttr
//
// This function creates a deep copy of the current TTNNLayoutAttr and
// replaces the element type with the given one.
//
// param context The MLIR context.
// param elementType The new element type.
// param tensorShape The shape of the tensor.
// param collapseIntervals The intervals to collapse (i.e. {{0, -1}})
// return The new TTNNLayoutAttr with the given element type.
TTNNLayoutAttr TTNNLayoutAttr::withElementType(
    Type elementType, ArrayRef<int64_t> tensorShape,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  return TTNNLayoutAttr::get(getContext(), tensorShape, elementType,
                             getBufferType(), getGrid(), getMemLayout(),
                             getTensorMesh(), collapseIntervals,
                             getIgnorePhysicalLayout());
}

// Construct a new TTNNLayoutAttr
//
// This function creates a deep copy of the current TTNNLayoutAttr and
// replaces the memory space with the given one.
//
// param context The MLIR context.
// param memorySpace The new memory space.
// return The new TTNNLayoutAttr with the given memory space.
TTNNLayoutAttr TTNNLayoutAttr::withBufferType(BufferType memorySpace) {
  TensorMemoryLayoutAttr memLayoutAttr = getMemLayout();
  mlir::tt::ttcore::GridAttr grid = getGrid();

  // For SystemMemory we need to clear memory layout and set grid to 1x1.
  if (memorySpace == BufferType::SystemMemory) {
    memLayoutAttr = TensorMemoryLayoutAttr{};
    grid =
        mlir::tt::ttcore::GridAttr::get(getContext(), grid.getShape().size());
  }

  // For DRAM we need to set memory layout to interleaved and set grid to 1x1.
  if (memorySpace == BufferType::DRAM) {
    memLayoutAttr = TensorMemoryLayoutAttr::get(
        getContext(), TensorMemoryLayout::Interleaved);
    grid =
        mlir::tt::ttcore::GridAttr::get(getContext(), grid.getShape().size());
  }

  // For L1 we will inherit the memory layout if its set.
  // Otherwise we will set it to interleaved.
  if (memorySpace == BufferType::L1) {
    memLayoutAttr = getMemLayout()
                        ? getMemLayout()
                        : TensorMemoryLayoutAttr::get(
                              getContext(), TensorMemoryLayout::Interleaved);
  }

  return TTNNLayoutAttr::get(
      getContext(), getLinear(), grid,
      mlir::tt::ttcore::buildMemRef<BufferType, BufferTypeAttr>(
          getContext(), getScalarShardShape(), getElementType(), memorySpace),
      memLayoutAttr, getTensorMesh(), getIgnorePhysicalLayout());
}

// Construct a new TTNNLayoutAttr
//
// This function creates a deep copy of the current TTNNLayoutAttr and
// replaces the memory layout with the given one.
//
// param context The MLIR context.
// param memLayoutAttr The new memory layout.
// return The new TTNNLayoutAttr with the given memory layout.
TTNNLayoutAttr
TTNNLayoutAttr::withMemoryLayout(TensorMemoryLayoutAttr memLayoutAttr) {
  return TTNNLayoutAttr::get(getContext(), getLinear(), getGrid(),
                             ttcore::buildMemRef<BufferType, BufferTypeAttr>(
                                 getContext(), getScalarShardShape(),
                                 getElementType(), getBufferType()),
                             memLayoutAttr, getTensorMesh(),
                             getIgnorePhysicalLayout());
}

// Construct a new TTNNLayoutAttr
//
// This function creates a deep copy of the current TTNNLayoutAttr and
// replaces the memory layout with the given one.
//
// param context The MLIR context.
// param memLayout The new memory layout.
// return The new TTNNLayoutAttr with the given memory layout.
TTNNLayoutAttr TTNNLayoutAttr::withMemoryLayout(TensorMemoryLayout memLayout) {

  TensorMemoryLayoutAttr memLayoutAttr =
      TensorMemoryLayoutAttr::get(getContext(), memLayout);
  return withMemoryLayout(memLayoutAttr);
}

// Construct a new TTNNLayoutAttr
//
// This function creates a deep copy of the current TTNNLayoutAttr and
// replaces shard shape with the given one.
//
// param context The MLIR context.
// param shardShape The new shard shape.
// return The new TTNNLayoutAttr with the given shard shape.
TTNNLayoutAttr
TTNNLayoutAttr::withShardShape(llvm::SmallVector<int64_t> shardShape) {
  return TTNNLayoutAttr::get(
      getContext(), getLinear(), getGrid(),
      mlir::tt::ttcore::buildMemRef<BufferType, BufferTypeAttr>(
          getContext(), shardShape, getElementType(), getBufferType()),
      getMemLayout(), getTensorMesh(), getIgnorePhysicalLayout());
}

// Construct a new TTNNLayoutAttr
//
// This function creates a deep copy of the current TTNNLayoutAttr and
// applies changes necessary to fit new tensor shape.
//
// param context The MLIR context.
// param tensorShape The new tensor shape.
// return The new TTNNLayoutAttr with the given tensor shape.
TTNNLayoutAttr TTNNLayoutAttr::withTensorShape(ArrayRef<int64_t> tensorShape) {
  // TODO(mrakita): This leaves default value of collapseIntervals parameter,
  // which might be different than the original value used to create the layout
  // attribute. This will work for now since we always use default value, but in
  // the future we would need to take this into account.
  return TTNNLayoutAttr::get(getContext(), tensorShape, getElementType(),
                             getBufferType(), getGrid(), getMemLayout(),
                             getTensorMesh(), getDefaultCollapseIntervals(),
                             getIgnorePhysicalLayout());
}

// Construct a new TTNNLayoutAttr
//
// This function creates a deep copy of the current TTNNLayoutAttr, setting the
// ignorePhysicalLayout property to the provided value. This is a status bit.
// The physical properties of the layout are preserved as calculated previously
// and remain accessible via getters
//
// param context The MLIR context.
// param ignorePhysicalLayout The new value for ignorePhysicalLayout.
// return The new TTNNLayoutAttr.
TTNNLayoutAttr
TTNNLayoutAttr::withIgnorePhysicalLayout(bool ignorePhysicalLayout) {
  return TTNNLayoutAttr::get(getContext(), getLinear(), getGrid(), getMemref(),
                             getMemLayout(), getTensorMesh(),
                             ignorePhysicalLayout);
};

TTNNLayoutAttr TTNNLayoutAttr::get(::mlir::MLIRContext *context,
                                   AffineMap linear, ttcore::GridAttr grid,
                                   MemRefType memref,
                                   TensorMemoryLayoutAttr mem_layout,
                                   ttcore::TensorMeshAttr tensor_mesh) {
  return TTNNLayoutAttr::get(context, linear, grid, memref, mem_layout,
                             tensor_mesh,
                             /*ignorePhysicalLayout=*/false);
}

// Construct a new TTNNLayoutAttr
//
// This function constructs a new TTNNLayoutAttr with the given parameters.
//
// param context The MLIR context.
// param tensorShape The shape of the tensor (i.e 6x10x10)
// param elementType The type of the element i.e TileType/FloatType/IntegerType
// param bufferType The type of the buffer
// param grid The grid where the tensor will be placed (i.e 2x3)
// param collapseIntervals The intervals to collapse (i.e. {{0, -1}})
// param memLayout The memory layout of the tensor
// return The constructed TTNNLayoutAttr
TTNNLayoutAttr TTNNLayoutAttr::get(
    ::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape,
    Type elementType, BufferType bufferType, mlir::tt::ttcore::GridAttr grid,
    TensorMemoryLayoutAttr memLayoutAttr,
    mlir::tt::ttcore::TensorMeshAttr tensorMesh,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals,
    bool ignorePhysicalLayout) {

  llvm::SmallVector<int64_t, 4> physicalShape(tensorShape.begin(),
                                              tensorShape.end());

  // If the tensor is tiled the last two dims need to be rounded up to tile size
  // before creating the affine map. E.g. (1, 2, 16, 16) -> (1, 2, 32, 32).
  if (llvm::isa<mlir::tt::ttcore::TileType>(elementType)) {
    physicalShape = utils::getTilePaddedShape(tensorShape);
  }

  // Construct a new affine map which will be used to map from logical
  // space to physical space.
  AffineMap linear = mlir::tt::ttcore::collapsedLinearAffineMap(
      context, physicalShape, grid.getShape(), collapseIntervals);

  // Calculate shard shape
  mlir::SmallVector<int64_t> shardShape;
  if (bufferType == BufferType::L1 &&
      memLayoutAttr.getValue() == TensorMemoryLayout::Interleaved) {
    shardShape = TTNNLayoutAttr::calculateLogicalShardShapeForL1Interleaved(
        physicalShape, elementType, linear, grid);
  } else {
    shardShape = TTNNLayoutAttr::calculateLogicalShardShapeForSharding(
        physicalShape, linear, grid);
  }

  // Build memref type with the given parameters
  MemRefType memRefType =
      mlir::tt::ttcore::buildMemRef<BufferType, BufferTypeAttr>(
          context, shardShape, elementType, bufferType);
  return get(context, linear, grid, memRefType, memLayoutAttr, tensorMesh,
             ignorePhysicalLayout);
}

llvm::LogicalResult TTNNLayoutAttr::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, AffineMap,
    mlir::tt::ttcore::GridAttr grid, MemRefType memref,
    TensorMemoryLayoutAttr memLayout,
    mlir::tt::ttcore::TensorMeshAttr tensorMesh, bool ignorePhysicalLayout) {
  BufferType bufferType =
      mlir::cast<BufferTypeAttr>(memref.getMemorySpace()).getValue();

  llvm::LogicalResult status = ::llvm::success();
  if (llvm::failed(verifyGridShape(emitError, grid, bufferType))) {
    status = llvm::failure();
  }
  if (llvm::failed(
          verifyBufferAndMemoryLayout(emitError, bufferType, memLayout))) {
    status = llvm::failure();
  }
  return status;
}

// Construct a new MemoryConfigAttr
//
// This function creates new MemoryConfigAttr from given TTNNLayoutAttr.
//
// param layoutAttr The TTNNLayoutAttr to create MemoryConfigAttr from.
// param deviceGrid Device grid to use for sharding spec.
// return The constructed MemoryConfigAttr.
MemoryConfigAttr MemoryConfigAttr::get(TTNNLayoutAttr layoutAttr,
                                       mlir::tt::ttcore::GridAttr deviceGrid) {
  BufferTypeAttr bufferTypeAttr =
      mlir::cast<BufferTypeAttr>(layoutAttr.getMemref().getMemorySpace());
  return MemoryConfigAttr::get(
      layoutAttr.getContext(), layoutAttr.getMemLayout(), bufferTypeAttr,
      utils::createShardSpecIfNeeded(layoutAttr, deviceGrid));
}

// Verify memory config attribute
::llvm::LogicalResult MemoryConfigAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    TensorMemoryLayoutAttr tensorMemoryLayout, BufferTypeAttr bufferType,
    std::optional<ShardSpecAttr> shardSpec) {
  // Verify buffer type, memory layout and sharding
  return ::llvm::success(verifyBufferAndMemoryLayout(emitError,
                                                     bufferType.getValue(),
                                                     tensorMemoryLayout)
                             .succeeded() &&
                         verifySharding(emitError, bufferType.getValue(),
                                        tensorMemoryLayout, shardSpec)
                             .succeeded());
}

bool CoreRangeAttr::intersects(CoreRangeAttr other) const {
  bool thisEndsBeforeOtherStarts =
      this->getEndCoord().getX() < other.getStartCoord().getX();
  bool thisStartsAfterOtherEnds =
      this->getStartCoord().getX() > other.getEndCoord().getX();
  bool thisEndsBelowOtherStarts =
      this->getEndCoord().getY() < other.getStartCoord().getY();
  bool thisStartsAboveOtherEnds =
      this->getStartCoord().getY() > other.getEndCoord().getY();

  return !(thisEndsBeforeOtherStarts || thisStartsAfterOtherEnds ||
           thisEndsBelowOtherStarts || thisStartsAboveOtherEnds);
}

::llvm::LogicalResult CoreRangeAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::tt::ttnn::CoreCoordAttr startCoord,
    mlir::tt::ttnn::CoreCoordAttr endCoord) {
  if (startCoord.getX() > endCoord.getX() ||
      startCoord.getY() > endCoord.getY()) {
    return emitError() << "Start coordinates " << startCoord
                       << " must be less than or equal to end coordinates "
                       << endCoord;
  }

  return ::llvm::success();
}

::llvm::LogicalResult CoreRangeSetAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr> coreRanges) {
  if (coreRanges.size() < 2) {
    return ::llvm::success();
  }

  // Check each pair of core ranges for intersections
  for (size_t i = 0; i < coreRanges.size() - 1; ++i) {
    for (size_t j = i + 1; j < coreRanges.size(); ++j) {
      CoreRangeAttr firstCoreRange = coreRanges[i];
      CoreRangeAttr secondCoreRange = coreRanges[j];
      if (firstCoreRange.intersects(secondCoreRange)) {
        return emitError() << "Core ranges overlap: " << firstCoreRange
                           << " and " << secondCoreRange;
      }
    }
  }

  return ::llvm::success();
}

// Returns empty configuration.
Conv2dConfigAttr Conv2dConfigAttr::get(::mlir::MLIRContext *context) {
  return Conv2dConfigAttr::get(context,
                               /*weightsDtype=*/std::nullopt,
                               /*activation=*/nullptr,
                               /*deallocateActivation=*/nullptr,
                               /*reallocateHaloOutput=*/nullptr,
                               /*actBlockHOverride=*/std::nullopt,
                               /*actBlockWDiv=*/std::nullopt,
                               /*reshardIfNotOptimal=*/nullptr,
                               /*overrideShardingConfig=*/nullptr,
                               /*shardLayout=*/std::nullopt,
                               /*coreGrid=*/nullptr,
                               /*transposeShards=*/nullptr,
                               /*outputLayout=*/std::nullopt,
                               /*enableActDoubleBuffer=*/nullptr,
                               /*enableWeightsDoubleBuffer=*/nullptr,
                               /*inPlace=*/nullptr);
}

// Returns default configuration.
Conv2dConfigAttr Conv2dConfigAttr::getDefault(::mlir::MLIRContext *context) {
  Conv2dConfigAttr convConfig = get(context);
  return Conv2dConfigParams(convConfig, /*partial=*/false)
      .buildConv2dConfigAttr(context);
}

Conv2dConfigAttr Conv2dConfigAttr::withActivation(StringRef activation) const {
  Conv2dConfigParams params(*this);
  params.activation = activation.str();
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr
Conv2dConfigAttr::withWeightsDtype(mlir::tt::ttcore::DataType dtype) const {
  Conv2dConfigParams params(*this);
  params.weightsDtype = dtype;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr Conv2dConfigAttr::withDeallocateActivation(bool value) const {
  Conv2dConfigParams params(*this);
  params.deallocateActivation = value;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr Conv2dConfigAttr::withReallocateHaloOutput(bool value) const {
  Conv2dConfigParams params(*this);
  params.reallocateHaloOutput = value;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr Conv2dConfigAttr::withActBlockHOverride(uint32_t value) const {
  Conv2dConfigParams params(*this);
  params.actBlockHOverride = value;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr Conv2dConfigAttr::withActBlockWDiv(uint32_t value) const {
  Conv2dConfigParams params(*this);
  params.actBlockWDiv = value;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr Conv2dConfigAttr::withReshardIfNotOptimal(bool value) const {
  Conv2dConfigParams params(*this);
  params.reshardIfNotOptimal = value;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr
Conv2dConfigAttr::withOverrideShardingConfig(bool value) const {
  Conv2dConfigParams params(*this);
  params.overrideShardingConfig = value;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr
Conv2dConfigAttr::withShardLayout(TensorMemoryLayout layout) const {
  Conv2dConfigParams params(*this);
  params.shardLayout = layout;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr Conv2dConfigAttr::withCoreGrid(CoreRangeSetAttr grid) const {
  Conv2dConfigParams params(*this);
  params.coreGrid = grid;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr Conv2dConfigAttr::withTransposeShards(bool value) const {
  Conv2dConfigParams params(*this);
  params.transposeShards = value;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr Conv2dConfigAttr::withOutputLayout(Layout layout) const {
  Conv2dConfigParams params(*this);
  params.outputLayout = layout;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr Conv2dConfigAttr::withEnableActDoubleBuffer(bool value) const {
  Conv2dConfigParams params(*this);
  params.enableActDoubleBuffer = value;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr
Conv2dConfigAttr::withEnableWeightsDoubleBuffer(bool value) const {
  Conv2dConfigParams params(*this);
  params.enableWeightsDoubleBuffer = value;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr Conv2dConfigAttr::withInPlace(bool value) const {
  Conv2dConfigParams params(*this);
  params.inPlace = value;
  return params.buildConv2dConfigAttr(getContext());
}

bool Conv2dConfigAttr::hasActivation() const {
  return getActivation() != nullptr && getActivation().getValue() != "";
}

bool Conv2dConfigAttr::hasWeightsDtype() const {
  return getWeightsDtype().has_value();
}

bool Conv2dConfigAttr::hasDeallocateActivation() const {
  return getDeallocateActivation() != nullptr;
}

bool Conv2dConfigAttr::hasReallocateHaloOutput() const {
  return getReallocateHaloOutput() != nullptr;
}

bool Conv2dConfigAttr::hasActBlockHOverride() const {
  return getActBlockHOverride().has_value();
}

bool Conv2dConfigAttr::hasActBlockWDiv() const {
  return getActBlockWDiv().has_value();
}

bool Conv2dConfigAttr::hasReshardIfNotOptimal() const {
  return getReshardIfNotOptimal() != nullptr;
}

bool Conv2dConfigAttr::hasOverrideShardingConfig() const {
  return getOverrideShardingConfig() != nullptr;
}

bool Conv2dConfigAttr::hasShardLayout() const {
  return getShardLayout().has_value();
}

bool Conv2dConfigAttr::hasCoreGrid() const { return getCoreGrid() != nullptr; }

bool Conv2dConfigAttr::hasTransposeShards() const {
  return getTransposeShards() != nullptr;
}

bool Conv2dConfigAttr::hasOutputLayout() const {
  return getOutputLayout().has_value();
}

bool Conv2dConfigAttr::hasEnableActDoubleBuffer() const {
  return getEnableActDoubleBuffer() != nullptr;
}

bool Conv2dConfigAttr::hasEnableWeightsDoubleBuffer() const {
  return getEnableWeightsDoubleBuffer() != nullptr;
}

bool Conv2dConfigAttr::hasInPlace() const { return getInPlace() != nullptr; }

CoreRangeSetAttr
ShardSpecAttr::getCoreRangeSet(mlir::MLIRContext *context,
                               mlir::tt::ttcore::GridAttr shardGrid,
                               mlir::tt::ttcore::GridAttr deviceGrid) {
  llvm::SmallVector<CoreRangeAttr> coreRangeSet;
  AffineMap mapping = (shardGrid.getMapping().isEmpty() == true)
                          ? deviceGrid.getMapping()
                          : shardGrid.getMapping();

  for (const auto &locsize2d :
       mlir::tt::ttcore::utils::toCoreRangeSet(shardGrid.getShape(), mapping)) {
    const auto &[loc, size] = locsize2d;
    coreRangeSet.push_back(
        CoreRangeAttr::get(context, CoreCoordAttr::get(context, loc[0], loc[1]),
                           CoreCoordAttr::get(context, loc[0] + size[0] - 1,
                                              loc[1] + size[1] - 1)));
  }

  return CoreRangeSetAttr::get(context, coreRangeSet);
}

struct DeviceComputeKernelConfigAttrParams {
  std::optional<mlir::tt::ttnn::MathFidelity> mathFidelity;
  mlir::BoolAttr mathApproxMode;
  mlir::BoolAttr fp32DestAccEn;
  mlir::BoolAttr packerL1Acc;
  mlir::BoolAttr dstFullSyncEn;

  DeviceComputeKernelConfigAttrParams() = delete;

  DeviceComputeKernelConfigAttrParams(DeviceComputeKernelConfigAttr attr) {
    mathFidelity = attr.getMathFidelity();
    mathApproxMode = attr.getMathApproxMode();
    fp32DestAccEn = attr.getFp32DestAccEn();
    packerL1Acc = attr.getPackerL1Acc();
    dstFullSyncEn = attr.getDstFullSyncEn();
  }

  DeviceComputeKernelConfigAttr
  buildDeviceComputeKernelConfigAttr(mlir::MLIRContext *ctx) const {
    return DeviceComputeKernelConfigAttr::get(ctx, mathFidelity, mathApproxMode,
                                              fp32DestAccEn, packerL1Acc,
                                              dstFullSyncEn);
  }
};

DeviceComputeKernelConfigAttr DeviceComputeKernelConfigAttr::withMathFidelity(
    mlir::tt::ttnn::MathFidelity mathFidelity) const {
  DeviceComputeKernelConfigAttrParams params(*this);
  params.mathFidelity = mathFidelity;
  return params.buildDeviceComputeKernelConfigAttr(getContext());
}

DeviceComputeKernelConfigAttr
DeviceComputeKernelConfigAttr::withMathApproxMode(bool value) const {
  DeviceComputeKernelConfigAttrParams params(*this);
  params.mathApproxMode = BoolAttr::get(getContext(), value);
  return params.buildDeviceComputeKernelConfigAttr(getContext());
}

DeviceComputeKernelConfigAttr
DeviceComputeKernelConfigAttr::withFp32DestAccEn(bool value) const {
  DeviceComputeKernelConfigAttrParams params(*this);
  params.fp32DestAccEn = BoolAttr::get(getContext(), value);
  return params.buildDeviceComputeKernelConfigAttr(getContext());
}

DeviceComputeKernelConfigAttr
DeviceComputeKernelConfigAttr::withPackerL1Acc(bool value) const {
  DeviceComputeKernelConfigAttrParams params(*this);
  params.packerL1Acc = BoolAttr::get(getContext(), value);
  return params.buildDeviceComputeKernelConfigAttr(getContext());
}

DeviceComputeKernelConfigAttr
DeviceComputeKernelConfigAttr::withDstFullSyncEn(bool value) const {
  DeviceComputeKernelConfigAttrParams params(*this);
  params.dstFullSyncEn = BoolAttr::get(getContext(), value);
  return params.buildDeviceComputeKernelConfigAttr(getContext());
}

::llvm::LogicalResult ProgramAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<mlir::Attribute> kernels,
    llvm::ArrayRef<mlir::tt::ttnn::KernelCBAttr> cbs,
    llvm::ArrayRef<mlir::tt::ttnn::KernelSemaphoreAttr> semaphores) {

  for (auto kernel : kernels) {
    if (!llvm::isa<mlir::tt::ttnn::ComputeKernelAttr,
                   mlir::tt::ttnn::ReadKernelAttr,
                   mlir::tt::ttnn::WriteKernelAttr>(kernel)) {
      return emitError() << "Unexpected kernel";
    }
  }

  return ::llvm::success();
}

::llvm::LogicalResult KernelCBAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    uint32_t totalSize, CoreRangeSetAttr coreRanges,
    llvm::ArrayRef<mlir::tt::ttnn::KernelCBFormatAttr> formats) {
  return ::llvm::success();
}

::llvm::LogicalResult KernelCBFormatAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    uint32_t bufferIndex, ttcore::DataType dtype, uint32_t pageSize) {
  return ::llvm::success();
}

::llvm::LogicalResult KernelSemaphoreAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    KernelCoreType coreType, ::mlir::tt::ttnn::CoreRangeSetAttr coreRanges,
    uint32_t initialValue) {
  return ::llvm::success();
}

::llvm::LogicalResult verifyCommonRuntimeArgs(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<mlir::Attribute> args) {

  for (auto arg : args) {
    if (!llvm::isa<mlir::tt::ttnn::KernelArgCBBufferIndexAttr,
                   mlir::tt::ttnn::KernelArgAddressOfTensorAttr,
                   mlir::tt::ttnn::KernelArgSemaphoreAtAttr>(arg)) {
      return emitError() << "Unexpected common runtime argument";
    }
  }

  return ::llvm::success();
}

::llvm::LogicalResult verifyCompileTimeArgs(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<mlir::Attribute> args) {
  for (auto arg : args) {
    if (!llvm::isa<mlir::tt::ttnn::KernelArgCBBufferIndexAttr,
                   mlir::tt::ttnn::KernelArgSemaphoreAtAttr>(arg)) {
      return emitError() << "Unexpected compile time argument";
    }
  }

  return ::llvm::success();
}

::llvm::LogicalResult ComputeKernelAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    SymbolRefAttr symbolRef, ::mlir::tt::ttnn::CoreRangeSetAttr coreRanges,
    ComputeKernelMathFidelity mathFidelity, bool fp32DestAccEn,
    bool dstFullSyncEn,
    ::llvm::ArrayRef<ComputeKernelUnpackToDestMode> unpackToDestModes,
    bool bfp8PackPrecise, bool mathApproxMode,
    ::llvm::ArrayRef<mlir::Attribute> commonRtArgs,
    ::llvm::ArrayRef<mlir::Attribute> ctArgs) {
  if (failed(verifyCommonRuntimeArgs(emitError, commonRtArgs)) ||
      failed(verifyCompileTimeArgs(emitError, ctArgs))) {
    return ::llvm::failure();
  }

  return ::llvm::success();
}

::llvm::LogicalResult ReadKernelAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::SymbolRefAttr symbolRef, CoreRangeSetAttr coreRanges,
    ::llvm::ArrayRef<mlir::Attribute> commonRtArgs,
    ::llvm::ArrayRef<mlir::Attribute> ctArgs) {
  if (failed(verifyCommonRuntimeArgs(emitError, commonRtArgs)) ||
      failed(verifyCompileTimeArgs(emitError, ctArgs))) {
    return ::llvm::failure();
  }

  return ::llvm::success();
}

::llvm::LogicalResult WriteKernelAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::SymbolRefAttr symbolRef, CoreRangeSetAttr coreRanges,
    ::llvm::ArrayRef<mlir::Attribute> commonRtArgs,
    ::llvm::ArrayRef<mlir::Attribute> ctArgs) {
  if (failed(verifyCommonRuntimeArgs(emitError, commonRtArgs)) ||
      failed(verifyCompileTimeArgs(emitError, ctArgs))) {
    return ::llvm::failure();
  }

  return ::llvm::success();
}

// Transform TTNNLayoutAttr with a different layout while preserving the
// element type.
//
// param layout The target layout (RowMajor or Tile).
// param tensorShape The shape of the tensor.
// return The new TTNNLayoutAttr with the specified layout.
TTNNLayoutAttr TTNNLayoutAttr::withLayout(Layout layout,
                                          ArrayRef<int64_t> tensorShape) {
  assert(layout == Layout::RowMajor || layout == Layout::Tile);
  Type elementType = utils::getElementType(getContext(), layout, getDataType());
  return withElementType(elementType, tensorShape);
}
