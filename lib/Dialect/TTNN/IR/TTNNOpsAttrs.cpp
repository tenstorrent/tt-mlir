// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include <numeric>

using namespace mlir::tt::ttnn;

// Check if the tensor is tiled
bool TTNNLayoutAttr::isTiled() const {
  return ::mlir::isa<::mlir::tt::TileType>(getElementType());
}

// Get layout of the tensor (RowMajor/Tile)
Layout TTNNLayoutAttr::getLayout() const {
  return isTiled() ? Layout::Tile : Layout::RowMajor;
}

// Get optinoal memory layout
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
    ArrayRef<int64_t> tensorShape, mlir::AffineMap linear, GridAttr grid) {
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
// tensor memory layout.
//
// All examples assume that the tensor is mapped to a 8x8 grid.
// Example: tensor<1x1024xbf16> ( -> 32 tiles ) -> {1, 1}
// Example: tensor<512x512xbf16> ( -> 256 tiles ) -> {1, 4}
// Example: tensor<32x2049xbf16> ( -> 65 tiles ) -> {1, 2}
//
// return The logical shard shape in case of interleaved tensor memory layout.
llvm::SmallVector<int64_t>
TTNNLayoutAttr::calculateLogicalShardShapeForL1Interleaved(
    ArrayRef<int64_t> tensorShape, mlir::Type elementType,
    mlir::AffineMap linear, mlir::tt::GridAttr grid) {
  assert(linear.getNumResults() == grid.getShape().size());
  assert(mlir::isa<mlir::tt::TileType>(elementType));

  mlir::SmallVector<std::int64_t> physicalShape =
      ttmlir::utils::evalShape(linear, tensorShape);
  mlir::SmallVector<std::int64_t> physicalTiledShape =
      mlir::cast<mlir::tt::TileType>(elementType).getTiledShape(physicalShape);
  uint64_t numOfTiles =
      std::accumulate(physicalTiledShape.begin(), physicalTiledShape.end(), 1,
                      std::multiplies<std::int64_t>());
  uint64_t numOfGridUnits =
      std::accumulate(grid.getShape().begin(), grid.getShape().end(), 1,
                      std::multiplies<std::int64_t>());

  mlir::SmallVector<std::int64_t> shardShape;
  shardShape.resize(grid.getShape().size() - 1, 1);
  shardShape.push_back((numOfTiles + numOfGridUnits - 1) / numOfGridUnits);
  return mlir::cast<mlir::tt::TileType>(elementType).getScalarShape(shardShape);
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
  if (mlir::isa<TileType>(elementType)) {
    return mlir::cast<TileType>(elementType).getElementType();
  }
  return elementType;
}

// Get scalar element type.
// Example: memref<2x2xf32> -> f32
// Example: memref<2x2x!tt.tile<32x32xf32>> -> f32
//
// return The scalar element type.
mlir::tt::DataType TTNNLayoutAttr::getDataType() const {
  Type elementType = getElementType();
  if (isTiled()) {
    TileType tileType = mlir::cast<TileType>(elementType);
    return tileType.getDataType();
  }

  return elementTypeToDataType(elementType);
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
    TileType tileType = mlir::cast<TileType>(elementType);
    return tileType.getSizeBytes();
  }
  return elementType.getIntOrFloatBitWidth() / 8;
}

// Get shard shape
//
// Return the shape of the shard.
// Example: memref<2x2x!tt.tile<32x32xf32>> -> { 2, 2 }
// Example: memref<128x128xf32> -> { 128, 128 }
// Example: memref<2x3x!tt.tile<32x32xf32>> -> { 2, 3 }
//
// return The shape of the shard.
llvm::SmallVector<int64_t> TTNNLayoutAttr::getShardShape() const {
  return SmallVector<int64_t>(getMemref().getShape());
}

// Get scalar shard shape
//
// If the element type is TileType, this function returns the scalar shape of
// the shard.
// Example: memref<2x2x!tt.tile<32x32xf32>> -> { 64, 64 }
// Example: memref<128x128xf32> -> { 128, 128 }
// Example: memref<2x3!tt.tile<32x32xf32>> -> { 64, 96 }
//
// return The scalar shape of the shard.
llvm::SmallVector<int64_t> TTNNLayoutAttr::getScalarShardShape() const {
  SmallVector<int64_t> shardShape(getMemref().getShape());
  if (isTiled()) {
    return mlir::cast<TileType>(getElementType()).getScalarShape(shardShape);
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

  TileType tileType = mlir::cast<TileType>(getElementType());
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
    ::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  return get(context, tensorShape, getElementType(), getBufferType(), grid,
             getMemLayout(), getTensorMeshSharding(), collapseIntervals);
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
    ::mlir::MLIRContext *context, RankedTensorType ty, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  assert(ty);
  return TTNNLayoutAttr::withGrid(context, ty.getShape(), grid,
                                  collapseIntervals);
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
    ::mlir::MLIRContext *context, Type elementType,
    ArrayRef<int64_t> tensorShape,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  return TTNNLayoutAttr::get(context, tensorShape, elementType, getBufferType(),
                             getGrid(), getMemLayout(), getTensorMeshSharding(),
                             collapseIntervals);
}

// Construct a new TTNNLayoutAttr
//
// This function creates a deep copy of the current TTNNLayoutAttr and
// replaces the memory space with the given one.
//
// param context The MLIR context.
// param memorySpace The new memory space.
// return The new TTNNLayoutAttr with the given memory space.
TTNNLayoutAttr TTNNLayoutAttr::withBufferType(::mlir::MLIRContext *context,
                                              BufferType memorySpace) {
  TensorMemoryLayoutAttr memLayoutAttr = getMemLayout();
  tt::GridAttr grid = getGrid();

  // For SystemMemory we need to clear memory layout and set grid to 1x1.
  if (memorySpace == BufferType::SystemMemory) {
    memLayoutAttr = TensorMemoryLayoutAttr{};
    grid = tt::GridAttr::get(context, grid.getShape().size());
  }

  // For DRAM we need to set memory layout to interleaved and set grid to 1x1.
  if (memorySpace == BufferType::DRAM) {
    memLayoutAttr =
        TensorMemoryLayoutAttr::get(context, TensorMemoryLayout::Interleaved);
    grid = tt::GridAttr::get(context, grid.getShape().size());
  }

  // For L1 we will inherit the memory layout if its set.
  // Otherwise we will set it to interleaved.
  if (memorySpace == BufferType::L1) {
    memLayoutAttr = getMemLayout()
                        ? getMemLayout()
                        : TensorMemoryLayoutAttr::get(
                              context, TensorMemoryLayout::Interleaved);
  }

  return TTNNLayoutAttr::get(
      context, getLinear(), grid,
      buildMemRef<BufferType, BufferTypeAttr>(context, getScalarShardShape(),
                                              getElementType(), memorySpace),
      memLayoutAttr, getTensorMeshSharding());
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
TTNNLayoutAttr::withMemoryLayout(::mlir::MLIRContext *context,
                                 TensorMemoryLayoutAttr memLayoutAttr) {
  return TTNNLayoutAttr::get(
      context, getLinear(), getGrid(),
      buildMemRef<BufferType, BufferTypeAttr>(
          context, getScalarShardShape(), getElementType(), getBufferType()),
      memLayoutAttr, getTensorMeshSharding());
}

// Construct a new TTNNLayoutAttr
//
// This function creates a deep copy of the current TTNNLayoutAttr and
// replaces the memory layout with the given one.
//
// param context The MLIR context.
// param memLayout The new memory layout.
// return The new TTNNLayoutAttr with the given memory layout.
TTNNLayoutAttr TTNNLayoutAttr::withMemoryLayout(::mlir::MLIRContext *context,
                                                TensorMemoryLayout memLayout) {

  TensorMemoryLayoutAttr memLayoutAttr =
      TensorMemoryLayoutAttr::get(context, memLayout);
  return withMemoryLayout(context, memLayoutAttr);
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
TTNNLayoutAttr::withShardShape(::mlir::MLIRContext *context,
                               llvm::SmallVector<int64_t> shardShape) {
  return TTNNLayoutAttr::get(
      context, getLinear(), getGrid(),
      buildMemRef<BufferType, BufferTypeAttr>(
          context, shardShape, getElementType(), getBufferType()),
      getMemLayout(), getTensorMeshSharding());
}

// Construct a new TTNNLayoutAttr
//
// This function creates a deep copy of the current TTNNLayoutAttr and
// applies changes necessary to fit new tensor shape.
//
// param context The MLIR context.
// param tensorShape The new tensor shape.
// return The new TTNNLayoutAttr with the given tensor shape.
TTNNLayoutAttr TTNNLayoutAttr::withTensorShape(::mlir::MLIRContext *context,
                                               ArrayRef<int64_t> tensorShape) {
  // TODO(mrakita): This leaves default value of collapseIntervals parameter,
  // which might be different than the original value used to create the layout
  // attribute. This will work for now since we always use default value, but in
  // the future we would need to take this into account.
  return TTNNLayoutAttr::get(context, tensorShape, getElementType(),
                             getBufferType(), getGrid(), getMemLayout(),
                             getTensorMeshSharding());
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
    Type elementType, BufferType bufferType, GridAttr grid,
    TensorMemoryLayoutAttr memLayoutAttr,
    TensorMeshShardingAttr tensorMeshSharding,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {

  llvm::SmallVector<int64_t, 4> physicalShape(tensorShape.begin(),
                                              tensorShape.end());

  // If the tensor is tiled the last two dims need to be rounded up to tile size
  // before creating the affine map. E.g. (1, 2, 16, 16) -> (1, 2, 32, 32).
  if (llvm::isa<TileType>(elementType)) {
    physicalShape = utils::getTilePaddedShape(tensorShape);
  }

  // Construct a new affine map which will be used to map from logical
  // space to physical space.
  AffineMap linear = collapsedLinearAffineMap(
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
  MemRefType memRefType = buildMemRef<BufferType, BufferTypeAttr>(
      context, shardShape, elementType, bufferType);
  return get(context, linear, grid, memRefType, memLayoutAttr,
             tensorMeshSharding);
}

::llvm::LogicalResult TTNNLayoutAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, AffineMap,
    GridAttr, MemRefType memref, TensorMemoryLayoutAttr memLayout,
    TensorMeshShardingAttr tensorMeshSharding) {
  BufferType bufferType =
      mlir::cast<BufferTypeAttr>(memref.getMemorySpace()).getValue();
  return verifyBufferAndMemoryLayout(emitError, bufferType, memLayout);
}

// Construct a new MemoryConfig
//
// This function creates a deep copy of the current MemoryConfigAttr and
// replaces the buffer type with the given one.
//
// param context The MLIR context.
// param buffer type The new buffer type.
// return The new MemoryConfigAttr with the given buffer type.
MemoryConfigAttr MemoryConfigAttr::withBufferType(::mlir::MLIRContext *context,
                                                  BufferType bufferType) {
  return MemoryConfigAttr::get(context,
                               BufferTypeAttr::get(context, bufferType),
                               getShardSpec(), getTensorMemoryLayout());
}

// Construct a new MemoryConfig
//
// This function creates a deep copy of the current MemoryConfig and
// replaces the memory layout with the given one.
//
// param context The MLIR context.
// param memLayout The new memory layout.
// return The new MemoryConfig with the given memory layout.
MemoryConfigAttr
MemoryConfigAttr::withMemoryLayout(::mlir::MLIRContext *context,
                                   TensorMemoryLayout memLayout) {
  return MemoryConfigAttr::get(context, getBufferType(), getShardSpec(),
                               TensorMemoryLayoutAttr::get(context, memLayout));
}

// Verify memory config attribute
::llvm::LogicalResult MemoryConfigAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    BufferTypeAttr bufferType, ShardSpecAttr shardSpec,
    TensorMemoryLayoutAttr tensorMemoryLayout) {
  return verifyBufferAndMemoryLayout(emitError, bufferType.getValue(),
                                     tensorMemoryLayout);

  // TODO(#2140): Once we complete #1628, we should add a verifier for
  // ShardSpecAttr. ShardSpecAttr is only valid if the buffer type is L1.
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

Conv2dConfigAttr Conv2dConfigAttr::get(::mlir::MLIRContext *context) {
  return Conv2dConfigAttr::get(
      context, std::nullopt, std::nullopt, nullptr, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, nullptr, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

template <size_t ParamIndex, typename T>
Conv2dConfigAttr Conv2dConfigAttr::withParam(T value) {
  std::tuple<std::optional<DataType>, std::optional<DataType>, StringAttr,
             std::optional<uint32_t>, std::optional<bool>, std::optional<bool>,
             std::optional<uint32_t>, std::optional<uint32_t>,
             std::optional<bool>, std::optional<bool>,
             std::optional<TensorMemoryLayout>, CoreRangeSetAttr,
             std::optional<bool>, std::optional<Layout>, std::optional<bool>,
             std::optional<bool>, std::optional<bool>, std::optional<bool>,
             std::optional<bool>, std::optional<bool>>
      params = {getDtype(),
                getWeightsDtype(),
                getActivation(),
                getInputChannelsAlignment(),
                getDeallocateActivation(),
                getReallocateHaloOutput(),
                getActBlockHOverride(),
                getActBlockWDiv(),
                getReshardIfNotOptimal(),
                getOverrideShardingConfig(),
                getShardLayout(),
                getCoreGrid(),
                getTransposeShards(),
                getOutputLayout(),
                getPreprocessWeightsOnDevice(),
                getAlwaysPreprocessWeights(),
                getEnableActDoubleBuffer(),
                getEnableWeightsDoubleBuffer(),
                getEnableSplitReader(),
                getEnableSubblockPadding()};

  std::get<ParamIndex>(params) = value;

  return std::apply(
      [this](auto &&...args) {
        return Conv2dConfigAttr::get(getContext(),
                                     std::forward<decltype(args)>(args)...);
      },
      params);
}

Conv2dConfigAttr
Conv2dConfigAttr::withDtype(std::optional<mlir::tt::DataType> value) {
  return withParam<0>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withWeightsDtype(std::optional<mlir::tt::DataType> value) {
  return withParam<1>(value);
}
Conv2dConfigAttr Conv2dConfigAttr::withActivation(mlir::StringAttr value) {
  return withParam<2>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withInputChannelsAlingment(std::optional<uint32_t> value) {
  return withParam<3>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withDeallocateActivation(std::optional<bool> value) {
  return withParam<4>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withReallocateHaloInput(std::optional<bool> value) {
  return withParam<5>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withActBlockHOverride(std::optional<uint32_t> value) {
  return withParam<6>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withActBlockWDiv(std::optional<uint32_t> value) {
  return withParam<7>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withReshardIfNotOptimal(std::optional<bool> value) {
  return withParam<8>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withOverrideShardingConfig(std::optional<bool> value) {
  return withParam<9>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withShardLayout(std::optional<TensorMemoryLayout> value) {
  return withParam<10>(value);
}
Conv2dConfigAttr Conv2dConfigAttr::withCoreGrid(CoreRangeSetAttr value) {
  return withParam<11>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withTransposeShards(std::optional<bool> value) {
  return withParam<12>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withOutputLayout(std::optional<Layout> value) {
  return withParam<13>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withPreprocessWeightsOnDevice(std::optional<bool> value) {
  return withParam<14>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withAlwaysPreprocessWeights(std::optional<bool> value) {
  return withParam<15>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withEnableActDoubleBuffer(std::optional<bool> value) {
  return withParam<16>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withEnableWeightsDoubleBuffer(std::optional<bool> value) {
  return withParam<17>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withEnableSplitReader(std::optional<bool> value) {
  return withParam<18>(value);
}
Conv2dConfigAttr
Conv2dConfigAttr::withEnableSubblockPadding(std::optional<bool> value) {
  return withParam<19>(value);
}
