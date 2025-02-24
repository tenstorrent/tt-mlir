// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <numeric>

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

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

// Get stride given tensor logical shape
llvm::SmallVector<int64_t>
TTNNLayoutAttr::getStride(ArrayRef<int64_t> logicalShape) const {

  llvm::SmallVector<int64_t> stride(logicalShape.size());
  AffineMap linearMap = getLinear();

  // Calculate the physical shape of the tensor.
  // Given tensor (6x15x10) and linear (d0, d1, d2) -> (d0 * 15 + d1, d2)
  // The physical shape is (90, 10)
  SmallVector<int64_t> physicalShape =
      ttmlir::utils::evalShape(linearMap, logicalShape);

  // Origin point in the logical space (0, 0)
  SmallVector<AffineExpr> originPoint(logicalShape.size(),
                                      getAffineConstantExpr(0, getContext()));

  size_t prevDimElems = 1;

  // Iterates through physical dimensions (starting from the inner one).
  for (int i = linearMap.getNumResults() - 1; i >= 0; i--) {
    AffineExpr expr = linearMap.getResult(i);

    // Get coordinate of the i-th dimension (in physical space) of the origin
    // (in logical space).
    AffineExpr constantExpr = expr.replaceDims(originPoint);
    std::int64_t valueAtZero =
        llvm::cast<AffineConstantExpr>(constantExpr).getValue();

    for (size_t j = 0; j < logicalShape.size(); j++) {
      if (!expr.isFunctionOfDim(j)) {
        continue;
      }

      // Move from the origin point by one in the j-th dimension,
      // and get the coordinate of the i-th dimension (in physical space).
      SmallVector<AffineExpr> newPoint = originPoint;
      newPoint[j] = getAffineConstantExpr(1, getContext());
      constantExpr = expr.replaceDims(newPoint);
      std::int64_t valueAtOne =
          llvm::cast<AffineConstantExpr>(constantExpr).getValue();

      // One step in the j-th dimension, jumps delta * prevDimElems elements in
      // the physical space.
      int64_t delta = valueAtOne - valueAtZero;
      stride[j] = prevDimElems * delta;
    }

    prevDimElems *= physicalShape[i];
  }

  return stride;
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
// Size is calculate by pluging the tensor shape into the linear map.
// This result is then divided by the tile shape.
// Example: tensor shape (6x15x10), linear map (d0, d1, d2) -> (d0 * 15 + d1,
// d2) and tile shape (32, 32) The result is (90, 10) which is then divided by
// tile shape (32, 32) -> (3, 1)
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
  return ttmlir::utils::evalShape(tiled, tensorShape);
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

// Get new identity affine map i.e (d0, d1) -> (d0, d1)
//
// This function returns a new identity affine map
// with the same number of dimensions as the linear map.
//
// return The new identity affine map.
mlir::AffineMap TTNNLayoutAttr::getIdentityTileLinearMap() const {
  assert(isTiled() && "Expected a tiled layout");

  return mlir::AffineMap::getMultiDimIdentityMap(getLinear().getNumResults(),
                                                 getContext());
}

// Takes phyisical memory map and replaces the symbols with the shard shape
//
// This function takes a physical memory map and replaces the symbols with the
// shard shape
//
// param physicalMemoryMap The physical memory map (d0, d1)[s0, s1]
// return New memory map with symbols replaced with shard shape.
mlir::AffineMap TTNNLayoutAttr::replaceMemoryMapSymbolsWithShardShape(
    AffineMap physicalMemoryMap) const {
  mlir::SmallVector<int64_t> shardShape = getShardShape();
  assert(physicalMemoryMap.getNumSymbols() == shardShape.size() &&
         "Physical memory map must have same number of symbols as logical "
         "shard rank");

  SmallVector<AffineExpr> symReplacements;
  for (size_t i = 0; i < physicalMemoryMap.getNumSymbols(); ++i) {
    symReplacements.push_back(
        getAffineConstantExpr(shardShape[i], getContext()));
  }

  SmallVector<AffineExpr> dimReplacements;
  for (size_t i = 0; i < physicalMemoryMap.getNumDims(); ++i) {
    dimReplacements.push_back(getAffineDimExpr(i, getContext()));
  }

  return physicalMemoryMap.replaceDimsAndSymbols(
      dimReplacements, symReplacements, physicalMemoryMap.getNumDims(), 0);
}

int64_t TTNNLayoutAttr::getTensorSizeInBytes(ArrayRef<int64_t> tensorShape,
                                             DeviceAttr device) const {
  SmallVector<int64_t> shape = isTiled() ? getTiledShape(tensorShape)
                                         : SmallVector<int64_t>(tensorShape);
  MemorySpace memorySpace = utils::toTTMemorySpace(getBufferType());
  AffineMap linearMap = isTiled() ? getIdentityTileLinearMap() : getLinear();
  mlir::SmallVector<std::int64_t> linearShape =
      ttmlir::utils::evalShape(linearMap, shape);
  AffineMap memoryMap = replaceMemoryMapSymbolsWithShardShape(
      device.getMapForMemorySpace(memorySpace));
  mlir::SmallVector<std::int64_t> physicalMemory =
      ttmlir::utils::evalShape(memoryMap, linearShape);
  std::int64_t elementSize = getElementSizeBytes();
  uint64_t sizeBytes =
      physicalMemory[MemoryMapResultIdx::ShardOffset] * elementSize;
  return sizeBytes;
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
             getMemLayout(), collapseIntervals);
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
                             getGrid(), getMemLayout(), collapseIntervals);
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
      memLayoutAttr);
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
      memLayoutAttr);
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
      getMemLayout());
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
                             getBufferType(), getGrid(), getMemLayout());
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
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  // Construct a new affine map which will be used to map from logical
  // space to physical space.
  AffineMap linear = collapsedLinearAffineMap(
      context, tensorShape, grid.getShape(), collapseIntervals);

  // Calculate shard shape
  mlir::SmallVector<int64_t> shardShape;
  if (bufferType == BufferType::L1 &&
      memLayoutAttr.getValue() == TensorMemoryLayout::Interleaved) {
    shardShape = TTNNLayoutAttr::calculateLogicalShardShapeForL1Interleaved(
        tensorShape, elementType, linear, grid);
  } else {
    shardShape = TTNNLayoutAttr::calculateLogicalShardShapeForSharding(
        tensorShape, linear, grid);
  }

  // Build memref type with the given parameters
  MemRefType memRefType = buildMemRef<BufferType, BufferTypeAttr>(
      context, shardShape, elementType, bufferType);
  return get(context, linear, grid, memRefType, memLayoutAttr);
}

::llvm::LogicalResult TTNNLayoutAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, AffineMap,
    GridAttr, MemRefType memref, TensorMemoryLayoutAttr memLayout) {
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
