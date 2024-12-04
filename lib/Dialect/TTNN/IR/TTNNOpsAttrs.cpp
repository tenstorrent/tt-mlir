// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <numeric>

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt::ttnn;

// Check if tensor is on host
inline bool isSystemBufferType(BufferType bufferType) {
  return bufferType == BufferType::SystemMemory;
}

// Check if the tensor is on device
inline bool isDeviceBufferType(BufferType bufferType) {
  return bufferType == BufferType::DRAM || bufferType == BufferType::L1;
}

// Check if tensor is in DRAM memory
inline bool isDRAMBufferType(BufferType bufferType) {
  return bufferType == BufferType::DRAM;
}

// Check if tensor is in L1 memory
inline bool isL1BufferType(BufferType bufferType) {
  return bufferType == BufferType::L1;
}

// Check if the tensor is tiled
bool TTNNLayoutAttr::isTiled() const {
  return ::mlir::isa<::mlir::tt::TileType>(getElementType());
}

// Get layout of the tensor (RowMajor/Tile)
Layout TTNNLayoutAttr::getLayout() const {
  return isTiled() ? Layout::Tile : Layout::RowMajor;
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
  return (getMemLayout() == TensorMemoryLayout::HeightSharded ||
          getMemLayout() == TensorMemoryLayout::WidthSharded ||
          getMemLayout() == TensorMemoryLayout::BlockSharded);
}

// Check if the tensor memory layout is sharded in L1 memory
bool TTNNLayoutAttr::hasShardedL1TensorMemoryLayout() const {
  return hasL1BufferType() &&
         (getMemLayout() == TensorMemoryLayout::HeightSharded ||
          getMemLayout() == TensorMemoryLayout::WidthSharded ||
          getMemLayout() == TensorMemoryLayout::BlockSharded);
}

// Check if the tensor memory layout is interleaved and in L1 memory
bool TTNNLayoutAttr::hasInterleavedL1TensorMemoryLayout() const {
  return hasL1BufferType() &&
         (getMemLayout() == TensorMemoryLayout::Interleaved);
}

// Check if the tensor memory layout is interleaved and in DRAM memory
bool TTNNLayoutAttr::hasInterleavedDRAMTensorMemoryLayout() const {
  return hasDRAMBufferType() &&
         (getMemLayout() == TensorMemoryLayout::Interleaved);
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

// Gets the size of shard in bytes
//
// This function returns the size of the shard in bytes.
// Size is calculated by multiplying shard shape with element size.
//
// return The size of the shard in bytes.
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
// Example: memref<2x3!tt.tile<32x32xf32>> -> { 2, 3 }
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
  for (unsigned i = 0; i < physicalMemoryMap.getNumSymbols(); ++i) {
    symReplacements.push_back(
        getAffineConstantExpr(shardShape[i], getContext()));
  }

  SmallVector<AffineExpr> dimReplacements;
  for (unsigned i = 0; i < physicalMemoryMap.getNumDims(); ++i) {
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
// return The new TTNNLayoutAttr with the given element type.
TTNNLayoutAttr TTNNLayoutAttr::withElementType(::mlir::MLIRContext *context,
                                               Type elementType) {
  return TTNNLayoutAttr::get(
      context, getLinear(), getGrid(),
      buildMemRef<BufferType, BufferTypeAttr>(context, getScalarShardShape(),
                                              elementType, getBufferType()),
      getMemLayout());
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
  return TTNNLayoutAttr::get(
      context, getLinear(), getGrid(),
      buildMemRef<BufferType, BufferTypeAttr>(context, getScalarShardShape(),
                                              getElementType(), memorySpace),
      getMemLayout());
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
  return TTNNLayoutAttr::get(
      context, getLinear(), getGrid(),
      buildMemRef<BufferType, BufferTypeAttr>(
          context, getScalarShardShape(), getElementType(), getBufferType()),
      memLayout);
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
    TensorMemoryLayout memLayout,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  // Construct a new affine map which will be used to map from logical
  // space to physical space
  AffineMap linear = collapsedLinearAffineMap(
      context, tensorShape, grid.getShape(), collapseIntervals);
  // Calculate shard shape by evaluating the linear map with last element
  // of the tensor shape and dividing it by the grid shape
  mlir::SmallVector<int64_t, 4> shardShape =
      calculateLogicalShardShape(tensorShape, linear, grid);
  // Build memref type with the given parameters
  MemRefType memRefType = buildMemRef<BufferType, BufferTypeAttr>(
      context, shardShape, elementType, bufferType);
  return get(context, linear, grid, memRefType, memLayout);
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
  return MemoryConfigAttr::get(context, getTensorMemoryLayout(),
                               BufferTypeAttr::get(context, bufferType),
                               getShardSpec());
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
  return MemoryConfigAttr::get(context,
                               TensorMemoryLayoutAttr::get(context, memLayout),
                               getBufferType(), getShardSpec());
}
