// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/Conv2dConfigParams.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Builders.h"

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

static const llvm::SmallVector<std::pair<std::int64_t, std::int64_t>>
    g_defaultCollapseIntervals = {std::pair<std::int64_t, std::int64_t>{0, -1}};

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
// 1. L1 buffers allow any grid shape (Interleaved spreads across the worker
//    grid; sharded layouts shard across the worker grid).
// 2. DRAM buffers allow any grid shape only for sharded layouts (the grid is
//    interpreted against the device's DRAM bank grid).  DRAM-Interleaved must
//    use a unit grid.
// 3. SystemMemory must use a unit grid.
llvm::LogicalResult
verifyGridShape(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                llvm::ArrayRef<int64_t> gridShape, BufferType bufferType,
                TensorMemoryLayoutAttr memLayoutAttr) {
  bool sharded =
      memLayoutAttr && isShardedMemoryLayout(memLayoutAttr.getValue());
  if (isL1BufferType(bufferType) || (isDRAMBufferType(bufferType) && sharded)) {
    return llvm::success();
  }

  llvm::SmallVector<int64_t> expectedGridShape({1, 1});
  if (llvm::equal(gridShape, expectedGridShape)) {
    return llvm::success();
  }
  return emitError() << "expected (" << expectedGridShape << ") grid shape for "
                     << stringifyBufferType(bufferType) << " buffer type"
                     << (isDRAMBufferType(bufferType) ? " with interleaved "
                                                        "memory layout"
                                                      : "")
                     << ", got (" << gridShape << ")";
}

// Checks:
// 1. If memory layout is present then:
//   - System memory buffer type is not allowed
//   - DRAM buffer type doesn't support BlockSharded layout
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
        memLayoutAttr.getValue() == TensorMemoryLayout::BlockSharded) {
      return emitError()
             << "BlockSharded layout is not supported for DRAM "
                "buffer type; use WidthSharded, HeightSharded or Interleaved";
    }
  } else if (bufferType != BufferType::SystemMemory) {
    return emitError()
           << "Memory layout is required for non-SystemMemory buffer type.";
  }

  return ::llvm::success();
}

// Checks:
// 1. If shard spec is present then:
//   - Buffer type must be a device buffer (L1 or DRAM)
//   - Tensor memory layout must be sharded: HeightSharded, WidthSharded,
//   BlockSharded
//   - DRAM buffer type only supports HeightSharded / WidthSharded layouts.
llvm::LogicalResult
verifySharding(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
               BufferType bufferType, TensorMemoryLayoutAttr memLayoutAttr,
               std::optional<ShardSpecAttr> shardSpec) {
  if (shardSpec && *shardSpec) {
    if (bufferType != BufferType::L1 && bufferType != BufferType::DRAM) {
      return emitError() << "Sharding is only valid for L1 or DRAM buffer type";
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

    if (bufferType == BufferType::DRAM &&
        memLayoutAttr.getValue() == TensorMemoryLayout::BlockSharded) {
      return emitError() << "BlockSharded layout is not supported for DRAM "
                            "buffer type; use WidthSharded or HeightSharded";
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
static llvm::SmallVector<int64_t>
calculateLogicalShardShapeForSharding(llvm::ArrayRef<int64_t> tensorShape,
                                      mlir::AffineMap linear,
                                      llvm::ArrayRef<int64_t> gridShape) {
  assert(linear.getNumResults() == gridShape.size());
  mlir::SmallVector<std::int64_t> physicalShape =
      ttmlir::utils::evalShape(linear, tensorShape);
  mlir::SmallVector<std::int64_t> shardShape(linear.getNumResults());
  for (size_t i = 0; i < linear.getNumResults(); ++i) {
    shardShape[i] = (physicalShape[i] + gridShape[i] - 1) / gridShape[i];
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
static llvm::SmallVector<int64_t> calculateLogicalShardShapeForL1Interleaved(
    llvm::ArrayRef<int64_t> tensorShape, mlir::Type elementType,
    mlir::AffineMap linear, llvm::ArrayRef<int64_t> gridShape) {
  assert(linear.getNumResults() == gridShape.size());

  mlir::SmallVector<std::int64_t> physicalShape =
      ttmlir::utils::evalShape(linear, tensorShape);

  uint64_t numOfGridUnits = ttmlir::utils::volume(gridShape);

  // Create shard shape with all dims set to 1 except the last one which is
  // set to shardVolume.
  mlir::SmallVector<std::int64_t> shardShape;
  shardShape.resize(gridShape.size() - 1, 1);

  if (!mlir::isa<mlir::tt::ttcore::TileType>(elementType)) {
    // If RowMajor, single shard should be TensorVolume / NumCores rounded up.
    // So in case of tensor<5120x1024xbf16> on 8x8 grid we have
    // 5120*1024/64 = 81920 -> shard shape is <1x81920xbf16>
    int64_t tensorVolume = ttmlir::utils::volume(physicalShape);
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

TTNNLayoutAttr
TTNNLayoutAttr::withIgnorePhysicalLayout(bool ignorePhysicalLayout) const {
  return TTNNLayoutAttr::get(getContext(), getLinear(), getGridShape(),
                             getMemref(), getMemLayout(), getTensorMesh(),
                             ignorePhysicalLayout, getCoreRangeSet());
}

llvm::LogicalResult TTNNLayoutAttr::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, AffineMap,
    llvm::ArrayRef<int64_t> gridShape, MemRefType memref,
    TensorMemoryLayoutAttr memLayout,
    mlir::tt::ttcore::TensorMeshAttr tensorMesh, bool ignorePhysicalLayout,
    CoreRangeSetAttr coreRangeSet) {
  BufferType bufferType =
      mlir::cast<BufferTypeAttr>(memref.getMemorySpace()).getValue();

  llvm::LogicalResult status = ::llvm::success();
  if (llvm::failed(
          verifyGridShape(emitError, gridShape, bufferType, memLayout))) {
    status = llvm::failure();
  }
  if (llvm::failed(
          verifyBufferAndMemoryLayout(emitError, bufferType, memLayout))) {
    status = llvm::failure();
  }

  // CRS / memLayout consistency:
  //   sharded memLayout => non-null core_range_set
  //   non-sharded / no memLayout => null core_range_set
  //
  // `ignorePhysicalLayout=true` opts out of the "sharded => non-null"
  // direction (the layout is intentionally a placeholder with unspecified
  // shard placement).
  bool isSharded = memLayout && isShardedMemoryLayout(memLayout.getValue());
  if (isSharded && !coreRangeSet && !ignorePhysicalLayout) {
    emitError() << "sharded TTNN layout ("
                << stringifyTensorMemoryLayout(memLayout.getValue())
                << ") must carry a core_range_set";
    status = llvm::failure();
  }
  if (!isSharded && coreRangeSet) {
    emitError()
        << "non-sharded TTNN layout must not carry a core_range_set (got "
        << (memLayout ? stringifyTensorMemoryLayout(memLayout.getValue())
                      : "no memory layout")
        << ")";
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
MemoryConfigAttr MemoryConfigAttr::get(TTNNLayoutAttr layoutAttr) {
  BufferTypeAttr bufferTypeAttr =
      mlir::cast<BufferTypeAttr>(layoutAttr.getMemref().getMemorySpace());
  TensorMemoryLayoutAttr tensorMemoryLayout = layoutAttr.getMemLayout();
  // A layout with `ignorePhysicalLayout` set models a sharded layout with
  // an unspecified shard shape; leave shardSpec unset so consumers get a
  // partial MemoryConfig and the backend can pick the physical layout.
  std::optional<ShardSpecAttr> shardSpec = std::nullopt;
  if (tensorMemoryLayout &&
      isShardedMemoryLayout(tensorMemoryLayout.getValue()) &&
      !layoutAttr.getIgnorePhysicalLayout()) {
    shardSpec = ShardSpecAttr::get(layoutAttr.getContext(), layoutAttr);
  }
  return MemoryConfigAttr::get(layoutAttr.getContext(), tensorMemoryLayout,
                               bufferTypeAttr, shardSpec);
}

MemoryConfigAttr MemoryConfigAttr::get(TTNNNDLayoutAttr layoutAttr) {
  return MemoryConfigAttr::get(
      layoutAttr.getContext(), layoutAttr.getMemLayout(),
      mlir::cast<BufferTypeAttr>(layoutAttr.getMemref().getMemorySpace()),
      /*shardSpec=*/std::nullopt, utils::createNDShardSpecIfNeeded(layoutAttr));
}

MemoryConfigAttr MemoryConfigAttr::get(
    ::mlir::MLIRContext *context, TensorMemoryLayoutAttr tensorMemoryLayout,
    BufferTypeAttr bufferType, std::optional<ShardSpecAttr> shardSpec) {
  return MemoryConfigAttr::get(context, tensorMemoryLayout, bufferType,
                               shardSpec, /*ndShardSpec=*/std::nullopt);
}

// Verify memory config attribute
::llvm::LogicalResult MemoryConfigAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    TensorMemoryLayoutAttr tensorMemoryLayout, BufferTypeAttr bufferType,
    std::optional<ShardSpecAttr> shardSpec,
    std::optional<NDShardSpecAttr> ndShardSpec) {

  if (ndShardSpec && shardSpec) {
    return emitError() << "Setting both NDShardSpecAttr and ShardSpecAttr is "
                          "not supported.";
  }
  // Verify buffer type, memory layout and sharding
  return ::llvm::success(verifyBufferAndMemoryLayout(emitError,
                                                     bufferType.getValue(),
                                                     tensorMemoryLayout)
                             .succeeded() &&
                         verifySharding(emitError, bufferType.getValue(),
                                        tensorMemoryLayout, shardSpec)
                             .succeeded());
}

// Manually parse MemoryConfigAttr to avoid various issues with the tablegen
// parser in dealing with multiple optional parameters. See PR #6512 for more
// details.
mlir::Attribute MemoryConfigAttr::parse(::mlir::AsmParser &parser,
                                        ::mlir::Type type) {
  ::llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLess()) {
    return {};
  }

  BufferTypeAttr bufferType;
  if (parser.parseCustomAttributeWithFallback(bufferType)) {
    return {};
  }

  TensorMemoryLayoutAttr tensorMemoryLayout;
  std::optional<ShardSpecAttr> shardSpec;
  std::optional<NDShardSpecAttr> ndShardSpec;

  while (parser.parseOptionalComma().succeeded()) {
    // Attempt to parse the optional param as a shardSpecAttr.
    ShardSpecAttr maybeShardSpec;
    OptionalParseResult shardSpecResult =
        parser.parseOptionalAttribute(maybeShardSpec);
    if (shardSpecResult.has_value()) {
      if (succeeded(*shardSpecResult)) {
        shardSpec = maybeShardSpec;
        continue;
      }
    }

    // Attempt to parse the optional param as a tensorMemoryLayoutAttr.
    TensorMemoryLayoutAttr tml;
    if (succeeded(parser.parseCustomAttributeWithFallback(tml))) {
      tensorMemoryLayout = tml;
      continue;
    }

    // Only attempt to parse the param as an NDShardSpecAttr if the keyword is
    // explicitly specified.
    if (parser.parseOptionalKeyword("ndShardSpec").succeeded()) {
      if (parser.parseEqual()) {
        return {};
      }
      NDShardSpecAttr attr;
      if (parser.parseCustomAttributeWithFallback(attr)) {
        return {};
      }
      ndShardSpec = attr;
      continue;
    }

    return {};
  }

  if (parser.parseGreater()) {
    return {};
  }

  // getChecked outputs verification errors instead of just crashing.
  return MemoryConfigAttr::getChecked(
      [loc, &parser]() { return parser.emitError(loc); }, parser.getContext(),
      tensorMemoryLayout, bufferType, shardSpec, ndShardSpec);
}

void MemoryConfigAttr::print(::mlir::AsmPrinter &p) const {
  p << "<";
  p.printStrippedAttrOrType(getBufferType());
  if (getTensorMemoryLayout()) {
    p << ", ";
    p.printStrippedAttrOrType(getTensorMemoryLayout());
  }
  if (getShardSpec()) {
    p << ", ";
    p.printStrippedAttrOrType(getShardSpec());
  } else if (getNdShardSpec()) {
    p << ", ndShardSpec = ";
    p.printStrippedAttrOrType(getNdShardSpec());
  }
  p << ">";
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

std::optional<CoreRangeAttr> CoreRangeSetAttr::getBoundingBox() const {
  llvm::ArrayRef<CoreRangeAttr> coreRanges = getCoreRanges();
  if (coreRanges.empty()) {
    return std::nullopt;
  }

  CoreCoordAttr firstStart = coreRanges.front().getStartCoord();
  CoreCoordAttr firstEnd = coreRanges.front().getEndCoord();
  uint64_t minStartX = firstStart.getX();
  uint64_t minStartY = firstStart.getY();
  uint64_t maxEndX = firstEnd.getX();
  uint64_t maxEndY = firstEnd.getY();
  for (CoreRangeAttr range : coreRanges.drop_front()) {
    minStartX = std::min(minStartX, range.getStartCoord().getX());
    minStartY = std::min(minStartY, range.getStartCoord().getY());
    maxEndX = std::max(maxEndX, range.getEndCoord().getX());
    maxEndY = std::max(maxEndY, range.getEndCoord().getY());
  }

  MLIRContext *context = getContext();
  return CoreRangeAttr::get(context,
                            CoreCoordAttr::get(context, minStartX, minStartY),
                            CoreCoordAttr::get(context, maxEndX, maxEndY));
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
                               /*enableKernelStrideFolding=*/nullptr,
                               /*configTensorsInDram=*/nullptr);
}

// Returns default configuration.
Conv2dConfigAttr Conv2dConfigAttr::getDefault(::mlir::MLIRContext *context) {
  Conv2dConfigAttr convConfig = get(context);
  return Conv2dConfigParams(convConfig, /*partial=*/false)
      .buildConv2dConfigAttr(context);
}

Conv2dConfigAttr
Conv2dConfigAttr::withActivation(UnaryOpType unaryOpType) const {
  Conv2dConfigParams params(*this);
  params.activation = unaryOpType;
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

Conv2dConfigAttr
Conv2dConfigAttr::withEnableKernelStrideFolding(bool value) const {
  Conv2dConfigParams params(*this);
  params.enableKernelStrideFolding = value;
  return params.buildConv2dConfigAttr(getContext());
}

Conv2dConfigAttr Conv2dConfigAttr::withConfigTensorsInDram(bool value) const {
  Conv2dConfigParams params(*this);
  params.configTensorsInDram = value;
  return params.buildConv2dConfigAttr(getContext());
}

bool Conv2dConfigAttr::hasActivation() const {
  return getActivation() != nullptr;
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

bool Conv2dConfigAttr::hasEnableKernelStrideFolding() const {
  return getEnableKernelStrideFolding() != nullptr;
}

bool Conv2dConfigAttr::hasConfigTensorsInDram() const {
  return getConfigTensorsInDram() != nullptr;
}

NDShardSpecAttr NDShardSpecAttr::get(TTNNNDLayoutAttr layout) {
  auto shardGrid = layout.getGrid();
  auto *context = layout.getContext();
  auto coreRangeSetAttr = CoreRangeSetAttr::get(
      context, CoreRangeAttr::get(
                   context, CoreCoordAttr::get(context, 0, 0),
                   CoreCoordAttr::get(context, shardGrid.getShape()[1] - 1,
                                      shardGrid.getShape()[0] - 1)));
  return NDShardSpecAttr::get(
      context, coreRangeSetAttr,
      ShapeAttr::get(context, layout.getScalarShardShape()),
      layout.getShardOrientation(), layout.getShardDistributionStrategy());
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
                   mlir::tt::ttnn::WriteKernelAttr,
                   mlir::tt::ttnn::DataMovementKernelAttr,
                   mlir::tt::ttnn::SourceComputeKernelAttr,
                   mlir::tt::ttnn::SourceReadKernelAttr,
                   mlir::tt::ttnn::SourceWriteKernelAttr>(kernel)) {
      return emitError() << "Unexpected kernel";
    }
  }

  return ::llvm::success();
}

::llvm::LogicalResult MeshProgramAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    MeshRangeAttr meshRange, ProgramAttr program) {
  return ::llvm::success();
}

::llvm::LogicalResult MeshProgramDescriptorAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<MeshProgramAttr> meshPrograms,
    ttcore::FabricConnectionConfigAttr fabricConnectionConfig) {
  return ::llvm::success();
}

::llvm::LogicalResult KernelCBAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    uint32_t totalSize, CoreRangeSetAttr coreRanges,
    llvm::ArrayRef<mlir::tt::ttnn::KernelCBFormatAttr> formats,
    mlir::tt::ttnn::KernelCBGlobalBufferAddressOfTensorAttr buffer) {
  return ::llvm::success();
}

::llvm::LogicalResult KernelCBFormatAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    uint32_t bufferIndex, ttcore::DataType dtype, uint32_t pageSize) {
  return ::llvm::success();
}

::llvm::LogicalResult KernelSemaphoreAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, uint32_t id,
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
                   mlir::tt::ttnn::KernelArgSemaphoreAtAttr,
                   mlir::tt::ttnn::KernelArgGlobalSemaphoreAttr,
                   mlir::tt::ttnn::KernelArgScalarAttr,
                   mlir::tt::ttnn::KernelArgNamedArgAttr>(arg)) {
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
                   mlir::tt::ttnn::KernelArgScalarAttr,
                   mlir::tt::ttnn::KernelArgSemaphoreAtAttr,
                   mlir::tt::ttnn::KernelArgTensorAccessorArgsAttr>(arg)) {
      return emitError() << "Unexpected compile time argument";
    }
  }

  return ::llvm::success();
}

::llvm::LogicalResult verifyRuntimeArgs(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<mlir::tt::ttnn::CoreRuntimeArgsAttr> rtArgs) {
  // Check for duplicate CoreCoords.
  llvm::DenseSet<std::pair<uint64_t, uint64_t>> seenCoords;
  for (CoreRuntimeArgsAttr rtArg : rtArgs) {
    CoreCoordAttr coord = rtArg.getCoreCoord();
    auto coordPair = std::make_pair(coord.getX(), coord.getY());
    if (!seenCoords.insert(coordPair).second) {
      return emitError() << "Duplicate CoreCoord (" << coord.getX() << ", "
                         << coord.getY() << ") in runtime arguments";
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
    ::llvm::ArrayRef<mlir::tt::ttnn::CoreRuntimeArgsAttr> rtArgs,
    ::llvm::ArrayRef<mlir::Attribute> ctArgs) {
  if (failed(verifyCommonRuntimeArgs(emitError, commonRtArgs)) ||
      failed(verifyRuntimeArgs(emitError, rtArgs)) ||
      failed(verifyCompileTimeArgs(emitError, ctArgs))) {
    return ::llvm::failure();
  }

  return ::llvm::success();
}

::llvm::LogicalResult ReadKernelAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::SymbolRefAttr symbolRef, CoreRangeSetAttr coreRanges,
    ::llvm::ArrayRef<mlir::Attribute> commonRtArgs,
    ::llvm::ArrayRef<mlir::tt::ttnn::CoreRuntimeArgsAttr> rtArgs,
    ::llvm::ArrayRef<mlir::Attribute> ctArgs) {
  if (failed(verifyCommonRuntimeArgs(emitError, commonRtArgs)) ||
      failed(verifyRuntimeArgs(emitError, rtArgs)) ||
      failed(verifyCompileTimeArgs(emitError, ctArgs))) {
    return ::llvm::failure();
  }

  return ::llvm::success();
}

::llvm::LogicalResult WriteKernelAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::SymbolRefAttr symbolRef, CoreRangeSetAttr coreRanges,
    ::llvm::ArrayRef<mlir::Attribute> commonRtArgs,
    ::llvm::ArrayRef<mlir::tt::ttnn::CoreRuntimeArgsAttr> rtArgs,
    ::llvm::ArrayRef<mlir::Attribute> ctArgs) {
  if (failed(verifyCommonRuntimeArgs(emitError, commonRtArgs)) ||
      failed(verifyRuntimeArgs(emitError, rtArgs)) ||
      failed(verifyCompileTimeArgs(emitError, ctArgs))) {
    return ::llvm::failure();
  }

  return ::llvm::success();
}

static ::llvm::LogicalResult verifyInlineKernelSource(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::StringAttr source) {
  if (!source || source.getValue().empty()) {
    return emitError() << "inline-source kernel requires a non-empty `source`";
  }
  return ::llvm::success();
}

::llvm::LogicalResult SourceComputeKernelAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::StringAttr source, ::mlir::tt::ttnn::CoreRangeSetAttr coreRanges,
    ComputeKernelMathFidelity mathFidelity, bool fp32DestAccEn,
    bool dstFullSyncEn,
    ::llvm::ArrayRef<ComputeKernelUnpackToDestMode> unpackToDestModes,
    bool bfp8PackPrecise, bool mathApproxMode,
    ::llvm::ArrayRef<mlir::Attribute> commonRtArgs,
    ::llvm::ArrayRef<mlir::tt::ttnn::CoreRuntimeArgsAttr> rtArgs,
    ::llvm::ArrayRef<mlir::Attribute> ctArgs) {
  if (failed(verifyInlineKernelSource(emitError, source)) ||
      failed(verifyCommonRuntimeArgs(emitError, commonRtArgs)) ||
      failed(verifyRuntimeArgs(emitError, rtArgs)) ||
      failed(verifyCompileTimeArgs(emitError, ctArgs))) {
    return ::llvm::failure();
  }
  return ::llvm::success();
}

::llvm::LogicalResult SourceReadKernelAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::StringAttr source, CoreRangeSetAttr coreRanges,
    ::llvm::ArrayRef<mlir::Attribute> commonRtArgs,
    ::llvm::ArrayRef<mlir::tt::ttnn::CoreRuntimeArgsAttr> rtArgs,
    ::llvm::ArrayRef<mlir::Attribute> ctArgs) {
  if (failed(verifyInlineKernelSource(emitError, source)) ||
      failed(verifyCommonRuntimeArgs(emitError, commonRtArgs)) ||
      failed(verifyRuntimeArgs(emitError, rtArgs)) ||
      failed(verifyCompileTimeArgs(emitError, ctArgs))) {
    return ::llvm::failure();
  }
  return ::llvm::success();
}

::llvm::LogicalResult SourceWriteKernelAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::StringAttr source, CoreRangeSetAttr coreRanges,
    ::llvm::ArrayRef<mlir::Attribute> commonRtArgs,
    ::llvm::ArrayRef<mlir::tt::ttnn::CoreRuntimeArgsAttr> rtArgs,
    ::llvm::ArrayRef<mlir::Attribute> ctArgs) {
  if (failed(verifyInlineKernelSource(emitError, source)) ||
      failed(verifyCommonRuntimeArgs(emitError, commonRtArgs)) ||
      failed(verifyRuntimeArgs(emitError, rtArgs)) ||
      failed(verifyCompileTimeArgs(emitError, ctArgs))) {
    return ::llvm::failure();
  }
  return ::llvm::success();
}

::llvm::LogicalResult DataMovementKernelAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    SymbolRefAttr symbolRef, CoreRangeSetAttr coreRanges,
    DataMovementProcessor processor, ttcore::NocIndex nocIndex, NocMode nocMode,
    llvm::ArrayRef<mlir::Attribute> commonRtArgs,
    llvm::ArrayRef<CoreRuntimeArgsAttr> rtArgs,
    llvm::ArrayRef<mlir::Attribute> ctArgs) {
  if (nocMode == NocMode::DynamicNoc) {
    return emitError() << "dynamic noc mode is not supported";
  }
  if (failed(verifyCommonRuntimeArgs(emitError, commonRtArgs)) ||
      failed(verifyRuntimeArgs(emitError, rtArgs)) ||
      failed(verifyCompileTimeArgs(emitError, ctArgs))) {
    return ::llvm::failure();
  }
  return ::llvm::success();
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
llvm::SmallVector<int64_t> TTNNNDLayoutAttr::getScalarShardShape() const {
  SmallVector<int64_t> shardShape(getMemref().getShape());
  if (isTiled()) {
    return mlir::cast<mlir::tt::ttcore::TileType>(getMemref().getElementType())
        .getScalarShape(shardShape);
  }

  return shardShape;
}

bool TTNNNDLayoutAttr::isTiled() const {
  return ::mlir::isa<::mlir::tt::ttcore::TileType>(
      getMemref().getElementType());
}

BufferType TTNNNDLayoutAttr::getBufferType() const {
  return mlir::cast<BufferTypeAttr>(getMemref().getMemorySpace()).getValue();
}

Layout TTNNNDLayoutAttr::getLayout() const {
  return isTiled() ? Layout::Tile : Layout::RowMajor;
}

mlir::tt::ttcore::DataType TTNNNDLayoutAttr::getDataType() const {
  Type elementType = getMemref().getElementType();
  if (isTiled()) {
    mlir::tt::ttcore::TileType tileType =
        mlir::cast<mlir::tt::ttcore::TileType>(elementType);
    return tileType.getDataType();
  }

  return mlir::tt::ttcore::elementTypeToDataType(elementType);
}

bool TTNNNDLayoutAttr::isInterleaved() const {
  return getMemLayout().getValue() == TensorMemoryLayout::Interleaved;
}

bool TTNNNDLayoutAttr::isSharded() const {
  return getMemLayout().getValue() != TensorMemoryLayout::Interleaved;
}

mlir::Type TTNNNDLayoutAttr::getElementType() const {
  return getMemref().getElementType();
}

//===----------------------------------------------------------------------===//
// TTNNLayoutAttr::Builder
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {

// Helper to derive the logical shard shape for a sharded/L1-interleaved layout.
//
static llvm::SmallVector<int64_t>
deriveShardShape(ArrayRef<int64_t> physicalShape, Type elementType,
                 AffineMap linear, ArrayRef<int64_t> gridShape,
                 BufferType bufferType, TensorMemoryLayoutAttr memLayout) {
  // SystemMemory and DRAM-Interleaved: shard shape == full physical shape
  // (no per-core split).
  bool isSharded = memLayout && isShardedMemoryLayout(memLayout.getValue());
  if (!isSharded && !isL1BufferType(bufferType)) {
    return ttmlir::utils::evalShape(linear, physicalShape);
  }

  TT_assertv(memLayout, "memLayout must be set for device buffers");

  switch (memLayout.getValue()) {
  case TensorMemoryLayout::Interleaved:
    // Note: the term "shard shape" is a bit of a misnomer for L1-interleaved
    // layouts, since the layout isn't "sharded" in the traditional sense, but
    // the rest of the codebase expects a "logical shard shape" to be defined
    // for L1-interleaved layouts as a per-core view of the physical shape.
    //
    return calculateLogicalShardShapeForL1Interleaved(
        physicalShape, elementType, linear, gridShape);

  case TensorMemoryLayout::BlockSharded:
  case TensorMemoryLayout::HeightSharded:
  case TensorMemoryLayout::WidthSharded:
    return calculateLogicalShardShapeForSharding(physicalShape, linear,
                                                 gridShape);
  default:
    llvm_unreachable("unexpected memory layout");
  }
}

// Helper to build a CoreRangeSet covering `numCores` L1 cores laid out
// row-major on a worker grid of shape `gridSize` ([H, W]), coalesced into at
// most two rectangles:
//   - one `W x fullRows` block for full rows,
//   - one `tail x 1` strip for the remainder.
//
static llvm::SmallVector<CoreRangeAttr>
buildRowMajorCoreRanges(mlir::MLIRContext *ctx, int64_t numCores,
                        ArrayRef<int64_t> gridSize) {
  TT_assertv(gridSize.size() == 2U, "expected 2D worker grid");
  const int64_t workerGridWidth = gridSize[1];
  llvm::SmallVector<CoreRangeAttr> ranges;
  int64_t fullRows = numCores / workerGridWidth;
  int64_t tail = numCores % workerGridWidth;
  if (fullRows > 0) {
    ranges.push_back(CoreRangeAttr::get(
        ctx, CoreCoordAttr::get(ctx, 0, 0),
        CoreCoordAttr::get(ctx, workerGridWidth - 1, fullRows - 1)));
  }
  if (tail > 0) {
    ranges.push_back(
        CoreRangeAttr::get(ctx, CoreCoordAttr::get(ctx, 0, fullRows),
                           CoreCoordAttr::get(ctx, tail - 1, fullRows)));
  }
  return ranges;
}

// Helper to derive the canonical CoreRangeSet for an L1-sharded layout
// by mapping the virtual `gridShape` onto physical worker cores per the
// `memLayout` flatten rule.
//
static CoreRangeSetAttr deriveCanonicalL1CoreRangeSet(
    mlir::MLIRContext *ctx, TensorMemoryLayout memLayout,
    ArrayRef<int64_t> gridShape, mlir::tt::ttcore::GridAttr workerGrid) {
  TT_assertv(isShardedMemoryLayout(memLayout),
             "CoreRangeSet can only be derived for sharded memory layouts");

  TT_assertv(gridShape.size() == 2U,
             "TTNNLayoutAttr shard grid must be 2D for L1 sharding");

  TT_assertv(workerGrid,
             "workerGrid is required to derive the canonical CoreRangeSet for "
             "an L1-sharded TTNN layout");

  llvm::ArrayRef<int64_t> workerGridShape = workerGrid.getShape();

  TT_assertv(workerGridShape.size() == 2U, "device worker grid must be 2D");

  int64_t workerGridVolume = ttmlir::utils::volume(workerGridShape);

  llvm::SmallVector<CoreRangeAttr> ranges;
  switch (memLayout) {
  case TensorMemoryLayout::BlockSharded: {
    // Virtual [H, W] maps identity onto physical cores (0,0)-(W-1, H-1).
    //
    TT_assertv((gridShape[0] <= workerGridShape[0] &&
                gridShape[1] <= workerGridShape[1]),
               "BlockSharded shard grid [{0}, {1}] does not fit in worker "
               "grid [{2}, {3}]",
               gridShape[0], gridShape[1], workerGridShape[0],
               workerGridShape[1]);
    ranges.push_back(CoreRangeAttr::get(
        ctx, CoreCoordAttr::get(ctx, 0, 0),
        CoreCoordAttr::get(ctx, gridShape[1] - 1, gridShape[0] - 1)));
    break;
  }
  case TensorMemoryLayout::HeightSharded: {
    // Virtual [M, 1] row-major flattens onto (m / W, m % W).
    //
    TT_assertv(gridShape[1] == 1U, "HeightSharded expects [M, 1] shard grid");
    TT_assertv(gridShape[0] <= workerGridVolume,
               "HeightSharded shard count {0} exceeds worker grid volume {1}",
               gridShape[0], workerGridVolume);
    ranges = buildRowMajorCoreRanges(ctx, gridShape[0], workerGridShape);
    break;
  }
  case TensorMemoryLayout::WidthSharded: {
    // Virtual [1, M] row-major flattens onto (m / W, m % W).
    //
    TT_assertv(gridShape[0] == 1U, "WidthSharded expects [1, M] shard grid");
    TT_assertv(gridShape[1] <= workerGridVolume,
               "WidthSharded shard count {0} exceeds worker grid volume {1}",
               gridShape[1], workerGridVolume);
    ranges = buildRowMajorCoreRanges(ctx, gridShape[1], workerGridShape);
    break;
  }
  default:
    llvm_unreachable("unexpected sharded memory layout");
  }

  return CoreRangeSetAttr::get(ctx, ranges);
}

// Helper to derive the canonical CoreRangeSet for a DRAM-sharded layout:
// a single rectangle covering the first `volume(gridShape)` DRAM banks of
// the device's `dramGrid`.
//
static CoreRangeSetAttr deriveCanonicalDramCoreRangeSet(
    mlir::MLIRContext *ctx, TensorMemoryLayout memLayout,
    ArrayRef<int64_t> gridShape, mlir::tt::ttcore::GridAttr dramGrid) {
  TT_assertv(dramGrid,
             "dramGrid is required to derive the canonical CoreRangeSet for a "
             "DRAM-sharded TTNN layout");
  TT_assertv((memLayout == TensorMemoryLayout::HeightSharded ||
              memLayout == TensorMemoryLayout::WidthSharded),
             "DRAM-sharded layout only supports HeightSharded / WidthSharded; "
             "got {0}",
             stringifyTensorMemoryLayout(memLayout));

  llvm::ArrayRef<int64_t> dramGridShape = dramGrid.getShape();
  TT_assertv(dramGridShape.size() == 2U, "device DRAM grid must be 2D");
  TT_assertv(dramGridShape[0] == 1,
             "device DRAM grid is expected to be a single row [1, N]; got "
             "[{0}, {1}]",
             dramGridShape[0], dramGridShape[1]);

  int64_t shardVolume = ttmlir::utils::volume(gridShape);
  int64_t dramVolume = ttmlir::utils::volume(dramGridShape);
  TT_assertv(shardVolume <= dramVolume,
             "DRAM-sharded layout cannot exceed available DRAM banks; got "
             "shard grid volume {0} vs dram grid volume {1}",
             shardVolume, dramVolume);
  TT_assertv(shardVolume >= 1,
             "DRAM-sharded layout must use at least one DRAM bank");

  llvm::SmallVector<CoreRangeAttr> ranges{
      CoreRangeAttr::get(ctx, CoreCoordAttr::get(ctx, 0, 0),
                         CoreCoordAttr::get(ctx, shardVolume - 1, 0))};
  return CoreRangeSetAttr::get(ctx, ranges);
}

// Constuctor which initializes the builder with the minimum required
// information to build a TTNNLayoutAttr instance: the MLIR context, the logical
// tensor shape, and the element type.
//
// Defaults to DRAM Interleaved layout with an unit grid shape.
//
// Used for building a TTNNLayoutAttr from scratch, without an existing layout
// to copy/inherit from.
//
//
// @param context The MLIR context in which the TTNNLayoutAttr will be created.
// @param tensorShapeIn The logical shape of the tensor for the layout being
// constructed.
// @param elementTypeIn The element type of the tensor for the layout being
// constructed.
//
TTNNLayoutAttr::Builder::Builder(MLIRContext *context,
                                 ArrayRef<int64_t> tensorShapeIn,
                                 Type elementTypeIn)
    : ctx(context), tensorShape(tensorShapeIn.begin(), tensorShapeIn.end()),
      collapseIntervals(g_defaultCollapseIntervals), elementType(elementTypeIn),
      bufferType(BufferType::DRAM),
      memLayout(TensorMemoryLayoutAttr::get(context,
                                            TensorMemoryLayout::Interleaved)),
      gridShape{1, 1} {}

// Constructor which initializes the builder with an existing
// TTNNLayoutAttr instance and the tensor shape corresponding to the new layout
// to be built.
//
// The provided TTNNLayoutAttr serves as a template for the new
// layout, allowing the builder to inherit properties such as memory layout,
// buffer type, and core range set, while the tensor shape can be customized for
// the new layout being constructed.
//
// @param layout An existing TTNNLayoutAttr instance to serve as a template for
// the new layout being built.
// @param tensorShapeIn The logical shape of the tensor for the new layout being
// constructed.
//
TTNNLayoutAttr::Builder::Builder(TTNNLayoutAttr layout,
                                 ArrayRef<int64_t> tensorShapeIn)
    : ctx(layout.getContext()),
      tensorShape(tensorShapeIn.begin(), tensorShapeIn.end()),
      collapseIntervals(g_defaultCollapseIntervals),
      elementType(layout.getElementType()), bufferType(layout.getBufferType()),
      memLayout(layout.getMemLayout()),
      gridShape(layout.getGridShape().begin(), layout.getGridShape().end()),
      coreRangeSet(layout.getCoreRangeSet()),
      tensorMesh(layout.getTensorMesh()),
      ignorePhysicalLayout(layout.getIgnorePhysicalLayout()) {}

// Convenience constructor to build from an existing TTNNLayoutAttr-encoded
// RankedTensorType.
//
// @param type A RankedTensorType with a TTNNLayoutAttr encoding, from which
// the builder will inherit shape and encoding.
//
TTNNLayoutAttr::Builder::Builder(RankedTensorType type)
    : Builder(mlir::cast<TTNNLayoutAttr>(type.getEncoding()), type.getShape()) {
}

// Set the element type for the layout being built.
//
TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setElementType(Type elementTypeIn) {
  elementType = elementTypeIn;
  return *this;
}

// Set the layout flavor (Tile vs RowMajor) for the layout being built.
//
// This method updates the element type to match the requested layout flavor,
// while preserving the underlying data type.
//
TTNNLayoutAttr::Builder &TTNNLayoutAttr::Builder::setLayout(Layout layout) {
  elementType = utils::getElementType(
      ctx, layout, mlir::tt::ttcore::getDataType(elementType));
  return *this;
}

// Set the collapse intervals for the layout being built.
//
TTNNLayoutAttr::Builder &TTNNLayoutAttr::Builder::setCollapseIntervals(
    ArrayRef<std::pair<std::int64_t, std::int64_t>> intervals) {
  collapseIntervals.assign(intervals.begin(), intervals.end());
  return *this;
}

// Set the buffer type for the layout being built.
//
TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setBufferType(BufferType bufferTypeIn) {
  if (bufferTypeIn == bufferType) {
    return *this;
  }

  bufferType = bufferTypeIn;

  // CoreRangeSet is invalidated when the buffer type changes.
  //
  coreRangeSet = nullptr;

  return *this;
}

// Set the memory layout for the layout being built.
//
TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setMemoryLayout(TensorMemoryLayoutAttr ml) {
  if (ml == memLayout) {
    return *this;
  }

  memLayout = ml;

  // CoreRangeSet is invalidated when the memory layout changes.
  //
  coreRangeSet = nullptr;

  return *this;
}

// Overload of setMemoryLayout to accept a TensorMemoryLayout enum directly.
//
TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setMemoryLayout(TensorMemoryLayout ml) {
  return setMemoryLayout(TensorMemoryLayoutAttr::get(ctx, ml));
}

// Set the logical grid shape for the layout being built.
//
// The logical grid shape defines how the tensor is partitioned across cores for
// sharded / L1-interleaved layouts.
//
TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setGridShape(ArrayRef<int64_t> gridShapeIn) {
  if (llvm::ArrayRef<int64_t>(gridShapeIn) ==
      llvm::ArrayRef<int64_t>(gridShape)) {
    return *this;
  }

  gridShape.assign(gridShapeIn.begin(), gridShapeIn.end());

  // CoreRangeSet is invalidated when the grid shape changes.
  //
  coreRangeSet = nullptr;

  return *this;
}

// Set the tensor mesh for the layout being built.
//
TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setTensorMesh(ttcore::TensorMeshAttr mesh) {
  tensorMesh = mesh;
  return *this;
}

// Set an explicit CoreRangeSet for the layout being built.
//
// Required for sharded layouts.
//
// If not set explicitly, it can be either:
// - Inherited from an existing TTNNLayoutAttr, or
// - Computed canonically using buildWithCanonicalCorePlacement
//
TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setCoreRangeSet(CoreRangeSetAttr crs) {
  coreRangeSet = crs;
  return *this;
}

// Set whether to ignore physical layout constraints when building the layout.
//
TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setIgnorePhysicalLayout(bool ignore) {
  ignorePhysicalLayout = ignore;
  return *this;
}

// Terminator method which builds the TTNNLayoutAttr instance based on the
// current state of the builder.
//
TTNNLayoutAttr TTNNLayoutAttr::Builder::build() {
  TT_assertv(elementType, "elementType must be set on the Builder.");

  // Coerce buffer-type-driven invariants before validation, so callers can
  // construct a layout from a builder seeded with an unrelated source layout
  // without having to manually drop the inherited memLayout / gridShape:
  //   - SystemMemory has no associated memory layout and is always 1x1.
  //   - DRAM-Interleaved must use a unit grid (the verifier rejects anything
  //     else).
  if (bufferType == BufferType::SystemMemory) {
    memLayout = TensorMemoryLayoutAttr{};
    gridShape = {1, 1};
  } else if (isDRAMBufferType(bufferType) && memLayout &&
             memLayout.getValue() == TensorMemoryLayout::Interleaved) {
    gridShape = {1, 1};
  }

  if (memLayout && isShardedMemoryLayout(memLayout.getValue())) {
    TT_assertv(gridShape.size() != 0U,
               "gridShape must be set for sharded memory layouts.");

    TT_assertv(coreRangeSet != nullptr,
               "coreRangeSet must be set for sharded memory layouts. "
               "Use buildWithCanonicalCorePlacement, "
               "or provide coreRangeSet explicitly before calling build().");
  }

  if (isL1BufferType(bufferType) && memLayout &&
      memLayout.getValue() == TensorMemoryLayout::Interleaved) {
    TT_assertv(gridShape.size() != 0U,
               "gridShape must be set for L1 buffers with Interleaved memory "
               "layout.");
  }

  llvm::SmallVector<int64_t, 4> physicalShape(tensorShape.begin(),
                                              tensorShape.end());
  if (mlir::isa<mlir::tt::ttcore::TileType>(elementType)) {
    physicalShape = utils::getTilePaddedShape(tensorShape);
  }

  AffineMap linear = mlir::tt::ttcore::collapsedLinearAffineMap(
      ctx, physicalShape, gridShape, collapseIntervals);

  llvm::SmallVector<int64_t> shardShape = deriveShardShape(
      physicalShape, elementType, linear, gridShape, bufferType, memLayout);

  MemRefType memref = mlir::tt::ttcore::buildMemRef<BufferType, BufferTypeAttr>(
      ctx, shardShape, elementType, bufferType);

  return TTNNLayoutAttr::get(ctx, linear, gridShape, memref, memLayout,
                             tensorMesh, ignorePhysicalLayout, coreRangeSet);
}

// Terminator method to build a TTNNLayoutAttr with a canonical CoreRangeSet:
//   - L1-sharded: rectangles derived from the device worker grid by flattening
//     `gridShape` per the memory-layout-specific rule.
//   - DRAM-sharded: a single rectangle spanning all DRAM banks of the device's
//     `dramGrid`.
//
TTNNLayoutAttr TTNNLayoutAttr::Builder::buildWithCanonicalCorePlacement(
    ttcore::DeviceAttr deviceAttr) {
  if (!coreRangeSet && memLayout &&
      isShardedMemoryLayout(memLayout.getValue())) {
    if (isDRAMBufferType(bufferType)) {
      coreRangeSet = deriveCanonicalDramCoreRangeSet(
          ctx, memLayout.getValue(), gridShape, deviceAttr.getDramGrid());
    } else if (isL1BufferType(bufferType)) {
      coreRangeSet = deriveCanonicalL1CoreRangeSet(
          ctx, memLayout.getValue(), gridShape, deviceAttr.getWorkerGrid());
    }
  }

  return build();
}

} // namespace mlir::tt::ttnn
