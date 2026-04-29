// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/Conv2dConfigParams.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

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
                llvm::ArrayRef<int64_t> gridShape, BufferType bufferType) {
  if (isL1BufferType(bufferType)) {
    return llvm::success();
  }

  llvm::SmallVector<int64_t> expectedGridShape({1, 1});
  if (llvm::equal(gridShape, expectedGridShape)) {
    return llvm::success();
  }
  return emitError() << "expected (" << expectedGridShape
                     << ") grid shape for non-L1 buffer type, got ("
                     << gridShape << ") for " << stringifyBufferType(bufferType)
                     << " buffer type";
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

llvm::LogicalResult TTNNLayoutAttr::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, AffineMap,
    llvm::ArrayRef<int64_t> gridShape, MemRefType memref,
    TensorMemoryLayoutAttr memLayout,
    mlir::tt::ttcore::TensorMeshAttr tensorMesh, bool ignorePhysicalLayout,
    CoreRangeSetAttr coreRangeSet) {
  BufferType bufferType =
      mlir::cast<BufferTypeAttr>(memref.getMemorySpace()).getValue();

  llvm::LogicalResult status = ::llvm::success();
  if (llvm::failed(verifyGridShape(emitError, gridShape, bufferType))) {
    status = llvm::failure();
  }
  if (llvm::failed(
          verifyBufferAndMemoryLayout(emitError, bufferType, memLayout))) {
    status = llvm::failure();
  }

  // CRS / memLayout consistency:
  //   sharded memLayout      ⇒ non-null core_range_set
  //   non-sharded / no memLayout ⇒ null core_range_set
  // `ignorePhysicalLayout=true` opts out of the "sharded ⇒ non-null"
  // direction (the layout is intentionally a placeholder with unspecified
  // shard placement); the inverse direction is still enforced.
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
  std::optional<ShardSpecAttr> shardSpec = std::nullopt;
  if (tensorMemoryLayout &&
      isShardedMemoryLayout(tensorMemoryLayout.getValue())) {
    shardSpec = ShardSpecAttr::get(layoutAttr.getContext(), layoutAttr);
  }
  return MemoryConfigAttr::get(layoutAttr.getContext(), tensorMemoryLayout,
                               bufferTypeAttr, shardSpec);
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

// Build a CoreRangeSet covering `numCores` L1 cores laid out row-major on a
// worker grid of width `workerWidth`, coalesced into at most two rectangles:
//   - one `workerWidth x fullRows` block for full rows,
//   - one `tail x 1` strip for the remainder.
// This is the canonical row-major flattening for H/W-sharded tensors: shard m
// lands on physical core (m / W, m % W).
static llvm::SmallVector<CoreRangeAttr>
buildRowMajorCoreRanges(mlir::MLIRContext *ctx, int64_t numCores,
                        int64_t workerWidth) {
  llvm::SmallVector<CoreRangeAttr> ranges;
  int64_t fullRows = numCores / workerWidth;
  int64_t tail = numCores % workerWidth;
  if (fullRows > 0) {
    ranges.push_back(CoreRangeAttr::get(
        ctx, CoreCoordAttr::get(ctx, 0, 0),
        CoreCoordAttr::get(ctx, workerWidth - 1, fullRows - 1)));
  }
  if (tail > 0) {
    ranges.push_back(
        CoreRangeAttr::get(ctx, CoreCoordAttr::get(ctx, 0, fullRows),
                           CoreCoordAttr::get(ctx, tail - 1, fullRows)));
  }
  return ranges;
}

// Static helper to compute the canonical CoreRangeSet for a sharded layout
// by mapping the virtual `gridShape` onto physical cores per the
// `memLayout` flatten rule.
CoreRangeSetAttr TTNNLayoutAttr::computeCanonicalCoreRangeSet(
    mlir::MLIRContext *ctx, TensorMemoryLayout memLayout,
    ArrayRef<int64_t> gridShape, mlir::tt::ttcore::GridAttr deviceGrid) {
  assert(isShardedMemoryLayout(memLayout) &&
         "CoreRangeSet can only be derived for sharded memory layouts");
  assert(gridShape.size() == 2 &&
         "TTNNLayoutAttr shard grid must be 2D for L1 sharding");
  assert(deviceGrid &&
         "deviceGrid is required to derive the canonical CoreRangeSet for a "
         "sharded TTNN layout");

  llvm::ArrayRef<int64_t> deviceGridShape = deviceGrid.getShape();
  assert(deviceGridShape.size() == 2 && "device worker grid must be 2D");
  int64_t workerWidth = deviceGridShape[1];

  llvm::SmallVector<CoreRangeAttr> ranges;
  switch (memLayout) {
  case TensorMemoryLayout::BlockSharded: {
    // Virtual [H, W] maps identity onto physical cores (0,0)-(W-1, H-1).
    ranges.push_back(CoreRangeAttr::get(
        ctx, CoreCoordAttr::get(ctx, 0, 0),
        CoreCoordAttr::get(ctx, gridShape[1] - 1, gridShape[0] - 1)));
    break;
  }
  case TensorMemoryLayout::HeightSharded: {
    // Virtual [M, 1] row-major flattens onto (m / W, m % W).
    assert(gridShape[1] == 1 && "HeightSharded expects [M, 1] shard grid");
    ranges = buildRowMajorCoreRanges(ctx, gridShape[0], workerWidth);
    break;
  }
  case TensorMemoryLayout::WidthSharded: {
    // Virtual [1, M] row-major flattens onto (m / W, m % W).
    assert(gridShape[0] == 1 && "WidthSharded expects [1, M] shard grid");
    ranges = buildRowMajorCoreRanges(ctx, gridShape[1], workerWidth);
    break;
  }
  default:
    llvm_unreachable("unexpected sharded memory layout");
  }

  return CoreRangeSetAttr::get(ctx, ranges);
}

// Static helper to compute the CoreRangeSet for a sharded layout
// with an explicitly specified `gridShape`
// (corresponds to the legacy `exactGrid` mechanism).
// `gridShape` is interpreted as a physical core grid shape.
CoreRangeSetAttr
TTNNLayoutAttr::computeExactCoreRangeSet(mlir::MLIRContext *ctx,
                                         ArrayRef<int64_t> gridShape) {
  // gridShape is already the physical worker-grid footprint. Emit a single
  // rectangle at origin sized to the grid.
  assert(gridShape.size() == 2 &&
         "TTNNLayoutAttr shard grid must be 2D for L1 sharding");
  return CoreRangeSetAttr::get(
      ctx, CoreRangeAttr::get(
               ctx, CoreCoordAttr::get(ctx, 0, 0),
               CoreCoordAttr::get(ctx, gridShape[1] - 1, gridShape[0] - 1)));
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
                   mlir::tt::ttnn::DataMovementKernelAttr>(kernel)) {
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
                   mlir::tt::ttnn::KernelArgSemaphoreAtAttr>(arg)) {
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

namespace {

// Calculate the logical shape of the shard for sharded memory layouts.
//
// All examples assume that the tensor is mapped to a 8x8 grid.
// Example: tensor<32x32xbf16> -> {4, 4}
// Example: tensor<65x65xbf16> -> {9, 9}
llvm::SmallVector<int64_t>
calculateLogicalShardShapeForSharding(mlir::ArrayRef<int64_t> tensorShape,
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

// Calculate the logical shape of the shard for L1-Interleaved memory layout.
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
llvm::SmallVector<int64_t> calculateLogicalShardShapeForL1Interleaved(
    mlir::ArrayRef<int64_t> tensorShape, mlir::Type elementType,
    mlir::AffineMap linear, llvm::ArrayRef<int64_t> gridShape) {
  assert(linear.getNumResults() == gridShape.size());

  mlir::SmallVector<std::int64_t> physicalShape =
      ttmlir::utils::evalShape(linear, tensorShape);

  uint64_t numOfGridUnits = std::accumulate(gridShape.begin(), gridShape.end(),
                                            1, std::multiplies<std::int64_t>());

  mlir::SmallVector<std::int64_t> shardShape;
  shardShape.resize(gridShape.size() - 1, 1);

  if (!mlir::isa<mlir::tt::ttcore::TileType>(elementType)) {
    int64_t tensorVolume =
        std::accumulate(physicalShape.begin(), physicalShape.end(), 1,
                        std::multiplies<int64_t>());
    int64_t shardVolume = (tensorVolume + numOfGridUnits - 1) / numOfGridUnits;
    shardShape.push_back(shardVolume);
    return shardShape;
  }

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

llvm::SmallVector<int64_t> deriveShardShape(ArrayRef<int64_t> physicalShape,
                                            Type elementType, AffineMap linear,
                                            ArrayRef<int64_t> gridShape,
                                            BufferType bufferType,
                                            TensorMemoryLayoutAttr memLayout) {
  if (bufferType == BufferType::L1 && memLayout &&
      memLayout.getValue() == TensorMemoryLayout::Interleaved) {
    return calculateLogicalShardShapeForL1Interleaved(
        physicalShape, elementType, linear, gridShape);
  }
  return calculateLogicalShardShapeForSharding(physicalShape, linear,
                                               gridShape);
}

} // namespace

TTNNLayoutAttr::Builder::Builder(TTNNLayoutAttr layout)
    : context(layout.getContext()),
      collapseIntervals(1, layout.getDefaultCollapseIntervals()),
      elementType(layout.getElementType()), bufferType(layout.getBufferType()),
      gridShape(layout.getGridShape().begin(), layout.getGridShape().end()),
      memLayout(layout.getMemLayout()), tensorMesh(layout.getTensorMesh()),
      ignorePhysicalLayout(layout.getIgnorePhysicalLayout()),
      coreRangeSet(layout.getCoreRangeSet()) {}

TTNNLayoutAttr::Builder::Builder(RankedTensorType type)
    : Builder(mlir::cast<TTNNLayoutAttr>(type.getEncoding())) {
  setTensorShape(type.getShape());
}

TTNNLayoutAttr::Builder::Builder(MLIRContext *ctx,
                                 ArrayRef<int64_t> tensorShapeIn,
                                 Type elementTypeIn)
    : context(ctx), tensorShape(tensorShapeIn.begin(), tensorShapeIn.end()),
      collapseIntervals(1, std::pair<std::int64_t, std::int64_t>{0, -1}),
      elementType(elementTypeIn), bufferType(BufferType::DRAM), gridShape{1, 1},
      memLayout(
          TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::Interleaved)),
      tensorMesh(nullptr), ignorePhysicalLayout(false), coreRangeSet(nullptr) {}

TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setElementType(Type newElementType) {
  elementType = newElementType;
  return *this;
}

TTNNLayoutAttr::Builder &TTNNLayoutAttr::Builder::setLayout(Layout layout) {
  elementType = utils::getElementType(
      context, layout, mlir::tt::ttcore::getDataType(elementType));
  return *this;
}

TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setTensorShape(ArrayRef<int64_t> newTensorShape) {
  tensorShape.assign(newTensorShape.begin(), newTensorShape.end());
  return *this;
}

TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setTensorMesh(ttcore::TensorMeshAttr mesh) {
  tensorMesh = mesh;
  return *this;
}

TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setIgnorePhysicalLayout(bool ignore) {
  ignorePhysicalLayout = ignore;
  return *this;
}

TTNNLayoutAttr::Builder &TTNNLayoutAttr::Builder::setCollapseIntervals(
    ArrayRef<std::pair<std::int64_t, std::int64_t>> newCollapseIntervals) {
  collapseIntervals.assign(newCollapseIntervals.begin(),
                           newCollapseIntervals.end());
  return *this;
}

TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setBufferType(BufferType newBufferType) {
  bufferType = newBufferType;
  if (newBufferType == BufferType::SystemMemory) {
    memLayout = TensorMemoryLayoutAttr{};
    gridShape = {1, 1};
  } else if (newBufferType == BufferType::DRAM) {
    memLayout =
        TensorMemoryLayoutAttr::get(context, TensorMemoryLayout::Interleaved);
    gridShape = {1, 1};
  } else if (newBufferType == BufferType::L1 && !memLayout) {
    memLayout =
        TensorMemoryLayoutAttr::get(context, TensorMemoryLayout::Interleaved);
  }
  coreRangeSet = nullptr;
  return *this;
}

TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setMemoryLayout(TensorMemoryLayoutAttr ml) {
  if (ml) {
    int64_t volume = std::accumulate(gridShape.begin(), gridShape.end(),
                                     int64_t{1}, std::multiplies<>());
    switch (ml.getValue()) {
    case TensorMemoryLayout::HeightSharded:
      gridShape = {volume, 1};
      break;
    case TensorMemoryLayout::WidthSharded:
      gridShape = {1, volume};
      break;
    case TensorMemoryLayout::BlockSharded:
    case TensorMemoryLayout::Interleaved:
    case TensorMemoryLayout::NDSharded:
      break;
    }
  }
  memLayout = ml;
  coreRangeSet = nullptr;
  return *this;
}

TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setMemoryLayout(TensorMemoryLayout ml) {
  return setMemoryLayout(TensorMemoryLayoutAttr::get(context, ml));
}

TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setGridShape(ArrayRef<int64_t> newGridShape) {
  gridShape.assign(newGridShape.begin(), newGridShape.end());
  coreRangeSet = nullptr;
  return *this;
}

TTNNLayoutAttr::Builder &
TTNNLayoutAttr::Builder::setCoreRangeSet(CoreRangeSetAttr crs) {
  coreRangeSet = crs;
  return *this;
}

TTNNLayoutAttr TTNNLayoutAttr::Builder::build() {
  bool resultIsSharded =
      memLayout && isShardedMemoryLayout(memLayout.getValue());

  if (resultIsSharded) {
    assert(coreRangeSet &&
           "coreRangeSet must be set for sharded memory layouts. Use "
           "buildWithCanonicalCorePlacement, buildWithExactCorePlacement, or "
           "set coreRangeSet explicitly before calling build().");
  }

  llvm::SmallVector<int64_t, 4> physicalShape(tensorShape.begin(),
                                              tensorShape.end());
  if (mlir::isa<mlir::tt::ttcore::TileType>(elementType)) {
    physicalShape = utils::getTilePaddedShape(tensorShape);
  }
  AffineMap linear = mlir::tt::ttcore::collapsedLinearAffineMap(
      context, physicalShape, gridShape, collapseIntervals);

  llvm::SmallVector<int64_t> shardShape = deriveShardShape(
      physicalShape, elementType, linear, gridShape, bufferType, memLayout);

  MemRefType memref = mlir::tt::ttcore::buildMemRef<BufferType, BufferTypeAttr>(
      context, shardShape, elementType, bufferType);

  return TTNNLayoutAttr::get(context, linear, gridShape, memref, memLayout,
                             tensorMesh, ignorePhysicalLayout, coreRangeSet);
}

TTNNLayoutAttr TTNNLayoutAttr::Builder::buildWithCanonicalCorePlacement(
    ttcore::DeviceAttr deviceAttr) {
  if (memLayout && isShardedMemoryLayout(memLayout.getValue())) {
    coreRangeSet = TTNNLayoutAttr::computeCanonicalCoreRangeSet(
        context, memLayout.getValue(), gridShape, deviceAttr.getWorkerGrid());
  }

  return build();
}

TTNNLayoutAttr TTNNLayoutAttr::Builder::buildWithExactCorePlacement() {
  if (memLayout && isShardedMemoryLayout(memLayout.getValue())) {
    coreRangeSet = TTNNLayoutAttr::computeExactCoreRangeSet(context, gridShape);
  }

  return build();
}

BufferType TTNNLayoutAttr::Builder::getBufferType() const { return bufferType; }

TensorMemoryLayoutAttr TTNNLayoutAttr::Builder::getMemLayout() const {
  return memLayout;
}

ArrayRef<int64_t> TTNNLayoutAttr::Builder::getGridShape() const {
  return gridShape;
}

Type TTNNLayoutAttr::Builder::getElementType() const { return elementType; }

} // namespace mlir::tt::ttnn
