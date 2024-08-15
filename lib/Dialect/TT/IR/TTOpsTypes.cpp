// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <numeric>

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Target/Common/system_desc_generated.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ttmlir/Utils.h"

using namespace mlir::tt;

#include "ttmlir/Dialect/TT/IR/TTOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.cpp.inc"

mlir::tt::SystemDescAttr
mlir::tt::SystemDescAttr::getDefault(MLIRContext *context) {
  return tt::SystemDescAttr::get(
      context,
      // Chip Descriptors
      {
          tt::ChipDescAttr::get(
              context, tt::ArchAttr::get(context, tt::Arch::WormholeB0), {8, 8},
              1499136, 12, (1 << 30), 16, 32, 32,
              tt::ChipPhysicalCoresAttr::get(
                  context, tt::CoreCoordAttr::get(context, 0, 0),
                  tt::CoreCoordAttr::get(context, 0, 0),
                  tt::CoreCoordAttr::get(context, 0, 0),
                  tt::CoreCoordAttr::get(context, 0, 0))),
      },
      // Chip Descriptor Indices
      {
          0,
      },
      // Chip capabilities
      {
          tt::ChipCapabilityAttr::get(context,
                                      // NOLINTNEXTLINE
                                      tt::ChipCapability::PCIE |
                                          tt::ChipCapability::HostMMIO),
      },
      // Chip Mesh Coordinates
      {
          tt::ChipCoordAttr::get(context, 0, 0, 0, 0),
      },
      // Chip Channel Connections
      {});
}

mlir::tt::SystemDescAttr
mlir::tt::SystemDescAttr::getFromPath(MLIRContext *context, std::string &path) {
  // Check if file exists
  assert(!path.empty() && "cluster desc path must not be empty!");
  std::ifstream fbb(path, std::ios::binary | std::ios::ate);
  assert(fbb.good() && "cluster desc does not exist!");
  std::streampos size = fbb.tellg();
  fbb.seekg(0, std::ios::beg);
  auto buffer = std::shared_ptr<void>(std::malloc(size), std::free);
  fbb.read(static_cast<char *>(buffer.get()), size);

  // Read relevant information from binary
  auto binary_system_desc =
      ::tt::target::GetSizePrefixedSystemDescRoot(buffer.get())->system_desc();
  auto const *binary_chip_desc = binary_system_desc->chip_descs();
  auto const *binary_chip_desc_indices =
      binary_system_desc->chip_desc_indices();
  auto const *chip_capabilities = binary_system_desc->chip_capabilities();
  auto const *binary_chip_coords = binary_system_desc->chip_coords();

  // Acquire chip descs
  std::vector<tt::ChipDescAttr> chip_desc_list;
  for (auto element : *binary_chip_desc) {

    std::vector<tt::CoreCoordAttr> worker_cores, dram_cores, eth_cores,
        eth_inactive_cores;
    auto physical_cores = element->physical_cores();

    // Populate all vecrors with CoreCoordAttr instances
    for (auto const &core : *physical_cores->worker()) {
      worker_cores.emplace_back(
          tt::CoreCoordAttr::get(context, core->y(), core->x()));
    }
    for (auto const &core : *physical_cores->dram()) {
      dram_cores.emplace_back(
          tt::CoreCoordAttr::get(context, core->y(), core->x()));
    }
    for (auto const &core : *physical_cores->eth()) {
      eth_cores.emplace_back(
          tt::CoreCoordAttr::get(context, core->y(), core->x()));
    }
    for (auto const &core : *physical_cores->eth_inactive()) {
      eth_inactive_cores.emplace_back(
          tt::CoreCoordAttr::get(context, core->y(), core->x()));
    }
    // Create ChipPhysicalCoresAttr from the list of CoreCoordAttr instances
    auto chip_physical_cores_attr = tt::ChipPhysicalCoresAttr::get(
        context, worker_cores, dram_cores, eth_cores, eth_inactive_cores);

    auto current_chip_desc_attr = tt::ChipDescAttr::get(
        context, tt::ArchAttr::get(context, tt::Arch::WormholeB0),
        {element->grid_size()->y(), element->grid_size()->x()},
        element->l1_size(), element->num_dram_channels(),
        element->dram_channel_size(), element->noc_l1_address_align_bytes(),
        element->pcie_address_align_bytes(),
        element->noc_dram_address_align_bytes(), chip_physical_cores_attr);
    chip_desc_list.push_back(current_chip_desc_attr);
  }

  // Acquire chip indices
  std::vector<uint32_t> chip_indices_list;
  for (auto element : *binary_chip_desc_indices) {
    chip_indices_list.push_back(element);
  }

  // Acquire chip capabilities
  std::vector<tt::ChipCapabilityAttr> chip_capabilities_list;
  for (auto element : *chip_capabilities) {
    static_assert(
        static_cast<std::underlying_type_t<::tt::target::ChipCapability>>(
            ::mlir::tt::ChipCapability::PCIE) ==
        static_cast<std::underlying_type_t<::tt::target::ChipCapability>>(
            ::tt::target::ChipCapability::PCIE));
    static_assert(
        static_cast<std::underlying_type_t<::tt::target::ChipCapability>>(
            ::mlir::tt::ChipCapability::HostMMIO) ==
        static_cast<std::underlying_type_t<::tt::target::ChipCapability>>(
            ::tt::target::ChipCapability::HostMMIO));

    auto chip_capabilities_attr = tt::ChipCapabilityAttr::get(
        context, static_cast<::mlir::tt::ChipCapability>(element));
    chip_capabilities_list.push_back(chip_capabilities_attr);
  }

  // Acquire chip coordinates
  std::vector<tt::ChipCoordAttr> chip_coordinate_list;
  for (auto element : *binary_chip_coords) {
    auto chip_coordinate_attr = tt::ChipCoordAttr::get(
        context, element->rack(), element->shelf(), element->y(), element->x());
    chip_coordinate_list.push_back(chip_coordinate_attr);
  }

  // Generate system desc attribute
  auto system_desc_attr =
      tt::SystemDescAttr::get(context, chip_desc_list, chip_indices_list,
                              chip_capabilities_list, chip_coordinate_list, {});

  return system_desc_attr;
}

unsigned SystemDescAttr::getAddressAlignBytes(unsigned chipIndex) const {
  return std::max(std::initializer_list<unsigned>{
      getNocL1AddressAlignBytes(),
      getNocDRAMAddressAlignBytes(),
      getPcieAddressAlignBytes(),
  });
}

unsigned SystemDescAttr::getNocL1AddressAlignBytes(unsigned chipIndex) const {
  return getChipDescs()[chipIndex].getNocL1AddressAlignBytes();
}

unsigned SystemDescAttr::getNocDRAMAddressAlignBytes(unsigned chipIndex) const {
  return getChipDescs()[chipIndex].getNocDRAMAddressAlignBytes();
}

unsigned SystemDescAttr::getPcieAddressAlignBytes(unsigned chipIndex) const {
  return getChipDescs()[chipIndex].getPcieAddressAlignBytes();
}

static mlir::MemRefType buildMemRef(::mlir::MLIRContext *context,
                                    ::llvm::ArrayRef<int64_t> shardShape,
                                    ::mlir::Type elementType,
                                    MemorySpace memorySpace) {
  ::llvm::SmallVector<int64_t> scalarShardShape(shardShape);
  if (mlir::isa<TileType>(elementType)) {
    scalarShardShape =
        mlir::cast<TileType>(elementType).getTiledShape(scalarShardShape);
  }
  return mlir::MemRefType::get(
      scalarShardShape, elementType,
      mlir::AffineMap::getMultiDimIdentityMap(scalarShardShape.size(), context),
      MemorySpaceAttr::get(context, memorySpace));
}

//
// This function creates an affine map that represents collapsing the tensor
// dims onto an n-dimensional grid. E.g. (Where <> is some join operator)
//
//   - 3D tensor onto a 2D grid:
//       (d0, d1, d2) -> (d0 <> d1, d2)
//
//   - 4D tensor onto a 2D grid:
//       (d0, d1, d2, d3) -> (d0 <> d1 <> d2, d3)
//
// Note there are many ways we could collapse the above dims, by default we
// just collapse the interval [0, -1), which collapses dim0 up to but not
// including the last dim.  By using collapseIntervals we can achieve flexible
// collapsing of any set of consecutive dimension ranges.
//
//   - 4D tensor onto a 3D grid collapseIntervals=[(1, -1)]:
//       (d0, d1, d2, d3) -> (d0, d1 <> d2, d3)
//
//   - 4D tensor onto a 3D grid collapseIntervals=[(0, 2)]:
//       (d0, d1, d2, d3) -> (d0 <> d1, d2, d3)
//
//   - 7D tensor onto a 4D grid collapseIntervals=[(0, 3), (-3, -1)]:
//       (d0, d1, d2, d3, d4, d5, d6) -> (d0 <> d1 <> d2, d3, d4 <> d5, d6)
//
static mlir::AffineMap collapsedLinearAffineMap(
    ::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape,
    ::llvm::ArrayRef<int64_t> gridShape,
    ::llvm::ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  assert(shape.size() >= gridShape.size() && "shape must be >= gridShape");

  auto map = mlir::AffineMap::getMinorIdentityMap(shape.size(),
                                                  gridShape.size(), context);

  std::int64_t minimumDim = static_cast<std::int64_t>(shape.size());
  for (auto [begin, end] : collapseIntervals) {
    if (begin < 0) {
      begin += shape.size();
    }
    if (end < 0) {
      end += shape.size();
    }
    assert(end > 0);
    minimumDim = std::min(minimumDim, begin);
    auto collapsed = getAffineConstantExpr(0, context);
    int multiplier = 1;
    for (std::int64_t d = end - 1; d >= begin; --d) {
      collapsed = getAffineDimExpr(d, context) * multiplier + collapsed;
      multiplier *= shape[d];
    }
    map = map.dropResult(begin);
    map = map.insertResult(collapsed, begin);
  }

  // Fill in implicit lower dims
  for (std::int64_t d = 0; d < minimumDim; ++d) {
    map = map.dropResult(d);
    map = map.insertResult(getAffineDimExpr(d, context), d);
  }

  // Assert that all dims are represented on the RHS of the AffineMap
  for (std::size_t d = 0; d < shape.size(); ++d) {
    bool found = false;
    for (auto result : map.getResults()) {
      found |= result.isFunctionOfDim(d);
    }
    assert(found && "Dim does not participate in AffineMap RHS");
  }
  return map;
}

LayoutAttr LayoutAttr::get(
    ::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape,
    Type elementType, MemorySpace memorySpace, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals,
    OOBVal oobVal) {
  if (not grid) {
    grid = tensorShape.size() == 1 ? GridAttr::get(context, {1})
                                   : GridAttr::get(context, {1, 1});
  }

  SmallVector<AffineExpr> tensorShapeExprs(
      llvm::map_range(tensorShape, [context](std::int64_t e) {
        return getAffineConstantExpr(e - 1, context);
      }));
  auto linear = collapsedLinearAffineMap(context, tensorShape, grid.getShape(),
                                         collapseIntervals);
  SmallVector<std::int64_t> shardShape(linear.getNumResults());
  assert(linear.getNumResults() == grid.getShape().size());
  for (unsigned i = 0; i < linear.getNumResults(); ++i) {
    AffineExpr expr = linear.getResult(i);
    AffineExpr constantExpr = expr.replaceDims(tensorShapeExprs);
    std::int64_t constant =
        llvm::cast<AffineConstantExpr>(constantExpr).getValue() + 1;
    shardShape[i] = (constant + grid.getShape()[i] - 1) / grid.getShape()[i];
  }
  auto memref = buildMemRef(context, shardShape, elementType, memorySpace);
  return get(context, linear, oobVal, grid, memref);
}

LayoutAttr LayoutAttr::get(
    ::mlir::MLIRContext *context, RankedTensorType ty, MemorySpace memorySpace,
    GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals,
    OOBVal oobVal) {
  assert(ty);
  return get(context, ty.getShape(), ty.getElementType(), memorySpace, grid,
             collapseIntervals, oobVal);
}

LayoutAttr LayoutAttr::get(::mlir::MLIRContext *context, RankedTensorType ty,
                           MemorySpace memorySpace, Type elementType) {
  assert(ty);
  return get(context, ty.getShape(), elementType, memorySpace, {}, {{0, -1}},
             OOBVal::Undef);
}

// From the logical shape of the tensor and the affine map of the layout,
// compute the physical shape of the tensor, i.e the shape of the tensor
// after the dimensions have been collapsed onto a grid.
llvm::SmallVector<int64_t>
LayoutAttr::getPhysicalShape(ArrayRef<int64_t> logicalShape) const {
  llvm::SmallVector<int64_t> physicalShape(getGrid().getShape().size());
  SmallVector<AffineExpr> logicalShapeExprs(
      llvm::map_range(logicalShape, [context = getContext()](std::int64_t e) {
        return getAffineConstantExpr(e - 1, context);
      }));

  for (size_t i = 0; i < physicalShape.size(); i++) {
    AffineExpr expr = getLinear().getResult(i);
    AffineExpr constantExpr = expr.replaceDims(logicalShapeExprs);
    std::int64_t constant =
        llvm::cast<AffineConstantExpr>(constantExpr).getValue() + 1;
    physicalShape[i] = constant;
  }

  return physicalShape;
}

llvm::SmallVector<int64_t>
LayoutAttr::getStride(ArrayRef<int64_t> logicalShape) const {

  llvm::SmallVector<int64_t> stride(logicalShape.size());

  auto physicalShape = getPhysicalShape(logicalShape);

  // Origin point in the logical space (0, 0, ...)
  SmallVector<AffineExpr> originPoint(logicalShape.size(),
                                      getAffineConstantExpr(0, getContext()));

  auto linearMap = getLinear();
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
      auto newPoint = originPoint;
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

llvm::SmallVector<int64_t> LayoutAttr::getShardShape() const {
  SmallVector<int64_t> shardShape(getMemref().getShape());
  auto elementType = getElementType();
  if (mlir::isa<TileType>(elementType)) {
    return mlir::cast<TileType>(elementType).getScalarShape(shardShape);
  }
  return shardShape;
}

mlir::Type LayoutAttr::getElementType() const {
  return getMemref().getElementType();
}

bool LayoutAttr::isTiled() const {
  return ::mlir::isa<::mlir::tt::TileType>(getElementType());
}

uint64_t LayoutAttr::getElementSizeBytes() const {
  mlir::Type elementType = getElementType();
  if (mlir::isa<TileType>(elementType)) {
    auto tileType = mlir::cast<TileType>(elementType);
    return tileType.getSizeBytes();
  }
  return elementType.getIntOrFloatBitWidth() / 8;
}

LayoutAttr LayoutAttr::withGrid(
    ::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  return get(context, tensorShape, getElementType(), getMemorySpace(), grid,
             collapseIntervals, getOobVal());
}

LayoutAttr LayoutAttr::withGrid(
    ::mlir::MLIRContext *context, RankedTensorType ty, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  assert(ty);
  return LayoutAttr::withGrid(context, ty.getShape(), grid, collapseIntervals);
}

LayoutAttr LayoutAttr::withElementType(::mlir::MLIRContext *context,
                                       Type elementType) {
  return LayoutAttr::get(
      context, getLinear(), getOobVal(), getGrid(),
      buildMemRef(context, getShardShape(), elementType, getMemorySpace()));
}

MemorySpace LayoutAttr::getMemorySpace() const {
  return mlir::cast<mlir::tt::MemorySpaceAttr>(getMemref().getMemorySpace())
      .getValue();
}

DeviceAttr DeviceAttr::get(::mlir::MLIRContext *context,
                           SystemDescAttr systemDesc,
                           ArrayRef<unsigned> chipIds) {
  assert(not chipIds.empty() && "expected at least one chip");
  ChipDescAttr chipDesc = systemDesc.getChipDescs()[chipIds.front()];
  ArrayRef<int64_t> physicalGrid(chipDesc.getGrid());
  assert(physicalGrid.size() == 2 && "expected 2D grid");
  for (unsigned chipId : chipIds) {
    ChipDescAttr chip = systemDesc.getChipDescs()[chipId];
    if (chip.getGrid() != physicalGrid) {
      llvm::report_fatal_error("all chips must have the same grid shape");
    }
  }

  SmallVector<int64_t> virtualGrid(physicalGrid);
  if (chipIds.size() > 1) {
    virtualGrid.insert(virtualGrid.begin(), chipIds.size());
  }
  assert(virtualGrid.size() >= 2 && "expected at least 2D grid");

  // auto c0 = getAffineConstantExpr(physicalGrid[0], context);
  // auto c1 = getAffineConstantExpr(physicalGrid[1], context);
  auto dZ = getAffineConstantExpr(0, context);
  auto dY = getAffineDimExpr(virtualGrid.size() - 2, context);
  auto dX = getAffineDimExpr(virtualGrid.size() - 1, context);

  SmallVector<AffineExpr> gridExprs = {dZ, dY, dX};
  auto gridMap = AffineMap::get(virtualGrid.size(), 0, gridExprs, context);

  return get(context, GridAttr::get(context, virtualGrid, gridMap), chipIds);
}

DeviceAttr DeviceAttr::get(::mlir::MLIRContext *context,
                           SystemDescAttr systemDesc, bool enableMultichip) {
  assert(systemDesc.getChipDescIndices().size() > 0 &&
         "expected at least one chip");
  SmallVector<unsigned> chipIds(
      enableMultichip ? systemDesc.getChipDescIndices().size() : 1);
  std::iota(chipIds.begin(), chipIds.end(), 0);
  return get(context, systemDesc, chipIds);
}

::mlir::LogicalResult
DeviceAttr::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                   ::mlir::tt::GridAttr grid,
                   ::llvm::ArrayRef<unsigned> chipIds) {
  if (chipIds.empty()) {
    emitError() << "expected at least one chip";
    return ::mlir::failure();
  }

  auto gridShape = grid.getShape();
  for (auto dim : gridShape) {
    if (dim <= 0) {
      emitError() << "expected positive grid dimensions";
      return ::mlir::failure();
    }
  }

  auto physicalGridMapping = grid.getMapping();
  if (physicalGridMapping.getNumResults() != 3) {
    emitError() << "expected physical grid mapping to have 3 results";
    return ::mlir::failure();
  }

  return ::mlir::success();
}

::mlir::LogicalResult
TileType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                 ArrayRef<int64_t> shape, DataType dataType) {
  if (shape.size() != 2) {
    emitError() << "expected 2D shape";
    return ::mlir::failure();
  }
  return ::mlir::success();
}

TileType TileType::get(::mlir::MLIRContext *context, Type elementType,
                       ArrayRef<int64_t> shape) {
  return get(context, shape, elementTypeToDataType(elementType));
}

llvm::SmallVector<int64_t>
TileType::getScalarShape(SmallVector<int64_t> tiledShape) const {
  assert(tiledShape.size() >= 2 && "expected at least 2D shape");
  tiledShape[tiledShape.size() - 2] *= getHeight();
  tiledShape[tiledShape.size() - 1] *= getWidth();
  return tiledShape;
}

llvm::SmallVector<int64_t>
TileType::getTiledShape(SmallVector<int64_t> scalarShape) const {
  assert(scalarShape.size() >= 2 && "expected at least 2D shape");
  scalarShape[scalarShape.size() - 2] =
      (scalarShape[scalarShape.size() - 2] + getHeight() - 1) / getHeight();
  scalarShape[scalarShape.size() - 1] =
      (scalarShape[scalarShape.size() - 1] + getWidth() - 1) / getWidth();
  return scalarShape;
}

uint64_t TileType::getSizeBytes() const {
  switch (getDataType()) {
  case DataType::Float32:
    return getHeight() * getWidth() * 4;
  case DataType::Float16:
    return getHeight() * getWidth() * 2;
  case DataType::BFloat16:
    return getHeight() * getWidth() * 2;
  case DataType::BFP_Float8:
    assert(getHeight() == 32 && getWidth() == 32);
    return 1024;
  case DataType::BFP_BFloat8:
    assert(getHeight() == 32 && getWidth() == 32);
    return 1024;
  case DataType::BFP_Float4:
    assert(getHeight() == 32 && getWidth() == 32);
    return 512;
  case DataType::BFP_BFloat4:
    assert(getHeight() == 32 && getWidth() == 32);
    return 512;
  case DataType::BFP_Float2:
    assert(getHeight() == 32 && getWidth() == 32);
    return 256;
  case DataType::BFP_BFloat2:
    assert(getHeight() == 32 && getWidth() == 32);
    return 256;
  case DataType::UInt32:
    return getHeight() * getWidth() * 4;
  case DataType::UInt16:
    return getHeight() * getWidth() * 2;
  case DataType::UInt8:
    return getHeight() * getWidth();
  }
}

SystemDescAttr mlir::tt::getCurrentScopeSystemDesc(mlir::Operation *op) {
  // Walk up scope levels until we find the top level ModuleOp which carries the
  // system desc
  while (op) {
    if (mlir::isa<mlir::ModuleOp>(op)) {
      auto systemDesc = op->getAttrOfType<SystemDescAttr>(SystemDescAttr::name);
      assert(systemDesc && "expected system desc to be present on the module");
      return systemDesc;
    }
    op = op->getParentOp();
  }
  assert(false && "expected system desc to be present in the scope");
  return nullptr;
}

DeviceAttr mlir::tt::getCurrentScopeDevice(mlir::Operation *op) {
  while (op) {
    if (auto device = op->getAttrOfType<DeviceAttr>(DeviceAttr::name)) {
      return device;
    }
    op = op->getParentOp();
  }
  return nullptr;
}

void TTDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.cpp.inc"
      >();
}
