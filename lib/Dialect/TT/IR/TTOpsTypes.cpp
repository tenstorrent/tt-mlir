// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Target/Common/Target.h"
#include "ttmlir/Utils.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <fstream>
#include <numeric>

using namespace mlir::tt;

#include "ttmlir/Dialect/TT/IR/TTOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.cpp.inc"

unsigned mlir::tt::ChipDescAttr::getScratchL1RegionSize() const {
  // 4KB is the default size for the scratch L1 region.
  constexpr uint32_t kScratchL1RegionSize = 1 << 12;
  return kScratchL1RegionSize;
}

unsigned mlir::tt::ChipDescAttr::getScratchL1RegionAddress() const {
  return getL1Size() - getScratchL1RegionSize();
}

mlir::tt::SystemDescAttr
mlir::tt::SystemDescAttr::getDefault(MLIRContext *context) {
  // Populate a dummy n150
  SmallVector<std::int64_t> gridShape = {8, 8};

  // populate a placeholder for supported tile sizes
  SmallVector<tt::DataTypeAttr> supported_data_types;
  supported_data_types.push_back(
      tt::DataTypeAttr::get(context, tt::DataType::Float32));
  supported_data_types.push_back(
      tt::DataTypeAttr::get(context, tt::DataType::Float16));
  supported_data_types.push_back(
      tt::DataTypeAttr::get(context, tt::DataType::BFloat16));
  supported_data_types.push_back(
      tt::DataTypeAttr::get(context, tt::DataType::BFP_Float8));
  supported_data_types.push_back(
      tt::DataTypeAttr::get(context, tt::DataType::BFP_BFloat8));
  supported_data_types.push_back(
      tt::DataTypeAttr::get(context, tt::DataType::BFP_Float4));
  supported_data_types.push_back(
      tt::DataTypeAttr::get(context, tt::DataType::BFP_BFloat4));
  supported_data_types.push_back(
      tt::DataTypeAttr::get(context, tt::DataType::BFP_Float2));
  supported_data_types.push_back(
      tt::DataTypeAttr::get(context, tt::DataType::BFP_BFloat2));
  supported_data_types.push_back(
      tt::DataTypeAttr::get(context, tt::DataType::UInt32));
  supported_data_types.push_back(
      tt::DataTypeAttr::get(context, tt::DataType::UInt16));
  supported_data_types.push_back(
      tt::DataTypeAttr::get(context, tt::DataType::UInt8));

  // populate a placeholder for supported tile sizes
  SmallVector<tt::TileSizeAttr> supported_tile_sizes;
  supported_tile_sizes.push_back(tt::TileSizeAttr::get(context, 4, 16));
  supported_tile_sizes.push_back(tt::TileSizeAttr::get(context, 16, 16));
  supported_tile_sizes.push_back(tt::TileSizeAttr::get(context, 32, 16));
  supported_tile_sizes.push_back(tt::TileSizeAttr::get(context, 4, 32));
  supported_tile_sizes.push_back(tt::TileSizeAttr::get(context, 16, 32));
  supported_tile_sizes.push_back(tt::TileSizeAttr::get(context, 32, 32));

  SmallVector<CoreCoordAttr> workerCores;
  workerCores.reserve(gridShape[0] * gridShape[1]);
  for (std::int64_t y = 0; y < gridShape[0]; ++y) {
    for (std::int64_t x = 0; x < gridShape[1]; ++x) {
      workerCores.push_back(CoreCoordAttr::get(context, y, x));
    }
  }
  SmallVector<CoreCoordAttr> dramCores;
  for (std::int64_t x = 0; x < 4; ++x) {
    for (std::int64_t y = 0; y < 3; ++y) {
      dramCores.push_back(CoreCoordAttr::get(context, y + gridShape[0], x));
    }
  }
  return tt::SystemDescAttr::get(
      context,
      // CPU Descriptors
      {tt::CPUDescAttr::get(
          context, tt::CPURole::Host,
          mlir::StringAttr::get(context, "x86_64-pc-linux-gnu"))},
      // Chip Descriptors
      {
          tt::ChipDescAttr::get(
              context, tt::ArchAttr::get(context, tt::Arch::WormholeB0),
              gridShape, 1499136, 12, (1 << 30), 16, 32, 32, 1024, 1024, 1024,
              (1 << 30),
              tt::ChipPhysicalCoresAttr::get(context, workerCores, dramCores,
                                             {}, {}),
              supported_data_types, supported_tile_sizes, 32),
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
  assert(!path.empty() && "system desc path must not be empty!");
  std::ifstream fbb(path, std::ios::binary | std::ios::ate);
  assert(fbb.good() && "system desc does not exist!");
  std::streampos size = fbb.tellg();
  fbb.seekg(0, std::ios::beg);
  auto buffer = std::shared_ptr<void>(std::malloc(size), std::free);
  fbb.read(static_cast<char *>(buffer.get()), size);

  // Read relevant information from binary
  auto const *binary_system_desc =
      ::tt::target::GetSizePrefixedSystemDescRoot(buffer.get())->system_desc();
  auto const *binary_cpu_desc = binary_system_desc->cpu_descs();
  auto const *binary_chip_desc = binary_system_desc->chip_descs();
  auto const *binary_chip_desc_indices =
      binary_system_desc->chip_desc_indices();
  auto const *chip_capabilities = binary_system_desc->chip_capabilities();
  auto const *binary_chip_coords = binary_system_desc->chip_coords();
  auto const *chip_channel_connections = binary_system_desc->chip_channels();

  // Acquire cpu descs
  std::vector<tt::CPUDescAttr> cpu_desc_list;
  for (auto const *element : *binary_cpu_desc) {
    static_assert(static_cast<std::underlying_type_t<::tt::target::CPURole>>(
                      ::mlir::tt::CPURole::Device) ==
                  static_cast<std::underlying_type_t<::tt::target::CPURole>>(
                      ::tt::target::CPURole::Device));
    static_assert(static_cast<std::underlying_type_t<::tt::target::CPURole>>(
                      ::mlir::tt::CPURole::Host) ==
                  static_cast<std::underlying_type_t<::tt::target::CPURole>>(
                      ::tt::target::CPURole::Host));
    const auto *flatbufferTargetTripleString = element->target_triple();
    cpu_desc_list.emplace_back(tt::CPUDescAttr::get(
        context, static_cast<mlir::tt::CPURole>(element->role()),
        mlir::StringAttr::get(
            context, std::string(flatbufferTargetTripleString->c_str(),
                                 flatbufferTargetTripleString->size()))));
  }

  // Acquire chip descs
  std::vector<tt::ChipDescAttr> chip_desc_list;
  for (auto const *element : *binary_chip_desc) {
    std::vector<tt::CoreCoordAttr> worker_cores, dram_cores, eth_cores,
        eth_inactive_cores;
    auto const *physical_cores = element->physical_cores();

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

    tt::Arch arch;
    switch (element->arch()) {
    case ::tt::target::Arch::Grayskull:
      arch = tt::Arch::Grayskull;
      break;
    case ::tt::target::Arch::Wormhole_b0:
      arch = tt::Arch::WormholeB0;
      break;
    case ::tt::target::Arch::Blackhole:
      arch = tt::Arch::Blackhole;
      break;
    }

    std::vector<tt::DataTypeAttr> supported_data_types_attr;

    for (auto it : *(element->supported_data_types())) {
      switch (it) {
      case ::tt::target::DataType::Float32:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::Float32));
        break;
      case ::tt::target::DataType::Float16:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::Float16));
        break;
      case ::tt::target::DataType::BFloat16:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::BFloat16));
        break;
      case ::tt::target::DataType::BFP_Float8:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::BFP_Float8));
        break;
      case ::tt::target::DataType::BFP_BFloat8:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::BFP_BFloat8));
        break;
      case ::tt::target::DataType::BFP_Float4:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::BFP_Float4));
        break;
      case ::tt::target::DataType::BFP_BFloat4:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::BFP_BFloat4));
        break;
      case ::tt::target::DataType::BFP_Float2:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::BFP_Float2));
        break;
      case ::tt::target::DataType::BFP_BFloat2:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::BFP_BFloat2));
        break;
      case ::tt::target::DataType::UInt32:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::UInt32));
        break;
      case ::tt::target::DataType::UInt16:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::UInt16));
        break;
      case ::tt::target::DataType::UInt8:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::UInt8));
        break;
      }
    }

    SmallVector<tt::TileSizeAttr> supported_tile_sizes_attr;

    for (auto const *it : *(element->supported_tile_sizes())) {
      supported_tile_sizes_attr.push_back(
          tt::TileSizeAttr::get(context, it->y(), it->x()));
    }

    auto current_chip_desc_attr = tt::ChipDescAttr::get(
        context, tt::ArchAttr::get(context, arch),
        {element->grid_size()->y(), element->grid_size()->x()},
        element->l1_size(), element->num_dram_channels(),
        element->dram_channel_size(), element->noc_l1_address_align_bytes(),
        element->pcie_address_align_bytes(),
        element->noc_dram_address_align_bytes(), element->l1_unreserved_base(),
        element->erisc_l1_unreserved_base(), element->dram_unreserved_base(),
        element->dram_unreserved_end(), chip_physical_cores_attr,
        supported_data_types_attr, supported_tile_sizes_attr,
        element->num_cbs());
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
  for (auto const *element : *binary_chip_coords) {
    auto chip_coordinate_attr = tt::ChipCoordAttr::get(
        context, element->rack(), element->shelf(), element->y(), element->x());
    chip_coordinate_list.push_back(chip_coordinate_attr);
  }

  std::vector<tt::ChipChannelAttr> chip_channel_list;
  for (auto const *element : *chip_channel_connections) {
    std::vector<int64_t> ethernet_core_coord0_vec = {
        element->ethernet_core_coord0().y(),
        element->ethernet_core_coord0().x()};

    std::vector<int64_t> ethernet_core_coord1_vec = {
        element->ethernet_core_coord1().y(),
        element->ethernet_core_coord1().x()};

    auto chip_channel_attr = tt::ChipChannelAttr::get(
        context, element->device_id0(), ethernet_core_coord0_vec,
        element->device_id1(), ethernet_core_coord1_vec);
    chip_channel_list.push_back(chip_channel_attr);
  }

  // Generate system desc attribute
  auto system_desc_attr = tt::SystemDescAttr::get(
      context, cpu_desc_list, chip_desc_list, chip_indices_list,
      chip_capabilities_list, chip_coordinate_list, chip_channel_list);

  return system_desc_attr;
}

unsigned SystemDescAttr::getAddressAlignBytes(unsigned chipIndex) const {
  return std::max(std::initializer_list<unsigned>{
      getNocL1AddressAlignBytes(),
      getNocDRAMAddressAlignBytes(),
      getPcieAddressAlignBytes(),
  });
}

unsigned SystemDescAttr::getAddressAlignBytes(MemorySpace memorySpace,
                                              unsigned chipIndex) const {
  switch (memorySpace) {
  case MemorySpace::DeviceL1:
    return getNocL1AddressAlignBytes(chipIndex);
  case MemorySpace::DeviceDRAM:
    return getNocDRAMAddressAlignBytes(chipIndex);
  case MemorySpace::SystemMMIO:
    return getPcieAddressAlignBytes(chipIndex);
  default:
    return 1;
  }
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

::llvm::LogicalResult StreamLayoutAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    AffineMap affineMap, StreamMode streamMode, uint32_t numBuffers) {
  if (streamMode == StreamMode::Alias) {
    if (numBuffers != 1) {
      emitError() << "'Alias' mode must imply no buffering: numBuffers = "
                  << numBuffers;
      return ::mlir::failure();
    }
  }
  return ::mlir::success();
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
mlir::AffineMap collapsedLinearAffineMap(
    ::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape,
    ::llvm::ArrayRef<int64_t> gridShape,
    ::llvm::ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  int64_t numResultsClamped = std::min(shape.size(), gridShape.size());
  auto map = mlir::AffineMap::getMinorIdentityMap(shape.size(),
                                                  numResultsClamped, context);

  std::int64_t minimumDim = static_cast<std::int64_t>(shape.size());
  for (auto [begin, end] : collapseIntervals) {
    if (begin < 0) {
      begin += shape.size();
    }
    if (end < 0) {
      end += shape.size();
    }
    if (begin == end) {
      continue;
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

  while (map.getNumResults() < gridShape.size()) {
    map = map.insertResult(getAffineConstantExpr(0, context), 0);
  }

  return map;
}

mlir::SmallVector<std::int64_t>
calculateLogicalShardShape(mlir::ArrayRef<int64_t> tensorShape,
                           mlir::AffineMap linear, GridAttr grid) {
  assert(linear.getNumResults() == grid.getShape().size());
  mlir::SmallVector<std::int64_t> logicalShape =
      ttmlir::utils::evalShape(linear, tensorShape);
  mlir::SmallVector<std::int64_t> shardShape(linear.getNumResults());
  for (unsigned i = 0; i < linear.getNumResults(); ++i) {
    shardShape[i] =
        (logicalShape[i] + grid.getShape()[i] - 1) / grid.getShape()[i];
  }
  return shardShape;
}

MetalLayoutAttr MetalLayoutAttr::get(
    ::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape,
    Type elementType, MemorySpace memorySpace, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals,
    OOBVal oobVal, TensorMemoryLayout memLayout) {
  if (not grid) {
    grid = GridAttr::get(context, tensorShape.size());
  }

  auto linear = collapsedLinearAffineMap(context, tensorShape, grid.getShape(),
                                         collapseIntervals);
  auto shardShape = calculateLogicalShardShape(tensorShape, linear, grid);
  auto memref = buildMemRef<MemorySpace, MemorySpaceAttr>(
      context, shardShape, elementType, memorySpace);
  return get(context, linear, oobVal, grid, memref, memLayout);
}

MetalLayoutAttr MetalLayoutAttr::get(
    ::mlir::MLIRContext *context, RankedTensorType ty, MemorySpace memorySpace,
    GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals,
    OOBVal oobVal, TensorMemoryLayout memLayout) {
  assert(ty);
  return get(context, ty.getShape(), ty.getElementType(), memorySpace, grid,
             collapseIntervals, oobVal, memLayout);
}

MetalLayoutAttr MetalLayoutAttr::get(::mlir::MLIRContext *context,
                                     RankedTensorType ty,
                                     MemorySpace memorySpace, GridAttr grid,
                                     Type elementType,
                                     TensorMemoryLayout memLayout) {
  assert(ty);
  assert(grid);
  return get(context, ty.getShape(), elementType, memorySpace, grid, {{0, -1}},
             OOBVal::Undef, memLayout);
}

// From the logical shape of the tensor and the affine map of the layout,
// compute the physical shape of the tensor, i.e the shape of the tensor
// after the dimensions have been collapsed onto a grid.
llvm::SmallVector<int64_t>
MetalLayoutAttr::getPhysicalShape(ArrayRef<int64_t> logicalShape) const {
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
MetalLayoutAttr::getStride(ArrayRef<int64_t> logicalShape) const {

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

llvm::SmallVector<int64_t>
MetalLayoutAttr::getShardShape(bool convertTileToScalar) const {
  SmallVector<int64_t> shardShape(getMemref().getShape());
  auto elementType = getElementType();
  if (mlir::isa<TileType>(elementType) && convertTileToScalar) {
    return mlir::cast<TileType>(elementType).getScalarShape(shardShape);
  }
  return shardShape;
}

StreamMode MetalLayoutAttr::getStreamMode() const {
  StreamLayoutAttr layout =
      llvm::dyn_cast<StreamLayoutAttr>(getMemref().getLayout());
  assert(layout != nullptr && "expected a StreamLayoutAttr layout");
  return layout.getStreamMode();
}

mlir::Type MetalLayoutAttr::getElementType() const {
  return getMemref().getElementType();
}

mlir::Type MetalLayoutAttr::getScalarElementType() const {
  auto elementType = getElementType();
  if (mlir::isa<TileType>(elementType)) {
    return mlir::cast<TileType>(elementType).getElementType();
  }
  return elementType;
}

bool MetalLayoutAttr::hasShardedTensorMemoryLayout() const {
  return (getMemLayout() == TensorMemoryLayout::HeightSharded or
          getMemLayout() == TensorMemoryLayout::WidthSharded or
          getMemLayout() == TensorMemoryLayout::BlockSharded);
}

bool MetalLayoutAttr::hasInterleavedTensorMemoryLayout() const {
  return (getMemLayout() == TensorMemoryLayout::Interleaved);
}

bool MetalLayoutAttr::hasShardedL1TensorMemoryLayout() const {
  return ::mlir::tt::isL1MemorySpace(getMemorySpace()) and
         (getMemLayout() == TensorMemoryLayout::HeightSharded or
          getMemLayout() == TensorMemoryLayout::WidthSharded or
          getMemLayout() == TensorMemoryLayout::BlockSharded);
}

bool MetalLayoutAttr::hasInterleavedL1TensorMemoryLayout() const {
  return ::mlir::tt::isL1MemorySpace(getMemorySpace()) and
         (getMemLayout() == TensorMemoryLayout::Interleaved);
}

bool MetalLayoutAttr::isTiled() const {
  return ::mlir::isa<::mlir::tt::TileType>(getElementType());
}

uint64_t MetalLayoutAttr::getElementSizeBytes() const {
  mlir::Type elementType = getElementType();
  if (mlir::isa<TileType>(elementType)) {
    auto tileType = mlir::cast<TileType>(elementType);
    return tileType.getSizeBytes();
  }
  return elementType.getIntOrFloatBitWidth() / 8;
}

uint64_t MetalLayoutAttr::getMemrefSizeBytes() const {
  MemRefType ty = getMemref();
  auto shape = ty.getShape();
  uint64_t size = getElementSizeBytes();
  return std::accumulate(shape.begin(), shape.end(), size,
                         std::multiplies<uint64_t>());
}

MetalLayoutAttr MetalLayoutAttr::withGrid(
    ::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  return get(context, tensorShape, getElementType(), getMemorySpace(), grid,
             collapseIntervals, getOobVal(), getMemLayout());
}

MetalLayoutAttr MetalLayoutAttr::withGrid(
    ::mlir::MLIRContext *context, RankedTensorType ty, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  assert(ty);
  return MetalLayoutAttr::withGrid(context, ty.getShape(), grid,
                                   collapseIntervals);
}

MetalLayoutAttr MetalLayoutAttr::withElementType(::mlir::MLIRContext *context,
                                                 Type elementType) {
  return MetalLayoutAttr::get(
      context, getLinear(), getOobVal(), getGrid(),
      buildMemRef<MemorySpace, MemorySpaceAttr>(context, getShardShape(),
                                                elementType, getMemorySpace()),
      getMemLayout());
}

MetalLayoutAttr MetalLayoutAttr::withMemorySpace(::mlir::MLIRContext *context,
                                                 MemorySpace memorySpace) {
  return MetalLayoutAttr::get(
      context, getLinear(), getOobVal(), getGrid(),
      buildMemRef<MemorySpace, MemorySpaceAttr>(context, getShardShape(),
                                                getElementType(), memorySpace),
      getMemLayout());
}

MetalLayoutAttr
MetalLayoutAttr::withMemoryLayout(::mlir::MLIRContext *context,
                                  TensorMemoryLayout memLayout) {
  return MetalLayoutAttr::get(
      context, getLinear(), getOobVal(), getGrid(),
      buildMemRef<MemorySpace, MemorySpaceAttr>(
          context, getShardShape(), getElementType(), getMemorySpace()),
      memLayout);
}

MetalLayoutAttr
MetalLayoutAttr::withShardShape(::mlir::MLIRContext *context,
                                llvm::SmallVector<int64_t> shardShape) {
  return MetalLayoutAttr::get(
      context, getLinear(), getOobVal(), getGrid(),
      buildMemRef<MemorySpace, MemorySpaceAttr>(
          context, shardShape, getElementType(), getMemorySpace()),
      getMemLayout());
}

// TODO(vroubtsovTT): remove this, it's difficult/unsafe to use
MetalLayoutAttr MetalLayoutAttr::withStreamLayout(::mlir::MLIRContext *context,
                                                  StreamLayoutAttr layout) {
  return MetalLayoutAttr::get(
      context, getLinear(), getOobVal(), getGrid(),
      buildMemRef<MemorySpace, MemorySpaceAttr>(
          context, getShardShape(), getElementType(), getMemorySpace(), layout),
      getMemLayout());
}

MetalLayoutAttr MetalLayoutAttr::withOuterScale(
    ::mlir::MLIRContext *context, llvm::ArrayRef<int64_t> outerScale,
    StreamMode streamMode, std::uint32_t numBuffers) {

  auto innerShape = getShardShape();
  std::size_t innerShapeSize = innerShape.size();

  llvm::SmallVector<int64_t> fullShape(2 * innerShapeSize); // rank doubles
  for (std::size_t d = 0; d < innerShapeSize; ++d) {
    fullShape[d] = 1;
    fullShape[innerShapeSize + d] = innerShape[d];
  }

  auto fullAffineMap =
      mlir::AffineMap::getMultiDimIdentityMap(fullShape.size(), context);
  auto fullLayout =
      StreamLayoutAttr::get(context, fullAffineMap, streamMode, numBuffers);
  auto fullMemRef = buildMemRef<MemorySpace, MemorySpaceAttr>(
      context, fullShape, getElementType(), getMemorySpace(), fullLayout);

  return MetalLayoutAttr::get(context, getLinear(), getOobVal(), getGrid(),
                              fullMemRef, getMemLayout());
}

MemorySpace MetalLayoutAttr::getMemorySpace() const {
  return mlir::cast<mlir::tt::MemorySpaceAttr>(getMemref().getMemorySpace())
      .getValue();
}

// Returns shape of the tensor after tilization is applied to the two inner most
// dimensions.
llvm::SmallVector<int64_t>
MetalLayoutAttr::getTiledShape(llvm::ArrayRef<int64_t> tensorShape) const {
  assert(isTiled() && "Expected a tiled layout");

  mlir::AffineMap linear = getLinear();
  uint32_t rank = linear.getNumResults();
  assert(rank >= 2 && "Expected at least two results in linear map");
  mlir::AffineExpr y = linear.getResult(rank - 2);
  mlir::AffineExpr x = linear.getResult(rank - 1);

  TileType tileType = mlir::cast<TileType>(getElementType());
  int64_t tileH = tileType.getHeight();
  int64_t tileW = tileType.getWidth();

  mlir::AffineMap tiled =
      linear.replace(mlir::DenseMap<mlir::AffineExpr, mlir::AffineExpr>{
          {y, y.floorDiv(tileH)}, {x, x.floorDiv(tileW)}});

  return ttmlir::utils::evalShape(tiled, tensorShape);
}

mlir::AffineMap MetalLayoutAttr::getIdentityTileLinearMap() const {
  assert(isTiled() && "Expected a tiled layout");

  return mlir::AffineMap::getMultiDimIdentityMap(getLinear().getNumResults(),
                                                 getContext());
}

//
// Memory affine maps use symbols as placeholders for the shard shape. This
// function replaces those symbols with the actual shard shape for this layout.
//
// E.g. the l1Map before replacement:
//   (d0, d1)[s0, s1] ->
//     (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1)
//
// After replacement with shard shape [2, 3]:
//   (d0, d1)[2, 3] ->
//     (0, d0 floordiv 2, d1 floordiv 3, (d0 mod 2) * 3 + d1 mod 3)
//
mlir::AffineMap MetalLayoutAttr::replaceMemoryMapSymbolsWithShardShape(
    AffineMap physicalMemoryMap) const {
  mlir::SmallVector<int64_t> shardShape =
      getShardShape(false /*convertTileToScalar*/);
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

// Projects tensor layout onto a physical memory map. Uses given linear map to
// derive the shard shape and the projection of shard indexes onto the logical
// grid. Then it composes the logical grid projection with physical memory
// mapping.
mlir::AffineMap
MetalLayoutAttr::projectOnto(mlir::AffineMap linearMap,
                             mlir::AffineMap physicalMemoryMap) const {
  assert(getGrid().getShape().size() == physicalMemoryMap.getNumDims() &&
         "Layout and device grids must have same number of dimensions");
  assert(getLinear().getNumResults() == physicalMemoryMap.getNumDims() &&
         "Linear map and physical map must have same number of dimensions");
  return replaceMemoryMapSymbolsWithShardShape(physicalMemoryMap)
      .compose(linearMap);
}

//
// This function creates an affine map that represents mapping the tensor's
// linear layout onto the 2d physical device grid. A typical example will look
// like:
//   (d0, d1)[s0, s1] -> ( # Uses affine symbols s0, s1 to represent shard dims
//     0,                           # Device index
//     d0 floordiv s0,              # CoreCoordY
//     d1 floordiv s1,              # CoreCoordX
//     (d0 mod s0) * s1 + d1 mod s1 # Element offset within shard
//   )
//
static mlir::AffineMap createL1Map(::mlir::MLIRContext *context,
                                   GridAttr workerGrid,
                                   SystemDescAttr systemDesc,
                                   ::llvm::ArrayRef<unsigned> chipIds) {
  mlir::AffineMap workerMap = workerGrid.getMapping();
  mlir::SmallVector<mlir::AffineExpr> l1MapResults(workerMap.getNumDims());
  mlir::AffineExpr shardIndexing = getAffineConstantExpr(0, context);
  mlir::AffineExpr shardVolumeExpr = getAffineConstantExpr(1, context);

  // Compute the projection of the layout onto its own logical grid.
  // Simultaneously compute the indexing of shards within each core.
  for (int i = workerMap.getNumDims() - 1; i >= 0; i--) {
    mlir::AffineExpr linearIdx = getAffineDimExpr(i, context);
    mlir::AffineExpr shardDim = getAffineSymbolExpr(i, context);
    l1MapResults[i] = linearIdx.floorDiv(shardDim);
    shardIndexing = (linearIdx % shardDim) * shardVolumeExpr + shardIndexing;
    shardVolumeExpr = shardVolumeExpr * shardDim;
  }

  // Compose the logical grid projection with the device grid mapping, now we
  // have a projection onto the physical grid.
  mlir::AffineMap gridProjection = workerMap.compose(mlir::AffineMap::get(
      workerMap.getNumDims(), workerMap.getNumDims(), l1MapResults, context));

  // Finally we append the indexing of shards within each core.
  mlir::SmallVector<mlir::AffineExpr> l1Map(gridProjection.getResults());
  l1Map.push_back(shardIndexing);
  return mlir::AffineMap::get(workerMap.getNumDims(), workerMap.getNumDims(),
                              l1Map, context);
}

static GridAttr createWorkerGrid(::mlir::MLIRContext *context,
                                 mlir::ArrayRef<int64_t> chipGrid,
                                 mlir::ArrayRef<int64_t> meshShapeRef) {
  assert(chipGrid.size() == 2 && "expected 2D grid");
  mlir::SmallVector<int64_t> meshShape(meshShapeRef);

  // Fill in missing dimensions with 1 so that the mesh shape is at least 2D.
  while (meshShape.size() < chipGrid.size()) {
    meshShape.push_back(1);
  }

  mlir::SmallVector<int64_t> virtualGrid(chipGrid);
  // Fill in missing dimensions with 1 so that the virtual grid is at has
  // meshShape dimensions.
  while (virtualGrid.size() < meshShape.size()) {
    virtualGrid.insert(virtualGrid.begin(), 1);
  }

  // Multiply out the mesh shape to get the real virtual grid shape.
  for (int i = meshShape.size() - 1; i >= 0; --i) {
    virtualGrid[i] *= meshShape[i];
  }

  // Calculate the affine expressions for the worker grid.
  auto workerDeviceIdx = getAffineConstantExpr(0, context);
  auto workerCoreY = getAffineDimExpr(virtualGrid.size() - 2, context);
  auto workerCoreX = getAffineDimExpr(virtualGrid.size() - 1, context);
  assert(virtualGrid.size() == meshShape.size());

  // Special case the inner 2 dimensions of the device indexing to support
  // horizonally/vertically stacked virtual grids.  For these cases we need an
  // affine expression that rolls over the device index when we reach the end of
  // single-chip boundaries.
  int meshStride = 1;
  if (meshShape[meshShape.size() - 1] > 1) {
    workerDeviceIdx = workerCoreX.floorDiv(chipGrid[1]) + workerDeviceIdx;
    workerCoreX = workerCoreX % meshShape[meshShape.size() - 1];
  }

  meshStride *= meshShape[meshShape.size() - 1];
  if (meshShape[meshShape.size() - 2] > 1) {
    workerDeviceIdx =
        workerCoreY.floorDiv(chipGrid[0]) * meshStride + workerDeviceIdx;
    workerCoreY = workerCoreY % meshShape[meshShape.size() - 2];
  }

  // The rest of the dimensions are just a simple linearization.
  meshStride *= meshShape[meshShape.size() - 2];
  for (int i = static_cast<int>(meshShape.size()) - 3; i >= 0; --i) {
    workerDeviceIdx =
        getAffineDimExpr(i, context) * meshStride + workerDeviceIdx;
    meshStride *= meshShape[i];
  }

  mlir::SmallVector<mlir::AffineExpr> workerGridExprs = {
      workerDeviceIdx, workerCoreY, workerCoreX};
  auto workerGridMap =
      mlir::AffineMap::get(virtualGrid.size(), 0, workerGridExprs, context);
  return GridAttr::get(context, virtualGrid, workerGridMap);
}

//
// This function creates an affine map that represents mapping the tensor's
// linear layout onto physical dram banks. A typical example will end up looking
// pretty complicated:
//   (d0, d1)[s0, s1] -> (
//     0,                                  # Device index
//     0,                                  # CoreCoordY
//     (addr floordiv 8192) mod 12,        # Channel Idx / CoreCoordX
//     addr floordiv 98304 + addr mod 8192 # Offset within channel
//   )
//
// Where `addr` is the linearized address as though it were indexing all of DRAM
// flat.  Then we do some additional calculations to break up the channels into
// interleaved pages:
//   addr = (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) +
//          (d0 mod s0) * s1 + d1 mod s1)
//
static mlir::AffineMap createDramMap(::mlir::MLIRContext *context,
                                     GridAttr workerGrid, ArchAttr arch,
                                     mlir::ArrayRef<CoreCoordAttr> dramCores,
                                     unsigned dramPageSize) {
  mlir::AffineMap workerMap = workerGrid.getMapping();
  assert(workerMap.getNumResults() == PhysGridResultIdx::NumIndices);
  mlir::AffineExpr addr = getAffineConstantExpr(0, context);
  mlir::AffineExpr shardIndexing = getAffineConstantExpr(0, context);
  mlir::AffineExpr shardVolumeExpr = getAffineConstantExpr(1, context);
  mlir::AffineExpr gridVolumeExpr = getAffineConstantExpr(1, context);

  for (int i = workerMap.getNumDims() - 1; i >= 0; i--) {
    mlir::AffineExpr linearIdx = getAffineDimExpr(i, context);
    mlir::AffineExpr shardDim = getAffineSymbolExpr(i, context);
    addr = linearIdx.floorDiv(shardDim) * gridVolumeExpr + addr;
    shardIndexing = (linearIdx % shardDim) * shardVolumeExpr + shardIndexing;
    shardVolumeExpr = shardVolumeExpr * shardDim;
    gridVolumeExpr = gridVolumeExpr * workerGrid.getShape()[i];
  }

  addr = addr * shardVolumeExpr + shardIndexing;

  mlir::AffineExpr pageSizeExpr = getAffineConstantExpr(dramPageSize, context);
  mlir::AffineExpr numDramCores =
      getAffineConstantExpr(dramCores.size(), context);
  mlir::SmallVector<mlir::AffineExpr> dramMapResults = {
      addr.floorDiv(pageSizeExpr) % numDramCores,
      addr.floorDiv(pageSizeExpr * numDramCores) + addr % pageSizeExpr,
  };

  // Dram logical coords are 1d, so constant 0 index for
  // MemMapResultIdx::CoreCoordY
  dramMapResults.insert(dramMapResults.begin(),
                        getAffineConstantExpr(0, context));
  dramMapResults.insert(dramMapResults.begin(),
                        workerMap.getResult(MemoryMapResultIdx::DeviceIdx));
  assert(dramMapResults.size() == MemoryMapResultIdx::NumIndices);

  return mlir::AffineMap::get(workerMap.getNumDims(), workerMap.getNumDims(),
                              dramMapResults, context);
}

static mlir::AffineMap createDramMap(::mlir::MLIRContext *context,
                                     GridAttr workerGrid,
                                     SystemDescAttr systemDesc,
                                     ::llvm::ArrayRef<unsigned> chipIds,
                                     unsigned dramPageSize) {
  auto chipDesc = systemDesc.getChipDescs().front();
  auto chipPhysicalCores = chipDesc.getChipPhysicalCores();
  auto firstDramCores = chipPhysicalCores.getDram();
  assert(!firstDramCores.empty() && "expected at least one dram core");

  for (unsigned chipId : chipIds) {
    auto chipDesc = systemDesc.getChipDescs()[chipId];
    auto chipPhysicalCores = chipDesc.getChipPhysicalCores();
    auto dramCores = chipPhysicalCores.getDram();
    assert(dramCores.size() == firstDramCores.size());
  }

  return createDramMap(context, workerGrid, chipDesc.getArch(), firstDramCores,
                       dramPageSize);
}

DeviceAttr DeviceAttr::get(::mlir::MLIRContext *context,
                           SystemDescAttr systemDesc,
                           ArrayRef<int64_t> meshShape,
                           ArrayRef<unsigned> chipIds) {
  assert(not chipIds.empty() && "expected at least one chip");
  ChipDescAttr chipDesc = systemDesc.getChipDescs()[chipIds.front()];
  SmallVector<int64_t> chipGrid(chipDesc.getGrid());
  assert(chipGrid.size() == 2 && "expected 2D grid");

  // Take the min grid shape across all chips, this can happen if 1 chip has
  // more harvested rows than another.  We take the min because we want to
  // ensure that the logical grid is the same across all chips and this vastly
  // simplifies the grid /memory mappings across the board.
  for (unsigned chipId : chipIds) {
    ChipDescAttr chip = systemDesc.getChipDescs()[chipId];
    ArrayRef<int64_t> iChipGrid = chip.getGrid();
    assert(iChipGrid.size() == 2 && "expected 2D grid");
    chipGrid[0] = std::min(chipGrid[0], iChipGrid[0]);
    chipGrid[1] = std::min(chipGrid[1], iChipGrid[1]);
  }

  auto workerGrid = createWorkerGrid(context, chipGrid, meshShape);
  auto l1Map = createL1Map(context, workerGrid, systemDesc, chipIds);
  constexpr unsigned dramPageSize = 8192;
  auto dramMap =
      createDramMap(context, workerGrid, systemDesc, chipIds, dramPageSize);
  return get(context, workerGrid, l1Map, dramMap, meshShape, chipIds);
}

DeviceAttr DeviceAttr::get(::mlir::MLIRContext *context,
                           SystemDescAttr systemDesc,
                           ArrayRef<int64_t> meshShape) {
  int64_t numChips = ttmlir::utils::volume(meshShape);
  assert(systemDesc.getChipDescIndices().size() >=
             static_cast<size_t>(numChips) &&
         "expected at least one chip");
  SmallVector<unsigned> chipIds(numChips);
  std::iota(chipIds.begin(), chipIds.end(), 0);
  return get(context, systemDesc, meshShape, chipIds);
}

// Sample the last index in the tensor to get the last addressable element of
// the tensor to determine its footprint in memory.
uint64_t DeviceAttr::getLayoutSizeBytes(ArrayRef<int64_t> tensorScalarShape,
                                        MetalLayoutAttr layout,
                                        MemorySpace memorySpace) const {
  SmallVector<int64_t> shape = layout.isTiled()
                                   ? layout.getTiledShape(tensorScalarShape)
                                   : SmallVector<int64_t>(tensorScalarShape);
  AffineMap linearMap =
      layout.isTiled() ? layout.getIdentityTileLinearMap() : layout.getLinear();
  mlir::SmallVector<std::int64_t> linearShape =
      ttmlir::utils::evalShape(linearMap, shape);
  AffineMap memoryMap = layout.replaceMemoryMapSymbolsWithShardShape(
      getMapForMemorySpace(memorySpace));
  mlir::SmallVector<std::int64_t> physicalMemory =
      ttmlir::utils::evalShape(memoryMap, linearShape);
  std::int64_t elementSize = layout.getElementSizeBytes();
  uint64_t sizeBytes =
      physicalMemory[MemoryMapResultIdx::ShardOffset] * elementSize;
  return sizeBytes;
}

uint64_t DeviceAttr::getTensorSizeBytes(RankedTensorType tensorType,
                                        MemorySpace memorySpace) const {
  assert(tensorType.getEncoding());
  return getLayoutSizeBytes(
      tensorType.getShape(),
      mlir::cast<MetalLayoutAttr>(tensorType.getEncoding()), memorySpace);
}

::mlir::LogicalResult
DeviceAttr::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                   ::mlir::tt::GridAttr workerGrid, ::mlir::AffineMap l1Map,
                   ::mlir::AffineMap dramMap,
                   ::llvm::ArrayRef<int64_t> meshShape,
                   ::llvm::ArrayRef<unsigned> chipIds) {
  if (chipIds.empty()) {
    emitError() << "expected at least one chip";
    return ::mlir::failure();
  }

  std::int64_t meshVolume = ttmlir::utils::volume(meshShape);
  if (chipIds.size() != static_cast<size_t>(meshVolume)) {
    emitError() << "expected chipIds size to match the volume of meshShape";
    return ::mlir::failure();
  }

  auto workerGridShape = workerGrid.getShape();
  for (auto dim : workerGridShape) {
    if (dim <= 0) {
      emitError() << "expected positive grid dimensions";
      return ::mlir::failure();
    }
  }

  auto physicalGridMapping = workerGrid.getMapping();
  if (physicalGridMapping.getNumResults() != PhysGridResultIdx::NumIndices) {
    emitError() << "expected physical grid mapping to have "
                   "PhysGridResultIdx::NumIndices results";
    return ::mlir::failure();
  }

  if (l1Map.getNumResults() != MemoryMapResultIdx::NumIndices) {
    emitError()
        << "expected l1Map to have MemoryMapResultIdx::NumIndices results";
    return ::mlir::failure();
  }

  if (dramMap.getNumResults() != MemoryMapResultIdx::NumIndices) {
    emitError()
        << "expected dramMap to have MemoryMapResultIdx::NumIndices results";
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

mlir::Type TileType::getElementType() const {
  switch (getDataType()) {
  case DataType::Float32:
    return FloatType::getF32(getContext());
  case DataType::Float16:
    return FloatType::getF16(getContext());
  case DataType::BFloat16:
    return FloatType::getBF16(getContext());
  case DataType::BFP_Float8:
    return FloatType::getF16(getContext());
  case DataType::BFP_BFloat8:
    return FloatType::getBF16(getContext());
  case DataType::BFP_Float4:
    return FloatType::getF16(getContext());
  case DataType::BFP_BFloat4:
    return FloatType::getBF16(getContext());
  case DataType::BFP_Float2:
    return FloatType::getF16(getContext());
  case DataType::BFP_BFloat2:
    return FloatType::getBF16(getContext());
  case DataType::UInt32:
    return IntegerType::get(getContext(), 32,
                            IntegerType::SignednessSemantics::Unsigned);
  case DataType::UInt16:
    return IntegerType::get(getContext(), 16,
                            IntegerType::SignednessSemantics::Unsigned);
  case DataType::UInt8:
    return IntegerType::get(getContext(), 8,
                            IntegerType::SignednessSemantics::Unsigned);
  }
}

SystemDescAttr mlir::tt::getCurrentScopeSystemDesc(mlir::Operation *op) {
  // Walk up scope levels until we find the top level ModuleOp which carries
  // the system desc
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
