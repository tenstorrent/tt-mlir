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

mlir::tt::SystemDescAttr mlir::tt::SystemDescAttr::getDefault(
    MLIRContext *context, const ::llvm::SmallVector<int64_t> &meshShape) {
  // Set default values
  constexpr auto l1Size = 1499136;
  constexpr auto numDramChannels = 12;
  constexpr auto dramChannelSize = 1 << 30;
  constexpr auto nocL1AddressAlignBytes = 16;
  constexpr auto pcieAddressAlignBytes = 32;
  constexpr auto nocDRAMAddressAlignBytes = 32;
  constexpr auto l1UnreservedBase = 1024;
  constexpr auto eriscL1UnreservedBase = 1024;
  constexpr auto dramUnreservedBase = 1024;
  constexpr auto dramUnreservedEnd = 1 << 30;
  constexpr auto numCBs = 32;
  constexpr auto numComputeThreads = 1;
  constexpr auto numDatamovementThreads = 2;

  // Get number of chips in mesh.
  int64_t numberOfChips =
      std::accumulate(meshShape.begin(), meshShape.end(), int64_t{1},
                      std::multiplies<int64_t>());

  // Populate dummy values for single chip or multi chip config.
  llvm::SmallVector<std::int64_t> gridShape = {8, 8};

  // Populate a placeholder for supported tile sizes.
  llvm::SmallVector<DataTypeAttr> supported_data_types = {
      DataTypeAttr::get(context, DataType::Float32),
      DataTypeAttr::get(context, DataType::Float16),
      DataTypeAttr::get(context, DataType::BFloat16),
      DataTypeAttr::get(context, DataType::BFP_Float8),
      DataTypeAttr::get(context, DataType::BFP_BFloat8),
      DataTypeAttr::get(context, DataType::BFP_Float4),
      DataTypeAttr::get(context, DataType::BFP_BFloat4),
      DataTypeAttr::get(context, DataType::BFP_Float2),
      DataTypeAttr::get(context, DataType::BFP_BFloat2),
      DataTypeAttr::get(context, DataType::UInt32),
      DataTypeAttr::get(context, DataType::UInt16),
      DataTypeAttr::get(context, DataType::UInt8),
      DataTypeAttr::get(context, DataType::Int32),
  };

  // Populate a placeholder for supported tile sizes.
  llvm::SmallVector<TileSizeAttr> supported_tile_sizes = {
      TileSizeAttr::get(context, 4, 16),  TileSizeAttr::get(context, 16, 16),
      TileSizeAttr::get(context, 32, 16), TileSizeAttr::get(context, 4, 32),
      TileSizeAttr::get(context, 16, 32), TileSizeAttr::get(context, 32, 32),
  };

  llvm::SmallVector<CoreCoordAttr> workerCores;
  workerCores.reserve(gridShape[0] * gridShape[1]);
  for (std::int64_t y = 0; y < gridShape[0]; ++y) {
    for (std::int64_t x = 0; x < gridShape[1]; ++x) {
      workerCores.push_back(CoreCoordAttr::get(context, y, x));
    }
  }

  llvm::SmallVector<CoreCoordAttr> dramCores;
  for (std::int64_t x = 0; x < 4; ++x) {
    for (std::int64_t y = 0; y < 3; ++y) {
      dramCores.push_back(CoreCoordAttr::get(context, y + gridShape[0], x));
    }
  }

  // Get number of chips indices.
  llvm::SmallVector<uint32_t> chipIndicesList =
      llvm::to_vector(llvm::seq<uint32_t>(numberOfChips));

  // Duplicate number of chip desc attributes based on number of chips.
  llvm::SmallVector<tt::ChipDescAttr> chipDescs;
  chipDescs.reserve(numberOfChips);

  for (auto i = 0; i < numberOfChips; i++) {
    chipDescs.push_back(ChipDescAttr::get(
        context, ArchAttr::get(context, Arch::WormholeB0), gridShape, l1Size,
        numDramChannels, dramChannelSize, nocL1AddressAlignBytes,
        pcieAddressAlignBytes, nocDRAMAddressAlignBytes, l1UnreservedBase,
        eriscL1UnreservedBase, dramUnreservedBase, dramUnreservedEnd,
        ChipPhysicalCoresAttr::get(context, workerCores, dramCores, {}, {}),
        supported_data_types, supported_tile_sizes, numCBs, numComputeThreads,
        numDatamovementThreads));
  }

  // Duplicate number of chip capabilities based on number of chips.
  llvm::SmallVector<tt::ChipCapabilityAttr> chipCapabilities;
  chipCapabilities.reserve(numberOfChips);

  for (auto i = 0; i < numberOfChips; i++) {
    chipCapabilities.push_back(tt::ChipCapabilityAttr::get(
        context,
        // NOLINTNEXTLINE
        tt::ChipCapability::PCIE | tt::ChipCapability::HostMMIO));
  }

  // Update chip channels based on number of chips.
  llvm::SmallVector<tt::ChipChannelAttr> chipChannelList;
  chipChannelList.reserve(numberOfChips);

  if (numberOfChips != 1) {
    for (auto i = 0; i < numberOfChips; i++) {
      // Assume a default ring topology where final chip connects with initial
      // chip.
      chipChannelList.push_back(tt::ChipChannelAttr::get(
          context, i, {0, 0}, (i + 1) % numberOfChips, {0, 0}));
    }
  }

  return tt::SystemDescAttr::get(
      context,
      // CPU Descriptors
      {tt::CPUDescAttr::get(
          context, tt::CPURole::Host,
          mlir::StringAttr::get(context, "x86_64-pc-linux-gnu"))},
      // Chip Descriptors
      chipDescs,
      // Chip Descriptor Indices
      chipIndicesList,
      // Chip capabilities
      chipCapabilities,
      // Chip Mesh Coordinates
      {
          tt::ChipCoordAttr::get(context, 0, 0, 0, 0),
      },
      // Chip Channel Connections
      chipChannelList);
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
      case ::tt::target::DataType::Int32:
        supported_data_types_attr.push_back(
            tt::DataTypeAttr::get(context, tt::DataType::Int32));
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
        element->num_cbs(), element->num_compute_threads(),
        element->num_datamovement_threads());
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

mlir::AffineMap ShardLayoutAttr::getAffineMap() const {
  auto *context = getContext();
  int64_t rank = getStride().size();
  SmallVector<mlir::AffineExpr> mapExprs(rank + 1);

  for (int64_t i = 0; i < rank; i++) {
    mapExprs[i] = getAffineDimExpr(i, context);
  }

  mapExprs[rank] = getAffineConstantExpr(0, context);
  for (int64_t i = rank - 1; i >= 0; i--) {
    mlir::AffineExpr shardDim = getAffineDimExpr(rank + i, context);
    mlir::AffineExpr stride = getAffineConstantExpr(getStride()[i], context);
    mapExprs[rank] = shardDim * stride + mapExprs[rank];
  }

  return mlir::AffineMap::get(getStride().size() * 2, 0, mapExprs, context);
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
    OOBVal oobVal) {
  if (not grid) {
    grid = GridAttr::get(context, tensorShape.size());
  }

  auto linear = collapsedLinearAffineMap(context, tensorShape, grid.getShape(),
                                         collapseIntervals);
  auto shardShape = calculateLogicalShardShape(tensorShape, linear, grid);
  auto memref = buildMemRef<MemorySpace, MemorySpaceAttr>(
      context, shardShape, elementType, memorySpace);
  return get(context, linear, oobVal, grid, memref);
}

MetalLayoutAttr MetalLayoutAttr::get(
    ::mlir::MLIRContext *context, RankedTensorType ty, MemorySpace memorySpace,
    GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals,
    OOBVal oobVal) {
  assert(ty);
  SmallVector<int64_t> tensorShape(ty.getShape());
  if (mlir::isa<TileType>(ty.getElementType())) {
    tensorShape =
        mlir::cast<TileType>(ty.getElementType()).getScalarShape(tensorShape);
  }
  return get(context, tensorShape, ty.getElementType(), memorySpace, grid,
             collapseIntervals, oobVal);
}

MetalLayoutAttr MetalLayoutAttr::get(::mlir::MLIRContext *context,
                                     RankedTensorType ty,
                                     MemorySpace memorySpace, GridAttr grid,
                                     Type elementType) {
  assert(ty);
  assert(grid);
  return get(context, ty.getShape(), elementType, memorySpace, grid, {{0, -1}},
             OOBVal::Undef);
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

llvm::SmallVector<int64_t> MetalLayoutAttr::getShardStride() const {
  SmallVector<int64_t> shardShape =
      getShardShape(/*convertTileToScalar=*/false);
  SmallVector<int64_t> shardStride(shardShape.size());
  shardStride[shardStride.size() - 1] = getElementSizeBytes();
  for (int64_t i = static_cast<int64_t>(shardStride.size()) - 2; i >= 0; i--) {
    shardStride[i] = shardShape[i + 1] * shardStride[i + 1];
  }
  return shardStride;
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
             collapseIntervals, getOobVal());
}

MetalLayoutAttr MetalLayoutAttr::withGrid(
    ::mlir::MLIRContext *context, RankedTensorType ty, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  assert(ty);
  SmallVector<int64_t> tensorShape(ty.getShape());
  auto tileType = mlir::dyn_cast<TileType>(ty.getElementType());
  if (tileType) {
    tensorShape = tileType.getScalarShape(tensorShape);
  }
  return MetalLayoutAttr::withGrid(context, tensorShape, grid,
                                   collapseIntervals);
}

MetalLayoutAttr MetalLayoutAttr::withElementType(::mlir::MLIRContext *context,
                                                 Type elementType) {
  return MetalLayoutAttr::get(
      context, getLinear(), getOobVal(), getGrid(),
      buildMemRef<MemorySpace, MemorySpaceAttr>(context, getShardShape(true),
                                                elementType, getMemorySpace()));
}

MetalLayoutAttr MetalLayoutAttr::withMemorySpace(::mlir::MLIRContext *context,
                                                 MemorySpace memorySpace) {
  return MetalLayoutAttr::get(
      context, getLinear(), getOobVal(), getGrid(),
      buildMemRef<MemorySpace, MemorySpaceAttr>(context, getShardShape(true),
                                                getElementType(), memorySpace));
}

MetalLayoutAttr
MetalLayoutAttr::withShardShape(::mlir::MLIRContext *context,
                                llvm::SmallVector<int64_t> shardShape) {
  return MetalLayoutAttr::get(
      context, getLinear(), getOobVal(), getGrid(),
      buildMemRef<MemorySpace, MemorySpaceAttr>(
          context, shardShape, getElementType(), getMemorySpace()));
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

mlir::MemRefType MetalLayoutAttr::getBufferType() const {
  SmallVector<int64_t> fullMemrefShape;
  auto gridShape = getGrid().getShape();
  auto shardShape = getShardShape(/*convertTileToScalar*/ false);
  fullMemrefShape.append(gridShape.begin(), gridShape.end());
  fullMemrefShape.append(shardShape.begin(), shardShape.end());
  return MemRefType::get(
      fullMemrefShape, getElementType(),
      ShardLayoutAttr::get(getContext(), getShardStride(), /*buffered=*/1),
      MemorySpaceAttr::get(getContext(), getMemorySpace()));
}

//
// This function creates an affine map that represents mapping shards onto the
// 2d physical device grid. A typical example would nearly be identity:
//   (d0, d1, d2)[s0] -> ( # affine symbol baseOffset
//     0,                  # Device index
//     d0,                 # GridDimY
//     d1,                 # GridDimX
//     s0 + d2             # L1 offset
//   )
//
//  Note the symbols are only there to maintain a congruent API with
//  createDramMap which actually does use the shard shape for address
//  calculation.  Perhaps in the future a more sophisticated mapping might use
//  them.
//
static mlir::AffineMap createL1Map(mlir::MLIRContext *context,
                                   GridAttr workerGrid) {
  mlir::AffineMap workerMap = workerGrid.getMapping();
  // Take the workerMap and just add an additional dimension for the L1 shard
  // offset for each core.
  mlir::SmallVector<mlir::AffineExpr> workerMapExprs(workerMap.getResults());
  mlir::AffineExpr baseOffset = getAffineSymbolExpr(0, context);
  workerMapExprs.push_back(baseOffset +
                           getAffineDimExpr(workerMap.getNumDims(), context));
  return mlir::AffineMap::get(workerMap.getNumDims() + 1, 1, workerMapExprs,
                              context);
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
// linear layout onto physical dram banks. The affine map round robin's the bank
// in pages of page size.
//   (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (
//                |   |   |   |   |   |
//                |   |   |   |   |   +- Base Address
//                |   |   |   |   +- Page size
//                |   |   |   +- Shard Dim X
//                |   |   +- Shard Dim Y
//                |   +- Grid Dim X
//                +- Grid Dim Y
//     0,                                                 # Device index
//     0,                                                 # Not Applicable
//     (addr floordiv s4) mod 12,                         # Channel Idx
//     (addr floordiv (s4 * 12)) * s4 + addr mod s4 + s5  # Channel Offset
//   )
//
// Where `addr` is the linearized address as though it were indexing all of DRAM
// flat:
//   addr = (d0 * s2 * s3 * s1) + (d1 * s2 * s3) + d2
//
static mlir::AffineMap createDramMap(::mlir::MLIRContext *context,
                                     GridAttr workerGrid, size_t numDramCores,
                                     size_t dramPageSize) {
  mlir::AffineMap workerMap = workerGrid.getMapping();
  assert(workerMap.getNumResults() == PhysGridResultIdx::NumIndices);

  mlir::AffineExpr shardVolumeExpr = getAffineConstantExpr(1, context);
  for (int i = workerMap.getNumDims() - 1; i >= 0; i--) {
    mlir::AffineExpr shardDim =
        getAffineSymbolExpr(workerMap.getNumDims() + i, context);
    shardVolumeExpr = shardDim * shardVolumeExpr;
  }

  mlir::AffineExpr addr = getAffineDimExpr(workerMap.getNumDims(), context);
  mlir::AffineExpr gridVolumeExpr = getAffineConstantExpr(1, context);
  for (int i = workerMap.getNumDims() - 1; i >= 0; i--) {
    mlir::AffineExpr dim = getAffineDimExpr(i, context);
    mlir::AffineExpr gridDim = getAffineSymbolExpr(i, context);
    addr = dim * gridVolumeExpr * shardVolumeExpr + addr;
    gridVolumeExpr = gridVolumeExpr * gridDim;
  }

  mlir::AffineExpr pageSizeExpr =
      getAffineSymbolExpr(workerMap.getNumDims() * 2, context);
  mlir::AffineExpr baseAddressExpr =
      getAffineSymbolExpr(workerMap.getNumDims() * 2 + 1, context);
  mlir::AffineExpr numDramCoresExpr =
      getAffineConstantExpr(numDramCores, context);
  mlir::SmallVector<mlir::AffineExpr> dramMapResults = {
      getAffineConstantExpr(0, context),
      getAffineConstantExpr(0, context),
      addr.floorDiv(pageSizeExpr) % numDramCoresExpr,
      addr.floorDiv(pageSizeExpr * numDramCoresExpr) + addr % pageSizeExpr +
          baseAddressExpr,
  };

  return mlir::AffineMap::get(workerMap.getNumDims() + 1,
                              workerMap.getNumDims() * 2 + 2, dramMapResults,
                              context);
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

  return createDramMap(context, workerGrid, firstDramCores.size(),
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
  auto l1Map = createL1Map(context, workerGrid);
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

mlir::AffineMap DeviceAttr::getMemoryMap(MemRefType memrefType, size_t pageSize,
                                         size_t baseOffset) const {
  tt::MemorySpace memorySpace =
      mlir::cast<MemorySpaceAttr>(memrefType.getMemorySpace()).getValue();
  AffineMap affineMap = memrefType.getLayout().getAffineMap();
  switch (memorySpace) {
  case MemorySpace::DeviceL1: {
    SmallVector<int64_t> symbols = {static_cast<int64_t>(baseOffset)};
    return ttmlir::utils::replaceAffineMapSymbols(getL1Map(), symbols)
        .compose(affineMap);
  }
  case MemorySpace::DeviceDRAM: {
    assert(pageSize > 0 && "expected positive page size");
    SmallVector<int64_t> symbols(memrefType.getShape());
    symbols.push_back(static_cast<int64_t>(pageSize));
    symbols.push_back(static_cast<int64_t>(baseOffset));
    return ttmlir::utils::replaceAffineMapSymbols(getDramMap(), symbols)
        .compose(affineMap);
  }
  default: {
    llvm_unreachable("Unsupported memory space");
  }
  }
}

size_t DeviceAttr::getMemrefSizeBytes(MemRefType memrefType,
                                      size_t pageSize) const {
  // TODO(nsmith): We need to implement this somehow
  assert(false);
  return 0;
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
      getMemoryMap(layout.getMemref(), 0));
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
  case DataType::Int32:
    return getHeight() * getWidth() * 4;
  case DataType::UInt16:
    return getHeight() * getWidth() * 2;
  case DataType::UInt8:
    return getHeight() * getWidth();
  }
}

mlir::Type TileType::getElementType() const {
  return dataTypeToElementType(getContext(), getDataType());
}

void TTDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.cpp.inc"
      >();
}
