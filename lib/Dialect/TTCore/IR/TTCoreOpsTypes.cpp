// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Target/Common/Target.h"
#include "ttmlir/Target/Common/system_desc_bfbs_hash_generated.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <numeric>

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsEnums.cpp.inc"

#include "ttmlir/Dialect/TTCore/IR/TTCoreAttrInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.cpp.inc"
#include "ttmlir/Target/Common/types_generated.h"

namespace mlir::tt::ttcore {

namespace {
SystemDescAttr createDefaultBlackholeSystemDesc(
    mlir::MLIRContext *context, const ::llvm::SmallVector<int64_t> &meshShape) {
  // Set default values
  constexpr auto l1Size = 1572864;
  constexpr auto numDramChannels = 8;
  constexpr auto dramChannelSize = 4278190080;
  constexpr auto nocL1AddressAlignBytes = 16;
  constexpr auto pcieAddressAlignBytes = 64;
  constexpr auto nocDRAMAddressAlignBytes = 64;
  constexpr auto l1UnreservedBase = 98304;
  constexpr auto eriscL1UnreservedBase = 100032;
  constexpr auto dramUnreservedBase = 64;
  constexpr auto dramUnreservedEnd = 4276383744;
  constexpr auto dstRegisterSizeTiles = 8;
  constexpr auto numCBs = 32;
  constexpr auto numComputeThreads = 1;
  constexpr auto numDatamovementThreads = 2;

  // Get number of chips in mesh.
  int64_t numberOfChips =
      std::accumulate(meshShape.begin(), meshShape.end(), int64_t{1},
                      std::multiplies<int64_t>());

  // Populate dummy values for single chip or multi chip config.
  llvm::SmallVector<std::int64_t> gridShape = {10, 13};

  // Physical-to-translated coordinate translation offsets
  llvm::SmallVector<std::int64_t> coordTranslationOffsets = {18, 18};

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

  // Get number of chips indices.
  llvm::SmallVector<uint32_t> chipIndicesList =
      llvm::to_vector(llvm::seq<uint32_t>(numberOfChips));

  // Duplicate number of chip desc attributes based on number of chips.
  llvm::SmallVector<ChipDescAttr> chipDescs;
  chipDescs.reserve(numberOfChips);

  for (auto i = 0; i < numberOfChips; i++) {
    chipDescs.push_back(ChipDescAttr::get(
        context, ArchAttr::get(context, Arch::Blackhole), gridShape,
        coordTranslationOffsets, l1Size, numDramChannels, dramChannelSize,
        nocL1AddressAlignBytes, pcieAddressAlignBytes, nocDRAMAddressAlignBytes,
        l1UnreservedBase, eriscL1UnreservedBase, dramUnreservedBase,
        dramUnreservedEnd, supported_data_types, supported_tile_sizes,
        dstRegisterSizeTiles, numCBs, numComputeThreads,
        numDatamovementThreads));
  }

  // Duplicate number of chip capabilities based on number of chips.
  llvm::SmallVector<ChipCapabilityAttr> chipCapabilities;
  chipCapabilities.reserve(numberOfChips);

  for (auto i = 0; i < numberOfChips; i++) {
    chipCapabilities.push_back(
        ChipCapabilityAttr::get(context, ChipCapability::HostMMIO));
  }

  // Update chip channels based on number of chips.
  llvm::SmallVector<ChipChannelAttr> chipChannelList;
  chipChannelList.reserve(numberOfChips);

  if (numberOfChips != 1) {
    for (auto i = 0; i < numberOfChips; i++) {
      // Assume a default ring topology where final chip connects with initial
      // chip.
      chipChannelList.push_back(ChipChannelAttr::get(
          context, i, {0, 0}, (i + 1) % numberOfChips, {0, 0}));
    }
  }

  return SystemDescAttr::get(
      context,
      // CPU Descriptors
      {CPUDescAttr::get(context, CPURole::Host,
                        mlir::StringAttr::get(context, "x86_64-pc-linux-gnu"))},
      // Chip Descriptors
      chipDescs,
      // Chip Descriptor Indices
      chipIndicesList,
      // Chip capabilities
      chipCapabilities,
      // Chip Mesh Coordinates
      {
          ChipCoordAttr::get(context, 0, 0, 0, 0),
      },
      // Chip Channel Connections
      chipChannelList);
}

SystemDescAttr
createDefaultWormholeSystemDesc(mlir::MLIRContext *context,
                                const ::llvm::SmallVector<int64_t> &meshShape) {
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
  constexpr auto dstRegisterSizeTiles = 8;
  constexpr auto numCBs = 32;
  constexpr auto numComputeThreads = 1;
  constexpr auto numDatamovementThreads = 2;

  // Get number of chips in mesh.
  int64_t numberOfChips =
      std::accumulate(meshShape.begin(), meshShape.end(), int64_t{1},
                      std::multiplies<int64_t>());

  // Populate dummy values for single chip or multi chip config.
  llvm::SmallVector<std::int64_t> gridShape = {8, 8};

  // Physical-to-translated coordinate translation offsets
  llvm::SmallVector<std::int64_t> coordTranslationOffsets = {18, 18};

  // Populate a placeholder for supported tile sizes.
  llvm::SmallVector<DataTypeAttr> supportedDataTypes = {
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
  llvm::SmallVector<TileSizeAttr> supportedTileSizes = {
      TileSizeAttr::get(context, 4, 16),  TileSizeAttr::get(context, 16, 16),
      TileSizeAttr::get(context, 32, 16), TileSizeAttr::get(context, 4, 32),
      TileSizeAttr::get(context, 16, 32), TileSizeAttr::get(context, 32, 32),
  };

  // Get number of chips indices.
  llvm::SmallVector<uint32_t> chipIndicesList =
      llvm::to_vector(llvm::seq<uint32_t>(numberOfChips));

  // Duplicate number of chip desc attributes based on number of chips.
  llvm::SmallVector<ChipDescAttr> chipDescs;
  chipDescs.reserve(numberOfChips);

  for (auto i = 0; i < numberOfChips; i++) {
    chipDescs.push_back(ChipDescAttr::get(
        context, ArchAttr::get(context, Arch::WormholeB0), gridShape,
        coordTranslationOffsets, l1Size, numDramChannels, dramChannelSize,
        nocL1AddressAlignBytes, pcieAddressAlignBytes, nocDRAMAddressAlignBytes,
        l1UnreservedBase, eriscL1UnreservedBase, dramUnreservedBase,
        dramUnreservedEnd, supportedDataTypes, supportedTileSizes,
        dstRegisterSizeTiles, numCBs, numComputeThreads,
        numDatamovementThreads));
  }

  // Duplicate number of chip capabilities based on number of chips.
  llvm::SmallVector<ChipCapabilityAttr> chipCapabilities;
  chipCapabilities.reserve(numberOfChips);

  for (auto i = 0; i < numberOfChips; i++) {
    chipCapabilities.push_back(
        ChipCapabilityAttr::get(context, ChipCapability::HostMMIO));
  }

  // Update chip channels based on number of chips.
  llvm::SmallVector<ChipChannelAttr> chipChannelList;
  chipChannelList.reserve(numberOfChips);

  if (numberOfChips != 1) {
    for (auto i = 0; i < numberOfChips; i++) {
      // Assume a default ring topology where final chip connects with initial
      // chip.
      chipChannelList.push_back(ChipChannelAttr::get(
          context, i, {0, 0}, (i + 1) % numberOfChips, {0, 0}));
    }
  }

  return SystemDescAttr::get(
      context,
      // CPU Descriptors
      {CPUDescAttr::get(context, CPURole::Host,
                        mlir::StringAttr::get(context, "x86_64-pc-linux-gnu"))},
      // Chip Descriptors
      chipDescs,
      // Chip Descriptor Indices
      chipIndicesList,
      // Chip capabilities
      chipCapabilities,
      // Chip Mesh Coordinates
      {
          ChipCoordAttr::get(context, 0, 0, 0, 0),
      },
      // Chip Channel Connections
      chipChannelList);
}
} // namespace

SystemDescAttr
SystemDescAttr::getDefault(MLIRContext *context, Arch arch,
                           const ::llvm::SmallVector<int64_t> &meshShape) {
  switch (arch) {
  case Arch::WormholeB0:
    return createDefaultWormholeSystemDesc(context, meshShape);
  case Arch::Blackhole:
    return createDefaultBlackholeSystemDesc(context, meshShape);
  default:
    llvm_unreachable("Unsupported arch");
  }
}

mlir::FailureOr<SystemDescAttr> SystemDescAttr::getFromPath(
    MLIRContext *context, StringRef path,
    llvm::function_ref<mlir::InFlightDiagnostic()> diagFn) {
  if (path.empty()) {
    diagFn() << "system desc path must not be empty";
    return failure();
  }

  std::ifstream fbb(path.data(), std::ios::binary | std::ios::ate);
  if (!fbb.good()) {
    diagFn() << "system desc does not exist: " << path;
    return failure();
  }
  std::streampos size = fbb.tellg();
  fbb.seekg(0, std::ios::beg);
  auto buffer = std::shared_ptr<void>(std::malloc(size), std::free);
  fbb.read(static_cast<char *>(buffer.get()), size);

  return SystemDescAttr::getFromBuffer(context, buffer.get(), diagFn);
}

mlir::FailureOr<SystemDescAttr> SystemDescAttr::getFromBuffer(
    MLIRContext *context, void *systemDesc,
    llvm::function_ref<mlir::InFlightDiagnostic()> diagFn) {
  // Read relevant information from binary
  const auto *binarySystemDescRoot =
      ::tt::target::GetSizePrefixedSystemDescRoot(systemDesc);
  if (binarySystemDescRoot->schema_hash()->string_view() !=
      ::tt::target::common::system_desc_bfbs_schema_hash) {
    diagFn() << "system desc schema mismatch, please collect a system desc "
                "with a runtime compiled with the same schema version";
    return failure();
  }

  const auto *binarySystemDesc = binarySystemDescRoot->system_desc();
  const auto *binaryCpuDesc = binarySystemDesc->cpu_descs();
  const auto *binaryChipDesc = binarySystemDesc->chip_descs();
  const auto *binaryChipDescIndices = binarySystemDesc->chip_desc_indices();
  const auto *chipCapabilities = binarySystemDesc->chip_capabilities();
  const auto *binaryChipCoords = binarySystemDesc->chip_coords();
  const auto *chipChannelConnections = binarySystemDesc->chip_channels();

  // Acquire cpu descs
  std::vector<CPUDescAttr> cpuDescList;
  for (const auto *element : *binaryCpuDesc) {
    static_assert(llvm::to_underlying(CPURole::Device) ==
                  llvm::to_underlying(CPURole::Device));
    static_assert(llvm::to_underlying(CPURole::Host) ==
                  llvm::to_underlying(::tt::target::CPURole::Host));
    const auto *flatbufferTargetTripleString = element->target_triple();
    cpuDescList.emplace_back(CPUDescAttr::get(
        context, static_cast<CPURole>(element->role()),
        mlir::StringAttr::get(
            context, std::string(flatbufferTargetTripleString->c_str(),
                                 flatbufferTargetTripleString->size()))));
  }

  // Acquire chip descs
  std::vector<ChipDescAttr> chipDescList;
  for (const auto *element : *binaryChipDesc) {
    Arch arch;
    switch (element->arch()) {
    case ::tt::target::Arch::Grayskull:
      arch = Arch::Grayskull;
      break;
    case ::tt::target::Arch::Wormhole_b0:
      arch = Arch::WormholeB0;
      break;
    case ::tt::target::Arch::Blackhole:
      arch = Arch::Blackhole;
      break;
    }

    std::vector<DataTypeAttr> supportedDataTypesAttr;

    for (auto it : *(element->supported_data_types())) {
      switch (it) {
      case ::tt::target::DataType::Float32:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::Float32));
        break;
      case ::tt::target::DataType::Float16:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::Float16));
        break;
      case ::tt::target::DataType::BFloat16:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::BFloat16));
        break;
      case ::tt::target::DataType::BFP_Float8:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::BFP_Float8));
        break;
      case ::tt::target::DataType::BFP_BFloat8:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::BFP_BFloat8));
        break;
      case ::tt::target::DataType::BFP_Float4:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::BFP_Float4));
        break;
      case ::tt::target::DataType::BFP_BFloat4:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::BFP_BFloat4));
        break;
      case ::tt::target::DataType::BFP_Float2:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::BFP_Float2));
        break;
      case ::tt::target::DataType::BFP_BFloat2:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::BFP_BFloat2));
        break;
      case ::tt::target::DataType::UInt32:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::UInt32));
        break;
      case ::tt::target::DataType::UInt16:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::UInt16));
        break;
      case ::tt::target::DataType::UInt8:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::UInt8));
        break;
      case ::tt::target::DataType::Int32:
        supportedDataTypesAttr.push_back(
            DataTypeAttr::get(context, DataType::Int32));
        break;
      // Unsupported data types
      // We will list the cases here (rather than use `default`) so that
      // if new supported data types are added, we will get a compile error.
      case ::tt::target::DataType::Float64:
      case ::tt::target::DataType::Int64:
      case ::tt::target::DataType::UInt64:
      case ::tt::target::DataType::Int16:
      case ::tt::target::DataType::Int8:
      case ::tt::target::DataType::Bool:
        break;
      }
    }

    SmallVector<TileSizeAttr> supportedTileSizesAttr;

    for (const auto *it : *(element->supported_tile_sizes())) {
      supportedTileSizesAttr.push_back(
          TileSizeAttr::get(context, it->y(), it->x()));
    }

    auto currentChipDescAttr = ChipDescAttr::get(
        context, ArchAttr::get(context, arch),
        {element->grid_size()->y(), element->grid_size()->x()},
        {element->coord_translation_offsets()->y(),
         element->coord_translation_offsets()->x()},
        element->l1_size(), element->num_dram_channels(),
        element->dram_channel_size(), element->noc_l1_address_align_bytes(),
        element->pcie_address_align_bytes(),
        element->noc_dram_address_align_bytes(), element->l1_unreserved_base(),
        element->erisc_l1_unreserved_base(), element->dram_unreserved_base(),
        element->dram_unreserved_end(), supportedDataTypesAttr,
        supportedTileSizesAttr, element->dst_register_size_tiles(),
        element->num_cbs(), element->num_compute_threads(),
        element->num_datamovement_threads());
    chipDescList.push_back(currentChipDescAttr);
  }

  // Acquire chip indices
  std::vector<uint32_t> chipIndicesList;
  for (auto element : *binaryChipDescIndices) {
    chipIndicesList.push_back(element);
  }

  // Acquire chip capabilities
  std::vector<ChipCapabilityAttr> chipCapabilitiesList;
  for (auto element : *chipCapabilities) {
    static_assert(llvm::to_underlying(ChipCapability::HostMMIO) ==
                  llvm::to_underlying(::tt::target::ChipCapability::HostMMIO));

    auto chipCapabilitiesAttr =
        ChipCapabilityAttr::get(context, static_cast<ChipCapability>(element));
    chipCapabilitiesList.push_back(chipCapabilitiesAttr);
  }

  // Acquire chip coordinates
  std::vector<ChipCoordAttr> chipCoordinateList;
  for (const auto *element : *binaryChipCoords) {
    auto chipCoordinateAttr = ChipCoordAttr::get(
        context, element->rack(), element->shelf(), element->y(), element->x());
    chipCoordinateList.push_back(chipCoordinateAttr);
  }

  std::vector<ChipChannelAttr> chipChannelList;
  for (const auto *element : *chipChannelConnections) {
    std::vector<int64_t> ethernetCoreCoord0Vec = {
        element->ethernet_core_coord0().y(),
        element->ethernet_core_coord0().x()};

    std::vector<int64_t> ethernetCoreCoord1Vec = {
        element->ethernet_core_coord1().y(),
        element->ethernet_core_coord1().x()};

    auto chipChannelAttr = ChipChannelAttr::get(
        context, element->device_id0(), ethernetCoreCoord0Vec,
        element->device_id1(), ethernetCoreCoord1Vec);
    chipChannelList.push_back(chipChannelAttr);
  }

  // Generate system desc attribute
  auto systemDescAttr = SystemDescAttr::get(
      context, cpuDescList, chipDescList, chipIndicesList, chipCapabilitiesList,
      chipCoordinateList, chipChannelList);

  return systemDescAttr;
}

ChipDescAttr SystemDescAttr::getChipDesc(unsigned chipIndex) const {
  return getChipDescs()[getChipDescIndices()[chipIndex]];
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
  return getChipDesc(chipIndex).getNocL1AddressAlignBytes();
}

unsigned SystemDescAttr::getNocDRAMAddressAlignBytes(unsigned chipIndex) const {
  return getChipDesc(chipIndex).getNocDRAMAddressAlignBytes();
}

unsigned SystemDescAttr::getPcieAddressAlignBytes(unsigned chipIndex) const {
  return getChipDesc(chipIndex).getPcieAddressAlignBytes();
}

ShardLayoutAttr ShardLayoutAttr::get(mlir::MLIRContext *context,
                                     ArrayRef<int64_t> shape,
                                     uint64_t elementSize, uint32_t buffers) {
  return get(
      context,
      ttmlir::utils::calculateStrides(shape, static_cast<int64_t>(elementSize)),
      buffers);
}

ShardLayoutAttr ShardLayoutAttr::get(ArrayRef<int64_t> shape, Type elementType,
                                     uint32_t buffers) {
  return get(elementType.getContext(), shape, getElementSizeBytes(elementType),
             buffers);
}

ShardLayoutAttr ShardLayoutAttr::get(mlir::MemRefType memrefType,
                                     uint32_t buffers) {
  ArrayRef<int64_t> shape = memrefType.getShape();
  if (auto layout =
          mlir::dyn_cast<DeviceLayoutInterface>(memrefType.getLayout())) {
    shape = layout.getShardShape(memrefType);
  }
  return get(shape, memrefType.getElementType(), buffers);
}

mlir::AffineMap ShardLayoutAttr::getAffineMap() const {
  return ttmlir::utils::generateAffineMapFromShardStrides(getStride(),
                                                          getContext());
}

InterleavedLayoutAttr InterleavedLayoutAttr::get(mlir::MLIRContext *context,
                                                 ArrayRef<int64_t> shape,
                                                 uint64_t elementSize) {
  return get(context, ttmlir::utils::calculateStrides(
                          shape, static_cast<int64_t>(elementSize)));
}

InterleavedLayoutAttr InterleavedLayoutAttr::get(ArrayRef<int64_t> shape,
                                                 Type elementType) {
  return get(elementType.getContext(), shape, getElementSizeBytes(elementType));
}

InterleavedLayoutAttr InterleavedLayoutAttr::get(mlir::MemRefType memrefType) {
  ArrayRef<int64_t> shape = memrefType.getShape();
  if (auto layout =
          mlir::dyn_cast<DeviceLayoutInterface>(memrefType.getLayout())) {
    shape = layout.getShardShape(memrefType);
  }
  return get(shape, memrefType.getElementType());
}

mlir::AffineMap InterleavedLayoutAttr::getAffineMap() const {
  return ttmlir::utils::generateAffineMapFromShardStrides(getStride(),
                                                          getContext());
}

ViewLayoutAttr ViewLayoutAttr::compose(ViewLayoutAttr g) const {
  return get(getContext(), getAffineMap().compose(g.getAffineMap()));
}

mlir::AffineMap HostLayoutAttr::getAffineMap() const {
  auto *context = getContext();
  const int64_t rank = getLogicalShape().size();
  SmallVector<int64_t> strides{getHostStrides()};
  assert(strides.size() == static_cast<size_t>(rank));

  // Expression for generating linearized/flattened offsets.
  mlir::AffineExpr linearAddrExpr = getAffineConstantExpr(0, context);
  for (int64_t i = rank - 1; i >= 0; i--) {
    auto dim = getAffineDimExpr(i, context);
    auto stride = getAffineConstantExpr(strides[i], context);
    linearAddrExpr = dim * stride + linearAddrExpr;
  }

  return mlir::AffineMap::get(rank, 0, linearAddrExpr, context);
}

LogicalResult
HostLayoutAttr::getStridesAndOffset(ArrayRef<int64_t> shape,
                                    SmallVectorImpl<int64_t> &strides,
                                    int64_t &offset) const {
  const int64_t rank = getHostStrides().size();
  assert(rank == static_cast<int64_t>(shape.size()));
  strides.resize_for_overwrite(rank);

  for (int64_t i = 0; i < rank; i++) {
    strides[i] = getHostStrides()[i];
  }
  offset = 0;
  return success();
}

bool HostLayoutAttr::isPadded() const {
  return getHostVolume() > ttmlir::utils::volume(getLogicalShape());
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
    if (begin >= end) {
      continue;
    }
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

  return mlir::simplifyAffineMap(map);
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

static llvm::SmallVector<int64_t>
applyCollapsedIntervalsAndAlignments(llvm::ArrayRef<int64_t> shape,
                                     mlir::DenseIntElementsAttr intervals,
                                     llvm::ArrayRef<int64_t> alignments) {
  assert(shape.size() == alignments.size() &&
         "Shape and alignments must have same size");

  llvm::SmallVector<int64_t> resultShape;

  // Process with collapse intervals.
  auto values = intervals.getValues<int64_t>();
  assert(intervals && intervals.getType().getShape()[1] == 2);
  auto numIntervals = intervals.getType().getShape()[0];

  int64_t currentIdx = 0;

  for (int64_t i = 0; i < numIntervals; ++i) {
    int64_t start = values[i * 2];
    int64_t end = values[i * 2 + 1];

    // Handle Python-like negative indexing.
    if (start < 0) {
      start = shape.size() + start;
    }
    if (end < 0) {
      end = shape.size() + end;
    }

    assert(start >= 0 && static_cast<size_t>(start) < shape.size() &&
           "Start index out of bounds");
    assert(end >= start && static_cast<size_t>(end) <= shape.size() &&
           "End index out of bounds");

    if (end - start == 1) {
      // Single dimension - apply alignment.
      resultShape.push_back(
          ttmlir::utils::alignUp(shape[start], alignments[start]));
    } else if (end > start) {
      // Start by aligning the innermost dimension.
      int64_t collapsedDim =
          ttmlir::utils::alignUp(shape[end - 1], alignments[end - 1]);

      // Process remaining dimensions from inner to outer w/ multiplication.
      for (int64_t j = end - 2; j >= start; --j) {
        // Multiply and then align up means that the alignment is in fact, the
        // intentionally imposed alignment for the current collapse stage of the
        // current collapsed interval.
        collapsedDim =
            ttmlir::utils::alignUp(shape[j] * collapsedDim, alignments[j]);
      }

      resultShape.push_back(collapsedDim);
    }
    currentIdx = end;
  }

  // Handle remaining dimensions with alignment.
  for (int64_t i = shape.size() - 1; i >= currentIdx; --i) {
    resultShape.push_back(ttmlir::utils::alignUp(shape[i], alignments[i]));
  }

  return resultShape;
}

llvm::SmallVector<int64_t>
MetalLayoutAttr::getPhysicalShape(ArrayRef<int64_t> tileShape) const {
  llvm::SmallVector<int64_t> physicalShape =
      applyCollapsedIntervalsAndAlignments(
          getLogicalShape(), getCollapsedIntervals(), getDimAlignments());
  if (!tileShape.empty()) {
    assert(physicalShape.size() >= 2);
    assert(tileShape.size() == 2);
    assert(physicalShape[physicalShape.size() - 2] % tileShape[0] == 0);
    physicalShape[physicalShape.size() - 2] /= tileShape[0];
    assert(physicalShape[physicalShape.size() - 1] % tileShape[1] == 0);
    physicalShape[physicalShape.size() - 1] /= tileShape[1];
  }
  return physicalShape;
}

// Takes various shape fields and returns the expected physical shape, which
// should be the actual tensor shape.
llvm::SmallVector<int64_t>
MetalLayoutAttr::getDeviceShape(ArrayRef<int64_t> gridShape,
                                ArrayRef<int64_t> tileShape) const {
  llvm::SmallVector<int64_t> physicalShape = getPhysicalShape(tileShape);
  llvm::SmallVector<int64_t> deviceShape(gridShape);
  deviceShape.reserve(physicalShape.size() * 2);

  assert(physicalShape.size() == gridShape.size() &&
         "Grid rank must equalcollapsed tensor rank");
  // Without tiling, distribute dimensions across grid.
  for (size_t i = 0; i < physicalShape.size(); ++i) {
    const int64_t dim = physicalShape[i];
    assert(dim % gridShape[i] == 0 &&
           "Collapsed dimension must be evenly divisible by grid dimension");
    deviceShape.push_back(dim / gridShape[i]);
  }
  return deviceShape;
}

static llvm::SmallVector<int64_t>
normalizeAndFlattenIntervals(const mlir::DenseIntElementsAttr &inputIntervals,
                             size_t inputRank) {
  auto values = inputIntervals.getValues<int64_t>();
  auto it = values.begin();

  // Store intervals as pairs for easier sorting.
  llvm::SmallVector<std::pair<int64_t, int64_t>> intervals;
  while (it != values.end()) {
    int64_t start = *it++;
    int64_t end = *it++;

    // Handle negative indexing (Python-style).
    if (start < 0) {
      start += inputRank;
    }
    if (end < 0) {
      end += inputRank;
    }

    intervals.push_back({start, end});
  }

  // Sort intervals by start position.
  llvm::sort(intervals,
             [](const auto &a, const auto &b) { return a.first < b.first; });

  llvm::SmallVector<int64_t> normalized;
  // Track highest endpoint we've covered s.t. we can handle uncovered dims
  // easily afterwards.
  int64_t coveredUpTo = 0;

  // Add sorted intervals and track coverage.
  for (const auto &[start, end] : intervals) {
    normalized.push_back(start);
    normalized.push_back(end);
    coveredUpTo = std::max(coveredUpTo, end);
  }

  // Any uncovered dim becomes it's own interval.
  while (coveredUpTo < static_cast<int64_t>(inputRank)) {
    normalized.push_back(coveredUpTo);
    normalized.push_back(++coveredUpTo);
  }

  return normalized;
}

llvm::SmallVector<int64_t> MetalLayoutAttr::getNormalizedIntervals() const {
  return normalizeAndFlattenIntervals(getCollapsedIntervals(),
                                      getLogicalShape().size());
}

mlir::AffineMap
MetalLayoutAttr::getIndexAffineMapOrIdentity(unsigned rank) const {
  mlir::AffineMap map = getIndexAffineMap();
  if (!map || map.getNumResults() == 0) {
    return mlir::AffineMap::getMultiDimIdentityMap(rank, getContext());
  }
  return map;
}

// When tile-alignment is needed and the height dim of the physical shape is
// collapsed from multiple logical dims, we record two alignment requirements:
// - One for the 2nd last logical dim (intervalEnd), and it's always the height
// of a tile.
// - One for the outermost logical dim (intervalStart), the value that: after
// applying it to the aligned & collapsed dim, the entire height dim becomes
// tile/grid-aligned, depending on the need.
//
// For example, tensor<4x43x7> + grid<8x8> + tile<32x32> gives:
// - Tile-aligned logical shape 4x64x32.
// - Collapsed physical 2D shape 256x32.
// - Dim alignments 1x32x32.
// - Unsharded device shape 256x32.
// - Grid & shard is 256x32 / 32x32 = 8x1.
// - Tiled device shape 8x1x1x1x!tile<32x32>.
// So there will be 8 workers, each handling a shard containing a single tile.
// This achieves high worker utilization, and ensures the majority of the
// padding are at the end of the tensor buffer so the memory access strides are
// small.
//
// This strategy has good work utilization but also has its issues. Consider the
// same example as above but with tensor<9x43x7>:
// - Tile-aligned logical shape 9x64x32.
// - Collapsed physical 2D shape 576x32.
// - Dim alignments 256x32x32.
// - Unsharded device shape [alignUp(9 * alignUp(43, 32), 256)]x32 = 768x32.
// - Grid & shard is 768x32 / 32x32 = 24x1.
// - Tiled device shape 8x1x3x1x!tile<32x32>.
// Similar to the example above, both result in 'unnatural' shard shapes like
// 1x1x!tile<32x32> and 3x1x!tile<32x32>. The 'natural' shard shape should be
// 2x1x!tile<32x32>, which has the potential to save some NoC traffic (e.g.
// reduction of 4x43x7 -> 4x1x7 is now worker-local).
llvm::SmallVector<int64_t>
MetalLayoutAttr::computeAlignments(ArrayRef<int64_t> logicalShape,
                                   ArrayRef<int64_t> deviceGridShape,
                                   ArrayRef<int64_t> normalizedIntervals) {
  constexpr std::array<int64_t, 2> tileShape = TileType::getDefaultShape();
  const int64_t logicalRank = logicalShape.size();
  const int64_t deviceGridRank = deviceGridShape.size();

  assert(logicalRank >= 2);
  assert(deviceGridRank == 2);
  assert(deviceGridRank * 2 ==
         static_cast<int64_t>(normalizedIntervals.size()));

  llvm::SmallVector<int64_t> alignments(logicalRank, 1);
  // Handle the last two intervals (which will map to tiles) with grid-aware
  // alignments.
  for (int64_t idx = -1; idx >= -2; idx--) {
    const int64_t tileIdx = tileShape.size() + idx;
    const int64_t tileDim = tileShape[tileIdx];

    const int64_t gridIdx = deviceGridRank + idx;
    const int64_t gridDim = deviceGridShape[gridIdx];

    const int64_t gridAlignmentThreshold = gridDim * tileDim;

    // Inclusive indices.
    const int64_t intervalStart = normalizedIntervals[gridIdx * 2];
    const int64_t intervalEnd = normalizedIntervals[gridIdx * 2 + 1] - 1;

    // Calculate collapsed size for this interval.
    int64_t collapsedSize = 1;
    for (int64_t i = intervalEnd; i >= intervalStart; i--) {
      if (i >= logicalRank - 2) {
        // Always tile-align the last two dimensions.
        collapsedSize *= ttmlir::utils::alignUp(logicalShape[i], tileDim);
      } else {
        collapsedSize *= logicalShape[i];
      }
    }

    // Align to grid boundary iff the collapsed size is strictly larger than
    // gridAlignmentThreshold, else align to tile boundary.
    const bool alignToGrid = collapsedSize > gridAlignmentThreshold;
    const int64_t alignment = alignToGrid ? gridAlignmentThreshold : tileDim;

    // Assume the collapsed intervals are always <[[0, N-2], [N-1, N]]>.
    if (intervalStart == intervalEnd) {
      alignments[intervalEnd] = alignment;
    } else {
      assert(idx == -2);
      assert(intervalEnd == logicalRank - 2);
      alignments[intervalEnd] = tileDim;
      // Avoid results like [32x32]x32, it should be [1x32]x32.
      if (alignToGrid) {
        alignments[intervalStart] = alignment;
      }
    }
  }
  assert(alignments[logicalRank - 1] % tileShape[1] == 0);
  assert(alignments[logicalRank - 2] % tileShape[0] == 0);
  return alignments;
}

// Getter with no intervals or alignments, we calculate them both.
MetalLayoutAttr MetalLayoutAttr::get(::mlir::MLIRContext *context,
                                     ArrayRef<int64_t> logicalShape,
                                     ArrayRef<int64_t> deviceGridShape,
                                     OOBVal oobVal, MemorySpace memorySpace,
                                     TensorMemoryLayout memoryLayout) {
  // Create collapse intervals.
  int64_t numDimsToCollapse = logicalShape.size() - deviceGridShape.size() + 1;
  llvm::SmallVector<int64_t> flattenedIntervals;

  // First interval will be [0, numDimsToCollapse).
  flattenedIntervals.push_back(0);
  flattenedIntervals.push_back(numDimsToCollapse);
  for (int64_t i = 1; i < static_cast<int64_t>(deviceGridShape.size()); ++i) {
    // Last gridRank - 1 intervals will be [i, i + 1).
    flattenedIntervals.push_back(numDimsToCollapse + i - 1);
    flattenedIntervals.push_back(numDimsToCollapse + i);
  }

  auto intervalType =
      RankedTensorType::get({static_cast<int64_t>(deviceGridShape.size()), 2},
                            IntegerType::get(context, 64));
  DenseIntElementsAttr collapsedIntervals =
      DenseIntElementsAttr::get(intervalType, flattenedIntervals);

  assert(collapsedIntervals.getType().getRank() == 2 &&
         "Collapse intervals must be a 2D array");

  // Set alignments based on the flattened intervals.
  llvm::SmallVector<int64_t> dimAlignmentsVec =
      computeAlignments(logicalShape, deviceGridShape, flattenedIntervals);

  return get(context, logicalShape, dimAlignmentsVec, collapsedIntervals,
             oobVal, memorySpace, memoryLayout, mlir::AffineMap::get(context));
}

// Getter with explicit collapsedIntervals, we calculate the alignments.
MetalLayoutAttr MetalLayoutAttr::get(::mlir::MLIRContext *context,
                                     ArrayRef<int64_t> logicalShape,
                                     ArrayRef<int64_t> deviceGridShape,
                                     OOBVal oobVal, MemorySpace memorySpace,
                                     TensorMemoryLayout memoryLayout,
                                     DenseIntElementsAttr collapsedIntervals) {
  llvm::SmallVector<int64_t> normalizedIntervals =
      normalizeAndFlattenIntervals(collapsedIntervals, logicalShape.size());
  llvm::SmallVector<int64_t> dimAlignmentsVec =
      computeAlignments(logicalShape, deviceGridShape, normalizedIntervals);

  return get(context, logicalShape, dimAlignmentsVec, collapsedIntervals,
             oobVal, memorySpace, memoryLayout, mlir::AffineMap::get(context));
}

// Getter with explicit collapsedIntervals and dimAlignments.
MetalLayoutAttr MetalLayoutAttr::get(::mlir::MLIRContext *context,
                                     ArrayRef<int64_t> logicalShape,
                                     ArrayRef<int64_t> deviceGridShape,
                                     OOBVal oobVal, MemorySpace memorySpace,
                                     TensorMemoryLayout memoryLayout,
                                     DenseIntElementsAttr collapsedIntervals,
                                     ArrayRef<int64_t> dimAlignments) {
  return get(context, logicalShape, dimAlignments, collapsedIntervals, oobVal,
             memorySpace, memoryLayout, mlir::AffineMap::get(context));
}

mlir::MemRefType
MetalLayoutAttr::getMemRefType(mlir::RankedTensorType tensorType) {

  if (!tensorType.getEncoding()) {
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  }

  auto layout = mlir::cast<MetalLayoutAttr>(tensorType.getEncoding());

  auto shardShape = layout.getShardShape(tensorType);

  return MemRefType::get(
      shardShape, tensorType.getElementType(), MemRefLayoutAttrInterface{},
      MemorySpaceAttr::get(tensorType.getContext(), layout.getMemorySpace()));
}

// 5-arg + explicit index_map convenience overload.
MetalLayoutAttr MetalLayoutAttr::get(::mlir::MLIRContext *context,
                                     ArrayRef<int64_t> logicalShape,
                                     ArrayRef<int64_t> deviceGridShape,
                                     OOBVal oobVal, MemorySpace memorySpace,
                                     TensorMemoryLayout memoryLayout,
                                     mlir::AffineMap indexAffineMap) {
  // Reuse the existing path that computes intervals/alignments, then attach
  // map.
  MetalLayoutAttr base = get(context, logicalShape, deviceGridShape, oobVal,
                             memorySpace, memoryLayout);
  return get(context, base.getLogicalShape(), base.getDimAlignments(),
             base.getCollapsedIntervals(), base.getOobVal(),
             base.getMemorySpace(), base.getMemoryLayout(), indexAffineMap);
}

// Get effective stride (use provided or calculate from shape)
llvm::SmallVector<int64_t>
MetalLayoutAttr::getShardStride(RankedTensorType tensorType) const {
  auto shardShape = getShardShape(tensorType);
  SmallVector<int64_t> shardStride(shardShape.size());
  shardStride[shardStride.size() - 1] =
      getElementSizeBytes(tensorType.getElementType());
  for (int64_t i = static_cast<int64_t>(shardStride.size()) - 2; i >= 0; i--) {
    shardStride[i] = shardShape[i + 1] * shardStride[i + 1];
  }
  return shardStride;
}

std::pair<llvm::SmallVector<int64_t>, int64_t>
MetalLayoutAttr::getHostStrideAndVolume() const {
  const auto logicalShape = getLogicalShape();
  const auto alignments = getDimAlignments();
  const auto normalizedIntervals = getNormalizedIntervals();

  llvm::SmallVector<int64_t> strides(logicalShape.size(), 0);

  int64_t currentStride = 1;
  for (int i = -1; i >= -2; i--) {
    const int64_t intervalIdx = normalizedIntervals.size() / 2 + i;
    // Inclusive indices.
    const int64_t intervalStart = normalizedIntervals[intervalIdx * 2];
    const int64_t intervalEnd = normalizedIntervals[intervalIdx * 2 + 1] - 1;

    int64_t collapsedSize = 1;
    // Both the alignments and the collapsed sizes are "cumulative" relative to
    // the current collapse interval. But to update the current stride we need
    // the true per-dim alignment, which is difficult to obtain, especially when
    // the aligned up new collapsed size is not a multiple of the old collapsed
    // size.
    // Solution: revert the current stride to before the current collapse
    // interval, and then update it straight to the current collapse stage.
    for (int64_t j = intervalEnd; j >= intervalStart; j--) {
      strides[j] = currentStride;
      currentStride /= collapsedSize;
      collapsedSize = ttmlir::utils::alignUp(collapsedSize * logicalShape[j],
                                             alignments[j]);
      currentStride *= collapsedSize;
    }
  }

  // At this point, currentStride == 'stride' of the entire tensor, i.e. volume.
  assert(currentStride >= ttmlir::utils::volume(logicalShape));
  return {strides, currentStride};
}

llvm::SmallVector<int64_t> MetalLayoutAttr::getHostStride() const {
  return getHostStrideAndVolume().first;
}

int64_t MetalLayoutAttr::getHostVolume() const {
  return getHostStrideAndVolume().second;
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
//   (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (
//                |   |   |   |   |   |   |
//                |   |   |   |   |   |   +- Element Size
//                |   |   |   |   |   +- Base Address
//                |   |   |   |   +- Page size
//                |   |   |   +- Shard Dim X
//                |   |   +- Shard Dim Y
//                |   +- Grid Dim X
//                +- Grid Dim Y
//     index[0]: Device ID
//     index[1]: _unused_
//     index[2]: Channel Idx
//       global_page_index % num_dram_banks
//     index[3]: Byte Offset In Channel
//       (channel_page_index * PAGE_SIZE) + (addr % PAGE_SIZE) + base_address
//   )
//
// Where `addr` is the linearized address as though it were indexing all of DRAM
// flat:
//   shard_vol = s2 * s3 * s6
//   addr = (d0 * shard_vol * s1) + (d1 * shard_vol) + d2
//
// Where global_page_index is the global page index corresponding to the
// address:
//   global_page_index  = addr floorDiv PAGE_SIZE
//
// Where channel_page_index is the page index within a bank
//   channel_page_index = global_page_index floorDiv NUM_DRAM_BANKS
//

static mlir::AffineMap createDramMap(::mlir::MLIRContext *context,
                                     GridAttr workerGrid, size_t numDramCores) {
  mlir::AffineMap workerMap = workerGrid.getMapping();
  assert(workerMap.getNumResults() == PhysGridResultIdx::NumIndices);

  size_t elemSizeIndex = workerMap.getNumDims() * 2 + 2;
  mlir::AffineExpr shardVolumeExpr =
      getAffineSymbolExpr(elemSizeIndex, context);
  for (int i = workerMap.getNumDims() - 1; i >= 0; i--) {
    mlir::AffineExpr shardDim =
        getAffineSymbolExpr(workerMap.getNumDims() + i, context);
    shardVolumeExpr = shardDim * shardVolumeExpr;
  }

  // flatAddr is an expression representing the address as-if the memory was
  // completely flat
  mlir::AffineExpr flatAddr = getAffineDimExpr(workerMap.getNumDims(), context);
  mlir::AffineExpr gridVolumeExpr = getAffineConstantExpr(1, context);
  for (int i = workerMap.getNumDims() - 1; i >= 0; i--) {
    mlir::AffineExpr dim = getAffineDimExpr(i, context);
    mlir::AffineExpr gridDim = getAffineSymbolExpr(i, context);
    flatAddr = dim * gridVolumeExpr * shardVolumeExpr + flatAddr;
    gridVolumeExpr = gridVolumeExpr * gridDim;
  }

  mlir::AffineExpr baseAddressExpr =
      getAffineSymbolExpr(workerMap.getNumDims() * 2 + 1, context);
  mlir::AffineExpr numDramBanksExpr =
      getAffineConstantExpr(numDramCores, context);
  mlir::AffineExpr dramPageSizeExpr =
      getAffineSymbolExpr(workerMap.getNumDims() * 2, context);
  mlir::AffineExpr pageIndex = flatAddr.floorDiv(dramPageSizeExpr);
  mlir::AffineExpr channelPageIndex = pageIndex.floorDiv(numDramBanksExpr);

  mlir::SmallVector<mlir::AffineExpr> dramMapResults = {
      getAffineConstantExpr(0, context), getAffineConstantExpr(0, context),
      pageIndex % numDramBanksExpr,
      (channelPageIndex * dramPageSizeExpr) + (flatAddr % dramPageSizeExpr) +
          baseAddressExpr};

  unsigned dimCount = workerMap.getNumDims() + 1;
  unsigned symbolCount = workerMap.getNumDims() * 2 + 3;
  return mlir::AffineMap::get(dimCount, symbolCount, dramMapResults, context);
}

static mlir::AffineMap createDramMap(::mlir::MLIRContext *context,
                                     GridAttr workerGrid,
                                     SystemDescAttr systemDesc,
                                     ::llvm::ArrayRef<unsigned> chipIds) {
  auto numDramChannels = systemDesc.getChipDescs().front().getNumDramChannels();
  return createDramMap(context, workerGrid, numDramChannels);
}

DeviceAttr DeviceAttr::get(::mlir::MLIRContext *context,
                           SystemDescAttr systemDesc,
                           ArrayRef<int64_t> meshShape,
                           ArrayRef<unsigned> chipIds) {
  assert(not chipIds.empty() && "expected at least one chip");
  ChipDescAttr chipDesc = systemDesc.getChipDesc(chipIds.front());
  SmallVector<int64_t> chipGrid(chipDesc.getGrid());
  assert(chipGrid.size() == 2 && "expected 2D grid");

  // Take the min grid shape across all chips, this can happen if 1 chip has
  // more harvested rows than another.  We take the min because we want to
  // ensure that the logical grid is the same across all chips and this vastly
  // simplifies the grid /memory mappings across the board.
  for (unsigned chipId : chipIds) {
    ChipDescAttr chip = systemDesc.getChipDesc(chipId);
    ArrayRef<int64_t> iChipGrid = chip.getGrid();
    assert(iChipGrid.size() == 2 && "expected 2D grid");
    chipGrid[0] = std::min(chipGrid[0], iChipGrid[0]);
    chipGrid[1] = std::min(chipGrid[1], iChipGrid[1]);
  }

  // Due mainly to SPMD programming model for multi-devices, workerGrid
  // currently is set to be limited to a single device {1, 1}.
  auto workerGrid = createWorkerGrid(context, chipGrid, {1, 1});
  auto l1Map = createL1Map(context, workerGrid);
  auto dramMap = createDramMap(context, workerGrid, systemDesc, chipIds);
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
                                         std::optional<AffineMap> view,
                                         size_t baseOffset) const {
  MemorySpace memorySpace =
      mlir::cast<MemorySpaceAttr>(memrefType.getMemorySpace()).getValue();
  AffineMap affineMap = memrefType.getLayout().getAffineMap();
  if (view) {
    affineMap = affineMap.compose(*view);
  }

  if (mlir::isa<ShardLayoutAttr>(memrefType.getLayout())) {

    switch (memorySpace) {
    case MemorySpace::DeviceL1: {
      SmallVector<int64_t> symbols = {static_cast<int64_t>(baseOffset)};
      return ttmlir::utils::replaceAffineMapSymbols(getL1Map(), symbols)
          .compose(affineMap);
    }
    case MemorySpace::DeviceDRAM: {
      // The DRAM page size is 1<->1 mapped to underlying memref shard size;
      // pageSize argument is ignored.
      pageSize = getMemrefSizeBytes(memrefType);
      assert(pageSize > 0 && "expected positive page size");
      SmallVector<int64_t> symbols(memrefType.getShape());
      symbols.push_back(static_cast<int64_t>(pageSize));
      symbols.push_back(static_cast<int64_t>(baseOffset));
      symbols.push_back(getElementSizeBytes(memrefType.getElementType()));
      return ttmlir::utils::replaceAffineMapSymbols(getDramMap(), symbols)
          .compose(affineMap);
    }
    default: {
      llvm_unreachable("Unsupported memory space");
    }
    }
  } else if (mlir::isa<ttcore::InterleavedLayoutAttr>(memrefType.getLayout())) {

    assert(memorySpace == MemorySpace::DeviceDRAM &&
           "interleavedLayoutAttr only supported for deviceDRAM memory space");

    auto interleavedLayout =
        mlir::cast<ttcore::InterleavedLayoutAttr>(memrefType.getLayout());

    // interleaved layout is either single tile or stride of outermost dim
    int64_t elementSizeBytes = getElementSizeBytes(memrefType.getElementType());
    pageSize = mlir::isa<ttcore::TileType>(memrefType.getElementType())
                   ? elementSizeBytes
                   : interleavedLayout.getStride().front();

    // interleaved layout for DRAM is constrained to have unit grid dims,
    // similar to TTNNLayoutAttr convention
    assert(ttmlir::utils::volume(interleavedLayout.getGridShape(memrefType)) ==
               1 &&
           "All dims in grid shape for DRAM interleaved memref must be 1 (i.e. "
           "1x1x...x1) ");

    SmallVector<int64_t> symbols(memrefType.getShape());
    symbols.push_back(static_cast<int64_t>(pageSize));
    symbols.push_back(static_cast<int64_t>(baseOffset));
    symbols.push_back(elementSizeBytes);

    return ttmlir::utils::replaceAffineMapSymbols(getDramMap(), symbols)
        .compose(affineMap);
  } else {
    assert(false && "Unsupported layout type on memref");
  }
}

mlir::AffineMap
DeviceAttr::getMemoryMap(std::pair<MemRefType, AffineMap> memrefAndView,
                         size_t pageSize, size_t baseOffset) const {
  return getMemoryMap(memrefAndView.first, pageSize, memrefAndView.second,
                      baseOffset);
}

size_t DeviceAttr::getShardSizeInBytes(MemRefType memrefType, size_t alignSize,
                                       bool includeBuffers) const {

  ShardLayoutAttr layout =
      mlir::dyn_cast<ShardLayoutAttr>(memrefType.getLayout());
  assert(
      (layout || !mlir::isa<DeviceLayoutInterface>(memrefType.getLayout())) &&
      "expected shard layout");
  bool isLocalMemref = (layout == nullptr);
  auto shardShape =
      isLocalMemref ? memrefType.getShape() : layout.getShardShape(memrefType);

  mlir::Type elementType = memrefType.getElementType();
  int64_t elementSizeBytes = getElementSizeBytes(elementType);
  auto numBuffers = (includeBuffers && layout) ? layout.getBuffers() : 1;
  auto bytesPerElem = elementSizeBytes * numBuffers;

  return ttmlir::utils::alignUp(
      static_cast<size_t>(ttmlir::utils::volume(shardShape, bytesPerElem)),
      alignSize);
}

size_t DeviceAttr::getMemrefSizeBytes(MemRefType memrefType, size_t pageSize,
                                      bool includeBuffers) const {
  size_t alignSize = pageSize;
  if (alignSize == 0) {
    auto memorySpace = getMemorySpace(memrefType);
    switch (memorySpace) {
    case MemorySpace::DeviceL1: {
      alignSize = getMemrefCBPageSizeBytes(memrefType);
      break;
    }
    case MemorySpace::DeviceDRAM: {
      alignSize = 1;
      break;
    }
    default: {
      llvm_unreachable("Unsupported memory space");
    }
    }
  }

  int64_t numBuffers = 1;
  ArrayRef<int64_t> shardShape;
  auto layout = memrefType.getLayout();
  if (auto devLayout = mlir::dyn_cast<DeviceLayoutInterface>(layout)) {
    shardShape = devLayout.getShardShape(memrefType);
    // if the layout is sharded, numBuffers is programmable
    if (auto shardLayout = mlir::dyn_cast<ShardLayoutAttr>(devLayout)) {
      numBuffers = (includeBuffers) ? shardLayout.getBuffers() : 1;
    }
  } else {
    // local memrefs have no layout attribute
    numBuffers = 1;
    shardShape = memrefType.getShape();
  }

  auto elementSizeBytes = getElementSizeBytes(memrefType.getElementType());
  return ttmlir::utils::alignUp(
      static_cast<size_t>(ttmlir::utils::volume(
          shardShape, static_cast<int64_t>(elementSizeBytes * numBuffers))),
      alignSize);
}

size_t DeviceAttr::getMemrefCBPageSizeBytes(MemRefType memrefType) const {
  mlir::Type elementType = memrefType.getElementType();
  TileType tileType = mlir::dyn_cast<TileType>(elementType);
  return tileType ? tileType.getSizeBytes()
                  : TileType::get(elementType).getSizeBytes();
}

size_t DeviceAttr::getMemrefCBNumPages(MemRefType memrefType) const {
  size_t sizeBytes =
      getMemrefSizeBytes(memrefType, getMemrefCBPageSizeBytes(memrefType),
                         /*includeBuffers=*/false);
  size_t pageSize = getMemrefCBPageSizeBytes(memrefType);
  assert(sizeBytes % pageSize == 0);
  return sizeBytes / pageSize;
}

::mlir::LogicalResult DeviceAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    GridAttr workerGrid, ::mlir::AffineMap l1Map, ::mlir::AffineMap dramMap,
    ::llvm::ArrayRef<int64_t> meshShape, ::llvm::ArrayRef<unsigned> chipIds) {
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

TileType TileType::get(Type elementType, ArrayRef<int64_t> shape) {
  return get(elementType.getContext(), shape,
             elementTypeToDataType(elementType));
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

void TTCoreDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.cpp.inc"
      >();
}
} // namespace mlir::tt::ttcore
