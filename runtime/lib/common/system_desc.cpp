// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"
#include "ttmlir/Target/Common/system_desc_bfbs_hash_generated.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"
#include "types_generated.h"
#include <cstdint>
#include <vector>

#define FMT_HEADER_ONLY
#include "hostdevcommon/common_values.hpp"
#include "tt-metalium/allocator.hpp"
#include "tt-metalium/hal.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_device.hpp"

namespace tt::runtime::system_desc {

using HalMemType = ::tt::tt_metal::HalMemType;
using BufferType = ::tt::tt_metal::BufferType;

static ::tt::target::Dim2d toFlatbuffer(const CoreCoord &coreCoord) {
  return ::tt::target::Dim2d(coreCoord.y, coreCoord.x);
}

static ::tt::target::Arch toFlatbuffer(::tt::ARCH arch) {
  switch (arch) {
  case ::tt::ARCH::GRAYSKULL:
    return ::tt::target::Arch::Grayskull;
  case ::tt::ARCH::WORMHOLE_B0:
    return ::tt::target::Arch::Wormhole_b0;
  case ::tt::ARCH::BLACKHOLE:
    return ::tt::target::Arch::Blackhole;
  default:
    break;
  }

  LOG_FATAL("Unsupported arch");
}

static std::vector<::tt::target::ChipChannel>
getAllDeviceConnections(const std::vector<::tt::tt_metal::IDevice *> &devices) {
  std::set<std::tuple<chip_id_t, CoreCoord, chip_id_t, CoreCoord>>
      connectionSet;

  auto addConnection = [&connectionSet](
                           chip_id_t deviceId0, CoreCoord ethCoreCoord0,
                           chip_id_t deviceId1, CoreCoord ethCoreCoord1) {
    if (deviceId0 > deviceId1) {
      std::swap(deviceId0, deviceId1);
      std::swap(ethCoreCoord0, ethCoreCoord1);
    }
    connectionSet.emplace(deviceId0, ethCoreCoord0, deviceId1, ethCoreCoord1);
  };

  for (const ::tt::tt_metal::IDevice *device : devices) {
    std::unordered_set<CoreCoord> activeEthernetCores =
        device->get_active_ethernet_cores(true);
    for (const CoreCoord &ethernetCore : activeEthernetCores) {
      bool getConnection = true;
      // Skip on blackhole. When link is down, get_connected_ethernet_core
      // will throw an exception.
      // See https://github.com/tenstorrent/tt-mlir/issues/3423 for BH
      if (workaround::Env::get().blackholeWorkarounds) {
        getConnection &= device->arch() != ::tt::ARCH::BLACKHOLE;
      }
      // See https://github.com/tenstorrent/tt-mlir/issues/3781 for WH
      getConnection &= device->arch() != ::tt::ARCH::WORMHOLE_B0;
      if (!getConnection) {
        continue;
      }
      std::tuple<chip_id_t, CoreCoord> connectedDevice =
          device->get_connected_ethernet_core(ethernetCore);
      addConnection(device->id(), ethernetCore, std::get<0>(connectedDevice),
                    std::get<1>(connectedDevice));
    }
  }

  std::vector<::tt::target::ChipChannel> allConnections;
  allConnections.resize(connectionSet.size());

  std::transform(
      connectionSet.begin(), connectionSet.end(), allConnections.begin(),
      [](const std::tuple<chip_id_t, CoreCoord, chip_id_t, CoreCoord>
             &connection) {
        return ::tt::target::ChipChannel(
            std::get<0>(connection), toFlatbuffer(std::get<1>(connection)),
            std::get<2>(connection), toFlatbuffer(std::get<3>(connection)));
      });

  return allConnections;
}

::tt::target::Dim2d
getCoordinateTranslationOffsets(const ::tt::tt_metal::IDevice *device) {
  const CoreCoord workerNWCorner =
      device->worker_core_from_logical_core({0, 0});
  const CoreCoord workerNWCornerTranslated =
      device->virtual_noc0_coordinate(0, workerNWCorner);
  return ::tt::target::Dim2d(workerNWCornerTranslated.y,
                             workerNWCornerTranslated.x);
}

// Calculate the end of the DRAM region that is not usable by compiler.  This
// upper region of memory is where kernel programs get allocated to.  This
// function intends to estimate some conservative max number.
static std::uint32_t
calculateDRAMUnreservedEnd(const ::tt::tt_metal::IDevice *device) {
  CoreCoord deviceGridSize = device->logical_grid_size();
  CoreCoord dramGridSize = device->dram_grid_size();
  std::uint32_t totalCores = deviceGridSize.x * deviceGridSize.y +
                             device->get_active_ethernet_cores().size();
  std::uint32_t totalDramCores = dramGridSize.x * dramGridSize.y;
  std::uint32_t programCarveOutPerCore =
      device->allocator()->get_base_allocator_addr(HalMemType::L1);
  std::uint32_t totalProgramCarveOut = programCarveOutPerCore * totalCores;
  // The total carve out can be interleaved between all dram channels
  std::uint32_t programCarveOutDramSpace =
      (totalProgramCarveOut + totalDramCores - 1) / totalDramCores;

  std::uint32_t dramAlignment =
      device->allocator()->get_alignment(BufferType::DRAM);
  LOG_ASSERT(dramAlignment > 0);
  LOG_ASSERT((dramAlignment & (dramAlignment - 1)) == 0);
  LOG_ASSERT(programCarveOutDramSpace < device->dram_size_per_channel());
  std::uint32_t dramUnreservedEnd =
      device->dram_size_per_channel() - programCarveOutDramSpace;
  // Align to dramAlignment
  dramUnreservedEnd = dramUnreservedEnd & ~(dramAlignment - 1);
  return dramUnreservedEnd;
}

static std::unique_ptr<::tt::runtime::SystemDesc> getCurrentSystemDescImpl(
    const ::tt::tt_metal::distributed::MeshDevice &meshDevice) {
  std::vector<::tt::tt_metal::IDevice *> devices = meshDevice.get_devices();
  std::sort(devices.begin(), devices.end(),
            [](const ::tt::tt_metal::IDevice *a,
               const ::tt::tt_metal::IDevice *b) { return a->id() < b->id(); });

  std::vector<::flatbuffers::Offset<tt::target::ChipDesc>> chipDescs;
  std::vector<uint32_t> chipDescIndices;
  std::vector<::tt::target::ChipCapability> chipCapabilities;

  // Ignore for now
  std::vector<::tt::target::ChipCoord> chipCoords = {
      ::tt::target::ChipCoord(0, 0, 0, 0)};
  ::flatbuffers::FlatBufferBuilder fbb;

  std::uint32_t pcieAlignment = ::tt::tt_metal::hal::get_pcie_alignment();
  std::uint32_t l1Alignment = ::tt::tt_metal::hal::get_l1_alignment();
  std::uint32_t dramAlignment = ::tt::tt_metal::hal::get_dram_alignment();

  for (const ::tt::tt_metal::IDevice *device : devices) {
    size_t l1UnreservedBase =
        device->allocator()->get_base_allocator_addr(HalMemType::L1);
    size_t dramUnreservedBase =
        device->allocator()->get_base_allocator_addr(HalMemType::DRAM);
    // Construct chip descriptor
    ::tt::target::Dim2d deviceGrid =
        toFlatbuffer(device->compute_with_storage_grid_size());

    assert(device->compute_with_storage_grid_size().x ==
               static_cast<size_t>(deviceGrid.x()) &&
           device->compute_with_storage_grid_size().y ==
               static_cast<size_t>(deviceGrid.y()));

    // Get the physical-to-translated coordinate translation offset of the
    // worker cores
    auto coordTranslationOffsets = getCoordinateTranslationOffsets(device);

    // The following is temporary place-holder value to be replaced by API
    // value.
    std::vector<::tt::target::DataType> supportedDataTypesVector = {
        ::tt::target::DataType::Float32,     ::tt::target::DataType::Float16,
        ::tt::target::DataType::BFloat16,    ::tt::target::DataType::BFP_Float8,
        ::tt::target::DataType::BFP_BFloat8, ::tt::target::DataType::BFP_Float4,
        ::tt::target::DataType::BFP_BFloat4, ::tt::target::DataType::BFP_Float2,
        ::tt::target::DataType::BFP_BFloat2, ::tt::target::DataType::UInt32,
        ::tt::target::DataType::UInt16,      ::tt::target::DataType::UInt8,
        ::tt::target::DataType::Int32};

    auto supportedDataTypes = fbb.CreateVector(supportedDataTypesVector);

    std::vector<::tt::target::Dim2d> supportedTileSizesVector = {
        ::tt::target::Dim2d(4, 16),  ::tt::target::Dim2d(16, 16),
        ::tt::target::Dim2d(32, 16), ::tt::target::Dim2d(4, 32),
        ::tt::target::Dim2d(16, 32), ::tt::target::Dim2d(32, 32)};

    auto supportedTileSizes =
        fbb.CreateVectorOfStructs(supportedTileSizesVector);

    auto dramUnreservedEnd = calculateDRAMUnreservedEnd(device);

    constexpr std::uint32_t kDstRegisterSizeTiles = 8;
    constexpr std::uint32_t kNumComputeThreads = 1;
    constexpr std::uint32_t kNumDatamovementThreads = 2;
    chipDescs.emplace_back(::tt::target::CreateChipDesc(
        fbb, toFlatbuffer(device->arch()), &deviceGrid,
        &coordTranslationOffsets, device->l1_size_per_core(),
        device->num_dram_channels(), device->dram_size_per_channel(),
        l1Alignment, pcieAlignment, dramAlignment, l1UnreservedBase,
        ::tt::tt_metal::hal::get_erisc_l1_unreserved_base(), dramUnreservedBase,
        dramUnreservedEnd, supportedDataTypes, supportedTileSizes,
        kDstRegisterSizeTiles, NUM_CIRCULAR_BUFFERS, kNumComputeThreads,
        kNumDatamovementThreads));
    chipDescIndices.push_back(chipDescIndices.size());
    // Derive chip capability
    ::tt::target::ChipCapability chipCapability =
        ::tt::target::ChipCapability::NONE;
    if (device->is_mmio_capable()) {
      chipCapability = chipCapability | ::tt::target::ChipCapability::HostMMIO;
    }
    chipCapabilities.push_back(chipCapability);
  }
  // Extract chip connected channels
  std::vector<::tt::target::ChipChannel> allConnections =
      getAllDeviceConnections(devices);
  // Store CPUDesc
  std::vector<::flatbuffers::Offset<tt::target::CPUDesc>> cpuDescs;
  cpuDescs.emplace_back(::tt::target::CreateCPUDesc(
      fbb, ::tt::target::CPURole::Host,
      fbb.CreateString(std::string(TARGET_TRIPLE))));

  // Create SystemDesc
  auto systemDesc = ::tt::target::CreateSystemDescDirect(
      fbb, &cpuDescs, &chipDescs, &chipDescIndices, &chipCapabilities,
      &chipCoords, &allConnections);
  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version version(ttmlirVersion.major, ttmlirVersion.minor,
                                ttmlirVersion.patch);
  auto root = ::tt::target::CreateSystemDescRootDirect(
      fbb, &version, tt::target::common::system_desc_bfbs_schema_hash,
      ::ttmlir::getGitHash(), "unknown", systemDesc);
  ::tt::target::FinishSizePrefixedSystemDescRootBuffer(fbb, root);
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  if (!::tt::target::VerifySizePrefixedSystemDescRootBuffer(verifier)) {
    LOG_FATAL("Failed to verify system desc root buffer");
  }
  uint8_t *buf = fbb.GetBufferPointer();
  auto size = fbb.GetSize();
  auto handle = ::tt::runtime::utils::mallocShared(size);
  std::memcpy(handle.get(), buf, size);
  return std::make_unique<::tt::runtime::SystemDesc>(handle);
}

static std::shared_ptr<::tt::tt_metal::distributed::MeshDevice>
createNewMeshDevice(std::optional<DispatchCoreType> dispatchCoreType) {

  ::tt::tt_metal::DispatchCoreType type =
      tt::runtime::common::getDispatchCoreType(dispatchCoreType);

  return ::tt::tt_metal::distributed::MeshDevice::create(
      ::tt::tt_metal::distributed::MeshDeviceConfig(
          /*mesh_shape=*/std::nullopt),
      DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, type);
}

::tt::runtime::SystemDesc
getCurrentSystemDesc(std::optional<DispatchCoreType> dispatchCoreType,
                     std::optional<Device> meshDevice) {

  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> meshDevicePtr;
  if (meshDevice.has_value()) {
    meshDevicePtr =
        meshDevice.value().asSharedPtr<::tt::tt_metal::distributed::MeshDevice>(
            getCurrentDeviceRuntime());
  } else {
    meshDevicePtr = createNewMeshDevice(dispatchCoreType);
  }

  LOG_DEBUG("Device grid size = { ",
            meshDevicePtr->compute_with_storage_grid_size().x, ", ",
            meshDevicePtr->compute_with_storage_grid_size().y, " }");

  std::exception_ptr eptr = nullptr;
  std::unique_ptr<::tt::runtime::SystemDesc> desc;
  try {
    desc = getCurrentSystemDescImpl(*meshDevicePtr);
  } catch (...) {
    eptr = std::current_exception();
  }
  if (!meshDevice.has_value()) {
    // Close if mesh device was created in this scope (not passed from caller)
    meshDevicePtr->close();
  }
  if (eptr) {
    LOG_ERROR("Exception occured when getting system descriptor");
    std::rethrow_exception(eptr);
  }
  return *desc;
}

} // namespace tt::runtime::system_desc
