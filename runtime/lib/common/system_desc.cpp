// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/detail/common.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/Common/system_desc_bfbs_hash_generated.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"
#include "types_generated.h"
#include <cstdint>
#include <exception>  // For std::exception_ptr
#include <execinfo.h> // For backtrace
#include <iostream>   // For debugging
#include <sstream>    // For std::stringstream
#include <vector>

#define FMT_HEADER_ONLY
#include "eth_l1_address_map.h"
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

  std::cerr << "[system_desc][FATAL] Unsupported arch" << std::endl;
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
      // Skip on blackhole. When link is down, get_connected_ethernet_core
      // will throw an exception.
      // See https://github.com/tenstorrent/tt-mlir/issues/3423
      if (device->arch() == ::tt::ARCH::BLACKHOLE) {
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

static void sort(std::vector<::tt::target::Dim2d> &vec) {
  std::sort(vec.begin(), vec.end(),
            [](const ::tt::target::Dim2d &a, const ::tt::target::Dim2d &b) {
              return a.y() < b.y() || (a.y() == b.y() && a.x() < b.x());
            });
}

// Gather physical helper cores by type for the device using metal device APIs
static flatbuffers::Offset<::tt::target::ChipPhysicalHelperCores>
createChipPhysicalHelperCores(const ::tt::tt_metal::IDevice *device,
                              flatbuffers::FlatBufferBuilder &fbb) {

  std::vector<::tt::target::Dim2d> dramCores, ethCores, ethInactiveCores;

  for (int dramChannel = 0; dramChannel < device->num_dram_channels();
       ++dramChannel) {
    CoreCoord logical = device->logical_core_from_dram_channel(dramChannel);
    dramCores.emplace_back(::tt::target::Dim2d(logical.y, logical.x));
  }

  for (const CoreCoord &logical : device->get_active_ethernet_cores(true)) {
    CoreCoord physical = device->ethernet_core_from_logical_core(logical);
    ethCores.emplace_back(::tt::target::Dim2d(physical.y, physical.x));
  }

  for (const CoreCoord &logical : device->get_inactive_ethernet_cores()) {
    CoreCoord physical = device->ethernet_core_from_logical_core(logical);
    ethInactiveCores.emplace_back(::tt::target::Dim2d(physical.y, physical.x));
  }

  sort(dramCores);
  sort(ethCores);
  sort(ethInactiveCores);

  return ::tt::target::CreateChipPhysicalHelperCores(
      fbb, fbb.CreateVectorOfStructs(dramCores),
      fbb.CreateVectorOfStructs(ethCores),
      fbb.CreateVectorOfStructs(ethInactiveCores));
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
  std::cerr << "[system_desc] Entering getCurrentSystemDescImpl" << std::endl;
  std::vector<::tt::tt_metal::IDevice *> devices = meshDevice.get_devices();
  std::cerr << "[system_desc] Number of devices in mesh: " << devices.size()
            << std::endl;
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
  std::cerr << "[system_desc] Alignments: pcie=" << pcieAlignment
            << ", l1=" << l1Alignment << ", dram=" << dramAlignment
            << std::endl;

  for (const ::tt::tt_metal::IDevice *device : devices) {
    std::cerr << "[system_desc][device " << device->id()
              << "] Getting L1 base address" << std::endl;
    size_t l1UnreservedBase =
        device->allocator()->get_base_allocator_addr(HalMemType::L1);
    std::cerr << "[system_desc][device " << device->id()
              << "] Getting DRAM base address" << std::endl;
    size_t dramUnreservedBase =
        device->allocator()->get_base_allocator_addr(HalMemType::DRAM);
    // Construct chip descriptor
    std::cerr << "[system_desc][device " << device->id()
              << "] Converting compute_with_storage_grid_size to flatbuffer"
              << std::endl;
    ::tt::target::Dim2d deviceGrid =
        toFlatbuffer(device->compute_with_storage_grid_size());

    std::cerr << "[system_desc][device " << device->id()
              << "] Asserting deviceGrid matches compute_with_storage_grid_size"
              << std::endl;
    assert(device->compute_with_storage_grid_size().x ==
               static_cast<size_t>(deviceGrid.x()) &&
           device->compute_with_storage_grid_size().y ==
               static_cast<size_t>(deviceGrid.y()));

    // Get the physical-to-translated coordinate translation offset of the
    // worker cores
    std::cerr << "[system_desc][device " << device->id()
              << "] Getting coordinate translation offsets" << std::endl;
    auto coordTranslationOffsets = getCoordinateTranslationOffsets(device);

    // Extract physical core coordinates for dram and eth cores
    std::cerr << "[system_desc][device " << device->id()
              << "] Creating chipPhysicalHelperCores" << std::endl;
    auto chipPhysicalHelperCores = createChipPhysicalHelperCores(device, fbb);

    // The following is temporary place-holder value to be replaced by API
    // value.
    std::cerr << "[system_desc][device " << device->id()
              << "] Setting up supportedDataTypesVector" << std::endl;
    std::vector<::tt::target::DataType> supportedDataTypesVector = {
        ::tt::target::DataType::Float32,     ::tt::target::DataType::Float16,
        ::tt::target::DataType::BFloat16,    ::tt::target::DataType::BFP_Float8,
        ::tt::target::DataType::BFP_BFloat8, ::tt::target::DataType::BFP_Float4,
        ::tt::target::DataType::BFP_BFloat4, ::tt::target::DataType::BFP_Float2,
        ::tt::target::DataType::BFP_BFloat2, ::tt::target::DataType::UInt32,
        ::tt::target::DataType::UInt16,      ::tt::target::DataType::UInt8,
        ::tt::target::DataType::Int32};

    std::cerr << "[system_desc][device " << device->id()
              << "] Creating flatbuffer vector for supportedDataTypes"
              << std::endl;
    auto supportedDataTypes = fbb.CreateVector(supportedDataTypesVector);

    std::cerr << "[system_desc][device " << device->id()
              << "] Setting up supportedTileSizesVector" << std::endl;
    std::vector<::tt::target::Dim2d> supportedTileSizesVector = {
        ::tt::target::Dim2d(4, 16),  ::tt::target::Dim2d(16, 16),
        ::tt::target::Dim2d(32, 16), ::tt::target::Dim2d(4, 32),
        ::tt::target::Dim2d(16, 32), ::tt::target::Dim2d(32, 32)};

    std::cerr << "[system_desc][device " << device->id()
              << "] Creating flatbuffer vector for supportedTileSizes"
              << std::endl;
    auto supportedTileSizes =
        fbb.CreateVectorOfStructs(supportedTileSizesVector);

    std::cerr << "[system_desc][device " << device->id()
              << "] Calculating DRAM unreserved end" << std::endl;
    auto dramUnreservedEnd = calculateDRAMUnreservedEnd(device);
    std::cerr << "[system_desc][device " << device->id()
              << "] dramUnreservedEnd=" << dramUnreservedEnd << std::endl;
    constexpr std::uint32_t kDstRegisterSizeTiles = 8;
    constexpr std::uint32_t kNumComputeThreads = 1;
    constexpr std::uint32_t kNumDatamovementThreads = 2;
    std::cerr << "[system_desc][device " << device->id()
              << "] Creating ChipDesc" << std::endl;
    chipDescs.emplace_back(::tt::target::CreateChipDesc(
        fbb, toFlatbuffer(device->arch()), &deviceGrid,
        &coordTranslationOffsets, device->l1_size_per_core(),
        device->num_dram_channels(), device->dram_size_per_channel(),
        l1Alignment, pcieAlignment, dramAlignment, l1UnreservedBase,
        ::eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, dramUnreservedBase,
        dramUnreservedEnd, chipPhysicalHelperCores, supportedDataTypes,
        supportedTileSizes, kDstRegisterSizeTiles, NUM_CIRCULAR_BUFFERS,
        kNumComputeThreads, kNumDatamovementThreads));
    std::cerr << "[system_desc][device " << device->id()
              << "] Pushing device id to chipDescIndices" << std::endl;
    chipDescIndices.push_back(device->id());
    // Derive chip capability
    std::cerr << "[system_desc][device " << device->id()
              << "] Deriving chip capability" << std::endl;
    ::tt::target::ChipCapability chipCapability =
        ::tt::target::ChipCapability::NONE;
    std::cerr << "[system_desc][device " << device->id()
              << "] Checking MMIO capability" << std::endl;
    if (device->is_mmio_capable()) {
      chipCapability = chipCapability | ::tt::target::ChipCapability::PCIE |
                       ::tt::target::ChipCapability::HostMMIO;
      std::cerr << "[system_desc][device " << device->id()
                << "] Device is MMIO capable" << std::endl;
    }
    std::cerr << "[system_desc][device " << device->id()
              << "] Pushing chipCapability to chipCapabilities" << std::endl;
    chipCapabilities.push_back(chipCapability);
    std::cerr << "[system_desc][device " << device->id()
              << "] Added chipCapability" << std::endl;
  }
  // Extract chip connected channels
  std::cerr << "[system_desc] Getting all device connections" << std::endl;
  std::vector<::tt::target::ChipChannel> allConnections =
      getAllDeviceConnections(devices);
  // Store CPUDesc
  std::cerr << "[system_desc] Creating CPUDesc vector" << std::endl;
  std::vector<::flatbuffers::Offset<tt::target::CPUDesc>> cpuDescs;
  std::cerr << "[system_desc] Creating CPUDesc entry" << std::endl;
  cpuDescs.emplace_back(::tt::target::CreateCPUDesc(
      fbb, ::tt::target::CPURole::Host,
      fbb.CreateString(std::string(TARGET_TRIPLE))));

  // Create SystemDesc
  std::cerr << "[system_desc] Creating SystemDescDirect" << std::endl;
  auto systemDesc = ::tt::target::CreateSystemDescDirect(
      fbb, &cpuDescs, &chipDescs, &chipDescIndices, &chipCapabilities,
      &chipCoords, &allConnections);
  std::cerr << "[system_desc] Created SystemDesc" << std::endl;
  std::cerr << "[system_desc] Getting ttmlir version" << std::endl;
  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  std::cerr << "[system_desc] Creating tt::target::Version" << std::endl;
  ::tt::target::Version version(ttmlirVersion.major, ttmlirVersion.minor,
                                ttmlirVersion.patch);
  std::cerr << "[system_desc] Creating SystemDescRootDirect" << std::endl;
  auto root = ::tt::target::CreateSystemDescRootDirect(
      fbb, &version, tt::target::common::system_desc_bfbs_schema_hash,
      ::ttmlir::getGitHash(), "unknown", systemDesc);
  std::cerr << "[system_desc] Created SystemDescRootDirect" << std::endl;
  std::cerr << "[system_desc] Finishing SizePrefixedSystemDescRootBuffer"
            << std::endl;
  ::tt::target::FinishSizePrefixedSystemDescRootBuffer(fbb, root);
  std::cerr << "[system_desc] Finished SizePrefixedSystemDescRootBuffer"
            << std::endl;
  std::cerr << "[system_desc] Verifying flatbuffer" << std::endl;
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  std::cerr << "[system_desc] Checking verification result" << std::endl;
  if (!::tt::target::VerifySizePrefixedSystemDescRootBuffer(verifier)) {
    std::cerr << "[system_desc][FATAL] Failed to verify system desc root buffer"
              << std::endl;
  }
  std::cerr << "[system_desc] Getting buffer pointer" << std::endl;
  uint8_t *buf = fbb.GetBufferPointer();
  auto size = fbb.GetSize();
  auto handle = ::tt::runtime::utils::malloc_shared(size);
  std::memcpy(handle.get(), buf, size);
  std::cerr << "[system_desc] Exiting getCurrentSystemDescImpl" << std::endl;
  return std::make_unique<::tt::runtime::SystemDesc>(handle);
}

static std::shared_ptr<::tt::tt_metal::distributed::MeshDevice>
createNewMeshDevice(size_t numDevices,
                    std::optional<DispatchCoreType> dispatchCoreType) {
  ::tt::tt_metal::distributed::MeshShape meshShape{
      1, static_cast<uint32_t>(numDevices)};

  ::tt::tt_metal::DispatchCoreType type =
      tt::runtime::common::getDispatchCoreType(dispatchCoreType);

  return ::tt::tt_metal::distributed::MeshDevice::create(
      ::tt::tt_metal::distributed::MeshDeviceConfig(meshShape),
      DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, type);
}

::tt::runtime::SystemDesc
getCurrentSystemDesc(std::optional<DispatchCoreType> dispatchCoreType,
                     std::optional<Device> meshDevice) {
  std::cerr << "[system_desc] Entering getCurrentSystemDesc" << std::endl;
  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> meshDevicePtr;
  if (meshDevice.has_value()) {
    meshDevicePtr =
        meshDevice.value().asSharedPtr<::tt::tt_metal::distributed::MeshDevice>(
            getCurrentRuntime());
  } else {
    const size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
    meshDevicePtr = createNewMeshDevice(numDevices, dispatchCoreType);
  }

  LOG_DEBUG("Device grid size = { ",
            meshDevicePtr->compute_with_storage_grid_size().x, ", ",
            meshDevicePtr->compute_with_storage_grid_size().y, " }");

  std::exception_ptr eptr = nullptr;
  std::unique_ptr<::tt::runtime::SystemDesc> desc;
  try {
    std::cerr << "[system_desc] Calling getCurrentSystemDescImpl" << std::endl;
    desc = getCurrentSystemDescImpl(*meshDevicePtr);
  } catch (...) {
    eptr = std::current_exception();
    std::cerr << "[system_desc] Exception caught in getCurrentSystemDesc"
              << std::endl;
    if (eptr) {
      try {
        std::rethrow_exception(eptr);
      } catch (const std::exception &e) {
        std::cerr << "[system_desc] Exception: " << e.what() << std::endl;
      }
    }
    void *array[10];
    size_t size = backtrace(array, 10);
    std::cerr << "[system_desc] Stack trace:" << std::endl;
    backtrace_symbols_fd(array, size, STDERR_FILENO);
  }
  if (!meshDevice.has_value()) {
    // Close if mesh device was created in this scope (not passed from caller)
    meshDevicePtr->close();
  }
  if (eptr) {
    std::cerr << "[system_desc][ERROR] Exception occurred when getting system "
                 "descriptor"
              << std::endl;
    std::rethrow_exception(eptr);
  }
  std::cerr << "[system_desc] Exiting getCurrentSystemDesc" << std::endl;
  return *desc;
}

} // namespace tt::runtime::system_desc
