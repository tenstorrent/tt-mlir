// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"
#include "types_generated.h"
#include <cstdint>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wctad-maybe-unsupported"
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wvla-extension"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wdeprecated-this-capture"
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
#pragma clang diagnostic ignored "-Wsuggest-override"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wreorder-ctor"
#pragma clang diagnostic ignored "-Wmismatched-tags"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#pragma clang diagnostic ignored "-Wzero-length-array"
#define FMT_HEADER_ONLY
#include "distributed/mesh_device.hpp"
#include "host_api.hpp"
#include "hostdevcommon/common_values.hpp"
#pragma clang diagnostic pop

namespace tt::runtime::system_desc {
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

  throw std::runtime_error("Unsupported arch");
}

static std::vector<::tt::target::ChipChannel>
getAllDeviceConnections(const std::vector<::tt::tt_metal::Device *> &devices) {
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

  for (const ::tt::tt_metal::Device *device : devices) {
    std::unordered_set<CoreCoord> activeEthernetCores =
        device->get_active_ethernet_cores(true);
    for (const CoreCoord &ethernetCore : activeEthernetCores) {
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

// Gather all physical cores by type for the device using metal device APIs
static flatbuffers::Offset<::tt::target::ChipPhysicalCores>
createChipPhysicalCores(const ::tt::tt_metal::Device *device,
                        flatbuffers::FlatBufferBuilder &fbb) {

  std::vector<::tt::target::Dim2d> worker_cores, dram_cores, eth_cores,
      eth_inactive_cores;

  CoreCoord logical_grid_size = device->compute_with_storage_grid_size();
  for (uint32_t y = 0; y < logical_grid_size.y; y++) {
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
      CoreCoord physical =
          device->worker_core_from_logical_core(CoreCoord(x, y));
      worker_cores.emplace_back(::tt::target::Dim2d(physical.y, physical.x));
    }
  }

  CoreCoord dram_grid_size = device->dram_grid_size();
  for (uint32_t y = 0; y < dram_grid_size.y; y++) {
    for (uint32_t x = 0; x < dram_grid_size.x; x++) {
      CoreCoord physical = device->dram_core_from_logical_core(CoreCoord(x, y));
      dram_cores.emplace_back(::tt::target::Dim2d(physical.y, physical.x));
    }
  }

  for (const CoreCoord &logical : device->get_active_ethernet_cores(true)) {
    CoreCoord physical = device->ethernet_core_from_logical_core(logical);
    eth_cores.emplace_back(::tt::target::Dim2d(physical.y, physical.x));
  }

  for (const CoreCoord &logical : device->get_inactive_ethernet_cores()) {
    CoreCoord physical = device->ethernet_core_from_logical_core(logical);
    eth_inactive_cores.emplace_back(
        ::tt::target::Dim2d(physical.y, physical.x));
  }

  sort(dram_cores);
  sort(eth_cores);
  sort(eth_inactive_cores);

  return ::tt::target::CreateChipPhysicalCores(
      fbb, fbb.CreateVectorOfStructs(worker_cores),
      fbb.CreateVectorOfStructs(dram_cores),
      fbb.CreateVectorOfStructs(eth_cores),
      fbb.CreateVectorOfStructs(eth_inactive_cores));
}

// Calculate the end of the DRAM region that is not usable by compiler.  This
// upper region of memory is where kernel programs get allocated to.  This
// function intends to estimate some conservative max number.
static std::uint32_t
calculateDRAMUnreservedEnd(const ::tt::tt_metal::Device *device) {
  CoreCoord deviceGridSize = device->logical_grid_size();
  CoreCoord dramGridSize = device->dram_grid_size();
  std::uint32_t totalCores = deviceGridSize.x * deviceGridSize.y +
                             device->get_active_ethernet_cores().size();
  std::uint32_t totalDramCores = dramGridSize.x * dramGridSize.y;
  std::uint32_t programCarveOutPerCore =
      device->get_base_allocator_addr(::tt::tt_metal::HalMemType::L1);
  std::uint32_t totalProgramCarveOut = programCarveOutPerCore * totalCores;
  // The total carve out can be interleaved between all dram channels
  std::uint32_t programCarveOutDramSpace =
      (totalProgramCarveOut + totalDramCores - 1) / totalDramCores;
  static_assert(DRAM_ALIGNMENT > 0);
  static_assert((DRAM_ALIGNMENT & (DRAM_ALIGNMENT - 1)) == 0);
  LOG_ASSERT(programCarveOutDramSpace < device->dram_size_per_channel());
  std::uint32_t dramUnreservedEnd =
      device->dram_size_per_channel() - programCarveOutDramSpace;
  // Align to DRAM_ALIGNMENT
  dramUnreservedEnd = dramUnreservedEnd & ~(DRAM_ALIGNMENT - 1);
  return dramUnreservedEnd;
}

static std::unique_ptr<::tt::runtime::SystemDesc> getCurrentSystemDescImpl(
    const ::tt::tt_metal::distributed::MeshDevice &meshDevice) {
  std::vector<::tt::tt_metal::Device *> devices = meshDevice.get_devices();
  std::sort(devices.begin(), devices.end(),
            [](const ::tt::tt_metal::Device *a,
               const ::tt::tt_metal::Device *b) { return a->id() < b->id(); });

  std::vector<::flatbuffers::Offset<tt::target::ChipDesc>> chipDescs;
  std::vector<uint32_t> chipDescIndices;
  std::vector<::tt::target::ChipCapability> chipCapabilities;

  // Ignore for now
  std::vector<::tt::target::ChipCoord> chipCoords = {
      ::tt::target::ChipCoord(0, 0, 0, 0)};
  ::flatbuffers::FlatBufferBuilder fbb;

  for (const ::tt::tt_metal::Device *device : devices) {
    size_t l1UnreservedBase =
        device->get_base_allocator_addr(::tt::tt_metal::HalMemType::L1);
    size_t dramUnreservedBase =
        device->get_base_allocator_addr(::tt::tt_metal::HalMemType::DRAM);

    // Construct chip descriptor
    ::tt::target::Dim2d deviceGrid =
        toFlatbuffer(device->compute_with_storage_grid_size());

    // Extract physical core coordinates for worker, dram, eth cores
    auto chipPhysicalCores = createChipPhysicalCores(device, fbb);

    // The following is temporary place-holder value to be replaced by API
    // value.
    std::vector<::tt::target::DataType> supportedDataTypesVector = {
        ::tt::target::DataType::Float32,     ::tt::target::DataType::Float16,
        ::tt::target::DataType::BFloat16,    ::tt::target::DataType::BFP_Float8,
        ::tt::target::DataType::BFP_BFloat8, ::tt::target::DataType::BFP_Float4,
        ::tt::target::DataType::BFP_BFloat4, ::tt::target::DataType::BFP_Float2,
        ::tt::target::DataType::BFP_BFloat2, ::tt::target::DataType::UInt32,
        ::tt::target::DataType::UInt16,      ::tt::target::DataType::UInt8};

    auto supportedDataTypes = fbb.CreateVector(supportedDataTypesVector);

    std::vector<::tt::target::Dim2d> supportedTileSizesVector = {
        ::tt::target::Dim2d(4, 16),  ::tt::target::Dim2d(16, 16),
        ::tt::target::Dim2d(32, 16), ::tt::target::Dim2d(4, 32),
        ::tt::target::Dim2d(16, 32), ::tt::target::Dim2d(32, 32)};

    auto supportedTileSizes =
        fbb.CreateVectorOfStructs(supportedTileSizesVector);

    auto dramUnreservedEnd = calculateDRAMUnreservedEnd(device);

    chipDescs.emplace_back(::tt::target::CreateChipDesc(
        fbb, toFlatbuffer(device->arch()), &deviceGrid,
        device->l1_size_per_core(), device->num_dram_channels(),
        device->dram_size_per_channel(), L1_ALIGNMENT, PCIE_ALIGNMENT,
        DRAM_ALIGNMENT, l1UnreservedBase,
        ::eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, dramUnreservedBase,
        dramUnreservedEnd, chipPhysicalCores, supportedDataTypes,
        supportedTileSizes, NUM_CIRCULAR_BUFFERS));
    chipDescIndices.push_back(device->id());
    // Derive chip capability
    ::tt::target::ChipCapability chipCapability =
        ::tt::target::ChipCapability::NONE;
    if (device->is_mmio_capable()) {
      chipCapability = chipCapability | ::tt::target::ChipCapability::PCIE |
                       ::tt::target::ChipCapability::HostMMIO;
    }
    chipCapabilities.push_back(chipCapability);
  }
  // Extract chip connected channels
  std::vector<::tt::target::ChipChannel> allConnections =
      getAllDeviceConnections(devices);
  // Store CPUDesc
  std::vector<::flatbuffers::Offset<tt::target::CPUDesc>> cpuDescs;
  cpuDescs.emplace_back(::tt::target::CPUDesc{::tt::target::CPURole::Host, std::string("TARGET_TRIPLE")});

  // Create SystemDesc
  auto systemDesc = ::tt::target::CreateSystemDescDirect(
      fbb, &chipDescs, &chipDescIndices, &chipCapabilities, &chipCoords,
      &allConnections, &cpuDescs);
  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version version(ttmlirVersion.major, ttmlirVersion.minor,
                                ttmlirVersion.patch);
  auto root = ::tt::target::CreateSystemDescRootDirect(
      fbb, &version, ::ttmlir::getGitHash(), "unknown", systemDesc);
  ::tt::target::FinishSizePrefixedSystemDescRootBuffer(fbb, root);
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  if (!::tt::target::VerifySizePrefixedSystemDescRootBuffer(verifier)) {
    throw std::runtime_error("Failed to verify system desc root buffer");
  }
  uint8_t *buf = fbb.GetBufferPointer();
  auto size = fbb.GetSize();
  auto handle = ::tt::runtime::utils::malloc_shared(size);
  std::memcpy(handle.get(), buf, size);
  return std::make_unique<::tt::runtime::SystemDesc>(handle);
}

std::pair<::tt::runtime::SystemDesc, DeviceIds> getCurrentSystemDesc() {
  size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
  std::vector<chip_id_t> deviceIds(numDevices);
  std::iota(deviceIds.begin(), deviceIds.end(), 0);
  ::tt::tt_metal::distributed::MeshShape meshShape =
      std::make_pair(1, numDevices);
  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> meshDevice =
      ::tt::tt_metal::distributed::MeshDevice::create(
          meshShape, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1,
          ::tt::tt_metal::DispatchCoreType::WORKER);
  std::exception_ptr eptr = nullptr;
  std::unique_ptr<::tt::runtime::SystemDesc> desc;
  try {
    desc = getCurrentSystemDescImpl(*meshDevice);
  } catch (...) {
    eptr = std::current_exception();
  }
  meshDevice->close_devices();
  if (eptr) {
    std::rethrow_exception(eptr);
  }
  return std::make_pair(*desc, deviceIds);
}

} // namespace tt::runtime::system_desc
