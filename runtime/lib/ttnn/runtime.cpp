// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/runtime.h"
#include "hostdevcommon/common_values.hpp"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/utils.h"
#include "utils.h"
#include "impl/device/device_mesh.hpp"
#include "hostdevcommon/common_values.hpp"

#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

namespace tt::runtime::ttnn {

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

static ::tt::target::Dim2d toFlatbuffer(const CoreCoord &coreCoord) {
  return ::tt::target::Dim2d(coreCoord.y, coreCoord.x);
}

static std::vector<::tt::target::ChipChannel>
getAllDeviceConnections(const vector<::ttnn::Device *> &devices) {
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

  for (const ::ttnn::Device *device : devices) {
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

static std::unique_ptr<SystemDesc>
getCurrentSystemDescImpl(const ::tt::tt_metal::DeviceMesh &deviceMesh) {
  std::vector<::ttnn::Device *> devices = deviceMesh.get_devices();
  std::sort(devices.begin(), devices.end(),
            [](const ::ttnn::Device *a, const ::ttnn::Device *b) {
              return a->id() < b->id();
            });

  std::vector<::flatbuffers::Offset<tt::target::ChipDesc>> chipDescs;
  std::vector<uint32_t> chipDescIndices;
  std::vector<::tt::target::ChipCapability> chipCapabilities;
  // Ignore for now
  std::vector<::tt::target::ChipCoord> chipCoords = {
      ::tt::target::ChipCoord(0, 0, 0, 0)};
  ::flatbuffers::FlatBufferBuilder fbb;

  for (const ::ttnn::Device *device : devices) {
    // Construct chip descriptor
    ::tt::target::Dim2d deviceGrid =
        toFlatbuffer(device->compute_with_storage_grid_size());
    chipDescs.push_back(::tt::target::CreateChipDesc(
        fbb, toFlatbuffer(device->arch()), &deviceGrid,
        device->l1_size_per_core(), device->num_dram_channels(),
        device->dram_size_per_channel(), L1_ALIGNMENT, PCIE_ALIGNMENT,
        DRAM_ALIGNMENT));
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
  // Create SystemDesc
  auto systemDesc = ::tt::target::CreateSystemDescDirect(
      fbb, &chipDescs, &chipDescIndices, &chipCapabilities, &chipCoords,
      &allConnections);
  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version version(ttmlirVersion.major, ttmlirVersion.minor,
                                ttmlirVersion.patch);
  auto root = ::tt::target::CreateSystemDescRootDirect(
      fbb, &version, ::ttmlir::getGitHash(), "unknown", systemDesc);
  ::tt::target::FinishSizePrefixedSystemDescRootBuffer(fbb, root);
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  if (not ::tt::target::VerifySizePrefixedSystemDescRootBuffer(verifier)) {
    throw std::runtime_error("Failed to verify system desc root buffer");
  }
  uint8_t *buf = fbb.GetBufferPointer();
  auto size = fbb.GetSize();
  auto handle = ::tt::runtime::utils::malloc_shared(size);
  std::memcpy(handle.get(), buf, size);
  return std::make_unique<SystemDesc>(handle);
}

std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc() {
  size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
  size_t numPciDevices = ::tt::tt_metal::GetNumPCIeDevices();
  TT_FATAL(numDevices % numPciDevices == 0,
           "Unexpected non-rectangular grid of devices");
  std::vector<chip_id_t> deviceIds(numDevices);
  std::iota(deviceIds.begin(), deviceIds.end(), 0);
  ::tt::tt_metal::DeviceGrid deviceGrid = std::make_pair(numDevices / numPciDevices, numPciDevices);
  ::tt::tt_metal::DeviceMesh deviceMesh(deviceGrid, deviceIds, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1);
  std::exception_ptr eptr = nullptr;
  std::unique_ptr<SystemDesc> desc;
  try {
    desc = getCurrentSystemDescImpl(deviceMesh);
  } catch (...) {
    eptr = std::current_exception();
  }
  deviceMesh.close_devices();
  if (eptr) {
    std::rethrow_exception(eptr);
  }
  return std::make_pair(*desc, deviceIds);
}

template <typename T>
static BorrowedStorage createStorage(void *ptr, std::uint32_t numElements) {
  return BorrowedStorage(
      borrowed_buffer::Buffer<T>(static_cast<T *>(ptr), numElements), [] {},
      [] {});
}

static BorrowedStorage createStorage(void *ptr, std::uint32_t numElements,
                                     ::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return createStorage<float>(ptr, numElements);
  // case ::tt::target::DataType::Float16:
  //   return createStorage<float16>(ptr, numElements);
  case ::tt::target::DataType::BFloat16:
    return createStorage<bfloat16>(ptr, numElements);
  case ::tt::target::DataType::UInt32:
    return createStorage<std::uint32_t>(ptr, numElements);
  case ::tt::target::DataType::UInt16:
    return createStorage<std::uint16_t>(ptr, numElements);
  // case ::tt::target::DataType::UInt8:
  //   return createStorage<std::uint8_t>(ptr, numElements);
  default:
    throw std::runtime_error("Unsupported data type");
  }
}

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType) {
  std::uint32_t numElements = shape[0] * stride[0];
  auto tensor = std::make_shared<::ttnn::Tensor>(
      createStorage(data.get(), numElements, dataType), shape,
      utils::toTTNNDataType(dataType), ::ttnn::Layout::ROW_MAJOR);
  return Tensor(tensor, data);
}

Device openDevice(std::vector<int> const &deviceIds,
                  std::vector<std::uint8_t> const &numHWCQs) {
  assert(deviceIds.size() == 1 && "Only one device is supported for now");
  assert(numHWCQs.empty() && "HWCQs are not supported for now");
  auto &device = ::ttnn::open_device(deviceIds.front());
  return Device::borrow(device);
}

void closeDevice(Device device) {
  auto &ttnn_device = device.as<::ttnn::Device>();
  ::ttnn::close_device(ttnn_device);
}

static ::tt::target::ttnn::TTNNBinary const *getBinary(Flatbuffer binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  if (not isTTNN) {
    throw std::runtime_error("Unsupported binary format");
  }
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex,
             std::vector<Tensor> const &inputHandles,
             std::vector<Tensor> const &outputHandles) {
  ::ttnn::Device &device = deviceHandle.as<::ttnn::Device>();
  ::tt::target::ttnn::TTNNBinary const &fbb = *getBinary(executableHandle);
  std::vector<::ttnn::Tensor *> inputs;
  inputs.reserve(inputHandles.size());
  for (auto &input : inputHandles) {
    inputs.push_back(static_cast<::ttnn::Tensor *>(input.handle.get()));
  }
  std::vector<::ttnn::Tensor *> outputs;
  outputs.reserve(outputHandles.size());
  for (auto &output : outputHandles) {
    outputs.push_back(static_cast<::ttnn::Tensor *>(output.handle.get()));
  }
  tt::runtime::ttnn::runProgram(device, fbb.programs()->Get(programIndex),
                                inputs, outputs);
  return Event(nullptr);
}

void wait(Event) {
  // Not implemented
}

} // namespace tt::runtime::ttnn
