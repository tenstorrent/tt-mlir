// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <variant>

#include "tracy/Tracy.hpp"
#include "tt/runtime/detail/common.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttmetal.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Version.h"

#include "executor.h"

namespace tt::runtime::ttmetal {

using ::tt::runtime::DeviceRuntime;
using Events = std::vector<std::shared_ptr<::tt::tt_metal::Event>>;
using DeviceList = std::vector<::tt::tt_metal::IDevice *>;

static ::tt::target::metal::TTMetalBinary const *getBinary(Flatbuffer binary) {
  bool isTTMetal =
      ::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          binary.handle.get());
  if (!isTTMetal) {
    LOG_FATAL("Unsupported binary format");
  }
  return ::tt::target::metal::GetSizePrefixedTTMetalBinary(binary.handle.get());
}

static Tensor createNullTensor() {
  return Tensor(nullptr, nullptr, DeviceRuntime::TTMetal);
}

static tt::runtime::MemoryView
createMemoryView(tt::tt_metal::detail::MemoryView const &memoryView) {
  return tt::runtime::MemoryView{
      .numBanks = memoryView.num_banks,
      .totalBytesPerBank = memoryView.total_bytes_per_bank,
      .totalBytesAllocatedPerBank = memoryView.total_bytes_allocated_per_bank,
      .totalBytesFreePerBank = memoryView.total_bytes_free_per_bank,
      .largestContiguousBytesFreePerBank =
          memoryView.largest_contiguous_bytes_free_per_bank,
      .blockTable = memoryView.block_table,
  };
}

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType) {
  TensorDesc desc;
  desc.shape = shape;
  desc.stride = stride;
  desc.itemsize = itemsize;
  desc.dataType = dataType;
  std::shared_ptr<MetalTensor> tensor = std::make_shared<MetalTensor>(desc);
  return Tensor(static_pointer_cast<void>(tensor), data,
                DeviceRuntime::TTMetal);
}

bool isTensorAllocated(Tensor tensor) {
  LOG_FATAL("isTensorAllocated not implemented for metal runtime");
}

tt::target::DataType getTensorDataType(Tensor tensor) {
  const MetalTensor &metalTensor =
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal);
  return std::visit(
      utils::overloaded{
          [&](TensorDesc const &desc) { return desc.dataType; },
          [&](DeviceBuffer const &buffer) {
            LOG_FATAL("Datatype mapping from buffer not supported yet.");
            return ::tt::target::DataType::Float32;
          },
      },
      metalTensor);
}

bool getTensorRetain(Tensor tensor) {
  LOG_FATAL("getTensorRetain not implemented for metal runtime");
}

void setTensorRetain(Tensor tensor, bool retain) {
  LOG_FATAL("setTensorRetain not implemented for metal runtime");
}

size_t getNumAvailableDevices() {
  return ::tt::tt_metal::GetNumAvailableDevices();
}

Device openMeshDevice(const std::vector<uint32_t> &meshShape,
                      const MeshDeviceOptions &options) {
  LOG_ASSERT(meshShape.size() == 2, "Mesh shape must be 2D for now");
  ::tt::tt_metal::distributed::MeshShape shape(meshShape);

  LOG_ASSERT(options.meshOffset.size() == 2, "Mesh offset must be 2D for now");
  ::tt::tt_metal::distributed::MeshCoordinate offset(options.meshOffset);

  size_t l1SmallSize = options.l1SmallSize.value_or(DEFAULT_L1_SMALL_SIZE);
  ::tt::tt_metal::DispatchCoreType dispatchCoreType =
      tt::runtime::common::getDispatchCoreType(options.dispatchCoreType);

  ::tt::tt_metal::distributed::MeshDeviceConfig meshConfig(shape, offset,
                                                           options.deviceIds);

  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> meshDevice =
      ::tt::tt_metal::distributed::MeshDevice::create(
          meshConfig, l1SmallSize, DEFAULT_TRACE_REGION_SIZE, options.numHWCQs,
          dispatchCoreType);

  LOG_DEBUG("Device grid size = { ",
            meshDevice->compute_with_storage_grid_size().x, ", ",
            meshDevice->compute_with_storage_grid_size().y, " }");

  return Device(std::static_pointer_cast<void>(meshDevice),
                DeviceRuntime::TTMetal);
}

void closeMeshDevice(Device parentMesh) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      parentMesh.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);

  LOG_ASSERT(metalMeshDevice.is_parent_mesh(),
             "Mesh device must be a parent mesh");

  if (uint32_t numSubMeshes = metalMeshDevice.get_submeshes().size()) {
    LOG_WARNING("Calling close on parent mesh device ", metalMeshDevice,
                " that has ", numSubMeshes, " unreleased submeshes.");
  }

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE)
  for (::tt::tt_metal::IDevice *ttmetalDevice : metalMeshDevice.get_devices()) {
    ::tt::tt_metal::detail::DumpDeviceProfileResults(ttmetalDevice);
  }
#endif
  metalMeshDevice.close();
}

Device createSubMeshDevice(
    Device parentMesh, const std::pair<uint32_t, uint32_t> &meshShape,
    const std::optional<const std::pair<uint32_t, uint32_t>> &meshOffset) {
  ::tt::tt_metal::distributed::MeshDevice &parentMeshDevice =
      parentMesh.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  LOG_ASSERT(parentMeshDevice.is_parent_mesh(),
             "Mesh device must be a parent mesh");

  ::tt::tt_metal::distributed::MeshShape shape{meshShape.first,
                                               meshShape.second};

  std::optional<::tt::tt_metal::distributed::MeshCoordinate> offset =
      std::nullopt;
  if (meshOffset.has_value()) {
    offset = ::tt::tt_metal::distributed::MeshCoordinate{
        meshOffset.value().first, meshOffset.value().second};
  }

  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> subMeshDevice =
      parentMeshDevice.create_submesh(shape, offset);

  return Device(std::static_pointer_cast<void>(subMeshDevice),
                DeviceRuntime::TTMetal);
}

void releaseSubMeshDevice(Device subMesh) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      subMesh.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);

  LOG_ASSERT(!metalMeshDevice.is_parent_mesh(),
             "Mesh device must be a submesh");

  metalMeshDevice.close();
  subMesh.handle.reset();
}

void deallocateBuffers(Device deviceHandle) {
  ::tt::tt_metal::distributed::MeshDevice &meshDevice =
      deviceHandle.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);

  for (::tt::tt_metal::IDevice *device : meshDevice.get_devices()) {
    device->allocator()->deallocate_buffers();
  }
}

void dumpMemoryReport(Device deviceHandle) {
  ::tt::tt_metal::distributed::MeshDevice &meshDevice =
      deviceHandle.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);

  for (::tt::tt_metal::IDevice *device : meshDevice.get_devices()) {
    ::tt::tt_metal::detail::DumpDeviceMemoryState(device);
  }
}

std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device deviceHandle, int deviceID) {
  std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
      memoryMap;
  ::tt::tt_metal::distributed::MeshDevice &meshDevice =
      deviceHandle.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);

  auto *device = meshDevice.get_device(deviceID);

  auto dramMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      device, tt::tt_metal::BufferType::DRAM);
  auto l1MemoryView = ::tt::tt_metal::detail::GetMemoryView(
      device, tt::tt_metal::BufferType::L1);
  auto l1SmallMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      device, tt::tt_metal::BufferType::L1_SMALL);
  auto traceMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      device, tt::tt_metal::BufferType::TRACE);

  memoryMap[tt::runtime::MemoryBufferType::DRAM] =
      createMemoryView(dramMemoryView);
  memoryMap[tt::runtime::MemoryBufferType::L1] = createMemoryView(l1MemoryView);
  memoryMap[tt::runtime::MemoryBufferType::L1_SMALL] =
      createMemoryView(l1SmallMemoryView);
  memoryMap[tt::runtime::MemoryBufferType::TRACE] =
      createMemoryView(traceMemoryView);

  return memoryMap;
}

void wait(Event event) {
  ::tt::tt_metal::EventSynchronize(
      event.handle_as<::tt::tt_metal::Event>(DeviceRuntime::TTMetal));
}

void wait(Tensor tensor) { ::tt::runtime::ttmetal::wait(tensor.event); }

void wait(std::vector<Tensor> const &tensors) {
  for (Tensor tensor : tensors) {
    ::tt::runtime::ttmetal::wait(tensor);
  }
}

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> &inputs) {
  ::tt::target::metal::TTMetalBinary const &fbb = *getBinary(executableHandle);
  ::tt::target::metal::Program const *program =
      fbb.programs()->Get(programIndex);
  ::tt::tt_metal::distributed::MeshDevice &meshDevice =
      deviceHandle.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  DeviceList allDevices = meshDevice.get_devices();
  LOG_ASSERT(allDevices.size() > 0, "Unexpected empty device mesh");
  DeviceList deviceList = {allDevices[0]};
  LOG_ASSERT(deviceList.size() == 1, "Only one device is supported for now");
  LOG_ASSERT(program->device_programs()->size() == deviceList.size(),
             "Device programs size mismatch");

  std::vector<Tensor> outputs;
  for (std::size_t i = 0; i < program->device_programs()->size(); ++i) {
    ::tt::tt_metal::IDevice *device = deviceList[i];

    ZoneScoped;
    std::string zoneName = "submit_" + std::string(program->name()->c_str()) +
                           "_device_" + std::to_string(device->id());
    ZoneName(zoneName.c_str(), zoneName.size());

    LOG_ASSERT(outputs.empty(), "Multi-device outputs not supported");
    outputs = executeDeviceProgram(device, program->device_programs()->Get(i),
                                   inputs);
    assert(outputs.size() == program->outputs()->size() &&
           "Outputs size mismatch");
  }

  return outputs;
}

std::string getOpDebugString(OpContext opContextHandle) {
  // Not implemented
  LOG_WARNING("obtaining op debug string for metal runtime not implemented");
  return "";
}

std::string getOpLocInfo(OpContext opContextHandle) {
  // Not implemented
  LOG_WARNING("obtaining op location info for metal runtime not implemented");
  return "";
}

Tensor getOpOutputTensor(OpContext opContextHandle,
                         CallbackContext programContextHandle) {
  // Not implemented
  LOG_WARNING("obtaining op output tensor for metal runtime not implemented");
  return createNullTensor();
}

std::vector<std::byte> getTensorDataBuffer(::tt::runtime::Tensor tensor) {
  LOG_WARNING("getDataBuffer not implemented for metal runtime");
  return {};
}

std::vector<std::uint32_t> getTensorShape(::tt::runtime::Tensor tensor) {
  LOG_WARNING("getShape not implemented for metal runtime");
  return {};
}

std::vector<std::uint32_t> getTensorStride(::tt::runtime::Tensor tensor) {
  LOG_WARNING("getStride not implemented for metal runtime");
  return {};
}

std::uint32_t getTensorElementSize(::tt::runtime::Tensor tensor) {
  LOG_WARNING("getElementSize not implemented for metal runtime");
  return 0;
}

std::uint32_t getTensorVolume(::tt::runtime::Tensor tensor) {
  LOG_WARNING("getVolume not implemented for metal runtime");
  return 0;
}

TensorDesc getTensorDesc(::tt::runtime::Tensor tensor) {
  LOG_WARNING("getTensorDesc not implemented for metal runtime");
  return {};
}

} // namespace tt::runtime::ttmetal
