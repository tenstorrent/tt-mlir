// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <variant>

#include "tracy/Tracy.hpp"
#include "tt/runtime/detail/common.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Version.h"

namespace tt::runtime::ttmetal {
using ::tt::runtime::DeviceRuntime;
constexpr inline std::size_t kHostBufferCommandQueueId = 0;
using Events = std::vector<std::shared_ptr<::tt::tt_metal::Event>>;
using DeviceList = std::vector<::tt::tt_metal::IDevice *>;
using MetalTensor =
    std::variant<TensorDesc, std::shared_ptr<::tt::tt_metal::Buffer>>;

static const ::tt::target::metal::TTMetalBinary *getBinary(Flatbuffer binary) {
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
createMemoryView(const tt::tt_metal::detail::MemoryView &memoryView) {
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
                    const std::vector<std::uint32_t> &shape,
                    const std::vector<std::uint32_t> &stride,
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
  if (std::holds_alternative<TensorDesc>(metalTensor)) {
    TensorDesc desc = std::get<TensorDesc>(metalTensor);
    return desc.dataType;
  }
  if (std::holds_alternative<std::shared_ptr<::tt::tt_metal::Buffer>>(
          metalTensor)) {
    LOG_FATAL("Datatype mapping from buffer not supported yet.");
  }
  LOG_ASSERT(false, "Unsupported tensor type");
  return ::tt::target::DataType::Float32;
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
    Device parentMesh, const std::vector<uint32_t> &meshShape,
    const std::optional<const std::vector<uint32_t>> &meshOffset) {
  ::tt::tt_metal::distributed::MeshDevice &parentMeshDevice =
      parentMesh.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  LOG_ASSERT(parentMeshDevice.is_parent_mesh(),
             "Mesh device must be a parent mesh");
  LOG_ASSERT(meshShape.size() == 2, "Mesh shape must be 2D for now");
  ::tt::tt_metal::distributed::MeshShape shape{meshShape[0], meshShape[1]};

  std::optional<::tt::tt_metal::distributed::MeshCoordinate> offset =
      std::nullopt;
  if (meshOffset.has_value()) {
    LOG_ASSERT(meshOffset.value().size() == 2,
               "Mesh offset must be 2D for now");
    offset = ::tt::tt_metal::distributed::MeshCoordinate{meshOffset.value()[0],
                                                         meshOffset.value()[1]};
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

void reshapeMeshDevice(Device meshDevice,
                       const std::vector<uint32_t> &meshShape) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      meshDevice.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);

  metalMeshDevice.reshape(
      ::tt::tt_metal::distributed::MeshShape(meshShape[0], meshShape[1]));
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
  Events events = event.as<Events>(DeviceRuntime::TTMetal);
  for (auto e : events) {
    ::tt::tt_metal::EventSynchronize(e);
  }
}

void wait(Tensor tensor) { ::tt::runtime::ttmetal::wait(tensor.event); }

void wait(const std::vector<Tensor> &tensors) {
  for (Tensor tensor : tensors) {
    ::tt::runtime::ttmetal::wait(tensor);
  }
}

static std::pair<std::shared_ptr<::tt::tt_metal::Buffer>,
                 std::shared_ptr<::tt::tt_metal::Event>>
prepareInput(::tt::tt_metal::IDevice *device, const MetalTensor &metalTensor,
             void *data, const ::tt::target::metal::TensorRef *tensorRef) {
  if (std::holds_alternative<TensorDesc>(metalTensor)) {
    // todo assert that tensorDesc matches hostTensorDesc
    std::shared_ptr<::tt::tt_metal::Buffer> buffer =
        createBufferFromTensorRef(device, tensorRef);
    auto event = std::make_shared<::tt::tt_metal::Event>();
    ::tt::tt_metal::CommandQueue &cq =
        device->command_queue(kHostBufferCommandQueueId);
    const bool blocking = false;
    ::tt::tt_metal::EnqueueWriteBuffer(cq, buffer, data, blocking);
    ::tt::tt_metal::EnqueueRecordEvent(cq, event);
    return std::make_pair(buffer, event);
  }

  if (std::holds_alternative<std::shared_ptr<::tt::tt_metal::Buffer>>(
          metalTensor)) {
    std::shared_ptr<::tt::tt_metal::Buffer> buffer =
        std::get<std::shared_ptr<::tt::tt_metal::Buffer>>(metalTensor);
    LOG_FATAL("Input from buffer not supported yet");
  }
  LOG_ASSERT(false, "Unsupported tensor type");
  return std::make_pair(nullptr, nullptr);
}

static std::shared_ptr<::tt::tt_metal::Buffer>
prepareOutput(::tt::tt_metal::IDevice *device, const MetalTensor *metalTensor,
              const ::tt::target::metal::TensorRef *tensorRef) {
  LOG_ASSERT(metalTensor != nullptr);
  if (const TensorDesc *hostTensorDesc = std::get_if<TensorDesc>(metalTensor);
      hostTensorDesc) {
    return createBufferFromTensorRef(device, tensorRef);
  }

  if (const std::shared_ptr<::tt::tt_metal::Buffer> *buffer =
          std::get_if<std::shared_ptr<::tt::tt_metal::Buffer>>(metalTensor);
      buffer) {
    return *buffer;
  }
  LOG_ASSERT(false, "Unsupported tensor type");
  return nullptr;
}

Events maybeCopyHostOutputs(::tt::tt_metal::IDevice *device,
                            const std::vector<Tensor> &outputHandles,
                            std::vector<OutputBuffer> submitOutputs,
                            Events submitEvents) {
  Events copyEvents;
  int i = 0;
  for (const Tensor &outputHandle : outputHandles) {
    if (const TensorDesc *hostTensor = std::get_if<TensorDesc>(
            &outputHandle.as<MetalTensor>(DeviceRuntime::TTMetal));
        hostTensor) {
      ::tt::tt_metal::CommandQueue &cq =
          device->command_queue(kHostBufferCommandQueueId);
      for (auto submitEvent : submitEvents) {
        ::tt::tt_metal::EnqueueWaitForEvent(cq, submitEvent);
      }
      submitEvents.clear();
      auto event = std::make_shared<::tt::tt_metal::Event>();
      const bool blocking = false;
      auto [global_id, buffer] = submitOutputs[i];
      ::tt::tt_metal::EnqueueReadBuffer(cq, buffer, outputHandle.data.get(),
                                        blocking);
      ::tt::tt_metal::EnqueueRecordEvent(cq, event);
      copyEvents.push_back(event);
    }
    ++i;
  }
  return copyEvents;
}

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex,
             const std::vector<Tensor> &inputHandles,
             const std::vector<Tensor> &outputHandles) {
  const ::tt::target::metal::TTMetalBinary &fbb = *getBinary(executableHandle);
  const ::tt::target::metal::Program *program =
      fbb.programs()->Get(programIndex);
  ::tt::tt_metal::distributed::MeshDevice &meshDevice =
      deviceHandle.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  DeviceList allDevices = meshDevice.get_devices();
  LOG_ASSERT(allDevices.size() > 0, "Unexpected empty device mesh");
  DeviceList deviceList = {allDevices[0]};
  LOG_ASSERT(deviceList.size() == 1, "Only one device is supported for now");
  std::shared_ptr<Events> events = std::make_shared<Events>();
  LOG_ASSERT(program->device_programs()->size() == deviceList.size(),
             "Device programs size mismatch");
  for (std::size_t i = 0; i < program->device_programs()->size(); ++i) {
    ::tt::tt_metal::IDevice *device = deviceList[i];

    ZoneScoped;
    std::string zoneName = "submit_" + std::string(program->name()->c_str()) +
                           "_device_" + std::to_string(device->id());
    ZoneName(zoneName.c_str(), zoneName.size());

    const ::tt::target::metal::DeviceProgram *deviceProgram =
        program->device_programs()->Get(i);
    Events deviceEvents;

    std::vector<InputBuffer> inputs;
    inputs.reserve(inputHandles.size());
    LOG_ASSERT(inputHandles.size() == deviceProgram->inputs()->size(),
               "Input size mismatch");
    for (unsigned i = 0; i < inputHandles.size(); ++i) {
      const ::tt::target::metal::TensorRef *tensorRef =
          deviceProgram->inputs()->Get(i);
      auto [buffer, event] = prepareInput(
          device, inputHandles[i].as<MetalTensor>(DeviceRuntime::TTMetal),
          inputHandles[i].data.get(), tensorRef);
      inputs.emplace_back(deviceProgram->inputs()->Get(i)->global_id(), buffer,
                          event);
    }

    std::vector<OutputBuffer> outputs;
    outputs.reserve(outputHandles.size());
    LOG_ASSERT(outputHandles.size() == deviceProgram->outputs()->size(),
               "Output size mismatch");
    for (unsigned i = 0; i < outputHandles.size(); ++i) {
      const ::tt::target::metal::TensorRef *tensorRef =
          deviceProgram->outputs()->Get(i);
      std::shared_ptr<::tt::tt_metal::Buffer> buffer = prepareOutput(
          device, &outputHandles[i].as<MetalTensor>(DeviceRuntime::TTMetal),
          tensorRef);
      outputs.emplace_back(deviceProgram->outputs()->Get(i)->global_id(),
                           buffer);
    }

    std::size_t cq_id = 0;
    for (const ::tt::target::metal::CommandQueue *cq :
         *deviceProgram->command_queues()) {
      FrameMark;
      deviceEvents.push_back(
          executeCommandQueue(device, cq, cq_id, inputs, outputs));
      ++cq_id;
      FrameMark;
    }

    Events copyEvents =
        maybeCopyHostOutputs(device, outputHandles, outputs, deviceEvents);
    if (!copyEvents.empty()) {
      std::swap(deviceEvents, copyEvents);
    }

    events->insert(events->end(), deviceEvents.begin(), deviceEvents.end());
  }

  return Event(static_pointer_cast<void>(events), DeviceRuntime::TTMetal);
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
