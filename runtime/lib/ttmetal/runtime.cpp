// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <variant>

#include "tt/runtime/detail/ttmetal.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"

#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Version.h"

namespace tt::runtime::ttmetal {
using ::tt::runtime::DeviceRuntime;
constexpr inline std::size_t kHostBufferCommandQueueId = 0;
using Events = std::vector<std::shared_ptr<::tt::tt_metal::Event>>;
using DeviceList = std::vector<::tt::tt_metal::Device *>;
using MetalTensor =
    std::variant<TensorDesc, std::shared_ptr<::tt::tt_metal::Buffer>>;

static ::tt::target::metal::TTMetalBinary const *getBinary(Flatbuffer binary) {
  bool isTTMetal =
      ::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          binary.handle.get());
  if (not isTTMetal) {
    throw std::runtime_error("Unsupported binary format");
  }
  return ::tt::target::metal::GetSizePrefixedTTMetalBinary(binary.handle.get());
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

tt::target::DataType getTensorDataType(Tensor tensor) {
  const MetalTensor &metalTensor =
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal);
  if (std::holds_alternative<TensorDesc>(metalTensor)) {
    TensorDesc desc = std::get<TensorDesc>(metalTensor);
    return desc.dataType;
  }
  if (std::holds_alternative<std::shared_ptr<::tt::tt_metal::Buffer>>(
          metalTensor)) {
    throw std::runtime_error("Datatype mapping from buffer not supported yet.");
  }
  assert(false && "Unsupported tensor type");
  return ::tt::target::DataType::Float32;
}

size_t getNumAvailableDevices() {
  return ::tt::tt_metal::GetNumAvailableDevices();
}

Device openDevice(DeviceIds const &deviceIds, size_t numHWCQs) {
  assert(deviceIds.size() && "No devices specified");
  ::tt::tt_metal::MeshShape grid = std::make_pair(1, deviceIds.size());
  std::shared_ptr<::tt::tt_metal::MeshDevice> meshDevice =
      ::tt::tt_metal::MeshDevice::create(
          grid, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, numHWCQs,
          ::tt::tt_metal::DispatchCoreType::WORKER);

  return Device(std::static_pointer_cast<void>(meshDevice),
                DeviceRuntime::TTMetal);
}

void closeDevice(Device device) {
  ::tt::tt_metal::MeshDevice &ttmetalMeshDevice =
      device.as<::tt::tt_metal::MeshDevice>(DeviceRuntime::TTMetal);
  ttmetalMeshDevice.close_devices();
}

void deallocateBuffers(Device deviceHandle) {
  ::tt::tt_metal::MeshDevice &meshDevice =
      deviceHandle.as<::tt::tt_metal::MeshDevice>(DeviceRuntime::TTMetal);

  for (::tt::tt_metal::Device *device : meshDevice.get_devices()) {
    device->deallocate_buffers();
  }
}

static std::pair<std::shared_ptr<::tt::tt_metal::Buffer>,
                 std::shared_ptr<::tt::tt_metal::Event>>
prepareInput(::tt::tt_metal::Device *device, MetalTensor const &metalTensor,
             void *data, ::tt::target::TensorRef const *tensorRef) {
  if (std::holds_alternative<TensorDesc>(metalTensor)) {
    // todo assert that tensorDesc matches hostTensorDesc
    std::shared_ptr<::tt::tt_metal::Buffer> buffer =
        createBufferFromTensorRef(device, tensorRef);
    auto event = std::make_shared<::tt::tt_metal::Event>();
    ::tt::tt_metal::CommandQueue &cq =
        device->command_queue(kHostBufferCommandQueueId);
    bool const blocking = false;
    ::tt::tt_metal::EnqueueWriteBuffer(cq, buffer, data, blocking);
    ::tt::tt_metal::EnqueueRecordEvent(cq, event);
    return std::make_pair(buffer, event);
  }

  if (std::holds_alternative<std::shared_ptr<::tt::tt_metal::Buffer>>(
          metalTensor)) {
    std::shared_ptr<::tt::tt_metal::Buffer> buffer =
        std::get<std::shared_ptr<::tt::tt_metal::Buffer>>(metalTensor);
    throw std::runtime_error("Input from buffer not supported yet");
  }
  assert(false && "Unsupported tensor type");
  return std::make_pair(nullptr, nullptr);
}

static std::shared_ptr<::tt::tt_metal::Buffer>
prepareOutput(::tt::tt_metal::Device *device, MetalTensor const *metalTensor,
              ::tt::target::TensorRef const *tensorRef) {
  assert(metalTensor != nullptr);
  if (TensorDesc const *hostTensorDesc = std::get_if<TensorDesc>(metalTensor);
      hostTensorDesc) {
    return createBufferFromTensorRef(device, tensorRef);
  }

  if (std::shared_ptr<::tt::tt_metal::Buffer> const *buffer =
          std::get_if<std::shared_ptr<::tt::tt_metal::Buffer>>(metalTensor);
      buffer) {
    return *buffer;
  }
  assert(false && "Unsupported tensor type");
  return nullptr;
}

Events maybeCopyHostOutputs(::tt::tt_metal::Device *device,
                            std::vector<Tensor> const &outputHandles,
                            std::vector<OutputBuffer> submitOutputs,
                            Events submitEvents) {
  Events copyEvents;
  int i = 0;
  for (Tensor const &outputHandle : outputHandles) {
    if (TensorDesc const *hostTensor = std::get_if<TensorDesc>(
            &outputHandle.as<MetalTensor>(DeviceRuntime::TTMetal));
        hostTensor) {
      ::tt::tt_metal::CommandQueue &cq =
          device->command_queue(kHostBufferCommandQueueId);
      for (auto submitEvent : submitEvents) {
        ::tt::tt_metal::EnqueueWaitForEvent(cq, submitEvent);
      }
      submitEvents.clear();
      auto event = std::make_shared<::tt::tt_metal::Event>();
      bool const blocking = false;
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
             std::vector<Tensor> const &inputHandles,
             std::vector<Tensor> const &outputHandles) {
  ::tt::target::metal::TTMetalBinary const &fbb = *getBinary(executableHandle);
  ::tt::target::metal::Program const *program =
      fbb.programs()->Get(programIndex);
  ::tt::tt_metal::MeshDevice &meshDevice =
      deviceHandle.as<::tt::tt_metal::MeshDevice>(DeviceRuntime::TTMetal);
  DeviceList allDevices = meshDevice.get_devices();
  assert(allDevices.size() > 0 && "Unexpected empty device mesh");
  DeviceList deviceList = {allDevices[0]};
  assert(deviceList.size() == 1 && "Only one device is supported for now");
  std::shared_ptr<Events> events = std::make_shared<Events>();
  assert(program->device_programs()->size() == deviceList.size() &&
         "Device programs size mismatch");
  for (std::size_t i = 0; i < program->device_programs()->size(); ++i) {
    ::tt::tt_metal::Device *device = deviceList[i];

    ZoneScoped;
    std::string zoneName = "submit_" + std::string(program->name()->c_str()) +
                           "_device_" + std::to_string(device->id());
    ZoneName(zoneName.c_str(), zoneName.size());

    ::tt::target::metal::DeviceProgram const *deviceProgram =
        program->device_programs()->Get(i);
    Events deviceEvents;

    std::vector<InputBuffer> inputs;
    inputs.reserve(inputHandles.size());
    assert(inputHandles.size() == deviceProgram->inputs()->size() &&
           "Input size mismatch");
    for (unsigned i = 0; i < inputHandles.size(); ++i) {
      ::tt::target::TensorRef const *tensorRef =
          deviceProgram->inputs()->Get(i);
      auto [buffer, event] = prepareInput(
          device, inputHandles[i].as<MetalTensor>(DeviceRuntime::TTMetal),
          inputHandles[i].data.get(), tensorRef);
      inputs.emplace_back(deviceProgram->inputs()->Get(i)->global_id(), buffer,
                          event);
    }

    std::vector<OutputBuffer> outputs;
    outputs.reserve(outputHandles.size());
    assert(outputHandles.size() == deviceProgram->outputs()->size() &&
           "Output size mismatch");
    for (unsigned i = 0; i < outputHandles.size(); ++i) {
      ::tt::target::TensorRef const *tensorRef =
          deviceProgram->outputs()->Get(i);
      std::shared_ptr<::tt::tt_metal::Buffer> buffer = prepareOutput(
          device, &outputHandles[i].as<MetalTensor>(DeviceRuntime::TTMetal),
          tensorRef);
      outputs.emplace_back(deviceProgram->outputs()->Get(i)->global_id(),
                           buffer);
    }

    std::size_t cq_id = 0;
    for (::tt::target::metal::CommandQueue const *cq :
         *deviceProgram->command_queues()) {
      FrameMark;
      deviceEvents.push_back(
          executeCommandQueue(device, cq, cq_id, inputs, outputs));
      ++cq_id;
      FrameMark;
    }

    Events copyEvents =
        maybeCopyHostOutputs(device, outputHandles, outputs, deviceEvents);
    if (not copyEvents.empty()) {
      std::swap(deviceEvents, copyEvents);
    }

    events->insert(events->end(), deviceEvents.begin(), deviceEvents.end());
  }

  return Event(static_pointer_cast<void>(events), DeviceRuntime::TTMetal);
}

void wait(Event event) {
  Events events = event.as<Events>(DeviceRuntime::TTMetal);
  for (auto e : events) {
    ::tt::tt_metal::EventSynchronize(e);
  }
}

} // namespace tt::runtime::ttmetal
