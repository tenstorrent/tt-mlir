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

constexpr inline std::size_t kHostBufferCommandQueueId = 0;
using Events = std::vector<std::shared_ptr<::tt::tt_metal::Event>>;
using DeviceMesh = std::vector<::tt::tt_metal::Device *>;
using MetalTensor =
    std::variant<TensorDesc, std::shared_ptr<::tt::tt_metal::Buffer>>;

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

static ::tt::target::Dim2d toFlatbuffer(CoreCoord coreCoord) {
  return ::tt::target::Dim2d(coreCoord.y, coreCoord.x);
}

static ::tt::target::metal::TTMetalBinary const *getBinary(Flatbuffer binary) {
  bool isTTMetal =
      ::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          binary.handle.get());
  if (not isTTMetal) {
    throw std::runtime_error("Unsupported binary format");
  }
  return ::tt::target::metal::GetSizePrefixedTTMetalBinary(binary.handle.get());
}

std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc() {
  ::tt::tt_metal::Device *device = ::tt::tt_metal::CreateDevice(0);
  std::vector<int> chipIds = {
      device->id(),
  };
  ::flatbuffers::FlatBufferBuilder fbb;
  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version version(ttmlirVersion.major, ttmlirVersion.minor,
                                ttmlirVersion.patch);
  ::tt::target::Dim2d deviceGrid =
      toFlatbuffer(device->compute_with_storage_grid_size());
  std::vector<::flatbuffers::Offset<tt::target::ChipDesc>> chipDescs = {
      ::tt::target::CreateChipDesc(
          fbb, toFlatbuffer(device->arch()), &deviceGrid, (1 << 20), 12,
          (1 << 20), L1_ALIGNMENT, PCIE_ALIGNMENT, DRAM_ALIGNMENT),
  };
  std::vector<uint32_t> chipDescIndices = {
      0,
  };
  ::tt::target::ChipCapability chipCapability =
      ::tt::target::ChipCapability::PCIE;
  if (device->is_mmio_capable()) {
    chipCapability = chipCapability | ::tt::target::ChipCapability::HostMMIO;
  }
  std::vector<::tt::target::ChipCapability> chipCapabilities = {
      chipCapability,
  };
  std::vector<::tt::target::ChipCoord> chipCoord = {
      ::tt::target::ChipCoord(0, 0, 0, 0),
  };
  std::vector<::tt::target::ChipChannel> chipChannel;
  auto systemDesc = ::tt::target::CreateSystemDescDirect(
      fbb, &chipDescs, &chipDescIndices, &chipCapabilities, &chipCoord,
      &chipChannel);
  auto root = ::tt::target::CreateSystemDescRootDirect(
      fbb, &version, ::ttmlir::getGitHash(), "unknown", systemDesc);
  ::tt::target::FinishSizePrefixedSystemDescRootBuffer(fbb, root);
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  if (not ::tt::target::VerifySizePrefixedSystemDescRootBuffer(verifier)) {
    throw std::runtime_error("Failed to verify system desc root buffer");
  }
  uint8_t *buf = fbb.GetBufferPointer();
  auto size = fbb.GetSize();
  auto handle = utils::malloc_shared(size);
  std::memcpy(handle.get(), buf, size);
  ::tt::tt_metal::CloseDevice(device);
  return std::make_pair(SystemDesc(handle), chipIds);
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
  return Tensor(static_pointer_cast<void>(tensor), data);
}

Device openDevice(std::vector<int> const &deviceIds,
                  std::vector<std::uint8_t> const &numHWCQs) {
  assert(numHWCQs.empty() || numHWCQs.size() == deviceIds.size());
  std::shared_ptr<DeviceMesh> deviceMesh = std::make_shared<DeviceMesh>();
  int i = 0;
  for (int deviceId : deviceIds) {
    uint8_t num_hw_cqs = numHWCQs.empty() ? 1 : numHWCQs[i];
    deviceMesh->push_back(CreateDevice(deviceId, num_hw_cqs));
    ++i;
  }
  return static_pointer_cast<void>(deviceMesh);
}

void closeDevice(Device device) {
  DeviceMesh &deviceMesh = device.as<DeviceMesh>();
  for (::tt::tt_metal::Device *device : deviceMesh) {
    ::tt::tt_metal::CloseDevice(device);
  }
}

static std::pair<std::shared_ptr<::tt::tt_metal::Buffer>,
                 std::shared_ptr<::tt::tt_metal::Event>>
prepareInput(::tt::tt_metal::Device *device, MetalTensor const &metalTensor,
             void *data, ::tt::target::TensorRef const *tensorRef) {
  if (TensorDesc const *hostTensorDesc = std::get_if<TensorDesc>(&metalTensor);
      hostTensorDesc) {
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
  } else if (std::shared_ptr<::tt::tt_metal::Buffer> const *buffer =
                 std::get_if<std::shared_ptr<::tt::tt_metal::Buffer>>(
                     &metalTensor);
             buffer) {
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
  } else if (std::shared_ptr<::tt::tt_metal::Buffer> const *buffer =
                 std::get_if<std::shared_ptr<::tt::tt_metal::Buffer>>(
                     metalTensor);
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
    if (TensorDesc const *hostTensor =
            std::get_if<TensorDesc>(&outputHandle.as<MetalTensor>());
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
  DeviceMesh &deviceMesh = deviceHandle.as<DeviceMesh>();
  assert(deviceMesh.size() == 1 && "Only one device is supported for now");
  std::shared_ptr<Events> events = std::make_shared<Events>();
  std::size_t cq_id = 0;
  assert(program->device_programs()->size() == deviceMesh.size() &&
         "Device programs size mismatch");
  for (std::size_t i = 0; i < program->device_programs()->size(); ++i) {
    ::tt::tt_metal::Device *device = deviceMesh[i];
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
      auto [buffer, event] =
          prepareInput(device, inputHandles[i].as<MetalTensor>(),
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
      std::shared_ptr<::tt::tt_metal::Buffer> buffer =
          prepareOutput(device, &outputHandles[i].as<MetalTensor>(), tensorRef);
      outputs.emplace_back(deviceProgram->outputs()->Get(i)->global_id(),
                           buffer);
    }

    for (::tt::target::metal::CommandQueue const *cq :
         *deviceProgram->command_queues()) {
      deviceEvents.push_back(
          executeCommandQueue(device, cq, cq_id, inputs, outputs));
      ++cq_id;
    }

    Events copyEvents =
        maybeCopyHostOutputs(device, outputHandles, outputs, deviceEvents);
    if (not copyEvents.empty()) {
      std::swap(deviceEvents, copyEvents);
    }

    events->insert(events->end(), deviceEvents.begin(), deviceEvents.end());
  }

  return static_pointer_cast<void>(events);
}

void wait(Event event) {
  Events events = event.as<Events>();
  for (auto e : events) {
    ::tt::tt_metal::EventSynchronize(e);
  }
}

} // namespace tt::runtime::ttmetal
