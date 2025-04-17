// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <unordered_map>

#include "executor.h"
#include "executor_utils.h"

#include "tracy/Tracy.hpp"
#include "tt/runtime/detail/common.h"
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttmetal.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"

#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Target/TTMetal/types_generated.h"
#include "ttmlir/Version.h"

namespace tt::runtime::ttmetal {

namespace {
class CQExecutor {
public:
  CQExecutor(
      ::tt::tt_metal::IDevice *device,
      const ::flatbuffers::Vector<
          ::flatbuffers::Offset<tt::target::metal::BufferRef>> *programInputs,
      std::vector<Tensor> const &inputs);

  std::vector<Tensor> const &getOutputs() const { return outputs; }

  void execute(::tt::target::metal::CommandQueue const *commandQueue);

private:
  void execute(::tt::target::metal::Command const *command);
  void execute(::tt::target::metal::HostAllocCommand const *command);
  void execute(::tt::target::metal::ReturnCommand const *command);
  void execute(::tt::target::metal::EnqueueProgramCommand const *command,
               char const *debugInfo);
  void execute(::tt::target::metal::EnqueueWriteBufferCommand const *command);
  void execute(::tt::target::metal::EnqueueReadBufferCommand const *command);
  void execute(::tt::target::metal::CreateBufferCommand const *command);
  void execute(::tt::target::metal::DeallocateBufferCommand const *command);
  void execute(::tt::target::metal::CreateEventCommand const *command);
  void execute(::tt::target::metal::EnqueueRecordEventCommand const *command);
  void execute(::tt::target::metal::EnqueueWaitForEventCommand const *command);
  void execute(::tt::target::metal::EventSynchronizeCommand const *command);
  void execute(::tt::target::metal::EventQueryCommand const *command);
  void execute(::tt::target::metal::FinishCommand const *command);

  void processRuntimeArgs(
      ::tt::tt_metal::Program &program, ::tt::tt_metal::KernelHandle &handle,
      CoreRangeSet const &coreRange,
      const ::flatbuffers::Vector<
          ::flatbuffers::Offset<::tt::target::metal::KernelArg>> *rtArgs,
      const ::flatbuffers::Vector<
          ::flatbuffers::Offset<tt::target::metal::BufferRef>> *buffers,
      const ::flatbuffers::Vector<
          ::flatbuffers::Offset<tt::target::metal::CBRef>> *cbs) const;

  uint32_t validateDeviceAddress(uint32_t address,
                                 ::tt::target::MemorySpace memorySpace) const {
    if (!debug::Env::get().deviceAddressValidation) {
      return address;
    }

    std::size_t unreservedBase = 0;
    std::size_t size = 0;
    std::size_t alignment = 0;
    switch (memorySpace) {
    case ::tt::target::MemorySpace::DeviceDRAM: {
      unreservedBase = deviceInfo.dramUnreservedBase;
      size = deviceInfo.dramSize;
      alignment = deviceInfo.dramAlignment;
      break;
    }
    case ::tt::target::MemorySpace::DeviceL1: {
      unreservedBase = deviceInfo.l1UnreservedBase;
      size = deviceInfo.l1Size;
      alignment = deviceInfo.l1Alignment;
      break;
    }
    default: {
      LOG_FATAL("Unsupported memory space for device address validation");
      break;
    }
    }
    assert(unreservedBase > 0);
    assert(alignment > 0);

    LOG_ASSERT(address >= unreservedBase,
               "Device address out of bounds for memory space[",
               ::tt::target::EnumNameMemorySpace(memorySpace), "], address[",
               address, "] < unreserved base[", unreservedBase, "]");
    LOG_ASSERT(address < size, "Device address out of bounds for memory space[",
               ::tt::target::EnumNameMemorySpace(memorySpace), "], address[",
               address, "] >= size[", size, "]");
    LOG_ASSERT(address % alignment == 0,
               "Device address not aligned for memory space[",
               ::tt::target::EnumNameMemorySpace(memorySpace), "], address[",
               address, "] % alignment[", alignment, "]");
    return address;
  }

private:
  ::tt::tt_metal::IDevice *device;
  std::vector<std::shared_ptr<::tt::tt_metal::Event>> initEvents;
  std::unordered_map<std::uint32_t, DeviceBuffer> deviceBuffers;
  std::unordered_map<std::uint32_t, Tensor> hostBuffers;
  std::unordered_map<std::uint32_t, std::shared_ptr<::tt::tt_metal::Event>>
      events;
  std::vector<Tensor> outputs;
  ::tt::tt_metal::CommandQueue *cq;
  char const *currentProgramName;

  // Only used for validation.
  struct {
    std::size_t dramUnreservedBase = 0;
    std::size_t dramSize = 0;
    std::size_t dramAlignment = 0;
    std::size_t l1UnreservedBase = 0;
    std::size_t l1Size = 0;
    std::size_t l1Alignment = 0;
  } deviceInfo;
};
} // namespace

CQExecutor::CQExecutor(
    ::tt::tt_metal::IDevice *device,
    const ::flatbuffers::Vector<
        ::flatbuffers::Offset<tt::target::metal::BufferRef>> *programInputs,
    std::vector<Tensor> const &inputs)
    : device(device) {
  initEvents.reserve(inputs.size());

  std::uint32_t inputIndex = 0;
  for (Tensor const &input : inputs) {
    const ::tt::target::metal::BufferRef *ref =
        programInputs->Get(inputIndex++);

    std::visit(utils::overloaded{
                   [&](TensorDesc const &) {
                     auto [_, inserted] =
                         hostBuffers.try_emplace(ref->global_id(), input);
                     assert(inserted);
                   },
                   [&](DeviceBuffer const &buffer) {
                     auto [_, inserted] =
                         deviceBuffers.try_emplace(ref->global_id(), buffer);
                     assert(inserted);
                   },
               },
               input.as<MetalTensor>(DeviceRuntime::TTMetal));

    auto event =
        input.event.handle_as<::tt::tt_metal::Event>(DeviceRuntime::TTMetal);
    if (event) {
      initEvents.push_back(event);
    }
  }

  if (debug::Env::get().deviceAddressValidation) {
    deviceInfo.dramUnreservedBase =
        device->allocator()->get_base_allocator_addr(
            ::tt::tt_metal::HalMemType::DRAM);
    deviceInfo.dramSize = device->dram_size_per_channel();
    deviceInfo.dramAlignment =
        device->allocator()->get_alignment(::tt::tt_metal::BufferType::DRAM);
    deviceInfo.l1UnreservedBase = device->allocator()->get_base_allocator_addr(
        ::tt::tt_metal::HalMemType::L1);
    deviceInfo.l1Size = device->l1_size_per_core();
    deviceInfo.l1Alignment =
        device->allocator()->get_alignment(::tt::tt_metal::BufferType::L1);
  }
}

void CQExecutor::execute(
    ::tt::target::metal::CommandQueue const *commandQueue) {
  currentProgramName = commandQueue->name()->c_str();
  cq = &device->command_queue(commandQueue->queue_id());

  for (auto const &event : initEvents) {
    ::tt::tt_metal::EnqueueWaitForEvent(*cq, event);
  }
  initEvents.clear();

  for (::tt::target::metal::Command const *command :
       *commandQueue->commands()) {
    LOG_TRACE(logger::LogRuntimeTTMetalCommand,
              "Executing command: ", EnumNameCommandType(command->type_type()));
    execute(command);
  }
}

void CQExecutor::execute(::tt::target::metal::Command const *command) {
  switch (command->type_type()) {
  case ::tt::target::metal::CommandType::HostAllocCommand: {
    execute(command->type_as_HostAllocCommand());
    break;
  }
  case ::tt::target::metal::CommandType::ReturnCommand: {
    execute(command->type_as_ReturnCommand());
    break;
  }
  case ::tt::target::metal::CommandType::EnqueueProgramCommand: {
    execute(command->type_as_EnqueueProgramCommand(),
            command->debug_info()->c_str());
    break;
  }
  case ::tt::target::metal::CommandType::EnqueueWriteBufferCommand: {
    execute(command->type_as_EnqueueWriteBufferCommand());
    break;
  }
  case ::tt::target::metal::CommandType::EnqueueReadBufferCommand: {
    execute(command->type_as_EnqueueReadBufferCommand());
    break;
  }
  case ::tt::target::metal::CommandType::CreateBufferCommand: {
    execute(command->type_as_CreateBufferCommand());
    break;
  }
  case ::tt::target::metal::CommandType::DeallocateBufferCommand: {
    execute(command->type_as_DeallocateBufferCommand());
    break;
  }
  case ::tt::target::metal::CommandType::CreateEventCommand: {
    execute(command->type_as_CreateEventCommand());
    break;
  }
  case ::tt::target::metal::CommandType::EnqueueRecordEventCommand: {
    execute(command->type_as_EnqueueRecordEventCommand());
    break;
  }
  case ::tt::target::metal::CommandType::EnqueueWaitForEventCommand: {
    execute(command->type_as_EnqueueWaitForEventCommand());
    break;
  }
  case ::tt::target::metal::CommandType::EventSynchronizeCommand: {
    execute(command->type_as_EventSynchronizeCommand());
    break;
  }
  case ::tt::target::metal::CommandType::EventQueryCommand: {
    execute(command->type_as_EventQueryCommand());
    break;
  }
  case ::tt::target::metal::CommandType::FinishCommand: {
    execute(command->type_as_FinishCommand());
    break;
  }
  case ::tt::target::metal::CommandType::NONE: {
    LOG_FATAL("Unsupported CommandType::NONE");
    break;
  }
  }
}

void CQExecutor::execute(::tt::target::metal::HostAllocCommand const *command) {
  assert(command->dst()->address() == 0);
  auto const *bufferDesc = command->dst()->desc();
  assert(bufferDesc->sharded_buffer_config() == nullptr);
  assert(bufferDesc->shape()->size() > 0);

  TensorDesc desc;
  desc.shape = std::vector<std::uint32_t>(bufferDesc->shape()->begin(),
                                          bufferDesc->shape()->end());
  desc.stride = utils::calculateStride(desc.shape);
  desc.itemsize = utils::dataTypeElementSize(bufferDesc->data_type());
  desc.dataType = bufferDesc->data_type();

  size_t size = desc.shape[0] * desc.stride[0] * desc.itemsize;
  auto data = std::shared_ptr<void>(std::malloc(size), std::free);
  assert(data);

  std::shared_ptr<MetalTensor> tensor = std::make_shared<MetalTensor>(desc);
  auto [_, inserted] = hostBuffers.try_emplace(
      command->dst()->global_id(), static_pointer_cast<void>(tensor), data,
      DeviceRuntime::TTMetal);
  assert(inserted);
}

void CQExecutor::execute(::tt::target::metal::ReturnCommand const *command) {
  std::shared_ptr<::tt::tt_metal::Event> event =
      std::make_shared<::tt::tt_metal::Event>();
  ::tt::tt_metal::EnqueueRecordEvent(*cq, event);

  assert(outputs.empty() &&
         "Unexpected outputs, multiple returns not supported");
  outputs.reserve(command->results()->size());
  for (auto const *result : *command->results()) {
    auto deviceIter = deviceBuffers.find(result->global_id());
    auto hostIter = hostBuffers.find(result->global_id());
    bool deviceFound = deviceIter != deviceBuffers.end();
    bool hostFound = hostIter != hostBuffers.end();
    assert(deviceFound != hostFound);
    if (deviceFound) {
      outputs.emplace_back(static_pointer_cast<void>(deviceIter->second),
                           nullptr, static_pointer_cast<void>(event),
                           DeviceRuntime::TTMetal);
    } else {
      outputs.emplace_back(hostIter->second);
      outputs.back().event =
          Event(static_pointer_cast<void>(event), DeviceRuntime::TTMetal);
    }
  }
}

// Process various types of runtime args if present and call Metal APIs.
void CQExecutor::processRuntimeArgs(
    ::tt::tt_metal::Program &program, ::tt::tt_metal::KernelHandle &handle,
    CoreRangeSet const &coreRange,
    const ::flatbuffers::Vector<
        ::flatbuffers::Offset<::tt::target::metal::KernelArg>> *rtArgs,
    const ::flatbuffers::Vector<
        ::flatbuffers::Offset<tt::target::metal::BufferRef>> *buffers,
    const ::flatbuffers::Vector<::flatbuffers::Offset<tt::target::metal::CBRef>>
        *cbs) const {
  if (rtArgs == nullptr || rtArgs->size() == 0) {
    return;
  }
  std::vector<uint32_t> rtArgsVec;
  rtArgsVec.reserve(rtArgs->size());
  for (auto const* kernelArg : *rtArgs) {
    switch (kernelArg->arg_type()) {
    case ::tt::target::metal::KernelArgType::KernelArgCBPort: {
      auto const *arg = kernelArg->arg_as_KernelArgCBPort();
      LOG_ASSERT(arg->operand_idx() < cbs->size(), "invalid operand ",
                 arg->operand_idx());
      rtArgsVec.push_back(cbs->Get(arg->operand_idx())->port());
      break;
    }
    case ::tt::target::metal::KernelArgType::KernelArgBufferAddress: {
      auto const *arg = kernelArg->arg_as_KernelArgBufferAddress();
      const tt::target::metal::BufferRef* buffer = buffers->Get(arg->operand_idx());
      LOG_ASSERT(deviceBuffers.find(buffer->global_id()) != deviceBuffers.end(),
                 "Buffer id referenced by rt args is no longer alive or was "
                 "never created");
      rtArgsVec.push_back(validateDeviceAddress(
          buffer->address(), buffer->desc()->memory_space()));
      break;
    }
    case ::tt::target::metal::KernelArgType::KernelArgSemaphore: {
      const auto *arg = kernelArg->arg_as_KernelArgSemaphore();
      rtArgsVec.push_back(::tt::tt_metal::CreateSemaphore(
          program, coreRange, arg->initial_value(),
          toCoreType(arg->core_type())));
      break;
    }
    case ::tt::target::metal::KernelArgType::NONE:
      LOG_FATAL("Unsupported runtime arg type");
    }
  }

  ::tt::tt_metal::SetRuntimeArgs(program, handle, coreRange, rtArgsVec);
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueProgramCommand const *command,
    char const *debugInfo) {
  ZoneScopedN("EnqueueProgramCommand");
  ::tt::tt_metal::Program program = ::tt::tt_metal::CreateProgram();

  for (::tt::target::metal::KernelConfig const *kernelConfig :
       *command->program()->kernels()) {
    ::tt::target::metal::KernelSource const *kernelSource =
        kernelConfig->kernel_as_KernelSource();
    LOG_ASSERT(kernelSource, "Only source kernels supported for now");
    std::string kernelSourceString(kernelSource->source()->c_str(),
                                   kernelSource->source()->size());
    CoreRangeSet coreRangeSet =
        common::toCoreRangeSet(kernelConfig->core_range_set());

    ::tt::tt_metal::KernelHandle handle = createKernel(
        program, kernelSourceString, coreRangeSet,
        createKernelConfig(kernelConfig), currentProgramName, debugInfo);

    processRuntimeArgs(program, handle, coreRangeSet,
                       kernelConfig->args()->rt_args(), command->buffers(),
                       command->cbs());
  }

  for (::tt::target::metal::CBRef const *cbRef : *command->cbs()) {
    CoreRangeSet coreRangeSet =
        common::toCoreRangeSet(cbRef->buffer_ref()
                                   ->desc()
                                   ->circular_buffer_config()
                                   ->core_range_set());
    ::tt::tt_metal::CircularBufferConfig config =
        createCircularBufferConfig(cbRef, deviceBuffers);
    ::tt::tt_metal::CreateCircularBuffer(program, coreRangeSet, config);
  }

  ::tt::tt_metal::EnqueueProgram(*cq, program, /*blocking=*/false);
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueWriteBufferCommand const *command) {
  ZoneScopedN("EnqueueWriteBufferCommand");

  void *src = hostBuffers.at(command->src()->global_id()).data.get();
  assert(src);
  ::tt::tt_metal::EnqueueWriteBuffer(
      *cq, deviceBuffers.at(command->dst()->global_id()), src,
      /*blocking=*/false);
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueReadBufferCommand const *command) {
  ZoneScopedN("EnqueueReadBufferCommand");

  void *dst = hostBuffers.at(command->dst()->global_id()).data.get();
  assert(dst);
  ::tt::tt_metal::EnqueueReadBuffer(
      *cq, deviceBuffers.at(command->src()->global_id()), dst,
      /*blocking=*/false);
}

void CQExecutor::execute(
    ::tt::target::metal::CreateBufferCommand const *command) {
  ZoneScopedN("CreateBufferCommand");
  if (deviceBuffers.find(command->ref()->global_id()) == deviceBuffers.end()) {
    validateDeviceAddress(command->ref()->address(),
                          command->ref()->desc()->memory_space());
    deviceBuffers[command->ref()->global_id()] =
        createBufferFromBufferRef(device, command->ref());
  }
}

void CQExecutor::execute(
    ::tt::target::metal::DeallocateBufferCommand const *command) {
  ZoneScopedN("DeallocateBufferCommand");
  auto iter = deviceBuffers.find(command->ref()->global_id());
  LOG_ASSERT(iter != deviceBuffers.end(), "Buffer not allocated");
  LOG_ASSERT(iter->second != nullptr, "Buffer already deallocated");
  ::tt::tt_metal::DeallocateBuffer(*iter->second);
  deviceBuffers.erase(iter);
}

void CQExecutor::execute(
    ::tt::target::metal::CreateEventCommand const *command) {
  ZoneScopedN("CreateEventCommand");
  LOG_ASSERT(!events.contains(command->ref()->global_id()));
  events[command->ref()->global_id()] =
      std::make_shared<::tt::tt_metal::Event>();
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueRecordEventCommand const *command) {
  ZoneScopedN("EnqueueRecordEventCommand");
  auto event = events.at(command->ref()->global_id());
  ::tt::tt_metal::EnqueueRecordEvent(*cq, event);
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueWaitForEventCommand const *command) {
  ZoneScopedN("EnqueueWaitForEventCommand");
  auto event = events.at(command->ref()->global_id());
  ::tt::tt_metal::EnqueueWaitForEvent(*cq, event);
}

void CQExecutor::execute(
    ::tt::target::metal::EventSynchronizeCommand const *command) {
  ZoneScopedN("EventSynchronizeCommand");
  auto event = events.at(command->ref()->global_id());
  ::tt::tt_metal::EventSynchronize(event);
}

void CQExecutor::execute(
    ::tt::target::metal::EventQueryCommand const *command) {
  ZoneScopedN("EventQueryCommand");
  auto event = events.at(command->ref()->global_id());
  (void)::tt::tt_metal::EventQuery(
      event); // todo, we need flatbuffer support for tracking and doing
              // something with the result
}

void CQExecutor::execute(::tt::target::metal::FinishCommand const *) {
  ZoneScopedN("FinishCommand");
  ::tt::tt_metal::Finish(*cq);
}

std::vector<Tensor>
executeDeviceProgram(::tt::tt_metal::IDevice *device,
                     ::tt::target::metal::DeviceProgram const *program,
                     std::vector<Tensor> const &inputs) {
  assert(program->command_queues()->size() == 1 && "Only one CQ supported");

  CQExecutor executor(device, program->inputs(), inputs);
  for (::tt::target::metal::CommandQueue const *cq :
       *program->command_queues()) {
    FrameMark;
    ZoneScoped;
    std::string zoneName =
        "executeCommandQueue_cq_" + std::to_string(cq->queue_id());
    ZoneName(zoneName.c_str(), zoneName.size());

    executor.execute(cq);

    FrameMark;
  }

  return executor.getOutputs();
}
} // namespace tt::runtime::ttmetal
