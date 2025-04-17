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

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;

namespace {
class CQExecutor {
public:
  CQExecutor(
      tt_metal::IDevice *device,
      const flatbuffers::Vector<
          flatbuffers::Offset<tt::target::metal::BufferRef>> *programInputs,
      std::vector<Tensor> const &inputs, bool blockingCQ);

  std::vector<Tensor> const &getOutputs() const { return outputs; }

  void execute(target::metal::CommandQueue const *commandQueue);

private:
  void execute(target::metal::Command const *command);
  void execute(target::metal::HostAllocCommand const *command);
  void execute(target::metal::ReturnCommand const *command);
  void execute(target::metal::EnqueueProgramCommand const *command,
               char const *debugInfo);
  void execute(target::metal::EnqueueWriteBufferCommand const *command);
  void execute(target::metal::EnqueueReadBufferCommand const *command);
  void execute(target::metal::CreateBufferCommand const *command);
  void execute(target::metal::DeallocateBufferCommand const *command);
  void execute(target::metal::CreateEventCommand const *command);
  void execute(target::metal::EnqueueRecordEventCommand const *command);
  void execute(target::metal::EnqueueWaitForEventCommand const *command);
  void execute(target::metal::EventSynchronizeCommand const *command);
  void execute(target::metal::EventQueryCommand const *command);
  void execute(target::metal::FinishCommand const *command);

private:
  tt_metal::IDevice *device;
  std::vector<std::shared_ptr<tt_metal::Event>> initEvents;
  std::unordered_map<std::uint32_t, DeviceBuffer> deviceBuffers;
  std::unordered_map<std::uint32_t, Tensor> hostBuffers;
  std::unordered_map<std::uint32_t, std::shared_ptr<tt_metal::Event>> events;
  std::vector<Tensor> outputs;
  tt_metal::CommandQueue *cq;
  bool blockingCQ;
  char const *currentProgramName;
  DeviceAddressValidator deviceAddressValidator;
};
} // namespace

CQExecutor::CQExecutor(
    tt_metal::IDevice *device,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::metal::BufferRef>>
        *programInputs,
    std::vector<Tensor> const &inputs, bool blockingCQ)
    : device(device), blockingCQ(blockingCQ), deviceAddressValidator(device) {
  initEvents.reserve(inputs.size());

  std::uint32_t inputIndex = 0;
  for (Tensor const &input : inputs) {
    const target::metal::BufferRef *ref = programInputs->Get(inputIndex++);

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
        input.event.asSharedPtr<tt_metal::Event>(DeviceRuntime::TTMetal);
    if (event) {
      initEvents.push_back(event);
    }
  }
}

void CQExecutor::execute(target::metal::CommandQueue const *commandQueue) {
  currentProgramName = commandQueue->name()->c_str();
  cq = &device->command_queue(commandQueue->queue_id());

  for (auto const &event : initEvents) {
    tt_metal::EnqueueWaitForEvent(*cq, event);
  }
  initEvents.clear();

  for (target::metal::Command const *command : *commandQueue->commands()) {
    LOG_TRACE(logger::LogRuntimeTTMetalCommand,
              "Executing command: ", EnumNameCommandType(command->type_type()),
              "\n\t", command->debug_info()->c_str());
    execute(command);
  }
}

void CQExecutor::execute(target::metal::Command const *command) {
  switch (command->type_type()) {
  case target::metal::CommandType::HostAllocCommand: {
    execute(command->type_as_HostAllocCommand());
    break;
  }
  case target::metal::CommandType::ReturnCommand: {
    execute(command->type_as_ReturnCommand());
    break;
  }
  case target::metal::CommandType::EnqueueProgramCommand: {
    execute(command->type_as_EnqueueProgramCommand(),
            command->debug_info()->c_str());
    break;
  }
  case target::metal::CommandType::EnqueueWriteBufferCommand: {
    execute(command->type_as_EnqueueWriteBufferCommand());
    break;
  }
  case target::metal::CommandType::EnqueueReadBufferCommand: {
    execute(command->type_as_EnqueueReadBufferCommand());
    break;
  }
  case target::metal::CommandType::CreateBufferCommand: {
    execute(command->type_as_CreateBufferCommand());
    break;
  }
  case target::metal::CommandType::DeallocateBufferCommand: {
    execute(command->type_as_DeallocateBufferCommand());
    break;
  }
  case target::metal::CommandType::CreateEventCommand: {
    execute(command->type_as_CreateEventCommand());
    break;
  }
  case target::metal::CommandType::EnqueueRecordEventCommand: {
    execute(command->type_as_EnqueueRecordEventCommand());
    break;
  }
  case target::metal::CommandType::EnqueueWaitForEventCommand: {
    execute(command->type_as_EnqueueWaitForEventCommand());
    break;
  }
  case target::metal::CommandType::EventSynchronizeCommand: {
    execute(command->type_as_EventSynchronizeCommand());
    break;
  }
  case target::metal::CommandType::EventQueryCommand: {
    execute(command->type_as_EventQueryCommand());
    break;
  }
  case target::metal::CommandType::FinishCommand: {
    execute(command->type_as_FinishCommand());
    break;
  }
  case target::metal::CommandType::NONE: {
    LOG_FATAL("Unsupported CommandType::NONE");
    break;
  }
  }
}

void CQExecutor::execute(target::metal::HostAllocCommand const *command) {
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

void CQExecutor::execute(target::metal::ReturnCommand const *command) {
  std::shared_ptr<tt_metal::Event> event = std::make_shared<tt_metal::Event>();
  tt_metal::EnqueueRecordEvent(*cq, event);

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

void CQExecutor::execute(target::metal::EnqueueProgramCommand const *command,
                         char const *debugInfo) {
  ZoneScopedN("EnqueueProgramCommand");
  tt_metal::Program program = tt_metal::CreateProgram();

  for (target::metal::KernelConfig const *kernelConfig :
       *command->program()->kernels()) {
    target::metal::KernelSource const *kernelSource =
        kernelConfig->kernel_as_KernelSource();
    LOG_ASSERT(kernelSource, "Only source kernels supported for now");
    std::string kernelSourceString(kernelSource->source()->c_str(),
                                   kernelSource->source()->size());

    CoreRangeSet coreRangeSet =
        common::toCoreRangeSet(kernelConfig->core_range_set());

    tt_metal::KernelHandle handle = createKernel(
        program, kernelSourceString, coreRangeSet,
        createKernelConfig(kernelConfig, command->buffers(), deviceBuffers,
                           command->cbs(), deviceAddressValidator),
        currentProgramName, debugInfo, kernelConfig->debug_info()->c_str());

    auto createSemaphore = [&](std::uint32_t initialValue,
                               CoreType coreType) -> std::uint32_t {
      return tt_metal::CreateSemaphore(program, coreRangeSet, initialValue,
                                       coreType);
    };
    std::vector<uint32_t> rtArgsVec = processRuntimeArgs(
        kernelConfig->args()->rt_args(), command->buffers(), deviceBuffers,
        command->cbs(), deviceAddressValidator, createSemaphore);
    tt_metal::SetRuntimeArgs(program, handle, coreRangeSet, rtArgsVec);
  }

  for (target::metal::CBRef const *cbRef : *command->cbs()) {
    CoreRangeSet coreRangeSet =
        common::toCoreRangeSet(cbRef->buffer_ref()
                                   ->desc()
                                   ->circular_buffer_config()
                                   ->core_range_set());
    tt_metal::CircularBufferConfig config =
        createCircularBufferConfig(cbRef, deviceBuffers);
    tt_metal::CreateCircularBuffer(program, coreRangeSet, config);
  }

  tt_metal::EnqueueProgram(*cq, program, blockingCQ);
}

void CQExecutor::execute(
    target::metal::EnqueueWriteBufferCommand const *command) {
  ZoneScopedN("EnqueueWriteBufferCommand");

  void *src = hostBuffers.at(command->src()->global_id()).data.get();
  assert(src);
  tt_metal::EnqueueWriteBuffer(
      *cq, deviceBuffers.at(command->dst()->global_id()), src, blockingCQ);
}

void CQExecutor::execute(
    target::metal::EnqueueReadBufferCommand const *command) {
  ZoneScopedN("EnqueueReadBufferCommand");

  void *dst = hostBuffers.at(command->dst()->global_id()).data.get();
  assert(dst);
  tt_metal::EnqueueReadBuffer(
      *cq, deviceBuffers.at(command->src()->global_id()), dst, blockingCQ);
}

void CQExecutor::execute(target::metal::CreateBufferCommand const *command) {
  ZoneScopedN("CreateBufferCommand");
  if (deviceBuffers.find(command->ref()->global_id()) == deviceBuffers.end()) {
    deviceBuffers[command->ref()->global_id()] = createBufferFromBufferRef(
        device, command->ref(), deviceAddressValidator);
  }
}

void CQExecutor::execute(
    target::metal::DeallocateBufferCommand const *command) {
  ZoneScopedN("DeallocateBufferCommand");
  auto iter = deviceBuffers.find(command->ref()->global_id());
  LOG_ASSERT(iter != deviceBuffers.end(), "Buffer not allocated");
  LOG_ASSERT(iter->second != nullptr, "Buffer already deallocated");
  tt_metal::DeallocateBuffer(*iter->second);
  deviceBuffers.erase(iter);
}

void CQExecutor::execute(target::metal::CreateEventCommand const *command) {
  ZoneScopedN("CreateEventCommand");
  LOG_ASSERT(!events.contains(command->ref()->global_id()));
  events[command->ref()->global_id()] = std::make_shared<tt_metal::Event>();
}

void CQExecutor::execute(
    target::metal::EnqueueRecordEventCommand const *command) {
  ZoneScopedN("EnqueueRecordEventCommand");
  auto event = events.at(command->ref()->global_id());
  tt_metal::EnqueueRecordEvent(*cq, event);
}

void CQExecutor::execute(
    target::metal::EnqueueWaitForEventCommand const *command) {
  ZoneScopedN("EnqueueWaitForEventCommand");
  auto event = events.at(command->ref()->global_id());
  tt_metal::EnqueueWaitForEvent(*cq, event);
}

void CQExecutor::execute(
    target::metal::EventSynchronizeCommand const *command) {
  ZoneScopedN("EventSynchronizeCommand");
  auto event = events.at(command->ref()->global_id());
  tt_metal::EventSynchronize(event);
}

void CQExecutor::execute(target::metal::EventQueryCommand const *command) {
  ZoneScopedN("EventQueryCommand");
  auto event = events.at(command->ref()->global_id());
  (void)tt_metal::EventQuery(
      event); // todo, we need flatbuffer support for tracking and doing
              // something with the result
}

void CQExecutor::execute(target::metal::FinishCommand const *) {
  ZoneScopedN("FinishCommand");
  tt_metal::Finish(*cq);
}

std::vector<Tensor>
executeDeviceProgram(tt_metal::IDevice *device,
                     target::metal::DeviceProgram const *program,
                     std::vector<Tensor> const &inputs) {
  assert(program->command_queues()->size() == 1 && "Only one CQ supported");

  CQExecutor executor(device, program->inputs(), inputs,
                      debug::Env::get().blockingCQ);
  for (target::metal::CommandQueue const *cq : *program->command_queues()) {
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
