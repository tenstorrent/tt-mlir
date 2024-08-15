// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_map>

#include "tt/runtime/detail/ttmetal.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"

#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Version.h"

namespace tt::runtime::ttmetal {

struct CQExecutor {
  ::tt::tt_metal::Device *device;
  std::vector<std::shared_ptr<::tt::tt_metal::Event>> initEvents;
  std::unordered_map<std::uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>>
      buffers;
  std::unordered_map<std::uint32_t, std::shared_ptr<::tt::tt_metal::Event>>
      events;
  ::tt::tt_metal::CommandQueue *cq;

  CQExecutor(::tt::tt_metal::Device *device, std::size_t cq_id,
             std::vector<InputBuffer> const &inputs,
             std::vector<OutputBuffer> const &outputs);

  std::shared_ptr<::tt::tt_metal::Event>
  execute(::tt::target::metal::CommandQueue const *commandQueue);
  void execute(::tt::target::metal::Command const *command);
  void execute(::tt::target::metal::EnqueueProgramCommand const *command);
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
};

CQExecutor::CQExecutor(::tt::tt_metal::Device *device, std::size_t cq_id,
                       std::vector<InputBuffer> const &inputs,
                       std::vector<OutputBuffer> const &outputs)
    : device(device) {
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto [global_id, buffer, event] = inputs[i];
    buffers[global_id] = buffer;
    if (event) {
      initEvents.push_back(event);
    }
  }

  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto [global_id, buffer] = outputs[i];
    buffers[global_id] = buffer;
  }

  cq = &device->command_queue(cq_id);
}

std::shared_ptr<::tt::tt_metal::Event>
CQExecutor::execute(::tt::target::metal::CommandQueue const *commandQueue) {
  for (auto const &event : initEvents) {
    ::tt::tt_metal::EnqueueWaitForEvent(*cq, event);
  }
  initEvents.clear();

  for (::tt::target::metal::Command const *command :
       *commandQueue->commands()) {
    execute(command);
  }

  std::shared_ptr<::tt::tt_metal::Event> event =
      std::make_shared<::tt::tt_metal::Event>();
  ::tt::tt_metal::EnqueueRecordEvent(*cq, event);
  return event;
}

void CQExecutor::execute(::tt::target::metal::Command const *command) {
  switch (command->type_type()) {
  case ::tt::target::metal::CommandType::EnqueueProgramCommand: {
    execute(command->type_as_EnqueueProgramCommand());
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
  default:
    throw std::runtime_error("Unsupported command type");
    break;
  }
}

static void writeFile(std::string const &fileName, char const *data,
                      std::size_t size) {
  std::ofstream file(fileName);
  file.write(data, size);
  file.close();
}

static std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>
createKernelConfig(::tt::target::metal::KernelSource const *kernelSource) {
  switch (kernelSource->source_type()) {
  case ::tt::target::metal::SourceType::Noc0: {
    return ::tt::tt_metal::ReaderDataMovementConfig();
  }
  case ::tt::target::metal::SourceType::Noc1: {
    return ::tt::tt_metal::WriterDataMovementConfig();
  }
  case ::tt::target::metal::SourceType::Tensix: {
    return ::tt::tt_metal::ComputeConfig();
  }
  default:
    break;
  }
  throw std::runtime_error("Unsupported kernel source type");
}

static ::tt::DataFormat toDataFormat(::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return ::tt::DataFormat::Float32;
  case ::tt::target::DataType::Float16:
    return ::tt::DataFormat::Float16;
  case ::tt::target::DataType::BFloat16:
    return ::tt::DataFormat::Float16_b;
  case ::tt::target::DataType::UInt32:
    return ::tt::DataFormat::UInt32;
  case ::tt::target::DataType::UInt16:
    return ::tt::DataFormat::UInt16;
  case ::tt::target::DataType::UInt8:
    return ::tt::DataFormat::UInt8;
  default:
    throw std::runtime_error("Unsupported data type");
  }
}

static ::tt::tt_metal::CircularBufferConfig createCircularBufferConfig(
    ::tt::target::CBRef const *cbRef,
    std::unordered_map<std::uint32_t,
                       std::shared_ptr<::tt::tt_metal::Buffer>> const
        &buffers) {
  std::uint32_t totalSize =
      cbRef->desc()->memory_desc()->size() * cbRef->desc()->num_buffers();
  ::tt::DataFormat dataFormat =
      toDataFormat(cbRef->desc()->memory_desc()->data_type());
  assert(cbRef->tensor_ref());
  assert(cbRef->tensor_ref()->address() == cbRef->address());
  return CircularBufferConfig(totalSize, {{cbRef->desc()->port(), dataFormat}},
                              *buffers.at(cbRef->tensor_ref()->global_id()))
      .set_page_size(cbRef->desc()->port(), cbRef->desc()->page_size());
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueProgramCommand const *command) {
  static int gKernelId = 0;

  ::tt::tt_metal::Program program = ::tt::tt_metal::CreateProgram();

  for (::tt::target::metal::KernelDesc const *kernelDesc :
       *command->program()->kernels()) {
    ::tt::target::metal::KernelSource const *kernelSource =
        kernelDesc->kernel_as_KernelSource();
    assert(kernelSource && "Only source kernels supported for now");
    // We need a new API to create a kernel from source string, or directly from
    // binary
    std::string fileName =
        "/tmp/ttmlir_" + std::to_string(gKernelId++) + ".cpp";
    writeFile(fileName, kernelSource->source()->c_str(),
              kernelSource->source()->size());
    CoreRangeSet coreRange = toCoreRangeSet(kernelDesc->core_range_set());
    std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> config =
        createKernelConfig(kernelSource);
    ::tt::tt_metal::KernelHandle handle =
        ::tt::tt_metal::CreateKernel(program, fileName, coreRange, config);
    (void)handle; // only needed for runtime args, which aren't supported yet

    for (::tt::target::CBRef const *cbRef : *kernelDesc->cbs()) {
      ::tt::tt_metal::CircularBufferConfig config =
          createCircularBufferConfig(cbRef, buffers);
      ::tt::tt_metal::CreateCircularBuffer(program, coreRange, config);
    }
  }

  constexpr bool blocking = false;
  ::tt::tt_metal::EnqueueProgram(*cq, program, blocking);
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueWriteBufferCommand const *command) {
  assert(command->src()->desc()->constant_data() != nullptr &&
         "Only constant data supported");
  throw std::runtime_error("Unsupported EnqueueWriteBufferCommand");
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueReadBufferCommand const *command) {
  // Maybe we will need this in the future, like paging to system mem?
  throw std::runtime_error("Unsupported EnqueueReadBufferCommand");
}

void CQExecutor::execute(
    ::tt::target::metal::CreateBufferCommand const *command) {
  buffers[command->ref()->global_id()] =
      createBufferFromTensorRef(device, command->ref());
}

void CQExecutor::execute(
    ::tt::target::metal::DeallocateBufferCommand const *command) {
  auto iter = buffers.find(command->ref()->global_id());
  assert(iter != buffers.end() && "Buffer not allocated");
  assert(iter->second != nullptr && "Buffer already deallocated");
  ::tt::tt_metal::DeallocateBuffer(*iter->second);
  iter->second.reset();
}

void CQExecutor::execute(
    ::tt::target::metal::CreateEventCommand const *command) {
  assert(events.find(command->ref()->global_id()) == events.end());
  events[command->ref()->global_id()] =
      std::make_shared<::tt::tt_metal::Event>();
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueRecordEventCommand const *command) {
  auto event = events.at(command->ref()->global_id());
  ::tt::tt_metal::EnqueueRecordEvent(*cq, event);
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueWaitForEventCommand const *command) {
  auto event = events.at(command->ref()->global_id());
  ::tt::tt_metal::EnqueueWaitForEvent(*cq, event);
}

void CQExecutor::execute(
    ::tt::target::metal::EventSynchronizeCommand const *command) {
  auto event = events.at(command->ref()->global_id());
  ::tt::tt_metal::EventSynchronize(event);
}

void CQExecutor::execute(
    ::tt::target::metal::EventQueryCommand const *command) {
  auto event = events.at(command->ref()->global_id());
  (void)::tt::tt_metal::EventQuery(
      event); // todo, we need flatbuffer support for tracking and doing
              // something with the result
}

void CQExecutor::execute(::tt::target::metal::FinishCommand const *) {
  ::tt::tt_metal::Finish(*cq);
}

std::shared_ptr<::tt::tt_metal::Event>
executeCommandQueue(::tt::tt_metal::Device *device,
                    ::tt::target::metal::CommandQueue const *commandQueue,
                    std::size_t cq_id, std::vector<InputBuffer> const &inputs,
                    std::vector<OutputBuffer> const &outputs) {
  CQExecutor executor(device, cq_id, inputs, outputs);
  return executor.execute(commandQueue);
}
} // namespace tt::runtime::ttmetal
