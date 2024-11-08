// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <unordered_map>

#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/ttmetal.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"

#include "tt/runtime/detail/logger.h"
#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Version.h"
#include "types_generated.h"

namespace tt::runtime::ttmetal {

struct CQExecutor {
  ::tt::tt_metal::Device *device;
  std::vector<std::shared_ptr<::tt::tt_metal::Event>> initEvents;
  std::unordered_map<std::uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>>
      buffers;
  std::unordered_map<std::uint32_t, std::shared_ptr<::tt::tt_metal::Event>>
      events;
  ::tt::tt_metal::CommandQueue *cq;
  char const *currentProgramName;

  CQExecutor(::tt::tt_metal::Device *device, std::size_t cq_id,
             std::vector<InputBuffer> const &inputs,
             std::vector<OutputBuffer> const &outputs);

  std::shared_ptr<::tt::tt_metal::Event>
  execute(::tt::target::metal::CommandQueue const *commandQueue);
  void execute(::tt::target::metal::Command const *command);
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
  currentProgramName = commandQueue->name()->c_str();

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
  default:
    LOG_FATAL("Unsupported command type");
    break;
  }
}

static char const *
kernelSourceTypeString(::tt::target::metal::KernelSource const *kernelSource) {
  switch (kernelSource->config_type()) {
  case ::tt::target::metal::KernelConfig::NONE: {
    break;
  }
  case ::tt::target::metal::KernelConfig::NocConfig: {
    switch (kernelSource->config_as_NocConfig()->noc_index()) {
    case tt::target::metal::NocIndex::Noc0: {
      return "noc0";
    }
    case tt::target::metal::NocIndex::Noc1: {
      return "noc1";
    }
    }
  }
  case ::tt::target::metal::KernelConfig::EthernetConfig: {
    switch (kernelSource->config_as_EthernetConfig()->eth_type()) {
    case tt::target::metal::EthType::Sender: {
      return "ethSender";
    }
    case tt::target::metal::EthType::Receiver: {
      return "ethReceiver";
    }
    }
  }
  case ::tt::target::metal::KernelConfig::TensixConfig: {
    return "tensix";
  }
  }
  return "unknown";
}

static std::string parseLocFromDebugInfo(char const *programDebugInfo) {
  if (!programDebugInfo) {
    static int gUnknownId = 0;
    return std::string("%unknown") + std::to_string(gUnknownId++);
  }
  std::string debugInfo(programDebugInfo);
  std::size_t pos = debugInfo.find_first_of(' ');
  if (pos == std::string::npos) {
    return debugInfo;
  }
  return debugInfo.substr(0, pos);
}

// Produces string representation of CoreRangeSet that is suitable for embedding
// in file name. Encode core range set so that ranges are separated by
// double underscore '__'. Range is represented with start and end coordinates
// as "startY_startX-endY_endX".
static std::string coreRangeToString(const CoreRangeSet &coreRanges) {
  std::string result;
  for (const auto &coreRange : coreRanges.ranges()) {
    result += std::to_string(coreRange.start_coord.y) + "_" +
              std::to_string(coreRange.start_coord.x) + "-" +
              std::to_string(coreRange.end_coord.y) + "_" +
              std::to_string(coreRange.end_coord.x);
    result += "__";
  }
  result.pop_back();
  result.pop_back();

  return result;
}

static std::string createKernelFilePath(
    char const *currentProgramName, char const *programDebugInfo,
    const CoreRangeSet &coreRangeSet,
    ::tt::target::metal::KernelSource const *kernelSource,
    char const *prefix = "/tmp/ttmlir_", char const *extention = ".cpp") {
  std::string path(prefix);
  path += currentProgramName;
  path += "_";
  path += parseLocFromDebugInfo(programDebugInfo);
  path += "_";
  path += kernelSourceTypeString(kernelSource);

  // Double underscore to visually separate core ranges from the rest.
  path += "__";
  path += coreRangeToString(coreRangeSet);
  path += extention;
  return path;
}

static void writeFile(std::string const &fileName, char const *data,
                      std::size_t size) {
  if (debug::Env::get().loadKernelsFromDisk) {
    std::ifstream file(fileName);
    LOG_ASSERT(file.is_open(), "Kernel file ", fileName, " not found");
    return;
  }
  std::ofstream file(fileName);
  file.write(data, size);
  file.close();
}

static std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>
createKernelConfig(::tt::target::metal::KernelSource const *kernelSource) {
  switch (kernelSource->config_type()) {
  case ::tt::target::metal::KernelConfig::NocConfig: {
    switch (kernelSource->config_as_NocConfig()->noc_index()) {
    case tt::target::metal::NocIndex::Noc0: {
      return ::tt::tt_metal::ReaderDataMovementConfig();
    }
    case tt::target::metal::NocIndex::Noc1: {
      return ::tt::tt_metal::WriterDataMovementConfig();
    }
    }
  }
  case ::tt::target::metal::KernelConfig::EthernetConfig: {
    ::tt::tt_metal::EthernetConfig ethernetConfig;
    switch (kernelSource->config_as_EthernetConfig()->eth_type()) {
    case tt::target::metal::EthType::Sender: {
      ethernetConfig.eth_mode = Eth::SENDER;
      break;
    }
    case tt::target::metal::EthType::Receiver: {
      ethernetConfig.eth_mode = Eth::RECEIVER;
      break;
    }
    }

    switch (kernelSource->config_as_EthernetConfig()->noc_index()) {
    case tt::target::metal::NocIndex::Noc0: {
      ethernetConfig.noc = NOC::NOC_0;
      break;
    }
    case tt::target::metal::NocIndex::Noc1: {
      ethernetConfig.noc = NOC::NOC_1;
      break;
    }
    }
    return ethernetConfig;
  }

  case ::tt::target::metal::KernelConfig::TensixConfig: {
    ::tt::tt_metal::ComputeConfig computeConfig;
    switch (kernelSource->config_as_TensixConfig()->math_fidelity()) {
    case tt::target::MathFidelity::HiFi4: {
      computeConfig.math_fidelity = MathFidelity::HiFi4;
      break;
    }
    case tt::target::MathFidelity::HiFi3: {
      computeConfig.math_fidelity = MathFidelity::HiFi3;
      break;
    }
    case tt::target::MathFidelity::HiFi2: {
      computeConfig.math_fidelity = MathFidelity::HiFi2;
      break;
    }
    case tt::target::MathFidelity::LoFi: {
      computeConfig.math_fidelity = MathFidelity::LoFi;
      break;
    }
    }

    computeConfig.fp32_dest_acc_en =
        kernelSource->config_as_TensixConfig()->fp32_dest_acc_en();
    computeConfig.math_approx_mode =
        kernelSource->config_as_TensixConfig()->math_approx_mode();

    auto const *unpackToDestModeVec =
        kernelSource->config_as_TensixConfig()->unpack_to_dest_mode();
    computeConfig.unpack_to_dest_mode.reserve(unpackToDestModeVec->size());

    for (auto mode : *unpackToDestModeVec) {
      switch (mode) {
      case tt::target::metal::UnpackToDestMode::UnpackToDestFp32: {
        computeConfig.unpack_to_dest_mode.push_back(
            UnpackToDestMode::UnpackToDestFp32);
        break;
      }
      case tt::target::metal::UnpackToDestMode::Default: {
        computeConfig.unpack_to_dest_mode.push_back(UnpackToDestMode::Default);
        break;
      }
      }
    }

    return computeConfig;
  }

  case ::tt::target::metal::KernelConfig::NONE: {
    break;
  }
  }
  LOG_FATAL("Unsupported kernel source type");
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
    LOG_FATAL("Unsupported data type");
  }
}

// Convert from Flatbuffer CoreType to soc_descriptor CoreType.
static CoreType toCoreType(::tt::target::metal::CoreType coreType) {
  switch (coreType) {
  case ::tt::target::metal::CoreType::WORKER:
    return CoreType::WORKER;
  case ::tt::target::metal::CoreType::ETH:
    return CoreType::ETH;
  }
  LOG_FATAL("Unsupported core type");
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
  LOG_ASSERT(cbRef->tensor_ref());
  LOG_ASSERT(cbRef->tensor_ref()->address() == cbRef->address(),
             "Address mismatch between tensor ref and cb ref");
  return CircularBufferConfig(totalSize, {{cbRef->desc()->port(), dataFormat}},
                              *buffers.at(cbRef->tensor_ref()->global_id()))
      .set_page_size(cbRef->desc()->port(), cbRef->desc()->page_size());
}

// Process various types of runtime args if present and call Metal APIs.
static void processRuntimeArgs(
    ::tt::tt_metal::Program &program,
    ::tt::target::metal::KernelDesc const *kernelDesc,
    ::tt::tt_metal::KernelHandle &handle, CoreRangeSet &coreRange,
    const ::flatbuffers::Vector<::flatbuffers::Offset<tt::target::TensorRef>>
        *operands,
    std::unordered_map<std::uint32_t,
                       std::shared_ptr<::tt::tt_metal::Buffer>> const
        &buffers) {

  using SemaphoreAddr = ::tt::target::metal::RuntimeArgSemaphoreAddress;
  using TensorAddr = ::tt::target::metal::RuntimeArgTensorAddress;

  const auto *rt_args_types = kernelDesc->runtime_args_type();
  const auto *rt_args = kernelDesc->runtime_args();

  if (rt_args == nullptr || rt_args_types == nullptr || rt_args->size() == 0 ||
      rt_args_types->size() == 0) {
    return;
  }

  LOG_ASSERT(rt_args_types->size() == rt_args->size());
  std::vector<uint32_t> rt_args_vec;

  for (size_t i = 0; i < rt_args->size(); i++) {
    switch (rt_args_types->Get(i)) {
    case ::tt::target::metal::RuntimeArg::RuntimeArgTensorAddress: {
      const auto *rt_arg = static_cast<const TensorAddr *>(rt_args->Get(i));
      LOG_ASSERT(rt_arg->operand_idx() < operands->size(), "invalid operand ",
                 rt_arg->operand_idx());
      uint32_t global_id = operands->Get(rt_arg->operand_idx())->global_id();
      uint32_t addr = buffers.at(global_id)->address();
      rt_args_vec.push_back(addr);
      break;
    }
    case ::tt::target::metal::RuntimeArg::RuntimeArgSemaphoreAddress: {
      const auto *rt_arg = static_cast<const SemaphoreAddr *>(rt_args->Get(i));
      auto addr = ::tt::tt_metal::CreateSemaphore(
          program, coreRange, rt_arg->initial_value(),
          toCoreType(rt_arg->core_type()));
      rt_args_vec.push_back(addr);
      break;
    }
    case ::tt::target::metal::RuntimeArg::NONE:
      LOG_FATAL("Unsupported runtime arg type");
    }
  }

  ::tt::tt_metal::SetRuntimeArgs(program, handle, coreRange, rt_args_vec);
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueProgramCommand const *command,
    char const *debugInfo) {

  ZoneScopedN("EnqueueProgramCommand");
  ::tt::tt_metal::Program program = ::tt::tt_metal::CreateProgram();

  std::unordered_set<uint32_t> createdCBs;
  for (::tt::target::metal::KernelDesc const *kernelDesc :
       *command->program()->kernels()) {
    ::tt::target::metal::KernelSource const *kernelSource =
        kernelDesc->kernel_as_KernelSource();
    LOG_ASSERT(kernelSource, "Only source kernels supported for now");
    CoreRangeSet coreRangeSet = toCoreRangeSet(kernelDesc->core_range_set());
    // We need a new API to create a kernel from source string, or directly from
    // binary
    std::string fileName = createKernelFilePath(currentProgramName, debugInfo,
                                                coreRangeSet, kernelSource);
    writeFile(fileName, kernelSource->source()->c_str(),
              kernelSource->source()->size());
    std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> config =
        createKernelConfig(kernelSource);

    ::tt::tt_metal::KernelHandle handle =
        ::tt::tt_metal::CreateKernel(program, fileName, coreRangeSet, config);

    for (::tt::target::CBRef const *cbRef : *kernelDesc->cbs()) {
      if (createdCBs.count(cbRef->desc()->port())) {
        // Since kernels may share the same CB, we only need to create it once.
        continue;
      }
      ::tt::tt_metal::CircularBufferConfig config =
          createCircularBufferConfig(cbRef, buffers);
      ::tt::tt_metal::CreateCircularBuffer(program, coreRangeSet, config);
      createdCBs.insert(cbRef->desc()->port());
    }

    // Process Kernel's runtime args based on variant and call metal APIs.
    processRuntimeArgs(program, kernelDesc, handle, coreRangeSet,
                       command->operands(), buffers);
  }

  constexpr bool blocking = false;
  ::tt::tt_metal::EnqueueProgram(*cq, program, blocking);
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueWriteBufferCommand const *command) {
  ZoneScopedN("EnqueueWriteBufferCommand");
  LOG_ASSERT(buffers.contains(command->dst()->global_id()),
             "Buffer not allocated");
  auto buffer = buffers[command->dst()->global_id()];
  constexpr bool blocking = false;
  switch (command->src_type()) { // currently only supporting ConstantBuffer32
  case tt::target::metal::HostBuffer::ConstantBuffer32: {
    const auto *src = command->src_as_ConstantBuffer32();
    LOG_ASSERT(src->data() != nullptr && (*src->data()).size() == 1,
               "Only scalar constant supported");
    LOG_ASSERT(command->dst()->size() ==
                   buffers[command->dst()->global_id()]->size(),
               "Size mismatch: ", command->dst()->size(),
               " != ", buffers[command->dst()->global_id()]->size());
    int shapeAccumulate = std::accumulate(
        (*command->dst()->desc()->shape()).begin(),
        (*command->dst()->desc()->shape()).end(), 1, std::multiplies<int>());
    std::vector<uint32_t> vec(shapeAccumulate, (*src->data())[0]);
    ::tt::tt_metal::EnqueueWriteBuffer(*cq, buffer, vec, blocking);
    break;
  }
  default:
    LOG_FATAL("Unsupported HostBuffer type");
  }
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueReadBufferCommand const *command) {
  ZoneScopedN("EnqueueReadBufferCommand");
  // Maybe we will need this in the future, like paging to system mem?
  LOG_FATAL("Unsupported EnqueueReadBufferCommand");
}

void CQExecutor::execute(
    ::tt::target::metal::CreateBufferCommand const *command) {
  ZoneScopedN("CreateBufferCommand");
  if (buffers.find(command->ref()->global_id()) == buffers.end()) {
    buffers[command->ref()->global_id()] =
        createBufferFromTensorRef(device, command->ref());
  }
}

void CQExecutor::execute(
    ::tt::target::metal::DeallocateBufferCommand const *command) {
  ZoneScopedN("DeallocateBufferCommand");
  auto iter = buffers.find(command->ref()->global_id());
  LOG_ASSERT(iter != buffers.end(), "Buffer not allocated");
  LOG_ASSERT(iter->second != nullptr, "Buffer already deallocated");
  ::tt::tt_metal::DeallocateBuffer(*iter->second);
  buffers.erase(iter);
}

void CQExecutor::execute(
    ::tt::target::metal::CreateEventCommand const *command) {
  ZoneScopedN("CreateEventCommand");
  LOG_ASSERT(not events.contains(command->ref()->global_id()));
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

std::shared_ptr<::tt::tt_metal::Event>
executeCommandQueue(::tt::tt_metal::Device *device,
                    ::tt::target::metal::CommandQueue const *commandQueue,
                    std::size_t cq_id, std::vector<InputBuffer> const &inputs,
                    std::vector<OutputBuffer> const &outputs) {

  ZoneScoped;
  std::string zoneName = "executeCommandQueue_cq_" + std::to_string(cq_id);
  ZoneName(zoneName.c_str(), zoneName.size());

  CQExecutor executor(device, cq_id, inputs, outputs);
  return executor.execute(commandQueue);
}
} // namespace tt::runtime::ttmetal
