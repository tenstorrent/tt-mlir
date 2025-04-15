// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <unordered_map>

#include "tracy/Tracy.hpp"
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
  ::tt::tt_metal::IDevice *device;
  std::vector<std::shared_ptr<::tt::tt_metal::Event>> initEvents;
  std::unordered_map<std::uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>>
      buffers;
  std::unordered_map<std::uint32_t, std::shared_ptr<::tt::tt_metal::Event>>
      events;
  ::tt::tt_metal::CommandQueue *cq;
  char const *currentProgramName;

  CQExecutor(::tt::tt_metal::IDevice *device, std::size_t cq_id,
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

CQExecutor::CQExecutor(::tt::tt_metal::IDevice *device, std::size_t cq_id,
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

static std::string kernelConfigTypeString(
    std::variant<::tt::tt_metal::DataMovementConfig,
                 ::tt::tt_metal::ComputeConfig,
                 ::tt::tt_metal::EthernetConfig> const &kernelConfig) {
  // return a string representation of the kernel config type
  if (auto const *dataMovementConfig =
          std::get_if<::tt::tt_metal::DataMovementConfig>(&kernelConfig)) {
    return "data_movement" +
           std::to_string(static_cast<std::underlying_type_t<
                              ::tt::tt_metal::DataMovementProcessor>>(
               dataMovementConfig->processor)) +
           "_noc" + std::to_string(dataMovementConfig->noc);
  } else if (auto const *computeConfig =
                 std::get_if<::tt::tt_metal::ComputeConfig>(&kernelConfig)) {
    return "compute";
  } else if (auto const *ethernetConfig =
                 std::get_if<::tt::tt_metal::EthernetConfig>(&kernelConfig)) {
    return "ethernet" +
           std::to_string(static_cast<std::underlying_type_t<
                              ::tt::tt_metal::DataMovementProcessor>>(
               ethernetConfig->processor)) +
           "_noc" + std::to_string(ethernetConfig->noc) + "_" +
           (ethernetConfig->eth_mode == ::tt::tt_metal::Eth::SENDER
                ? "sender"
                : "receiver");
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
    std::variant<::tt::tt_metal::DataMovementConfig,
                 ::tt::tt_metal::ComputeConfig,
                 ::tt::tt_metal::EthernetConfig> const &kernelConfig,
    char const *prefix = "/tmp/ttmlir_", char const *extention = ".cpp") {
  std::string path(prefix);
  path += currentProgramName;
  path += "_";
  path += parseLocFromDebugInfo(programDebugInfo);
  path += "_";
  path += kernelConfigTypeString(kernelConfig);

  // Double underscore to visually separate core ranges from the rest.
  path += "__";
  path += coreRangeToString(coreRangeSet);
  path += extention;
  return path;
}

static void writeFile(std::string const &fileName, std::string const &source) {
  if (debug::Env::get().loadKernelsFromDisk) {
    std::ifstream file(fileName);
    LOG_ASSERT(file.is_open(), "Kernel file ", fileName, " not found");
    return;
  }
  std::ofstream file(fileName);
  file.write(source.c_str(), source.size());
  file.close();
}

static ::tt::tt_metal::KernelHandle
createKernel(::tt::tt_metal::Program &program, std::string const &kernelSource,
             CoreRangeSet const &coreRangeSet,
             std::variant<::tt::tt_metal::DataMovementConfig,
                          ::tt::tt_metal::ComputeConfig,
                          ::tt::tt_metal::EthernetConfig> const &kernelConfig,
             char const *currentProgramName, char const *debugInfo) {
  bool const kernelFromFile = debug::Env::get().dumpKernelsToDisk ||
                              debug::Env::get().loadKernelsFromDisk;
  std::string fileName;
  if (kernelFromFile) {
    fileName = createKernelFilePath(currentProgramName, debugInfo, coreRangeSet,
                                    kernelConfig);
    writeFile(fileName, kernelSource);
  }
  return kernelFromFile
             ? CreateKernel(program, fileName, coreRangeSet, kernelConfig)
             : CreateKernelFromString(program, kernelSource, coreRangeSet,
                                      kernelConfig);
}

static std::vector<uint32_t> processCompileArgs(
    const ::flatbuffers::Vector<
        ::flatbuffers::Offset<::tt::target::metal::KernelArg>> *ctArgs) {
  std::vector<uint32_t> args;
  args.reserve(ctArgs->size());
  for (auto const *ctArg : *ctArgs) {
    args.push_back(ctArg->ct_value());
  }
  return args;
}

static std::variant<::tt::tt_metal::DataMovementConfig,
                    ::tt::tt_metal::ComputeConfig,
                    ::tt::tt_metal::EthernetConfig>
createKernelConfig(::tt::target::metal::KernelConfig const *kernelConfig) {
  std::vector<uint32_t> compileArgs =
      processCompileArgs(kernelConfig->args()->ct_args());
  switch (kernelConfig->type_type()) {
  case ::tt::target::metal::KernelConfigType::NocConfig: {
    switch (kernelConfig->type_as_NocConfig()->noc_index()) {
    case tt::target::metal::NocIndex::Noc0: {
      return ::tt::tt_metal::ReaderDataMovementConfig(compileArgs);
    }
    case tt::target::metal::NocIndex::Noc1: {
      return ::tt::tt_metal::WriterDataMovementConfig(compileArgs);
    }
    }
  }
  case ::tt::target::metal::KernelConfigType::EthernetConfig: {
    ::tt::tt_metal::EthernetConfig ethernetConfig;
    ethernetConfig.compile_args = compileArgs;
    switch (kernelConfig->type_as_EthernetConfig()->eth_type()) {
    case tt::target::metal::EthType::Sender: {
      ethernetConfig.eth_mode = ::tt::tt_metal::Eth::SENDER;
      break;
    }
    case tt::target::metal::EthType::Receiver: {
      ethernetConfig.eth_mode = ::tt::tt_metal::Eth::RECEIVER;
      break;
    }
    }

    switch (kernelConfig->type_as_EthernetConfig()->noc_index()) {
    case tt::target::metal::NocIndex::Noc0: {
      ethernetConfig.noc = ::tt::tt_metal::NOC::NOC_0;
      break;
    }
    case tt::target::metal::NocIndex::Noc1: {
      ethernetConfig.noc = ::tt::tt_metal::NOC::NOC_1;
      break;
    }
    }
    return ethernetConfig;
  }
  case ::tt::target::metal::KernelConfigType::ComputeConfig: {
    auto const *fbComputeConfig = kernelConfig->type_as_ComputeConfig();
    ::tt::tt_metal::ComputeConfig computeConfig;
    computeConfig.compile_args = compileArgs;
    switch (fbComputeConfig->math_fidelity()) {
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

    computeConfig.fp32_dest_acc_en = fbComputeConfig->fp32_dest_acc_en();
    computeConfig.math_approx_mode = fbComputeConfig->math_approx_mode();

    computeConfig.unpack_to_dest_mode.reserve(fbComputeConfig->unpack_to_dest_mode()->size());
    for (auto mode : *fbComputeConfig->unpack_to_dest_mode()) {
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

  case ::tt::target::metal::KernelConfigType::NONE: {
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
  case ::tt::target::DataType::Int32:
    return ::tt::DataFormat::Int32;
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
    ::tt::target::metal::CBRef const *cbRef,
    std::unordered_map<std::uint32_t,
                       std::shared_ptr<::tt::tt_metal::Buffer>> const
        &buffers) {
  auto const* bufferDesc = cbRef->buffer_ref()->desc();
  std::uint32_t totalSize = bufferDesc->size();
  ::tt::DataFormat dataFormat = toDataFormat(bufferDesc->data_type());
  assert(cbRef->buffer_ref());
  return ::tt::tt_metal::CircularBufferConfig(
             totalSize, {{cbRef->port(), dataFormat}},
             *buffers.at(cbRef->buffer_ref()->global_id()))
      .set_page_size(cbRef->port(), bufferDesc->page_size());
}

// Process various types of runtime args if present and call Metal APIs.
static void processRuntimeArgs(
    ::tt::tt_metal::Program &program, ::tt::tt_metal::KernelHandle &handle,
    CoreRangeSet const &coreRange,
    const ::flatbuffers::Vector<
        ::flatbuffers::Offset<::tt::target::metal::KernelArg>> *rtArgs,
    const ::flatbuffers::Vector<
        ::flatbuffers::Offset<tt::target::metal::BufferRef>> *buffers,
    const ::flatbuffers::Vector<::flatbuffers::Offset<tt::target::metal::CBRef>>
        *cbs) {
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
      LOG_ASSERT(arg->operand_idx() < buffers->size(), "invalid operand ",
                 arg->operand_idx());
      rtArgsVec.push_back(buffers->Get(arg->operand_idx())->address());
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
    CoreRangeSet coreRangeSet = toCoreRangeSet(kernelConfig->core_range_set());

    ::tt::tt_metal::KernelHandle handle = createKernel(
        program, kernelSourceString, coreRangeSet,
        createKernelConfig(kernelConfig), currentProgramName, debugInfo);

    for (::tt::target::metal::CBRef const *cbRef : *command->cbs()) {
      ::tt::tt_metal::CircularBufferConfig config =
          createCircularBufferConfig(cbRef, buffers);
      ::tt::tt_metal::CreateCircularBuffer(program, coreRangeSet, config);
    }

    processRuntimeArgs(program, handle, coreRangeSet,
                       kernelConfig->args()->rt_args(), command->buffers(),
                       command->cbs());
  }

  ::tt::tt_metal::EnqueueProgram(*cq, program, /*blocking=*/false);
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueWriteBufferCommand const *command) {
  ZoneScopedN("EnqueueWriteBufferCommand");
  LOG_FATAL("Unsupported EnqueueWriteBufferCommand");
}

void CQExecutor::execute(
    ::tt::target::metal::EnqueueReadBufferCommand const *command) {
  ZoneScopedN("EnqueueReadBufferCommand");
  LOG_FATAL("Unsupported EnqueueReadBufferCommand");
}

void CQExecutor::execute(
    ::tt::target::metal::CreateBufferCommand const *command) {
  ZoneScopedN("CreateBufferCommand");
  if (buffers.find(command->ref()->global_id()) == buffers.end()) {
    buffers[command->ref()->global_id()] =
        createBufferFromBufferRef(device, command->ref());
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

std::shared_ptr<::tt::tt_metal::Event>
executeCommandQueue(::tt::tt_metal::IDevice *device,
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
