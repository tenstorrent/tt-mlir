// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/worker/command_executor.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/distributed/worker/response_factory.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include <thread>

namespace tt::runtime::distributed::worker {

static const ::tt::runtime::distributed::flatbuffer::Command *
getCommand(const SizedBuffer &command) {
  bool isDistributedCommand =
      ::tt::runtime::distributed::flatbuffer::CommandBufferHasIdentifier(
          command.data());
  LOG_ASSERT(isDistributedCommand, "Command is not a distributed command");
  return ::tt::runtime::distributed::flatbuffer::GetCommand(command.data());
}

void CommandExecutor::connect(const std::string &host, uint16_t port) {
  LOG_ASSERT(!workerSocket_, "WorkerSocket already connected");
  workerSocket_ = std::make_unique<WorkerSocket>(host, port);
}

::tt::runtime::Binary
CommandExecutor::getOrCreateBinary(const flatbuffers::Vector<uint8_t> *binary,
                                   uint64_t binaryId) {
  ::tt::runtime::Binary executable(nullptr);
  if (binaryPool_.contains(binaryId)) {
    executable = binaryPool_.at(binaryId);
  } else {
    executable =
        ::tt::runtime::Binary::loadFromMemory(binary->data(), binary->size());
    executable.setId(binaryId);
    binaryPool_.insert_or_assign(binaryId, executable);
  }
  return executable;
}

void CommandExecutor::run() {
  launchCommandReceiver();
  launchResponseSender();
  while (!shutdownRequested_.load(std::memory_order_relaxed)) {
    SizedBuffer commandData = commandQueue_.popBlocking();
    const ::tt::runtime::distributed::flatbuffer::Command *command =
        getCommand(commandData);

    executeCommand(command);
  }
}

void CommandExecutor::launchCommandReceiver() {
  LOG_ASSERT(!commandReceiverThread_.joinable(),
             "Command receiver thread already running");
  commandReceiverThread_ = std::thread([this]() { receiveCommands(); });
}

// This will get run on the command receiver thread
void CommandExecutor::receiveCommands() {
  while (!shutdownRequested_.load(std::memory_order_relaxed)) {
    if (!workerSocket_->hasDataToRead()) {
      continue;
    }
    SizedBuffer commandData = workerSocket_->sizePrefixedRead();
    LOG_ASSERT(commandData.size(), "Read null command from worker socket");
    commandQueue_.push(commandData);
  }
}

void CommandExecutor::launchResponseSender() {
  LOG_ASSERT(!responseSenderThread_.joinable(),
             "Response sender thread already running");
  responseSenderThread_ = std::thread([this]() { sendResponses(); });
}

// This will get run on the response sender thread
void CommandExecutor::sendResponses() {
  while (!(shutdownRequested_.load(std::memory_order_acquire) &&
           responseQueue_.empty())) {
    std::optional<std::unique_ptr<::flatbuffers::FlatBufferBuilder>>
        responseBuilderOpt = responseQueue_.popWithTimeout();
    if (!responseBuilderOpt.has_value()) {
      continue;
    }
    std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
        std::move(responseBuilderOpt.value());
    size_t responseSize = responseBuilder->GetSize();
    LOG_ASSERT(responseSize > 0, "Unexpected empty response");
    workerSocket_->sizePrefixedWrite(responseBuilder->GetBufferPointer(),
                                     responseSize);
  }
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::runtime::distributed::flatbuffer::GetSystemDescCommand
        *command) {

  auto responseBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType =
      std::nullopt;
  if (command->dispatch_core_type()) {
    dispatchCoreType = ::tt::runtime::utils::toRuntimeDispatchCoreType(
        command->dispatch_core_type().value());
  }

  std::optional<Device> device = std::nullopt;
  if (command->device()) {
    device = devicePool_.at(command->device()->global_id());
  }

  ::tt::runtime::SystemDesc systemDesc =
      ::tt::runtime::system_desc::getCurrentSystemDesc(dispatchCoreType,
                                                       device);

  ResponseFactory::buildGetSystemDescResponse(*responseBuilder, commandId,
                                              systemDesc);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::runtime::distributed::flatbuffer::OpenMeshDeviceCommand
        *command) {

  auto responseBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint32_t deviceGlobalId = command->device_global_id();
  const ::tt::runtime::distributed::flatbuffer::MeshDeviceOptions *options =
      command->options();

  ::tt::runtime::MeshDeviceOptions meshDeviceOptions;
  if (options->mesh_offset()) {
    LOG_ASSERT(options->mesh_offset()->size() == 2,
               "Currently only 2D mesh offsets are supported");
    meshDeviceOptions.meshOffset = {options->mesh_offset()->Get(0),
                                    options->mesh_offset()->Get(1)};
  }

  if (options->device_ids()) {
    meshDeviceOptions.deviceIds.resize(options->device_ids()->size());
    std::copy_n(options->device_ids()->begin(), options->device_ids()->size(),
                meshDeviceOptions.deviceIds.begin());
  }

  meshDeviceOptions.numHWCQs = options->num_hw_cqs();
  meshDeviceOptions.enableProgramCache = options->enable_program_cache();

  if (options->mesh_shape()) {
    LOG_ASSERT(options->mesh_shape()->size() == 2,
               "Currently only 2D mesh shapes are supported");
    meshDeviceOptions.meshShape = {options->mesh_shape()->Get(0),
                                   options->mesh_shape()->Get(1)};
  }

  if (options->l1_small_size().has_value()) {
    meshDeviceOptions.l1SmallSize = options->l1_small_size().value();
  }

  if (options->trace_region_size().has_value()) {
    meshDeviceOptions.traceRegionSize = options->trace_region_size().value();
  }

  if (options->dispatch_core_type().has_value()) {
    meshDeviceOptions.dispatchCoreType =
        ::tt::runtime::utils::toRuntimeDispatchCoreType(
            options->dispatch_core_type().value());
  }

  ::tt::runtime::Device device =
      ::tt::runtime::openMeshDevice(meshDeviceOptions);

  device.setGlobalId(deviceGlobalId);

  devicePool_.insert_or_assign(deviceGlobalId, device);

  ResponseFactory::buildOpenMeshDeviceResponse(*responseBuilder, commandId,
                                               device);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::runtime::distributed::flatbuffer::CloseMeshDeviceCommand
        *command) {

  auto responseBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint32_t deviceGlobalId = command->device()->global_id();

  ::tt::runtime::Device device = devicePool_.at(deviceGlobalId);

  ::tt::runtime::closeMeshDevice(device);

  devicePool_.erase(deviceGlobalId);

  ResponseFactory::buildCloseMeshDeviceResponse(*responseBuilder, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::runtime::distributed::flatbuffer::CreateHostTensorCommand
        *command) {

  auto responseBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t tensorGlobalId = command->output_global_id();
  const uint8_t *tensorData = command->data()->data();
  std::vector<uint32_t> shape(command->shape()->begin(),
                              command->shape()->end());
  std::vector<uint32_t> stride(command->stride()->begin(),
                               command->stride()->end());
  uint32_t itemSize = command->item_size();
  ::tt::target::DataType dataType = command->data_type();

  ::tt::runtime::Tensor tensor = ::tt::runtime::createOwnedHostTensor(
      tensorData, shape, stride, itemSize, dataType);

  tensor.setGlobalId(tensorGlobalId);

  tensorPool_.insert_or_assign(tensorGlobalId, tensor);

  ResponseFactory::buildCreateHostTensorResponse(*responseBuilder, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::runtime::distributed::flatbuffer::GetLayoutCommand *command) {

  auto responseBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  ::tt::runtime::Binary binary =
      getOrCreateBinary(command->binary(), command->binary_id());

  ::tt::runtime::Layout layout = ::tt::runtime::getLayout(
      binary, command->program_id(), command->input_id());

  layout.setGlobalId(command->output_layout_id());
  layoutPool_.insert_or_assign(command->output_layout_id(), layout);

  ResponseFactory::buildGetLayoutResponse(*responseBuilder, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::runtime::distributed::flatbuffer::ToLayoutCommand *command) {

  auto responseBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t inputGlobalId = command->input_global_id();
  uint64_t outputGlobalId = command->output_global_id();
  ::tt::runtime::Device device = devicePool_.at(command->device()->global_id());

  ::tt::runtime::Tensor inputTensor = tensorPool_.at(inputGlobalId);

  std::optional<bool> retain = std::nullopt;
  if (command->retain().has_value()) {
    retain = command->retain().value();
  }

  ::tt::runtime::Layout layout = layoutPool_.at(command->layout_global_id());

  ::tt::runtime::Tensor resultTensor =
      ::tt::runtime::toLayout(inputTensor, device, layout, retain);

  resultTensor.setGlobalId(outputGlobalId);

  tensorPool_.insert_or_assign(outputGlobalId, resultTensor);

  ResponseFactory::buildToLayoutResponse(*responseBuilder, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::runtime::distributed::flatbuffer::SubmitCommand *command) {

  auto responseBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  ::tt::runtime::Device device = devicePool_.at(command->device()->global_id());

  std::vector<::tt::runtime::Tensor> inputTensors;
  for (const auto &inputGlobalId : *command->input_global_ids()) {
    inputTensors.push_back(tensorPool_.at(inputGlobalId));
  }

  ::tt::runtime::Binary executable =
      getOrCreateBinary(command->binary(), command->binary_id());

  std::vector<::tt::runtime::Tensor> outputTensors = ::tt::runtime::submit(
      device, executable, command->program_id(), inputTensors);

  LOG_ASSERT(outputTensors.size() == command->output_global_ids()->size(),
             "Output tensors from submit does not match the number of output "
             "global ids: ",
             outputTensors.size(),
             " != ", command->output_global_ids()->size());

  for (size_t i = 0; i < outputTensors.size(); i++) {
    outputTensors[i].setGlobalId(command->output_global_ids()->Get(i));
    tensorPool_.insert_or_assign(command->output_global_ids()->Get(i),
                                 outputTensors[i]);
  }

  ResponseFactory::buildSubmitResponse(*responseBuilder, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::runtime::distributed::flatbuffer::GetNumShardsCommand
        *command) {
  auto responseBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();
  uint64_t tensorGlobalId = command->input_global_id();
  ::tt::runtime::Tensor tensor = tensorPool_.at(tensorGlobalId);

  uint32_t numBuffers = ::tt::runtime::detail::getNumShards(tensor);

  ResponseFactory::buildGetNumShardsResponse(*responseBuilder, commandId,
                                             numBuffers);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::runtime::distributed::flatbuffer::ToHostCommand *command) {

  auto responseBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t inputGlobalId = command->input_global_id();

  ::tt::runtime::Tensor inputTensor = tensorPool_.at(inputGlobalId);

  bool untilize = command->untilize();
  bool blocking = command->blocking();

  std::vector<::tt::runtime::Tensor> outputTensors =
      ::tt::runtime::toHost(inputTensor, untilize, blocking);

  LOG_ASSERT(outputTensors.size() == command->output_global_ids()->size(),
             "Output tensors from toHost does not match the number of output "
             "global ids: ",
             outputTensors.size(),
             " != ", command->output_global_ids()->size());

  for (size_t i = 0; i < outputTensors.size(); i++) {
    outputTensors[i].setGlobalId(command->output_global_ids()->Get(i));
    tensorPool_.insert_or_assign(command->output_global_ids()->Get(i),
                                 outputTensors[i]);
  }

  ResponseFactory::buildToHostResponse(*responseBuilder, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::runtime::distributed::flatbuffer::MemcpyCommand *command) {
  auto responseBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t srcGlobalId = command->src_global_id();
  ::tt::runtime::Tensor srcTensor = tensorPool_.at(srcGlobalId);

  std::optional<std::vector<std::uint8_t>> dstDataBuffer = std::nullopt;

  if (command->dst_global_id().has_value()) {
    ::tt::runtime::Tensor dstTensor =
        tensorPool_.at(command->dst_global_id().value());
    ::tt::runtime::memcpy(dstTensor, srcTensor);
  } else {
    size_t volume = ::tt::runtime::getTensorVolume(srcTensor);
    size_t dstElementSize = ::tt::runtime::getTensorElementSize(srcTensor);

    std::optional<::tt::target::DataType> dstDataType = std::nullopt;
    if (command->dst_data_type().has_value()) {
      dstDataType = command->dst_data_type().value();
      dstElementSize =
          ::tt::runtime::utils::dataTypeElementSize(dstDataType.value());
    }

    size_t dstSizeBytes = volume * dstElementSize;

    dstDataBuffer = std::vector<std::uint8_t>(dstSizeBytes);

    ::tt::runtime::memcpy(dstDataBuffer.value().data(), srcTensor, dstDataType);
  }

  ResponseFactory::buildMemcpyResponse(*responseBuilder, commandId,
                                       dstDataBuffer);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::runtime::distributed::flatbuffer::ShutdownCommand *command) {

  LOG_INFO("Shutdown command received, shutting down command executor");

  auto responseBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();
  ResponseFactory::buildShutdownResponse(*responseBuilder, commandId);

  responseQueue_.push(std::move(responseBuilder));

  shutdownRequested_.store(true, std::memory_order_release);
  commandReceiverThread_.join();
  responseSenderThread_.join();

  LOG_INFO("Command executor shutdown complete");
}

void CommandExecutor::executeCommand(
    const ::tt::runtime::distributed::flatbuffer::Command *command) {
  switch (command->type_type()) {
  case ::tt::runtime::distributed::flatbuffer::CommandType::
      GetSystemDescCommand: {
    return execute(command->command_id(),
                   command->type_as_GetSystemDescCommand());
  }
  case ::tt::runtime::distributed::flatbuffer::CommandType::
      OpenMeshDeviceCommand: {
    return execute(command->command_id(),
                   command->type_as_OpenMeshDeviceCommand());
  }
  case ::tt::runtime::distributed::flatbuffer::CommandType::
      CloseMeshDeviceCommand: {
    return execute(command->command_id(),
                   command->type_as_CloseMeshDeviceCommand());
  }
  case ::tt::runtime::distributed::flatbuffer::CommandType::
      CreateHostTensorCommand: {
    return execute(command->command_id(),
                   command->type_as_CreateHostTensorCommand());
  }
  case ::tt::runtime::distributed::flatbuffer::CommandType::GetLayoutCommand: {
    return execute(command->command_id(), command->type_as_GetLayoutCommand());
  }
  case ::tt::runtime::distributed::flatbuffer::CommandType::ToLayoutCommand: {
    return execute(command->command_id(), command->type_as_ToLayoutCommand());
  }
  case ::tt::runtime::distributed::flatbuffer::CommandType::SubmitCommand: {
    return execute(command->command_id(), command->type_as_SubmitCommand());
  }
  case ::tt::runtime::distributed::flatbuffer::CommandType::
      GetNumShardsCommand: {
    return execute(command->command_id(),
                   command->type_as_GetNumShardsCommand());
  }
  case ::tt::runtime::distributed::flatbuffer::CommandType::ToHostCommand: {
    return execute(command->command_id(), command->type_as_ToHostCommand());
  }
  case ::tt::runtime::distributed::flatbuffer::CommandType::MemcpyCommand: {
    return execute(command->command_id(), command->type_as_MemcpyCommand());
  }
  case ::tt::runtime::distributed::flatbuffer::CommandType::ShutdownCommand: {
    return execute(command->command_id(), command->type_as_ShutdownCommand());
  }
  case ::tt::runtime::distributed::flatbuffer::CommandType::NONE: {
    LOG_FATAL("Unhandled command type: ",
              ::tt::runtime::distributed::flatbuffer::EnumNameCommandType(
                  command->type_type()));
  }
  }

  LOG_FATAL("Unreachable code path, all commands should be handled in switch "
            "statement");
}
} // namespace tt::runtime::distributed::worker
