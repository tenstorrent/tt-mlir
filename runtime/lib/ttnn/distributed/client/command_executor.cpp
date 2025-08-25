// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/distributed/client/command_executor.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/common/system_desc.h"
#include "tt/runtime/detail/ttnn/distributed/client/response_factory.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include <thread>

namespace tt::runtime::ttnn::distributed::client {

static const ::tt::target::ttnn::distributed::Command *
getCommand(const CommandBytes &command) {
  bool isTTNNCommand =
      ::tt::target::ttnn::distributed::CommandBufferHasIdentifier(
          command.get());
  LOG_ASSERT(isTTNNCommand, "Command is not a TTNN command");
  return ::tt::target::ttnn::distributed::GetCommand(command.get());
}

static bool
isShutdownCommand(const ::tt::target::ttnn::distributed::Command *command) {
  return command->type_type() ==
         ::tt::target::ttnn::distributed::CommandType::ShutdownCommand;
}

void CommandExecutor::connect(std::string_view host, uint16_t port) {
  LOG_ASSERT(!clientSocket_, "ClientSocket already connected");
  clientSocket_ = std::make_unique<ClientSocket>(host, port);
  clientSocket_->connect();
}

void CommandExecutor::run() {
  launchCommandReceiver();
  while (true) {
    CommandBytes commandData = commandQueue_.popBlocking();
    const ::tt::target::ttnn::distributed::Command *command =
        getCommand(commandData);

    ::flatbuffers::FlatBufferBuilder responseBuilder;
    executeCommand(command, responseBuilder);
    sendResponse(responseBuilder);

    if (isShutdownCommand(command)) {
      handleShutdown();
      break;
    }
  }
}

void CommandExecutor::launchCommandReceiver() {
  LOG_ASSERT(!commandReceiverThread_.joinable(),
             "Command receiver thread already running");
  commandReceiverThread_ = std::thread([this]() { receiveCommands(); });
}

// This will get run on the command receiver thread
void CommandExecutor::receiveCommands() {
  while (true) {
    CommandBytes commandData = clientSocket_->sizePrefixedRead();
    LOG_ASSERT(commandData, "Read null command from client socket");
    commandQueue_.push(commandData);
    if (isShutdownCommand(getCommand(commandData))) {
      break;
    }
  }
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::target::ttnn::distributed::GetSystemDescCommand *command,
    ::flatbuffers::FlatBufferBuilder &responseBuilder) {

  ::tt::runtime::DispatchCoreType dispatchCoreType =
      ::tt::runtime::utils::toRuntimeDispatchCoreType(
          command->dispatch_core_type());

  std::shared_ptr<::ttnn::MeshDevice> meshDevice;
  if (!command->device()) {
    meshDevice = ::tt::runtime::common::createFullMeshDevice(dispatchCoreType);
  } else {
    meshDevice = meshDevicePool_.getMeshDevice(command->device()->global_id());
  }

  ::flatbuffers::Offset<tt::target::SystemDescRoot> systemDescRoot =
      ::tt::runtime::system_desc::buildSystemDescRoot(responseBuilder,
                                                      *meshDevice);

  ResponseFactory::buildGetSystemDescResponse(responseBuilder, commandId,
                                              systemDescRoot);
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::target::ttnn::distributed::OpenMeshDeviceCommand *command,
    ::flatbuffers::FlatBufferBuilder &responseBuilder) {

  uint64_t deviceGlobalId = command->device_global_id();
  const ::tt::target::ttnn::distributed::MeshDeviceOptions *options =
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
      ::tt::runtime::ttnn::openMeshDevice(meshDeviceOptions);

  meshDevicePool_.insertRuntimeDevice(deviceGlobalId, device);

  ResponseFactory::buildOpenMeshDeviceResponse(responseBuilder, commandId,
                                               device);
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::target::ttnn::distributed::CloseMeshDeviceCommand *command,
    ::flatbuffers::FlatBufferBuilder &responseBuilder) {

  uint64_t deviceGlobalId = command->device()->global_id();
  ::tt::runtime::Device device =
      meshDevicePool_.getRuntimeDevice(deviceGlobalId);

  ::tt::runtime::ttnn::closeMeshDevice(device);

  meshDevicePool_.erase(deviceGlobalId);

  ResponseFactory::buildCloseMeshDeviceResponse(responseBuilder, commandId,
                                                /*success=*/true);
}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::target::ttnn::distributed::CreateHostTensorCommand *command,
    ::flatbuffers::FlatBufferBuilder &responseBuilder) {

  uint64_t tensorGlobalId = command->output_global_id();
  const uint8_t *tensorData = command->data()->data();
  std::vector<uint32_t> shape(command->shape()->begin(),
                              command->shape()->end());
  std::vector<uint32_t> stride(command->stride()->begin(),
                               command->stride()->end());
  uint32_t itemSize = command->item_size();
  ::tt::target::DataType dataType = command->data_type();

  ::tt::runtime::Tensor tensor = ::tt::runtime::ttnn::createOwnedHostTensor(
      tensorData, shape, stride, itemSize, dataType);

  tensorPool_.insert_or_assign(tensorGlobalId, tensor);

  ResponseFactory::buildCreateHostTensorResponse(responseBuilder, commandId,
                                                 /*success=*/true);
}
void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::target::ttnn::distributed::ToLayoutCommand *command,
    ::flatbuffers::FlatBufferBuilder &responseBuilder) {}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::target::ttnn::distributed::SubmitCommand *command,
    ::flatbuffers::FlatBufferBuilder &responseBuilder) {}

void CommandExecutor::execute(
    uint64_t commandId,
    const ::tt::target::ttnn::distributed::ShutdownCommand *command,
    ::flatbuffers::FlatBufferBuilder &responseBuilder) {}

void CommandExecutor::executeCommand(
    const ::tt::target::ttnn::distributed::Command *command,
    ::flatbuffers::FlatBufferBuilder &responseBuilder) {
  switch (command->type_type()) {
  case ::tt::target::ttnn::distributed::CommandType::GetSystemDescCommand: {
    return execute(command->command_id(),
                   command->type_as_GetSystemDescCommand(), responseBuilder);
  }
  case ::tt::target::ttnn::distributed::CommandType::OpenMeshDeviceCommand: {
    return execute(command->command_id(),
                   command->type_as_OpenMeshDeviceCommand(), responseBuilder);
  }
  case ::tt::target::ttnn::distributed::CommandType::CloseMeshDeviceCommand: {
    return execute(command->command_id(),
                   command->type_as_CloseMeshDeviceCommand(), responseBuilder);
  }
  case ::tt::target::ttnn::distributed::CommandType::CreateHostTensorCommand: {
    return execute(command->command_id(),
                   command->type_as_CreateHostTensorCommand(), responseBuilder);
  }
  case ::tt::target::ttnn::distributed::CommandType::ToLayoutCommand: {
    return execute(command->command_id(), command->type_as_ToLayoutCommand(),
                   responseBuilder);
  }
  case ::tt::target::ttnn::distributed::CommandType::SubmitCommand: {
    return execute(command->command_id(), command->type_as_SubmitCommand(),
                   responseBuilder);
  }
  case ::tt::target::ttnn::distributed::CommandType::ShutdownCommand: {
    return execute(command->command_id(), command->type_as_ShutdownCommand(),
                   responseBuilder);
  }
  default: {
    LOG_FATAL("Unhandled command type: ",
              ::tt::target::ttnn::distributed::EnumNameCommandType(
                  command->type_type()));
  }
  }
}

void CommandExecutor::sendResponse(
    ::flatbuffers::FlatBufferBuilder &responseBuilder) {
  LOG_ASSERT(responseBuilder.GetSize() > 0,
             "Expected response from command execution");
  size_t responseSize = responseBuilder.GetSize();
  clientSocket_->sizePrefixedWrite(responseBuilder.GetBufferPointer(),
                                   responseSize);
}

void CommandExecutor::handleShutdown() {
  LOG_INFO("Shutdown command received, shutting down command executor");
  commandReceiverThread_.join();
}

} // namespace tt::runtime::ttnn::distributed::client
