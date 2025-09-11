// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/server/server.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/detail/distributed/server/command_factory.h"
#include "tt/runtime/detail/distributed/utils/utils.h"
#include "tt/runtime/runtime.h"

namespace tt::runtime::distributed::server {

namespace debug {
static void checkResponsesIdentical(const std::vector<SizedBuffer> &responses) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  const SizedBuffer &firstResponse = responses[0];
  for (const SizedBuffer &response : responses) {
    DEBUG_ASSERT(response.size() == firstResponse.size(),
                 "Responses sizes are not identical across clients");
    DEBUG_ASSERT(std::memcmp(response.data(), firstResponse.data(),
                             response.size()) == 0,
                 "Response data is not identical across clients");
  }
}
#endif
} // namespace debug

static const ::tt::runtime::distributed::flatbuffer::Response *
getResponse(const SizedBuffer &response) {
  bool isDistributedResponse =
      ::tt::runtime::distributed::flatbuffer::ResponseBufferHasIdentifier(
          response.data());
  LOG_ASSERT(isDistributedResponse, "Response is not a distributed response");
  return ::tt::runtime::distributed::flatbuffer::GetResponse(response.data());
}

static bool isErrorResponse(const SizedBuffer &response) {
  return getResponse(response)->type_type() ==
         ::tt::runtime::distributed::flatbuffer::ResponseType::ErrorResponse;
}

Server::~Server() {
  if (!shutdownRequested_.load(std::memory_order_relaxed)) {
    shutdown();
  }
}

void Server::launchLocalSubprocess(uint16_t serverPort) {
  constexpr size_t numClients = 1;
  LOG_ASSERT(!serverSocket_, "Server already launched");
  serverSocket_ = std::make_unique<ServerSocket>(serverPort);
  uint16_t bindPort = serverSocket_->port();

  std::string command =
      tt::runtime::distributed::utils::getClientExecutableCommand(bindPort);

  std::future<int> asyncFuture = std::async(
      std::launch::async, [command]() { return std::system(command.c_str()); });
  exitCodeFutures_.emplace_back(std::move(asyncFuture));

  clientConnections_ = serverSocket_->connectToClients(numClients);
  launchCommandDispatcher();
  launchResponseHandler();
}

void Server::shutdown() {
  if (shutdownRequested_.load(std::memory_order_relaxed)) {
    LOG_WARNING("Calling shutdown() on already shutdown Server, "
                "returning immediately");
    return;
  }

  shutdownRequested_.store(true, std::memory_order_release);
  commandDispatcherThread_.join();
  responseHandlerThread_.join();

  for (auto &exitCodeFuture : exitCodeFutures_) {
    auto exitCodeResult = exitCodeFuture.wait_for(std::chrono::seconds(60));
    LOG_ASSERT(exitCodeResult == std::future_status::ready,
               "Client subprocess failed to exit");

    int exitCode = exitCodeFuture.get();
    LOG_ASSERT(exitCode == 0,
               "Client subprocess failed with exit code: ", exitCode);
  }
  exitCodeFutures_.clear();
}

void Server::setWriteTimeout(const std::chrono::seconds &timeout) {
  writeTimeout_ = timeout;
}

void Server::setReadTimeout(const std::chrono::seconds &timeout) {
  readTimeout_ = timeout;
}

SystemDesc Server::getCurrentSystemDesc(
    std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType,
    std::optional<::tt::runtime::Device> deviceHandle) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildGetSystemDescCommand(
      *commandBuilder, dispatchCoreType, deviceHandle);

  auto awaitingHandles = std::make_unique<std::vector<SharedHandle>>();
  SharedHandle systemDescHandle;
  awaitingHandles->push_back(systemDescHandle);

  // This command will execute synchronously and block until the response is
  // received The systemDesc handle will get populated by the response handler
  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(commandId, std::move(commandBuilder),
                                 std::move(awaitingHandles),
                                 std::move(awaitingPromise));

  awaitingFuture.wait();

  return SystemDesc(systemDescHandle.getHandle());
}

::tt::runtime::Device
Server::openMeshDevice(const ::tt::runtime::MeshDeviceOptions &options) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  ::tt::runtime::Device outputDeviceHandle(getCurrentRuntime());

  uint64_t commandId = CommandFactory::buildOpenMeshDeviceCommand(
      *commandBuilder, outputDeviceHandle, options);

  pushToCommandAndResponseQueues(commandId, std::move(commandBuilder));

  return outputDeviceHandle;
}

void Server::closeMeshDevice(const ::tt::runtime::Device &deviceHandle) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildCloseMeshDeviceCommand(
      *commandBuilder, deviceHandle);

  pushToCommandAndResponseQueues(commandId, std::move(commandBuilder));
}

::tt::runtime::Tensor Server::createOwnedHostTensor(
    const void *data, const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
    ::tt::target::DataType dataType) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  ::tt::runtime::Tensor outputTensorHandle;

  uint64_t commandId = CommandFactory::buildCreateHostTensorCommand(
      *commandBuilder, outputTensorHandle, data, shape, stride, itemsize,
      dataType);

  pushToCommandAndResponseQueues(commandId, std::move(commandBuilder));

  return outputTensorHandle;
}

::tt::runtime::Layout
Server::getLayout(const ::tt::runtime::Binary &executableHandle,
                  std::uint32_t programIndex, std::uint32_t inputIndex) {
  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  ::tt::runtime::Layout outputLayoutHandle;

  uint64_t commandId = CommandFactory::buildGetLayoutCommand(
      *commandBuilder, executableHandle, programIndex, inputIndex,
      outputLayoutHandle);

  pushToCommandAndResponseQueues(commandId, std::move(commandBuilder));

  return outputLayoutHandle;
}

::tt::runtime::Tensor
Server::toLayout(const ::tt::runtime::Tensor &tensorHandle,
                 const ::tt::runtime::Device &deviceHandle,
                 const ::tt::runtime::Layout &layoutHandle,
                 std::optional<bool> retain) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  ::tt::runtime::Tensor outputTensorHandle;

  uint64_t commandId = CommandFactory::buildToLayoutCommand(
      *commandBuilder, tensorHandle, deviceHandle, layoutHandle,
      outputTensorHandle, retain);

  pushToCommandAndResponseQueues(commandId, std::move(commandBuilder));

  return tensorHandle;
}

std::vector<::tt::runtime::Tensor>
Server::submit(const ::tt::runtime::Device &deviceHandle,
               const ::tt::runtime::Binary &executableHandle,
               std::uint32_t programIndex,
               const std::vector<::tt::runtime::Tensor> &inputHandles) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint32_t numOutputs = executableHandle.getNumProgramOutputs(programIndex);
  std::vector<::tt::runtime::Tensor> outputHandles;
  outputHandles.reserve(numOutputs);
  DeviceRuntime currentRuntime = getCurrentRuntime();

  for (uint32_t i = 0; i < numOutputs; i++) {
    outputHandles.push_back(
        ::tt::runtime::Tensor(nullptr, nullptr, currentRuntime));
  }

  uint64_t commandId = CommandFactory::buildSubmitCommand(
      *commandBuilder, deviceHandle, executableHandle, programIndex,
      inputHandles, outputHandles);

  pushToCommandAndResponseQueues(commandId, std::move(commandBuilder));

  return outputHandles;
}

std::vector<::tt::runtime::Tensor>
Server::toHost(const ::tt::runtime::Tensor &tensorHandle, bool untilize,
               bool blocking) {
  // For toHost, we need to get the number of shards this tensor has first
  // so we can create the correct number of output tensors
  // Performance wise, this is not ideal, but it should be ok for now since
  // toHost is typically the last step and is not on the hot path
  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  SharedHandle numShardsHandle;

  auto awaitingHandles = std::make_unique<std::vector<SharedHandle>>();
  awaitingHandles->push_back(numShardsHandle);

  uint64_t numShardsCommandId =
      CommandFactory::buildGetNumShardsCommand(*commandBuilder, tensorHandle);

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(numShardsCommandId, std::move(commandBuilder),
                                 std::move(awaitingHandles),
                                 std::move(awaitingPromise));

  awaitingFuture.wait();

  commandBuilder->Clear();

  std::uint32_t numShards = numShardsHandle.as<uint32_t>();

  std::vector<::tt::runtime::Tensor> outputHandles;
  outputHandles.reserve(numShards);
  for (std::uint32_t i = 0; i < numShards; i++) {
    outputHandles.push_back(
        ::tt::runtime::Tensor(nullptr, nullptr, getCurrentRuntime()));
  }

  uint64_t toHostCommandId = CommandFactory::buildToHostCommand(
      *commandBuilder, tensorHandle, untilize, blocking, outputHandles);

  pushToCommandAndResponseQueues(toHostCommandId, std::move(commandBuilder));

  return outputHandles;
}

void Server::memcpy(void *dst, const ::tt::runtime::Tensor &srcHandle,
                    std::optional<tt::target::DataType> targetDataType) {
  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildMemcpyCommand(
      *commandBuilder, srcHandle, /*dstTensor=*/std::nullopt, targetDataType);

  auto awaitingHandles = std::make_unique<std::vector<SharedHandle>>();

  std::shared_ptr<void> dstPtr =
      ::tt::runtime::utils::unsafe_borrow_shared(dst);

  SharedHandle dstHandle(dstPtr);
  awaitingHandles->push_back(dstHandle);

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(commandId, std::move(commandBuilder),
                                 std::move(awaitingHandles),
                                 std::move(awaitingPromise));

  awaitingFuture.wait();
}

void Server::pushToCommandAndResponseQueues(
    uint64_t commandId,
    std::unique_ptr<::flatbuffers::FlatBufferBuilder> commandBuilder,
    std::unique_ptr<std::vector<SharedHandle>> awaitingHandles,
    std::unique_ptr<std::promise<void>> awaitingPromise) {
  // Need to push to the response queue first to ensure the command response is
  // waited for before the command is dispatched
  awaitingResponseQueue_.push(std::make_unique<AwaitingResponseQueueEntry>(
      commandId, std::move(awaitingHandles), std::move(awaitingPromise)));
  commandQueue_.push(
      std::make_unique<CommandQueueEntry>(std::move(commandBuilder)));
}

void Server::launchCommandDispatcher() {
  LOG_ASSERT(!commandDispatcherThread_.joinable(),
             "Command dispatcher thread already running");
  commandDispatcherThread_ = std::thread([this]() { dispatchCommands(); });
}

void Server::dispatchCommand(
    const ::flatbuffers::FlatBufferBuilder &commandBuilder) {
  std::vector<std::future<ssize_t>> writeFutures;
  writeFutures.reserve(clientConnections_.size());
  size_t commandSize = commandBuilder.GetSize();
  LOG_ASSERT(commandSize > 0, "Unexpected empty command");

  // Currently broadcasting command to all clients
  // Will need updates later to support multi mesh and MPMD
  for (const auto &clientConnection : clientConnections_) {
    writeFutures.emplace_back(clientConnection->sizePrefixedWriteAsync(
        commandBuilder.GetBufferPointer(), commandSize));
  }

  for (const std::future<ssize_t> &writeFuture : writeFutures) {
    if (writeFuture.wait_for(writeTimeout_) == std::future_status::timeout) {
      LOG_FATAL("Write timeout occurred while sending command to client");
    }
  }
}

void Server::dispatchCommands() {
  std::vector<std::future<ssize_t>> writeFutures;
  writeFutures.reserve(clientConnections_.size());
  while (!(shutdownRequested_.load(std::memory_order_acquire) &&
           commandQueue_.empty())) {
    std::optional<std::unique_ptr<CommandQueueEntry>> commandEntryOpt =
        commandQueue_.popWithTimeout();
    if (!commandEntryOpt.has_value()) {
      continue;
    }

    std::unique_ptr<CommandQueueEntry> commandQueueEntry =
        std::move(commandEntryOpt.value());

    dispatchCommand(*(commandQueueEntry->commandBuilder));
  }
}

void Server::launchResponseHandler() {
  LOG_ASSERT(!responseHandlerThread_.joinable(),
             "Response handler thread already running");
  responseHandlerThread_ = std::thread([this]() { handleResponses(); });
}

void Server::handleResponses() {
  while (!(shutdownRequested_.load(std::memory_order_acquire) &&
           awaitingResponseQueue_.empty())) {
    std::optional<std::unique_ptr<AwaitingResponseQueueEntry>>
        awaitingResponseQueueEntryOpt = awaitingResponseQueue_.popWithTimeout();
    if (!awaitingResponseQueueEntryOpt.has_value()) {
      continue;
    }

    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponseQueueEntry =
        std::move(awaitingResponseQueueEntryOpt.value());

    std::vector<std::future<SizedBuffer>> readFutures;
    readFutures.reserve(clientConnections_.size());

    for (const auto &clientConnection : clientConnections_) {
      readFutures.push_back(clientConnection->sizePrefixedReadAsync());
    }

    std::vector<SizedBuffer> responseBuffers;
    responseBuffers.reserve(readFutures.size());
    for (std::future<SizedBuffer> &readFuture : readFutures) {
      if (readFuture.wait_for(readTimeout_) == std::future_status::timeout) {
        LOG_FATAL("Read timeout occurred while receiving response from client");
      }
      responseBuffers.push_back(readFuture.get());
    }

    for (const SizedBuffer &responseBuffer : responseBuffers) {
      if (isErrorResponse(responseBuffer)) {
        handleErrorResponse(
            getResponse(responseBuffer)->type_as_ErrorResponse(),
            *awaitingResponseQueueEntry);
      }
    }

    debug::checkResponsesIdentical(responseBuffers);

    // Since we currently only support SPMD, client responses should be
    // identical therefore we can just handle the first response
    const ::tt::runtime::distributed::flatbuffer::Response *response =
        getResponse(responseBuffers[0]);
    LOG_ASSERT(
        response->command_id() == awaitingResponseQueueEntry->commandId,
        "Response command id does not match awaiting response command id");

    handleResponse(response, *awaitingResponseQueueEntry);
  }
}

void Server::handleErrorResponse(
    const ::tt::runtime::distributed::flatbuffer::ErrorResponse *response,
    AwaitingResponseQueueEntry &awaitingResponse) {
  if (awaitingResponse.awaitingPromise) {
    awaitingResponse.awaitingPromise->set_value();
  }
  LOG_FATAL("Error response received with message: ",
            response->message()->c_str());
}

void Server::handleGetSystemDescResponse(
    const ::tt::runtime::distributed::flatbuffer::GetSystemDescResponse
        *response,
    AwaitingResponseQueueEntry &awaitingResponse) {
  std::unique_ptr<std::vector<SharedHandle>> awaitingHandles =
      std::move(awaitingResponse.awaitingHandles);

  DEBUG_ASSERT(awaitingHandles && awaitingHandles->size() == 1,
               "GetSystemDesc: Awaiting handles must be populated and contain "
               "exactly one handle");

  DEBUG_ASSERT(awaitingResponse.awaitingPromise,
               "Awaiting promise must be populated");

  const auto *systemDescVec = response->system_desc();

  std::shared_ptr<void> systemDescPtr =
      ::tt::runtime::utils::mallocShared(systemDescVec->size());
  std::memcpy(systemDescPtr.get(), systemDescVec->data(),
              systemDescVec->size());

  SharedHandle &systemDescHandle = awaitingHandles->at(0);
  systemDescHandle.reset(std::move(systemDescPtr));

  awaitingResponse.awaitingPromise->set_value();
}

void Server::handleOpenMeshDeviceResponse(
    const ::tt::runtime::distributed::flatbuffer::OpenMeshDeviceResponse
        *response,
    AwaitingResponseQueueEntry &awaitingResponse) {
  DEBUG_ASSERT(!awaitingResponse.awaitingHandles,
               "Awaiting handles should not be populated for OpenMeshDevice");
  DEBUG_ASSERT(!awaitingResponse.awaitingPromise,
               "Awaiting promise should not be populated for OpenMeshDevice");
}

void Server::handleCloseMeshDeviceResponse(
    const ::tt::runtime::distributed::flatbuffer::CloseMeshDeviceResponse
        *response,
    AwaitingResponseQueueEntry &awaitingResponse) {
  DEBUG_ASSERT(!awaitingResponse.awaitingHandles,
               "Awaiting handles should not be populated for CloseMeshDevice");
  DEBUG_ASSERT(!awaitingResponse.awaitingPromise,
               "Awaiting promise should not be populated for CloseMeshDevice");
}

void Server::handleGetNumShardsResponse(
    const ::tt::runtime::distributed::flatbuffer::GetNumShardsResponse
        *response,
    AwaitingResponseQueueEntry &awaitingResponse) {

  std::unique_ptr<std::vector<SharedHandle>> awaitingHandles =
      std::move(awaitingResponse.awaitingHandles);
  DEBUG_ASSERT(awaitingHandles && awaitingHandles->size() == 1,
               "GetSystemDesc: Awaiting handles must be populated and contain "
               "exactly one handle");

  DEBUG_ASSERT(awaitingResponse.awaitingPromise,
               "Awaiting promise must be populated");

  std::uint32_t numBuffers = response->num_buffers();

  SharedHandle &numBuffersHandle = awaitingHandles->at(0);
  numBuffersHandle.reset(std::make_shared<std::uint32_t>(numBuffers));

  awaitingResponse.awaitingPromise->set_value();
}

void Server::handleCreateHostTensorResponse(
    const ::tt::runtime::distributed::flatbuffer::CreateHostTensorResponse
        *response,
    AwaitingResponseQueueEntry &awaitingResponse) {
  DEBUG_ASSERT(!awaitingResponse.awaitingHandles,
               "Awaiting handles should not be populated for CreateHostTensor");
  DEBUG_ASSERT(!awaitingResponse.awaitingPromise,
               "Awaiting promise should not be populated for CreateHostTensor");
}

void Server::handleGetLayoutResponse(
    const ::tt::runtime::distributed::flatbuffer::GetLayoutResponse *response,
    AwaitingResponseQueueEntry &awaitingResponse) {
  DEBUG_ASSERT(!awaitingResponse.awaitingHandles,
               "Awaiting handles should not be populated for GetLayout");
  DEBUG_ASSERT(!awaitingResponse.awaitingPromise,
               "Awaiting promise should not be populated for GetLayout");
}

void Server::handleToLayoutResponse(
    const ::tt::runtime::distributed::flatbuffer::ToLayoutResponse *response,
    AwaitingResponseQueueEntry &awaitingResponse) {
  DEBUG_ASSERT(!awaitingResponse.awaitingHandles,
               "Awaiting handles should not be populated for ToLayout");
  DEBUG_ASSERT(!awaitingResponse.awaitingPromise,
               "Awaiting promise should not be populated for ToLayout");
}

void Server::handleSubmitResponse(
    const ::tt::runtime::distributed::flatbuffer::SubmitResponse *response,
    AwaitingResponseQueueEntry &awaitingResponse) {
  DEBUG_ASSERT(!awaitingResponse.awaitingHandles,
               "Awaiting handles should not be populated for Submit");
  DEBUG_ASSERT(!awaitingResponse.awaitingPromise,
               "Awaiting promise should not be populated for Submit");
}

void Server::handleMemcpyResponse(
    const ::tt::runtime::distributed::flatbuffer::MemcpyResponse *response,
    AwaitingResponseQueueEntry &awaitingResponse) {
  if (!awaitingResponse.awaitingHandles && !awaitingResponse.awaitingPromise) {
    return;
  }

  DEBUG_ASSERT(awaitingResponse.awaitingHandles->size() == 1,
               "Awaiting handles must contain exactly one handle");
  DEBUG_ASSERT(awaitingResponse.awaitingPromise,
               "Awaiting promise must be populated");
  SharedHandle &dstHandle = awaitingResponse.awaitingHandles->at(0);

  DEBUG_ASSERT(response->data(),
               "Unexpected empty data from client in memcpy response");

  std::memcpy(dstHandle.get(), response->data()->data(),
              response->data()->size());
  awaitingResponse.awaitingPromise->set_value();
}

void Server::handleResponse(
    const ::tt::runtime::distributed::flatbuffer::Response *response,
    AwaitingResponseQueueEntry &awaitingResponse) {
  switch (response->type_type()) {
  case ::tt::runtime::distributed::flatbuffer::ResponseType::ErrorResponse: {
    return handleErrorResponse(response->type_as_ErrorResponse(),
                               awaitingResponse);
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::
      GetSystemDescResponse: {
    return handleGetSystemDescResponse(
        response->type_as_GetSystemDescResponse(), awaitingResponse);
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::
      OpenMeshDeviceResponse: {
    return handleOpenMeshDeviceResponse(
        response->type_as_OpenMeshDeviceResponse(), awaitingResponse);
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::
      CloseMeshDeviceResponse: {
    return handleCloseMeshDeviceResponse(
        response->type_as_CloseMeshDeviceResponse(), awaitingResponse);
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::
      CreateHostTensorResponse: {
    return handleCreateHostTensorResponse(
        response->type_as_CreateHostTensorResponse(), awaitingResponse);
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::
      GetLayoutResponse: {
    return handleGetLayoutResponse(response->type_as_GetLayoutResponse(),
                                   awaitingResponse);
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::ToLayoutResponse: {
    return handleToLayoutResponse(response->type_as_ToLayoutResponse(),
                                  awaitingResponse);
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::SubmitResponse: {
    return handleSubmitResponse(response->type_as_SubmitResponse(),
                                awaitingResponse);
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::
      GetNumShardsResponse: {
    return handleGetNumShardsResponse(response->type_as_GetNumShardsResponse(),
                                      awaitingResponse);
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::ToHostResponse: {
    return handleToHostResponse(response->type_as_ToHostResponse(),
                                awaitingResponse);
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::MemcpyResponse: {
    return handleMemcpyResponse(response->type_as_MemcpyResponse(),
                                awaitingResponse);
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::ShutdownResponse: {
    return handleShutdownResponse(response->type_as_ShutdownResponse(),
                                  awaitingResponse);
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::NONE: {
    LOG_FATAL("Unhandled response type: ",
              ::tt::runtime::distributed::flatbuffer::EnumNameResponseType(
                  response->type_type()));
  }
  }
}

} // namespace tt::runtime::distributed::server
