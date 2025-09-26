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
#endif
}

static void assertNoAwaitingState(const AwaitingResponseQueueEntry &entry,
                                  const std::string &apiName) {
  DEBUG_ASSERT(!entry.awaitingHandles,
               "Awaiting handles should not be populated for " + apiName);
  DEBUG_ASSERT(!entry.awaitingPromise,
               "Awaiting promise should not be populated for " + apiName);
}
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

Server::~Server() { shutdown(); }

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

void Server::setWriteTimeout(const std::chrono::seconds &timeout) {
  writeTimeout_ = timeout;
}

void Server::setReadTimeout(const std::chrono::seconds &timeout) {
  readTimeout_ = timeout;
}

void Server::setClientShutdownTimeout(const std::chrono::seconds &timeout) {
  clientShutdownTimeout_ = timeout;
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

  ::tt::runtime::Device outputDeviceHandle(getCurrentDeviceRuntime());

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

  ::tt::runtime::Layout outputLayoutHandle(getCurrentDeviceRuntime());

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

  return outputTensorHandle;
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
  DeviceRuntime currentRuntime = getCurrentDeviceRuntime();

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
  auto numShardsCommandBuilder =
      std::make_unique<::flatbuffers::FlatBufferBuilder>();

  SharedHandle numShardsHandle;

  auto awaitingHandles = std::make_unique<std::vector<SharedHandle>>();
  awaitingHandles->push_back(numShardsHandle);

  uint64_t numShardsCommandId = CommandFactory::buildGetNumShardsCommand(
      *numShardsCommandBuilder, tensorHandle);

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(
      numShardsCommandId, std::move(numShardsCommandBuilder),
      std::move(awaitingHandles), std::move(awaitingPromise));

  awaitingFuture.wait();

  auto toHostCommandBuilder =
      std::make_unique<::flatbuffers::FlatBufferBuilder>();

  std::uint32_t numShards = numShardsHandle.as<uint32_t>();

  std::vector<::tt::runtime::Tensor> outputHandles;
  outputHandles.reserve(numShards);
  for (std::uint32_t i = 0; i < numShards; i++) {
    outputHandles.push_back(
        ::tt::runtime::Tensor(nullptr, nullptr, getCurrentDeviceRuntime()));
  }

  uint64_t toHostCommandId = CommandFactory::buildToHostCommand(
      *toHostCommandBuilder, tensorHandle, untilize, blocking, outputHandles);

  pushToCommandAndResponseQueues(toHostCommandId,
                                 std::move(toHostCommandBuilder));

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
  while (!(shutdownRequested_.load(std::memory_order_relaxed) &&
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
  while (!(shutdownRequested_.load(std::memory_order_relaxed) &&
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
            std::move(awaitingResponseQueueEntry));
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

    handleResponse(response, std::move(awaitingResponseQueueEntry));
  }
}

// TODO (#4484): Need better error handling for when an error occurs
// Currently we just log a fatal error and exit the program
void Server::handleErrorResponse(
    const ::tt::runtime::distributed::flatbuffer::ErrorResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  if (awaitingPromise) {
    awaitingPromise->set_value();
  }
  LOG_FATAL("Error response received with message: ",
            response->message()->c_str());
}

void Server::handleGetSystemDescResponse(
    const ::tt::runtime::distributed::flatbuffer::GetSystemDescResponse
        *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  DEBUG_ASSERT(awaitingHandles && awaitingHandles->size() == 1,
               "GetSystemDesc: Awaiting handles must be populated and contain "
               "exactly one handle");

  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  const auto *systemDescVec = response->system_desc();

  std::shared_ptr<void> systemDescPtr =
      ::tt::runtime::utils::mallocShared(systemDescVec->size());
  std::memcpy(systemDescPtr.get(), systemDescVec->data(),
              systemDescVec->size());

  SharedHandle &systemDescHandle = awaitingHandles->at(0);
  systemDescHandle.reset(std::move(systemDescPtr));

  awaitingPromise->set_value();
}

void Server::handleOpenMeshDeviceResponse(
    const ::tt::runtime::distributed::flatbuffer::OpenMeshDeviceResponse
        *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "OpenMeshDevice");
}

void Server::handleCloseMeshDeviceResponse(
    const ::tt::runtime::distributed::flatbuffer::CloseMeshDeviceResponse
        *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "CloseMeshDevice");
}

void Server::handleGetNumShardsResponse(
    const ::tt::runtime::distributed::flatbuffer::GetNumShardsResponse
        *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  DEBUG_ASSERT(awaitingHandles && awaitingHandles->size() == 1,
               "GetSystemDesc: Awaiting handles must be populated and contain "
               "exactly one handle");

  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  std::uint32_t numBuffers = response->num_buffers();

  SharedHandle &numBuffersHandle = awaitingHandles->at(0);
  numBuffersHandle.reset(std::static_pointer_cast<void>(
      std::make_shared<std::uint32_t>(numBuffers)));

  awaitingPromise->set_value();
}

void Server::handleCreateHostTensorResponse(
    const ::tt::runtime::distributed::flatbuffer::CreateHostTensorResponse
        *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "CreateHostTensor");
}

void Server::handleGetLayoutResponse(
    const ::tt::runtime::distributed::flatbuffer::GetLayoutResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "GetLayout");
}

void Server::handleToLayoutResponse(
    const ::tt::runtime::distributed::flatbuffer::ToLayoutResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "ToLayout");
}

void Server::handleSubmitResponse(
    const ::tt::runtime::distributed::flatbuffer::SubmitResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "Submit");
}

void Server::handleToHostResponse(
    const ::tt::runtime::distributed::flatbuffer::ToHostResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "ToHost");
}

void Server::handleMemcpyResponse(
    const ::tt::runtime::distributed::flatbuffer::MemcpyResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  // Handle the case where the memcpy is performed asynchronously on the client
  // side
  if (!awaitingHandles && !awaitingPromise) {
    return;
  }

  DEBUG_ASSERT(response->data(),
               "Unexpected empty data from client in memcpy response");

  DEBUG_ASSERT(awaitingHandles && awaitingHandles->size() == 1,
               "Awaiting handles must contain exactly one handle");
  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  SharedHandle &dstHandle = awaitingHandles->at(0);

  std::memcpy(dstHandle.get(), response->data()->data(),
              response->data()->size());
  awaitingPromise->set_value();
}

void Server::handleShutdownResponse(
    const ::tt::runtime::distributed::flatbuffer::ShutdownResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  DEBUG_ASSERT(!awaitingHandles, "Awaiting handles must be empty");
  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  awaitingPromise->set_value();
}

void Server::handleResponse(
    const ::tt::runtime::distributed::flatbuffer::Response *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  switch (response->type_type()) {
  case ::tt::runtime::distributed::flatbuffer::ResponseType::ErrorResponse: {
    return handleErrorResponse(response->type_as_ErrorResponse(),
                               std::move(awaitingResponse));
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::
      GetSystemDescResponse: {
    return handleGetSystemDescResponse(
        response->type_as_GetSystemDescResponse(), std::move(awaitingResponse));
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::
      OpenMeshDeviceResponse: {
    return handleOpenMeshDeviceResponse(
        response->type_as_OpenMeshDeviceResponse(),
        std::move(awaitingResponse));
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::
      CloseMeshDeviceResponse: {
    return handleCloseMeshDeviceResponse(
        response->type_as_CloseMeshDeviceResponse(),
        std::move(awaitingResponse));
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::
      CreateHostTensorResponse: {
    return handleCreateHostTensorResponse(
        response->type_as_CreateHostTensorResponse(),
        std::move(awaitingResponse));
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::
      GetLayoutResponse: {
    return handleGetLayoutResponse(response->type_as_GetLayoutResponse(),
                                   std::move(awaitingResponse));
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::ToLayoutResponse: {
    return handleToLayoutResponse(response->type_as_ToLayoutResponse(),
                                  std::move(awaitingResponse));
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::SubmitResponse: {
    return handleSubmitResponse(response->type_as_SubmitResponse(),
                                std::move(awaitingResponse));
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::
      GetNumShardsResponse: {
    return handleGetNumShardsResponse(response->type_as_GetNumShardsResponse(),
                                      std::move(awaitingResponse));
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::ToHostResponse: {
    return handleToHostResponse(response->type_as_ToHostResponse(),
                                std::move(awaitingResponse));
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::MemcpyResponse: {
    return handleMemcpyResponse(response->type_as_MemcpyResponse(),
                                std::move(awaitingResponse));
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::ShutdownResponse: {
    return handleShutdownResponse(response->type_as_ShutdownResponse(),
                                  std::move(awaitingResponse));
  }
  case ::tt::runtime::distributed::flatbuffer::ResponseType::NONE: {
    LOG_FATAL("Unhandled response type: ",
              ::tt::runtime::distributed::flatbuffer::EnumNameResponseType(
                  response->type_type()));
  }
  }
}

void Server::shutdown() {
  LOG_INFO("Shutting down distributed server");

  LOG_ASSERT(!shutdownRequested_.load(std::memory_order_relaxed),
             "Server already shutdown before shutdown()");

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();
  auto command = CommandFactory::buildShutdownCommand(*commandBuilder);

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(command, std::move(commandBuilder),
                                 /*awaitingHandles=*/nullptr,
                                 std::move(awaitingPromise));

  // Wait for client shutdown
  awaitingFuture.wait();

  shutdownRequested_.store(true, std::memory_order_relaxed);
  commandDispatcherThread_.join();
  responseHandlerThread_.join();

  for (auto &exitCodeFuture : exitCodeFutures_) {
    auto exitCodeResult =
        exitCodeFuture.wait_for(std::chrono::seconds(clientShutdownTimeout_));
    LOG_ASSERT(exitCodeResult == std::future_status::ready,
               "Client subprocess failed to exit");

    int exitCode = exitCodeFuture.get();
    LOG_ASSERT(exitCode == 0,
               "Client subprocess failed with exit code: ", exitCode);
  }

  exitCodeFutures_.clear();
  LOG_INFO("Distributed server shutdown complete");
}

} // namespace tt::runtime::distributed::server
