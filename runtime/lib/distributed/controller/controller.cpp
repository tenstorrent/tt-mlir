// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/controller/controller.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/distributed/controller/command_factory.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/detail/distributed/utils/utils.h"
#include "tt/runtime/runtime.h"

namespace tt::runtime::distributed::controller {

namespace debug {
static void checkResponsesIdentical(const std::vector<SizedBuffer> &responses) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  const SizedBuffer &firstResponse = responses[0];
  for (const SizedBuffer &response : responses) {
    DEBUG_ASSERT(response.size() == firstResponse.size(),
                 "Responses sizes are not identical across workers");
    DEBUG_ASSERT(std::memcmp(response.data(), firstResponse.data(),
                             response.size()) == 0,
                 "Response data is not identical across workers");
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

Controller::~Controller() { shutdown(); }

void Controller::launchLocalSubprocess(uint16_t controllerPort) {
  constexpr size_t numWorkers = 1;
  LOG_ASSERT(!controllerSocket_, "Controller already launched");
  controllerSocket_ = std::make_unique<ControllerSocket>(controllerPort);
  uint16_t bindPort = controllerSocket_->port();

  std::string command =
      tt::runtime::distributed::utils::getWorkerExecutableCommand(bindPort);

  std::future<int> asyncFuture = std::async(
      std::launch::async, [command]() { return std::system(command.c_str()); });
  exitCodeFutures_.emplace_back(std::move(asyncFuture));

  workerConnections_ =
      controllerSocket_->connectToWorkers(numWorkers, writeTimeout_);
  launchCommandDispatcher();
  launchResponseHandler();
}

void Controller::setWriteTimeout(const std::chrono::seconds &timeout) {
  writeTimeout_ = timeout;
}

void Controller::setReadTimeout(const std::chrono::seconds &timeout) {
  readTimeout_ = timeout;
}

void Controller::setWorkerShutdownTimeout(const std::chrono::seconds &timeout) {
  workerShutdownTimeout_ = timeout;
}

SystemDesc Controller::getCurrentSystemDesc(
    std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType,
    std::optional<::tt::runtime::Device> deviceHandle) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildGetSystemDescCommand(
      *commandBuilder, dispatchCoreType, deviceHandle);

  auto awaitingHandles = std::make_unique<std::vector<std::shared_ptr<void>>>();
  std::shared_ptr<::tt::runtime::SystemDesc> systemDescHandle(
      /*handle=*/nullptr);
  awaitingHandles->push_back(std::static_pointer_cast<void>(systemDescHandle));

  // This command will execute synchronously and block until the response is
  // received The systemDesc handle will get populated by the response handler
  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(commandId, std::move(commandBuilder),
                                 std::move(awaitingHandles),
                                 std::move(awaitingPromise));

  awaitingFuture.wait();

  return *systemDescHandle;
}

::tt::runtime::Device
Controller::openMeshDevice(const ::tt::runtime::MeshDeviceOptions &options) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  ::tt::runtime::Device outputDeviceHandle(getCurrentDeviceRuntime());

  uint64_t commandId = CommandFactory::buildOpenMeshDeviceCommand(
      *commandBuilder, outputDeviceHandle, options);

  pushToCommandAndResponseQueues(commandId, std::move(commandBuilder));

  return outputDeviceHandle;
}

void Controller::closeMeshDevice(const ::tt::runtime::Device &deviceHandle) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildCloseMeshDeviceCommand(
      *commandBuilder, deviceHandle);

  pushToCommandAndResponseQueues(commandId, std::move(commandBuilder));
}

::tt::runtime::Tensor Controller::createOwnedHostTensor(
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
Controller::getLayout(const ::tt::runtime::Binary &executableHandle,
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
Controller::toLayout(const ::tt::runtime::Tensor &tensorHandle,
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
Controller::submit(const ::tt::runtime::Device &deviceHandle,
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
Controller::toHost(const ::tt::runtime::Tensor &tensorHandle, bool untilize,
                   bool blocking) {
  // For toHost, we need to get the number of shards this tensor has first
  // so we can create the correct number of output tensors
  // Performance wise, this is not ideal, but it should be ok for now since
  // toHost is typically the last step and is not on the hot path
  auto numShardsCommandBuilder =
      std::make_unique<::flatbuffers::FlatBufferBuilder>();

  auto numShardsHandle = std::make_shared<std::uint32_t>(0);

  auto awaitingHandles = std::make_unique<std::vector<std::shared_ptr<void>>>();
  awaitingHandles->push_back(std::static_pointer_cast<void>(numShardsHandle));

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

  std::uint32_t numShards = *numShardsHandle;

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

void Controller::memcpy(void *dst, const ::tt::runtime::Tensor &srcHandle,
                        std::optional<tt::target::DataType> targetDataType) {
  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildMemcpyCommand(
      *commandBuilder, srcHandle, /*dstTensor=*/std::nullopt, targetDataType);

  auto awaitingHandles = std::make_unique<std::vector<std::shared_ptr<void>>>();

  std::shared_ptr<void> dstPtr = ::tt::runtime::utils::unsafeBorrowShared(dst);

  awaitingHandles->push_back(dstPtr);

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(commandId, std::move(commandBuilder),
                                 std::move(awaitingHandles),
                                 std::move(awaitingPromise));

  awaitingFuture.wait();
}

void Controller::pushToCommandAndResponseQueues(
    uint64_t commandId,
    std::unique_ptr<::flatbuffers::FlatBufferBuilder> commandBuilder,
    std::unique_ptr<std::vector<std::shared_ptr<void>>> awaitingHandles,
    std::unique_ptr<std::promise<void>> awaitingPromise) {
  // Need to push to the response queue first to ensure the command response is
  // waited for before the command is dispatched
  awaitingResponseQueue_.push(std::make_unique<AwaitingResponseQueueEntry>(
      commandId, std::move(awaitingHandles), std::move(awaitingPromise)));
  commandQueue_.push(
      std::make_unique<CommandQueueEntry>(std::move(commandBuilder)));
}

void Controller::launchCommandDispatcher() {
  LOG_ASSERT(!commandDispatcherThread_.joinable(),
             "Command dispatcher thread already running");
  commandDispatcherThread_ = std::thread([this]() { dispatchCommands(); });
}

void Controller::dispatchCommand(
    const ::flatbuffers::FlatBufferBuilder &commandBuilder) {
  std::vector<std::future<ssize_t>> writeFutures;
  writeFutures.reserve(workerConnections_.size());
  size_t commandSize = commandBuilder.GetSize();
  LOG_ASSERT(commandSize > 0, "Unexpected empty command");

  // Currently broadcasting command to all workers
  // Will need updates later to support multi mesh and MPMD
  for (const auto &workerConnection : workerConnections_) {
    writeFutures.emplace_back(workerConnection->sizePrefixedWriteAsync(
        commandBuilder.GetBufferPointer(), commandSize));
  }

  for (const std::future<ssize_t> &writeFuture : writeFutures) {
    if (writeFuture.wait_for(writeTimeout_) == std::future_status::timeout) {
      LOG_FATAL("Write timeout occurred while sending command to worker");
    }
  }
}

void Controller::dispatchCommands() {
  std::vector<std::future<ssize_t>> writeFutures;
  writeFutures.reserve(workerConnections_.size());
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

void Controller::launchResponseHandler() {
  LOG_ASSERT(!responseHandlerThread_.joinable(),
             "Response handler thread already running");
  responseHandlerThread_ = std::thread([this]() { handleResponses(); });
}

void Controller::handleResponses() {
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
    readFutures.reserve(workerConnections_.size());

    for (const auto &workerConnection : workerConnections_) {
      readFutures.push_back(workerConnection->sizePrefixedReadAsync());
    }

    std::vector<SizedBuffer> responseBuffers;
    responseBuffers.reserve(readFutures.size());
    for (std::future<SizedBuffer> &readFuture : readFutures) {
      if (readFuture.wait_for(readTimeout_) == std::future_status::timeout) {
        LOG_FATAL("Read timeout occurred while receiving response from worker");
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

    // Since we currently only support SPMD, worker responses should be
    // identical therefore we can just handle the first response
    const ::tt::runtime::distributed::flatbuffer::Response *response =
        getResponse(responseBuffers[0]);
    LOG_ASSERT(
        response->command_id() == awaitingResponseQueueEntry->commandId,
        "Response command id does not match awaiting response command id");

    handleResponse(response, std::move(awaitingResponseQueueEntry));
  }
}

// TODO (#5136): Need better error handling for when an error occurs
// Currently we just log a fatal error and exit the program
void Controller::handleErrorResponse(
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

void Controller::handleGetSystemDescResponse(
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

  std::shared_ptr<::tt::runtime::SystemDesc> systemDesc =
      std::static_pointer_cast<::tt::runtime::SystemDesc>(
          awaitingHandles->at(0));
  systemDesc->handle = std::move(systemDescPtr);

  awaitingPromise->set_value();
}

void Controller::handleOpenMeshDeviceResponse(
    const ::tt::runtime::distributed::flatbuffer::OpenMeshDeviceResponse
        *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "OpenMeshDevice");
}

void Controller::handleCloseMeshDeviceResponse(
    const ::tt::runtime::distributed::flatbuffer::CloseMeshDeviceResponse
        *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "CloseMeshDevice");
}

void Controller::handleGetNumShardsResponse(
    const ::tt::runtime::distributed::flatbuffer::GetNumShardsResponse
        *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  DEBUG_ASSERT(awaitingHandles && awaitingHandles->size() == 1,
               "GetSystemDesc: Awaiting handles must be populated and contain "
               "exactly one handle");

  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  std::uint32_t numShards = response->num_shards();

  std::shared_ptr<std::uint32_t> numShardsHandle =
      std::static_pointer_cast<std::uint32_t>(awaitingHandles->at(0));
  *numShardsHandle = numShards;

  awaitingPromise->set_value();
}

void Controller::handleCreateHostTensorResponse(
    const ::tt::runtime::distributed::flatbuffer::CreateHostTensorResponse
        *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "CreateHostTensor");
}

void Controller::handleGetLayoutResponse(
    const ::tt::runtime::distributed::flatbuffer::GetLayoutResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "GetLayout");
}

void Controller::handleToLayoutResponse(
    const ::tt::runtime::distributed::flatbuffer::ToLayoutResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "ToLayout");
}

void Controller::handleSubmitResponse(
    const ::tt::runtime::distributed::flatbuffer::SubmitResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "Submit");
}

void Controller::handleToHostResponse(
    const ::tt::runtime::distributed::flatbuffer::ToHostResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::assertNoAwaitingState(*awaitingResponse, "ToHost");
}

void Controller::handleMemcpyResponse(
    const ::tt::runtime::distributed::flatbuffer::MemcpyResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  // Handle the case where the memcpy is performed asynchronously on the worker
  // side
  if (!awaitingHandles && !awaitingPromise) {
    return;
  }

  DEBUG_ASSERT(response->data(),
               "Unexpected empty data from worker in memcpy response");

  DEBUG_ASSERT(awaitingHandles && awaitingHandles->size() == 1,
               "Awaiting handles must contain exactly one handle");
  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  std::shared_ptr<void> dstHandle = awaitingHandles->at(0);

  std::memcpy(dstHandle.get(), response->data()->data(),
              response->data()->size());
  awaitingPromise->set_value();
}

void Controller::handleShutdownResponse(
    const ::tt::runtime::distributed::flatbuffer::ShutdownResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  DEBUG_ASSERT(!awaitingHandles, "Awaiting handles must be empty");
  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  awaitingPromise->set_value();
}

void Controller::handleResponse(
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

void Controller::shutdown() {
  LOG_INFO("Shutting down distributed controller");

  LOG_ASSERT(!shutdownRequested_.load(std::memory_order_relaxed),
             "Controller already shutdown before shutdown()");

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();
  auto command = CommandFactory::buildShutdownCommand(*commandBuilder);

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(command, std::move(commandBuilder),
                                 /*awaitingHandles=*/nullptr,
                                 std::move(awaitingPromise));

  // Wait for worker shutdown
  awaitingFuture.wait();

  shutdownRequested_.store(true, std::memory_order_relaxed);
  commandDispatcherThread_.join();
  responseHandlerThread_.join();

  for (auto &exitCodeFuture : exitCodeFutures_) {
    auto exitCodeResult =
        exitCodeFuture.wait_for(std::chrono::seconds(workerShutdownTimeout_));
    LOG_ASSERT(exitCodeResult == std::future_status::ready,
               "Worker subprocess failed to exit");

    int exitCode = exitCodeFuture.get();
    LOG_ASSERT(exitCode == 0,
               "Worker subprocess failed with exit code: ", exitCode);
  }

  exitCodeFutures_.clear();
  LOG_INFO("Distributed controller shutdown complete");
}

} // namespace tt::runtime::distributed::controller
