// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/controller/controller.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/detail/distributed/controller/command_factory.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/detail/distributed/utils/utils.h"
#include "tt/runtime/runtime.h"
namespace tt::runtime::distributed::controller {

namespace fb = ::tt::runtime::distributed::flatbuffer;

static const fb::Response *getResponse(const SizedBuffer &response) {
  bool isDistributedResponse = fb::ResponseBufferHasIdentifier(response.data());
  LOG_ASSERT(isDistributedResponse, "Response is not a distributed response");
  return fb::GetResponse(response.data());
}

static bool isResponseType(const SizedBuffer &response,
                           fb::ResponseType responseType) {
  return getResponse(response)->type_type() == responseType;
}

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

static void checkResponseTypes(const std::vector<SizedBuffer> &responseBuffers,
                               fb::ResponseType responseType) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  for (const SizedBuffer &responseBuffer : responseBuffers) {
    DEBUG_ASSERT(isResponseType(responseBuffer, responseType),
                 "Response is not a " +
                     std::string(EnumNameResponseType(responseType)));
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

Controller::~Controller() {
  auto currentState = controllerState_.load(std::memory_order_relaxed);
  if (currentState != ControllerState::Uninitialized &&
      currentState != ControllerState::Shutdown) {
    shutdown();
  }
}

void Controller::launch(const ::tt::runtime::DistributedOptions &options) {

  LOG_ASSERT(controllerState_.load(std::memory_order_relaxed) ==
                 ControllerState::Uninitialized,
             "Controller must be uninitialized to launch");

  controllerSocket_ =
      std::make_unique<ControllerSocket>(options.controllerPort);
  uint16_t bindPort = controllerSocket_->port();

  controllerState_.store(ControllerState::ControllerSocketBound,
                         std::memory_order_relaxed);

  size_t numWorkers;
  std::string command;

  switch (options.mode) {
  case DistributedMode::LocalSubprocess: {
    LOG_ASSERT(
        !options.multiProcessArgs.has_value(),
        "MultiProcess args should not be populated for LocalSubprocess mode");
    numWorkers = 1;
    command =
        ::tt::runtime::distributed::utils::getWorkerExecutableCommand(bindPort);
    break;
  }
  case DistributedMode::MultiProcess: {
    LOG_ASSERT(options.multiProcessArgs.has_value(),
               "MultiProcess mode requires MultiProcessArgs to be populated");
    const ::tt::runtime::MultiProcessArgs &multiProcessArgs =
        options.multiProcessArgs.value();
    const std::string rankBindingPath = multiProcessArgs.getRankBindingPath();
    LOG_ASSERT(!rankBindingPath.empty(), "Rank binding path cannot be empty");
    LOG_ASSERT(std::filesystem::exists(rankBindingPath), "Rank binding path ",
               rankBindingPath, " does not exist");
    numWorkers =
        ::tt::runtime::distributed::utils::getNumProcesses(rankBindingPath);
    command = ::tt::runtime::distributed::utils::getTTRunCommand(
        bindPort, multiProcessArgs);
    break;
  }
  }
  LOG_INFO("Launching distributed controller with command: ", command, " on ",
           numWorkers, " worker(s)");

  std::future<int> asyncFuture = std::async(
      std::launch::async, [command]() { return std::system(command.c_str()); });
  exitCodeFuture_ = std::move(asyncFuture);

  controllerState_.store(ControllerState::CommandLaunched,
                         std::memory_order_relaxed);

  workerConnections_ =
      controllerSocket_->connectToWorkers(numWorkers, writeTimeout_);

  controllerState_.store(ControllerState::ConnectedToWorkers,
                         std::memory_order_relaxed);

  launchCommandDispatcher();
  launchResponseHandler();

  configureRuntimeContext();

  controllerState_.store(ControllerState::FullyOperational,
                         std::memory_order_relaxed);
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
  std::shared_ptr<::tt::runtime::SystemDesc> systemDescHandle =
      std::make_shared<::tt::runtime::SystemDesc>(/*handle=*/nullptr);
  awaitingHandles->push_back(std::static_pointer_cast<void>(systemDescHandle));

  // This command will execute synchronously and block until the response is
  // received The systemDesc handle will get populated by the response handler
  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(
      commandId, fb::CommandType::GetSystemDescCommand,
      std::move(commandBuilder), std::move(awaitingHandles),
      std::move(awaitingPromise));

  awaitingFuture.wait();

  return *systemDescHandle;
}

void Controller::setFabricConfig(
    const ::tt::runtime::FabricConfig &fabricConfig) {
  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildSetFabricConfigCommand(
      *commandBuilder, fabricConfig);

  pushToCommandAndResponseQueues(commandId,
                                 fb::CommandType::SetFabricConfigCommand,
                                 std::move(commandBuilder));
}

size_t Controller::getNumAvailableDevices() {
  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId =
      CommandFactory::buildGetNumAvailableDevicesCommand(*commandBuilder);

  auto awaitingHandles = std::make_unique<std::vector<std::shared_ptr<void>>>();
  std::shared_ptr<size_t> numDevicesHandle = std::make_shared<size_t>(0);
  awaitingHandles->push_back(std::static_pointer_cast<void>(numDevicesHandle));

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(
      commandId, fb::CommandType::GetNumAvailableDevicesCommand,
      std::move(commandBuilder), std::move(awaitingHandles),
      std::move(awaitingPromise));

  awaitingFuture.wait();

  return *numDevicesHandle;
}

::tt::runtime::Device
Controller::openMeshDevice(const ::tt::runtime::MeshDeviceOptions &options) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  ::tt::runtime::Device outputDeviceHandle(getCurrentDeviceRuntime());

  uint64_t commandId = CommandFactory::buildOpenMeshDeviceCommand(
      *commandBuilder, outputDeviceHandle, options);

  pushToCommandAndResponseQueues(commandId,
                                 fb::CommandType::OpenMeshDeviceCommand,
                                 std::move(commandBuilder));

  return outputDeviceHandle;
}

void Controller::closeMeshDevice(::tt::runtime::Device &deviceHandle) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildCloseMeshDeviceCommand(
      *commandBuilder, deviceHandle);

  pushToCommandAndResponseQueues(commandId,
                                 fb::CommandType::CloseMeshDeviceCommand,
                                 std::move(commandBuilder));
}

::tt::runtime::Device Controller::createSubMeshDevice(
    const ::tt::runtime::Device &parentMesh,
    const std::vector<uint32_t> &meshShape,
    const std::optional<const std::vector<uint32_t>> &meshOffset) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  ::tt::runtime::Device outputSubMeshHandle(getCurrentDeviceRuntime());

  uint64_t commandId = CommandFactory::buildCreateSubMeshDeviceCommand(
      *commandBuilder, parentMesh, outputSubMeshHandle, meshShape, meshOffset);

  pushToCommandAndResponseQueues(commandId,
                                 fb::CommandType::CreateSubMeshDeviceCommand,
                                 std::move(commandBuilder));

  return outputSubMeshHandle;
}

void Controller::releaseSubMeshDevice(const ::tt::runtime::Device &subMesh) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildReleaseSubMeshDeviceCommand(
      *commandBuilder, subMesh);

  pushToCommandAndResponseQueues(commandId,
                                 fb::CommandType::ReleaseSubMeshDeviceCommand,
                                 std::move(commandBuilder));
}

std::vector<uint32_t>
Controller::getMeshShape(const ::tt::runtime::Device &deviceHandle) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId =
      CommandFactory::buildGetMeshShapeCommand(*commandBuilder, deviceHandle);

  auto awaitingHandles = std::make_unique<std::vector<std::shared_ptr<void>>>();
  std::shared_ptr<std::vector<uint32_t>> shapeHandle =
      std::make_shared<std::vector<uint32_t>>();
  awaitingHandles->push_back(std::static_pointer_cast<void>(shapeHandle));

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(
      commandId, fb::CommandType::GetMeshShapeCommand,
      std::move(commandBuilder), std::move(awaitingHandles),
      std::move(awaitingPromise));

  awaitingFuture.wait();

  return *shapeHandle;
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

  pushToCommandAndResponseQueues(commandId,
                                 fb::CommandType::CreateHostTensorCommand,
                                 std::move(commandBuilder));

  return outputTensorHandle;
}

bool Controller::isTensorAllocated(const ::tt::runtime::Tensor &tensorHandle) {
  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildIsTensorAllocatedCommand(
      *commandBuilder, tensorHandle);

  auto awaitingHandles = std::make_unique<std::vector<std::shared_ptr<void>>>();
  auto allocatedHandle = std::make_shared<bool>(false);
  awaitingHandles->push_back(std::static_pointer_cast<void>(allocatedHandle));

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(
      commandId, fb::CommandType::IsTensorAllocatedCommand,
      std::move(commandBuilder), std::move(awaitingHandles),
      std::move(awaitingPromise));

  awaitingFuture.wait();

  return *allocatedHandle;
}

std::uint32_t
Controller::getTensorVolume(const ::tt::runtime::Tensor &tensorHandle) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildGetTensorVolumeCommand(
      *commandBuilder, tensorHandle);

  auto volumeHandle = std::make_shared<std::uint32_t>(0);
  auto awaitingHandles = std::make_unique<std::vector<std::shared_ptr<void>>>();
  awaitingHandles->push_back(std::static_pointer_cast<void>(volumeHandle));

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(
      commandId, fb::CommandType::GetTensorVolumeCommand,
      std::move(commandBuilder), std::move(awaitingHandles),
      std::move(awaitingPromise));

  awaitingFuture.wait();

  return *volumeHandle;
}

bool Controller::getTensorRetain(const ::tt::runtime::Tensor &tensorHandle) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildGetTensorRetainCommand(
      *commandBuilder, tensorHandle);

  auto retainHandle = std::make_shared<bool>(false);

  auto awaitingHandles = std::make_unique<std::vector<std::shared_ptr<void>>>();
  awaitingHandles->push_back(std::static_pointer_cast<void>(retainHandle));

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(
      commandId, fb::CommandType::GetTensorRetainCommand,
      std::move(commandBuilder), std::move(awaitingHandles),
      std::move(awaitingPromise));

  awaitingFuture.wait();

  return *retainHandle;
}

void Controller::setTensorRetain(const ::tt::runtime::Tensor &tensorHandle,
                                 bool retain) {
  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildSetTensorRetainCommand(
      *commandBuilder, tensorHandle, retain);

  pushToCommandAndResponseQueues(commandId,
                                 fb::CommandType::SetTensorRetainCommand,
                                 std::move(commandBuilder));
}

::tt::runtime::Layout
Controller::getLayout(const ::tt::runtime::Binary &executableHandle,
                      std::uint32_t programIndex, std::uint32_t inputIndex) {
  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  ::tt::runtime::Layout outputLayoutHandle(getCurrentDeviceRuntime());

  uint64_t commandId = CommandFactory::buildGetLayoutCommand(
      *commandBuilder, executableHandle, programIndex, inputIndex,
      outputLayoutHandle);

  pushToCommandAndResponseQueues(commandId, fb::CommandType::GetLayoutCommand,
                                 std::move(commandBuilder));

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

  pushToCommandAndResponseQueues(commandId, fb::CommandType::ToLayoutCommand,
                                 std::move(commandBuilder));

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

  pushToCommandAndResponseQueues(commandId, fb::CommandType::SubmitCommand,
                                 std::move(commandBuilder));

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
      numShardsCommandId, fb::CommandType::GetNumShardsCommand,
      std::move(numShardsCommandBuilder), std::move(awaitingHandles),
      std::move(awaitingPromise));

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
                                 fb::CommandType::ToHostCommand,
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

  pushToCommandAndResponseQueues(
      commandId, fb::CommandType::MemcpyCommand, std::move(commandBuilder),
      std::move(awaitingHandles), std::move(awaitingPromise));

  awaitingFuture.wait();
}

void Controller::memcpy(const ::tt::runtime::Tensor &dstHandle,
                        const ::tt::runtime::Tensor &srcHandle) {
  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId =
      CommandFactory::buildMemcpyCommand(*commandBuilder, srcHandle, dstHandle);

  pushToCommandAndResponseQueues(commandId, fb::CommandType::MemcpyCommand,
                                 std::move(commandBuilder));
}

void Controller::deallocateTensor(::tt::runtime::Tensor &tensorHandle,
                                  bool force) {

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  uint64_t commandId = CommandFactory::buildDeallocateTensorCommand(
      *commandBuilder, tensorHandle, force);

  pushToCommandAndResponseQueues(commandId,
                                 fb::CommandType::DeallocateTensorCommand,
                                 std::move(commandBuilder));
}

void Controller::pushToCommandAndResponseQueues(
    uint64_t commandId, const fb::CommandType &commandType,
    std::unique_ptr<::flatbuffers::FlatBufferBuilder> commandBuilder,
    std::unique_ptr<std::vector<std::shared_ptr<void>>> awaitingHandles,
    std::unique_ptr<std::promise<void>> awaitingPromise) {
  LOG_DEBUG("Pushing command id: ", commandId,
            " and command type: ", fb::EnumNameCommandType(commandType),
            " to command and response queues");
  // Need to push to the response queue first to ensure the command response is
  // waited for before the command is dispatched
  awaitingResponseQueue_.push(std::make_unique<AwaitingResponseQueueEntry>(
      commandId, commandType, std::move(awaitingHandles),
      std::move(awaitingPromise)));
  commandQueue_.push(
      std::make_unique<CommandQueueEntry>(std::move(commandBuilder)));
}

void Controller::launchCommandDispatcher() {
  LOG_ASSERT(!commandDispatcherThread_.joinable(),
             "Command dispatcher thread already running");
  commandDispatcherThread_ = std::thread([this]() { processCommandQueue(); });
  controllerState_.store(ControllerState::DispatcherReady,
                         std::memory_order_relaxed);
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

void Controller::processCommandQueue() {
  std::vector<std::future<ssize_t>> writeFutures;
  writeFutures.reserve(workerConnections_.size());
  while (!(controllerState_.load(std::memory_order_relaxed) ==
               ControllerState::ShuttingDown &&
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
  responseHandlerThread_ = std::thread([this]() { processResponseQueue(); });
  controllerState_.store(ControllerState::ResponseHandlerReady,
                         std::memory_order_relaxed);
}

void Controller::processResponseQueue() {
  while (!(controllerState_.load(std::memory_order_relaxed) ==
               ControllerState::ShuttingDown &&
           awaitingResponseQueue_.empty())) {
    std::optional<std::unique_ptr<AwaitingResponseQueueEntry>>
        awaitingResponseQueueEntryOpt = awaitingResponseQueue_.popWithTimeout();
    if (!awaitingResponseQueueEntryOpt.has_value()) {
      continue;
    }

    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponseQueueEntry =
        std::move(awaitingResponseQueueEntryOpt.value());

    [[maybe_unused]] uint64_t commandId = awaitingResponseQueueEntry->commandId;
    [[maybe_unused]] fb::CommandType commandType =
        awaitingResponseQueueEntry->commandType;

    LOG_DEBUG("Handling response for command id: ", commandId,
              " and command type: ", fb::EnumNameCommandType(commandType));

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
      if (isResponseType(responseBuffer, fb::ResponseType::ErrorResponse)) {
        handleErrorResponse(
            getResponse(responseBuffer)->type_as_ErrorResponse(),
            std::move(awaitingResponseQueueEntry));
      }
    }

    handleResponse(responseBuffers, std::move(awaitingResponseQueueEntry));

    LOG_DEBUG("Finished handling response for command id: ", commandId,
              " and command type: ", fb::EnumNameCommandType(commandType));
  }
}

void Controller::configureRuntimeContext() {
  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  auto commandId = CommandFactory::buildConfigureRuntimeContextCommand(
      *commandBuilder, ::tt::runtime::RuntimeContext::instance().getMlirHome(),
      ::tt::runtime::RuntimeContext::instance().getMetalHome(),
      ::tt::runtime::RuntimeContext::instance().getCurrentDeviceRuntime());

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(
      commandId, fb::CommandType::ConfigureRuntimeContextCommand,
      std::move(commandBuilder), /*awaitingHandles=*/nullptr,
      std::move(awaitingPromise));

  awaitingFuture.wait();

  controllerState_.store(ControllerState::RuntimeContextConfigured,
                         std::memory_order_relaxed);
}

// TODO (#5136): Need better error handling for when an error occurs
// Currently we just log a fatal error and exit the program
void Controller::handleErrorResponse(
    const fb::ErrorResponse *response,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  if (awaitingPromise) {
    awaitingPromise->set_value();
  }

  LOG_FATAL("Error response received with message: ",
            response->message()->c_str());
}

void Controller::handleConfigureRuntimeContextResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::ConfigureRuntimeContextResponse);

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  DEBUG_ASSERT(
      !awaitingHandles,
      "ConfigureRuntimeContext: There shouldn't be any awaiting handles");

  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  awaitingPromise->set_value();
}

void Controller::handleGetSystemDescResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::GetSystemDescResponse);

  const fb::GetSystemDescResponse *response =
      getResponse(responseBuffers[0])->type_as_GetSystemDescResponse();

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

void Controller::handleSetFabricConfigResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::SetFabricConfigResponse);

  debug::assertNoAwaitingState(*awaitingResponse, "SetFabricConfig");
}

void Controller::handleGetNumAvailableDevicesResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::GetNumAvailableDevicesResponse);

  // Number of devices is local to each host process, so we need to sum them up
  uint32_t numDevices = 0;
  for (const SizedBuffer &responseBuffer : responseBuffers) {
    const fb::GetNumAvailableDevicesResponse *response =
        getResponse(responseBuffer)->type_as_GetNumAvailableDevicesResponse();
    numDevices += response->num_devices();
  }

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  DEBUG_ASSERT(
      awaitingHandles && awaitingHandles->size() == 1,
      "GetNumAvailableDevices: Awaiting handles must be populated and contain "
      "exactly one handle");

  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  std::shared_ptr<size_t> numDevicesHandle =
      std::static_pointer_cast<size_t>(awaitingHandles->at(0));
  *numDevicesHandle = numDevices;

  awaitingPromise->set_value();
}

void Controller::handleOpenMeshDeviceResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::OpenMeshDeviceResponse);

  debug::assertNoAwaitingState(*awaitingResponse, "OpenMeshDevice");
}

void Controller::handleCloseMeshDeviceResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::CloseMeshDeviceResponse);

  debug::assertNoAwaitingState(*awaitingResponse, "CloseMeshDevice");
}

void Controller::handleCreateSubMeshDeviceResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::CreateSubMeshDeviceResponse);

  debug::assertNoAwaitingState(*awaitingResponse, "CreateSubMeshDevice");
}

void Controller::handleReleaseSubMeshDeviceResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::ReleaseSubMeshDeviceResponse);

  debug::assertNoAwaitingState(*awaitingResponse, "ReleaseSubMeshDevice");
}

void Controller::handleGetMeshShapeResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::GetMeshShapeResponse);

  const fb::GetMeshShapeResponse *response =
      getResponse(responseBuffers[0])->type_as_GetMeshShapeResponse();

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  DEBUG_ASSERT(awaitingHandles && awaitingHandles->size() == 1,
               "GetMeshShape: Awaiting handles must be populated and contain "
               "exactly one handle");

  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  std::shared_ptr<std::vector<uint32_t>> shapeHandle =
      std::static_pointer_cast<std::vector<uint32_t>>(awaitingHandles->at(0));

  for (const uint32_t &dim : *response->shape()) {
    shapeHandle->push_back(dim);
  }

  awaitingPromise->set_value();
}

void Controller::handleGetNumShardsResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::GetNumShardsResponse);

  const fb::GetNumShardsResponse *response =
      getResponse(responseBuffers[0])->type_as_GetNumShardsResponse();

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
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::CreateHostTensorResponse);

  debug::assertNoAwaitingState(*awaitingResponse, "CreateHostTensor");
}

void Controller::handleIsTensorAllocatedResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::IsTensorAllocatedResponse);

  const fb::IsTensorAllocatedResponse *response =
      getResponse(responseBuffers[0])->type_as_IsTensorAllocatedResponse();

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();
  DEBUG_ASSERT(
      awaitingHandles && awaitingHandles->size() == 1,
      "IsTensorAllocated: Awaiting handles must be populated and contain "
      "exactly one handle");
  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  std::shared_ptr<bool> allocatedHandle =
      std::static_pointer_cast<bool>(awaitingHandles->at(0));
  *allocatedHandle = response->allocated();

  awaitingPromise->set_value();
}

void Controller::handleGetTensorVolumeResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::GetTensorVolumeResponse);

  const fb::GetTensorVolumeResponse *response =
      getResponse(responseBuffers[0])->type_as_GetTensorVolumeResponse();

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();
  DEBUG_ASSERT(
      awaitingHandles && awaitingHandles->size() == 1,
      "GetTensorVolume: Awaiting handles must be populated and contain "
      "exactly one handle");
  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  std::shared_ptr<std::uint32_t> volumeHandle =
      std::static_pointer_cast<std::uint32_t>(awaitingHandles->at(0));
  *volumeHandle = response->volume();

  awaitingPromise->set_value();
}

void Controller::handleGetTensorRetainResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::GetTensorRetainResponse);

  const fb::GetTensorRetainResponse *response =
      getResponse(responseBuffers[0])->type_as_GetTensorRetainResponse();

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();
  DEBUG_ASSERT(
      awaitingHandles && awaitingHandles->size() == 1,
      "GetTensorRetain: Awaiting handles must be populated and contain "
      "exactly one handle");
  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  std::shared_ptr<bool> retainHandle =
      std::static_pointer_cast<bool>(awaitingHandles->at(0));
  *retainHandle = response->retain();
  awaitingPromise->set_value();
}

void Controller::handleSetTensorRetainResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::SetTensorRetainResponse);

  debug::assertNoAwaitingState(*awaitingResponse, "SetTensorRetain");
}

void Controller::handleGetLayoutResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::GetLayoutResponse);

  debug::assertNoAwaitingState(*awaitingResponse, "GetLayout");
}

void Controller::handleToLayoutResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::ToLayoutResponse);

  debug::assertNoAwaitingState(*awaitingResponse, "ToLayout");
}

void Controller::handleSubmitResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers, fb::ResponseType::SubmitResponse);

  debug::assertNoAwaitingState(*awaitingResponse, "Submit");
}

void Controller::handleToHostResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers, fb::ResponseType::ToHostResponse);

  debug::assertNoAwaitingState(*awaitingResponse, "ToHost");
}

void Controller::handleMemcpyResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers, fb::ResponseType::MemcpyResponse);

  const fb::MemcpyResponse *response =
      getResponse(responseBuffers[0])->type_as_MemcpyResponse();

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

void Controller::handleDeallocateTensorResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::DeallocateTensorResponse);

  debug::assertNoAwaitingState(*awaitingResponse, "DeallocateTensor");
}

void Controller::handleShutdownResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {

  debug::checkResponsesIdentical(responseBuffers);

  debug::checkResponseTypes(responseBuffers,
                            fb::ResponseType::ShutdownResponse);

  auto [awaitingHandles, awaitingPromise] =
      awaitingResponse->popAwaitingState();

  DEBUG_ASSERT(!awaitingHandles, "Awaiting handles must be empty");
  DEBUG_ASSERT(awaitingPromise, "Awaiting promise must be populated");

  awaitingPromise->set_value();
}

void Controller::handleResponse(
    const std::vector<SizedBuffer> &responseBuffers,
    std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse) {
  switch (awaitingResponse->commandType) {
  case fb::CommandType::ConfigureRuntimeContextCommand: {
    return handleConfigureRuntimeContextResponse(responseBuffers,
                                                 std::move(awaitingResponse));
  }
  case fb::CommandType::GetSystemDescCommand: {
    return handleGetSystemDescResponse(responseBuffers,
                                       std::move(awaitingResponse));
  }
  case fb::CommandType::SetFabricConfigCommand: {
    return handleSetFabricConfigResponse(responseBuffers,
                                         std::move(awaitingResponse));
  }
  case fb::CommandType::GetNumAvailableDevicesCommand: {
    return handleGetNumAvailableDevicesResponse(responseBuffers,
                                                std::move(awaitingResponse));
  }
  case fb::CommandType::OpenMeshDeviceCommand: {
    return handleOpenMeshDeviceResponse(responseBuffers,
                                        std::move(awaitingResponse));
  }
  case fb::CommandType::CloseMeshDeviceCommand: {
    return handleCloseMeshDeviceResponse(responseBuffers,
                                         std::move(awaitingResponse));
  }
  case fb::CommandType::CreateSubMeshDeviceCommand: {
    return handleCreateSubMeshDeviceResponse(responseBuffers,
                                             std::move(awaitingResponse));
  }
  case fb::CommandType::ReleaseSubMeshDeviceCommand: {
    return handleReleaseSubMeshDeviceResponse(responseBuffers,
                                              std::move(awaitingResponse));
  }
  case fb::CommandType::GetMeshShapeCommand: {
    return handleGetMeshShapeResponse(responseBuffers,
                                      std::move(awaitingResponse));
  }
  case fb::CommandType::CreateHostTensorCommand: {
    return handleCreateHostTensorResponse(responseBuffers,
                                          std::move(awaitingResponse));
  }
  case fb::CommandType::GetLayoutCommand: {
    return handleGetLayoutResponse(responseBuffers,
                                   std::move(awaitingResponse));
  }
  case fb::CommandType::IsTensorAllocatedCommand: {
    return handleIsTensorAllocatedResponse(responseBuffers,
                                           std::move(awaitingResponse));
  }
  case fb::CommandType::GetTensorVolumeCommand: {
    return handleGetTensorVolumeResponse(responseBuffers,
                                         std::move(awaitingResponse));
  }
  case fb::CommandType::GetTensorRetainCommand: {
    return handleGetTensorRetainResponse(responseBuffers,
                                         std::move(awaitingResponse));
  }
  case fb::CommandType::SetTensorRetainCommand: {
    return handleSetTensorRetainResponse(responseBuffers,
                                         std::move(awaitingResponse));
  }
  case fb::CommandType::ToLayoutCommand: {
    return handleToLayoutResponse(responseBuffers, std::move(awaitingResponse));
  }
  case fb::CommandType::SubmitCommand: {
    return handleSubmitResponse(responseBuffers, std::move(awaitingResponse));
  }
  case fb::CommandType::GetNumShardsCommand: {
    return handleGetNumShardsResponse(responseBuffers,
                                      std::move(awaitingResponse));
  }
  case fb::CommandType::ToHostCommand: {
    return handleToHostResponse(responseBuffers, std::move(awaitingResponse));
  }
  case fb::CommandType::MemcpyCommand: {
    return handleMemcpyResponse(responseBuffers, std::move(awaitingResponse));
  }
  case fb::CommandType::DeallocateTensorCommand: {
    return handleDeallocateTensorResponse(responseBuffers,
                                          std::move(awaitingResponse));
  }
  case fb::CommandType::ShutdownCommand: {
    return handleShutdownResponse(responseBuffers, std::move(awaitingResponse));
  }
  case fb::CommandType::NONE: {
    LOG_FATAL("Unhandled response type for command type: ",
              fb::EnumNameCommandType(awaitingResponse->commandType));
  }
  }
}

ShutdownResult Controller::shutdown() {
  LOG_INFO("Shutting down distributed controller");

  ControllerState currentState =
      controllerState_.load(std::memory_order_relaxed);

  // Abnormal shutdown
  if (currentState < ControllerState::FullyOperational) {
    controllerState_.store(ControllerState::Shutdown,
                           std::memory_order_relaxed);
    return ShutdownResult{
        .success = false,
        .errorMessage =
            "Controller shutdown requested before fully operational, "
            "indicating that an error occurred in an earlier stage."};
  }

  if (currentState != ControllerState::FullyOperational) {
    controllerState_.store(ControllerState::Shutdown,
                           std::memory_order_relaxed);
    return ShutdownResult{
        .success = false,
        .errorMessage =
            "Unexpected controller state: " +
            std::to_string(static_cast<std::uint8_t>(currentState)) +
            " during shutdown"};
  }

  auto commandBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();
  auto command = CommandFactory::buildShutdownCommand(*commandBuilder);

  auto awaitingPromise = std::make_unique<std::promise<void>>();
  std::future<void> awaitingFuture = awaitingPromise->get_future();

  pushToCommandAndResponseQueues(
      command, fb::CommandType::ShutdownCommand, std::move(commandBuilder),
      /*awaitingHandles=*/nullptr, std::move(awaitingPromise));

  // Wait for worker shutdown
  awaitingFuture.wait();

  controllerState_.store(ControllerState::ShuttingDown,
                         std::memory_order_relaxed);
  commandDispatcherThread_.join();
  responseHandlerThread_.join();

  auto exitCodeResult =
      exitCodeFuture_.wait_for(std::chrono::seconds(workerShutdownTimeout_));

  if (exitCodeResult == std::future_status::timeout) {
    controllerState_.store(ControllerState::Shutdown,
                           std::memory_order_relaxed);
    return ShutdownResult{.success = false,
                          .errorMessage = "Worker subprocess shutdown timeout"};
  }

  int exitCode = exitCodeFuture_.get();
  if (exitCode != 0) {
    controllerState_.store(ControllerState::Shutdown,
                           std::memory_order_relaxed);
    return ShutdownResult{.success = false,
                          .errorMessage =
                              "Worker subprocess failed with exit code: " +
                              std::to_string(exitCode)};
  }

  LOG_INFO("Distributed controller shutdown successfully");

  controllerState_.store(ControllerState::Shutdown, std::memory_order_relaxed);
  return ShutdownResult{.success = true, .errorMessage = ""};
}

} // namespace tt::runtime::distributed::controller
