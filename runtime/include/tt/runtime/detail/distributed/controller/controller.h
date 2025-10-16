// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_CONTROLLER_CONTROLLER_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_CONTROLLER_CONTROLLER_H

#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/detail/distributed/types/spsc_queue.h"
#include "tt/runtime/types.h"
#include <thread>

namespace tt::runtime::distributed::controller {

enum class ControllerState : std::uint8_t {
  Uninitialized,         // Initial state before any setup
  ControllerSocketBound, // Controller socket bound to the local port
  CommandLaunched,       // Subcommand process started (e.g. ttrun)
  ConnectedToWorkers,    // Worker connections established
  DispatcherReady,       // Command dispatcher is operational
  ResponseHandlerReady,  // Response handler is operational
  FullyOperational, // All components running, everything launched successfully
  ShuttingDown,     // Graceful shutdown in progress
  Shutdown,         // Shutdown state
};

struct ShutdownResult {
  bool success;
  std::string errorMessage;
};

struct AwaitingResponseQueueEntry {
  // Contains the command id that the response is for
  uint64_t commandId;

  ::tt::runtime::distributed::flatbuffer::CommandType commandType;

  // Contains the handles that will be populated when a response is received
  std::unique_ptr<std::vector<std::shared_ptr<void>>> awaitingHandles;

  // Contains the promise that will be set when a response is received
  std::unique_ptr<std::promise<void>> awaitingPromise;

  AwaitingResponseQueueEntry(
      uint64_t commandId,
      const ::tt::runtime::distributed::flatbuffer::CommandType &commandType,
      std::unique_ptr<std::vector<std::shared_ptr<void>>> awaitingHandles,
      std::unique_ptr<std::promise<void>> awaitingPromise)
      : commandId(commandId), commandType(commandType),
        awaitingHandles(std::move(awaitingHandles)),
        awaitingPromise(std::move(awaitingPromise)) {}

  std::tuple<std::unique_ptr<std::vector<std::shared_ptr<void>>>,
             std::unique_ptr<std::promise<void>>>
  popAwaitingState() {
    return std::make_tuple(std::move(awaitingHandles),
                           std::move(awaitingPromise));
  }
};

struct CommandQueueEntry {
  // Contains the flatbuffer command to be sent to the worker
  std::unique_ptr<::flatbuffers::FlatBufferBuilder> commandBuilder;

  CommandQueueEntry(
      std::unique_ptr<::flatbuffers::FlatBufferBuilder> commandBuilder)
      : commandBuilder(std::move(commandBuilder)) {}
};

class Controller {
public:
  Controller() = default;
  ~Controller();

  Controller(const Controller &) = delete;
  Controller &operator=(const Controller &) = delete;
  Controller(Controller &&) = delete;
  Controller &operator=(Controller &&) = delete;

  void launch(const ::tt::runtime::DistributedOptions &options);

  void setWriteTimeout(const std::chrono::seconds &timeout);
  void setReadTimeout(const std::chrono::seconds &timeout);
  void setWorkerShutdownTimeout(const std::chrono::seconds &timeout);

  // Runtime APIs
  SystemDesc getCurrentSystemDesc(
      std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType =
          std::nullopt,
      std::optional<::tt::runtime::Device> deviceHandle = std::nullopt);

  void setFabricConfig(const ::tt::runtime::FabricConfig &fabricConfig);

  size_t getNumAvailableDevices();

  ::tt::runtime::Device
  openMeshDevice(const ::tt::runtime::MeshDeviceOptions &options = {});

  void closeMeshDevice(::tt::runtime::Device &parentMeshHandle);

  ::tt::runtime::Device createSubMeshDevice(
      const ::tt::runtime::Device &parentMesh,
      const std::vector<uint32_t> &meshShape,
      const std::optional<const std::vector<uint32_t>> &meshOffset =
          std::nullopt);

  void releaseSubMeshDevice(const ::tt::runtime::Device &subMesh);

  std::vector<uint32_t> getMeshShape(const ::tt::runtime::Device &deviceHandle);

  ::tt::runtime::Tensor createOwnedHostTensor(
      const void *data, const std::vector<std::uint32_t> &shape,
      const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
      ::tt::target::DataType dataType);

  std::uint32_t getTensorVolume(const ::tt::runtime::Tensor &tensorHandle);

  bool getTensorRetain(const ::tt::runtime::Tensor &tensorHandle);

  void setTensorRetain(const ::tt::runtime::Tensor &tensorHandle, bool retain);

  ::tt::runtime::Layout getLayout(const ::tt::runtime::Binary &executableHandle,
                                  std::uint32_t programIndex,
                                  std::uint32_t inputIndex);

  ::tt::runtime::Tensor toLayout(const ::tt::runtime::Tensor &tensorHandle,
                                 const ::tt::runtime::Device &deviceHandle,
                                 const ::tt::runtime::Layout &layoutHandle,
                                 std::optional<bool> retain = std::nullopt);

  std::vector<::tt::runtime::Tensor>
  submit(const ::tt::runtime::Device &deviceHandle,
         const ::tt::runtime::Binary &executableHandle,
         std::uint32_t programIndex,
         const std::vector<::tt::runtime::Tensor> &inputHandles);

  std::vector<::tt::runtime::Tensor>
  toHost(const ::tt::runtime::Tensor &tensorHandle, bool untilize = false,
         bool blocking = true);

  void
  memcpy(void *dst, const ::tt::runtime::Tensor &srcHandle,
         std::optional<tt::target::DataType> targetDataType = std::nullopt);

  void memcpy(const ::tt::runtime::Tensor &dstHandle,
              const ::tt::runtime::Tensor &srcHandle);

  ShutdownResult shutdown();

private:
  std::chrono::seconds writeTimeout_{60};
  std::chrono::seconds readTimeout_{60};
  std::chrono::seconds workerShutdownTimeout_{60};

  std::atomic<ControllerState> controllerState_{ControllerState::Uninitialized};
  std::future<int> exitCodeFuture_;

  std::unique_ptr<ControllerSocket> controllerSocket_;
  std::vector<std::unique_ptr<Socket>> workerConnections_;

  std::thread commandDispatcherThread_;
  SPSCQueue<std::unique_ptr<CommandQueueEntry>> commandQueue_;

  std::thread responseHandlerThread_;
  SPSCQueue<std::unique_ptr<AwaitingResponseQueueEntry>> awaitingResponseQueue_;

  void pushToCommandAndResponseQueues(
      uint64_t commandId,
      const ::tt::runtime::distributed::flatbuffer::CommandType &commandType,
      std::unique_ptr<::flatbuffers::FlatBufferBuilder> commandBuilder,
      std::unique_ptr<std::vector<std::shared_ptr<void>>> awaitingHandles =
          nullptr,
      std::unique_ptr<std::promise<void>> awaitingPromise = nullptr);

  void launchCommandDispatcher();
  void processCommandQueue();
  void dispatchCommand(const ::flatbuffers::FlatBufferBuilder &commandBuilder);

  void launchResponseHandler();
  void processResponseQueue();

  void handleErrorResponse(
      const ::tt::runtime::distributed::flatbuffer::ErrorResponse *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleGetSystemDescResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleSetFabricConfigResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleGetNumAvailableDevicesResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleOpenMeshDeviceResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleCloseMeshDeviceResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleCreateSubMeshDeviceResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleReleaseSubMeshDeviceResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleGetMeshShapeResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleCreateHostTensorResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleGetTensorVolumeResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleGetTensorRetainResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleSetTensorRetainResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleGetLayoutResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleToLayoutResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleSubmitResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleGetNumShardsResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleToHostResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleMemcpyResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleShutdownResponse(
      const std::vector<SizedBuffer> &responseBuffers,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void
  handleResponse(const std::vector<SizedBuffer> &responseBuffers,
                 std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);
};

} // namespace tt::runtime::distributed::controller

#endif
