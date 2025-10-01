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

struct AwaitingResponseQueueEntry {
  // Contains the command id that the response is for
  uint64_t commandId;

  // Contains the handles that will be populated when a response is received
  std::unique_ptr<std::vector<std::shared_ptr<void>>> awaitingHandles;

  // Contains the promise that will be set when a response is received
  std::unique_ptr<std::promise<void>> awaitingPromise;

  AwaitingResponseQueueEntry(
      uint64_t commandId,
      std::unique_ptr<std::vector<std::shared_ptr<void>>> awaitingHandles,
      std::unique_ptr<std::promise<void>> awaitingPromise)
      : commandId(commandId), awaitingHandles(std::move(awaitingHandles)),
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

  // Launches a local subprocess that will connect to the controller
  void launchLocalSubprocess(uint16_t controllerPort);

  // TODO (#5135): Add support for launching worker subprocesses
  // on remote hosts through MPI/TTRun

  void setWriteTimeout(const std::chrono::seconds &timeout);
  void setReadTimeout(const std::chrono::seconds &timeout);
  void setWorkerShutdownTimeout(const std::chrono::seconds &timeout);

  // Runtime APIs
  SystemDesc getCurrentSystemDesc(
      std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType =
          std::nullopt,
      std::optional<::tt::runtime::Device> deviceHandle = std::nullopt);

  ::tt::runtime::Device
  openMeshDevice(const ::tt::runtime::MeshDeviceOptions &options = {});
  void closeMeshDevice(const ::tt::runtime::Device &parentMeshHandle);

  ::tt::runtime::Tensor createOwnedHostTensor(
      const void *data, const std::vector<std::uint32_t> &shape,
      const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
      ::tt::target::DataType dataType);

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

private:
  std::chrono::seconds writeTimeout_{60};
  std::chrono::seconds readTimeout_{60};
  std::chrono::seconds workerShutdownTimeout_{60};

  std::atomic<bool> shutdownRequested_{false};
  std::vector<std::future<int>> exitCodeFutures_;

  std::unique_ptr<ControllerSocket> controllerSocket_;
  std::vector<std::unique_ptr<Socket>> workerConnections_;

  std::thread commandDispatcherThread_;
  SPSCQueue<std::unique_ptr<CommandQueueEntry>> commandQueue_;

  std::thread responseHandlerThread_;
  SPSCQueue<std::unique_ptr<AwaitingResponseQueueEntry>> awaitingResponseQueue_;

  void pushToCommandAndResponseQueues(
      uint64_t commandId,
      std::unique_ptr<::flatbuffers::FlatBufferBuilder> commandBuilder,
      std::unique_ptr<std::vector<std::shared_ptr<void>>> awaitingHandles =
          nullptr,
      std::unique_ptr<std::promise<void>> awaitingPromise = nullptr);

  void launchCommandDispatcher();
  void dispatchCommands();
  void dispatchCommand(const ::flatbuffers::FlatBufferBuilder &commandBuilder);

  void launchResponseHandler();
  void handleResponses();
  void handleErrorResponse(
      const ::tt::runtime::distributed::flatbuffer::ErrorResponse *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);
  void handleGetSystemDescResponse(
      const ::tt::runtime::distributed::flatbuffer::GetSystemDescResponse
          *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);
  void handleOpenMeshDeviceResponse(
      const ::tt::runtime::distributed::flatbuffer::OpenMeshDeviceResponse
          *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);
  void handleCloseMeshDeviceResponse(
      const ::tt::runtime::distributed::flatbuffer::CloseMeshDeviceResponse
          *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);
  void handleCreateHostTensorResponse(
      const ::tt::runtime::distributed::flatbuffer::CreateHostTensorResponse
          *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);
  void handleGetLayoutResponse(
      const ::tt::runtime::distributed::flatbuffer::GetLayoutResponse *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);
  void handleToLayoutResponse(
      const ::tt::runtime::distributed::flatbuffer::ToLayoutResponse *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);
  void handleSubmitResponse(
      const ::tt::runtime::distributed::flatbuffer::SubmitResponse *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);
  void handleGetNumShardsResponse(
      const ::tt::runtime::distributed::flatbuffer::GetNumShardsResponse
          *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);
  void handleToHostResponse(
      const ::tt::runtime::distributed::flatbuffer::ToHostResponse *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);
  void handleMemcpyResponse(
      const ::tt::runtime::distributed::flatbuffer::MemcpyResponse *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);
  void handleShutdownResponse(
      const ::tt::runtime::distributed::flatbuffer::ShutdownResponse *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void handleResponse(
      const ::tt::runtime::distributed::flatbuffer::Response *response,
      std::unique_ptr<AwaitingResponseQueueEntry> awaitingResponse);

  void shutdown();
};

} // namespace tt::runtime::distributed::controller

#endif
