// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_SERVER_SERVER_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_SERVER_SERVER_H

#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/detail/distributed/types/spsc_queue.h"
#include "tt/runtime/types.h"
#include <thread>

namespace tt::runtime::distributed::server {

struct AwaitingResponseQueueEntry {
  // Contains the command id that the response is for
  uint64_t commandId;

  // Contains the future that will be set when the response is received
  // once a response is received
  std::unique_ptr<std::vector<SharedHandle>> awaitingHandles;

  // Contains the promise that will be set when the command is executed
  // and a response is received
  std::unique_ptr<std::promise<void>> awaitingPromise;
};

struct CommandQueueEntry {
  // Contains the flatbuffer command to be sent to the client
  std::unique_ptr<::flatbuffers::FlatBufferBuilder> commandBuilder;
};

class Server {
public:
  Server() = default;
  ~Server();

  Server(const Server &) = delete;
  Server &operator=(const Server &) = delete;
  Server(Server &&) = delete;
  Server &operator=(Server &&) = delete;

  // Launches a local subprocess that will connect to the server
  void launchLocalSubprocess(uint16_t serverPort);

  // TODO (#4484): Add support for launching client subprocesses
  // on remote hosts through MPI/TTRun

  void shutdown();

  void setWriteTimeout(const std::chrono::seconds &timeout);
  void setReadTimeout(const std::chrono::seconds &timeout);

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

  std::atomic<bool> shutdownRequested_{false};
  std::vector<std::future<int>> exitCodeFutures_;

  std::unique_ptr<ServerSocket> serverSocket_;
  std::vector<std::unique_ptr<Socket>> clientConnections_;

  std::thread commandDispatcherThread_;
  SPSCQueue<std::unique_ptr<CommandQueueEntry>> commandQueue_;

  std::thread responseHandlerThread_;
  SPSCQueue<std::unique_ptr<AwaitingResponseQueueEntry>> awaitingResponseQueue_;

  void pushToCommandAndResponseQueues(
      uint64_t commandId,
      std::unique_ptr<::flatbuffers::FlatBufferBuilder> commandBuilder,
      std::unique_ptr<std::vector<SharedHandle>> awaitingHandles = nullptr,
      std::unique_ptr<std::promise<void>> awaitingPromise = nullptr);

  void launchCommandDispatcher();
  void dispatchCommands();
  void dispatchCommand(const ::flatbuffers::FlatBufferBuilder &commandBuilder);

  void launchResponseHandler();
  void handleResponses();
  void handleErrorResponse(
      const ::tt::runtime::distributed::flatbuffer::ErrorResponse *response,
      AwaitingResponseQueueEntry &awaitingResponse);
  void handleGetSystemDescResponse(
      const ::tt::runtime::distributed::flatbuffer::GetSystemDescResponse
          *response,
      AwaitingResponseQueueEntry &awaitingResponse);
  void handleOpenMeshDeviceResponse(
      const ::tt::runtime::distributed::flatbuffer::OpenMeshDeviceResponse
          *response,
      AwaitingResponseQueueEntry &awaitingResponse);
  void handleCloseMeshDeviceResponse(
      const ::tt::runtime::distributed::flatbuffer::CloseMeshDeviceResponse
          *response,
      AwaitingResponseQueueEntry &awaitingResponse);
  void handleCreateHostTensorResponse(
      const ::tt::runtime::distributed::flatbuffer::CreateHostTensorResponse
          *response,
      AwaitingResponseQueueEntry &awaitingResponse);
  void handleGetLayoutResponse(
      const ::tt::runtime::distributed::flatbuffer::GetLayoutResponse *response,
      AwaitingResponseQueueEntry &awaitingResponse);
  void handleToLayoutResponse(
      const ::tt::runtime::distributed::flatbuffer::ToLayoutResponse *response,
      AwaitingResponseQueueEntry &awaitingResponse);
  void handleSubmitResponse(
      const ::tt::runtime::distributed::flatbuffer::SubmitResponse *response,
      AwaitingResponseQueueEntry &awaitingResponse);
  void handleGetNumShardsResponse(
      const ::tt::runtime::distributed::flatbuffer::GetNumShardsResponse
          *response,
      AwaitingResponseQueueEntry &awaitingResponse);
  void handleToHostResponse(
      const ::tt::runtime::distributed::flatbuffer::ToHostResponse *response,
      AwaitingResponseQueueEntry &awaitingResponse);
  void handleMemcpyResponse(
      const ::tt::runtime::distributed::flatbuffer::MemcpyResponse *response,
      AwaitingResponseQueueEntry &awaitingResponse);
  void handleShutdownResponse(
      const ::tt::runtime::distributed::flatbuffer::ShutdownResponse *response,
      AwaitingResponseQueueEntry &awaitingResponse);

  void handleResponse(
      const ::tt::runtime::distributed::flatbuffer::Response *response,
      AwaitingResponseQueueEntry &awaitingResponse);
};

} // namespace tt::runtime::distributed::server

#endif
