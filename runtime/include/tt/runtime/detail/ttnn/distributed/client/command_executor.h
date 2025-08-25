// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_CLIENT_COMMAND_EXECUTOR_H
#define TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_CLIENT_COMMAND_EXECUTOR_H

#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/ttnn/distributed/types/spsc_queue.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn::distributed::client {

using CommandBytes = std::shared_ptr<void>;

class CommandExecutor {
public:
  CommandExecutor();
  ~CommandExecutor();

  CommandExecutor(const CommandExecutor &) = delete;
  CommandExecutor &operator=(const CommandExecutor &) = delete;
  CommandExecutor(CommandExecutor &&) = delete;
  CommandExecutor &operator=(CommandExecutor &&) = delete;

  void connect(std::string_view host, uint16_t port);

  void run();

private:
  std::unique_ptr<ClientSocket> clientSocket_;
  SPSCQueue<CommandBytes> commandQueue_;
  std::thread commandReceiverThread_;
  std::unordered_map<uint64_t, ::tt::runtime::Tensor> tensorPool_;
  ::tt::runtime::ttnn::MeshDevicePool meshDevicePool_;

  void launchCommandReceiver();
  void receiveCommands();

  void
  execute(uint64_t commandId,
          const ::tt::target::ttnn::distributed::GetSystemDescCommand *command,
          ::flatbuffers::FlatBufferBuilder &responseBuilder);
  void
  execute(uint64_t commandId,
          const ::tt::target::ttnn::distributed::OpenMeshDeviceCommand *command,
          ::flatbuffers::FlatBufferBuilder &responseBuilder);
  void execute(
      uint64_t commandId,
      const ::tt::target::ttnn::distributed::CloseMeshDeviceCommand *command,
      ::flatbuffers::FlatBufferBuilder &responseBuilder);
  void execute(
      uint64_t commandId,
      const ::tt::target::ttnn::distributed::CreateHostTensorCommand *command,
      ::flatbuffers::FlatBufferBuilder &responseBuilder);
  void execute(uint64_t commandId,
               const ::tt::target::ttnn::distributed::ToLayoutCommand *command,
               ::flatbuffers::FlatBufferBuilder &responseBuilder);
  void execute(uint64_t commandId,
               const ::tt::target::ttnn::distributed::SubmitCommand *command,
               ::flatbuffers::FlatBufferBuilder &responseBuilder);
  void execute(uint64_t commandId,
               const ::tt::target::ttnn::distributed::ShutdownCommand *command,
               ::flatbuffers::FlatBufferBuilder &responseBuilder);

  void executeCommand(const ::tt::target::ttnn::distributed::Command *command,
                      ::flatbuffers::FlatBufferBuilder &responseBuilder);
  void sendResponse(::flatbuffers::FlatBufferBuilder &responseBuilder);
  void handleShutdown();
};

} // namespace tt::runtime::ttnn::distributed::client
#endif // TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_CLIENT_COMMAND_EXECUTOR_H
