// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_CLIENT_COMMAND_EXECUTOR_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_CLIENT_COMMAND_EXECUTOR_H

#include <atomic>
#include <thread>
#include <unordered_map>

#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/detail/distributed/types/spsc_queue.h"
#include "tt/runtime/types.h"

namespace tt::runtime::distributed::client {

class CommandExecutor {
public:
  CommandExecutor() = default;
  ~CommandExecutor() = default;

  CommandExecutor(const CommandExecutor &) = delete;
  CommandExecutor &operator=(const CommandExecutor &) = delete;
  CommandExecutor(CommandExecutor &&) = delete;
  CommandExecutor &operator=(CommandExecutor &&) = delete;

  void connect(const std::string &host, uint16_t port);

  void run();

private:
  std::atomic<bool> shutdownRequested_{false};
  std::unique_ptr<ClientSocket> clientSocket_;
  SPSCQueue<SizedBuffer> commandQueue_;
  std::thread commandReceiverThread_;
  std::unordered_map<uint32_t, ::tt::runtime::Device> devicePool_;
  std::unordered_map<uint64_t, ::tt::runtime::Binary> binaryPool_;
  std::unordered_map<uint64_t, ::tt::runtime::Layout> layoutPool_;
  std::unordered_map<uint64_t, ::tt::runtime::Tensor> tensorPool_;

  ::tt::runtime::Binary
  getOrCreateBinary(const flatbuffers::Vector<uint8_t> *binary,
                    uint64_t binaryId);

  void launchCommandReceiver();
  void receiveCommands();

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::GetSystemDescCommand
              *command);
  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::OpenMeshDeviceCommand
              *command);
  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::CloseMeshDeviceCommand
              *command);
  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::CreateHostTensorCommand
              *command);
  void execute(
      uint64_t commandId,
      const ::tt::runtime::distributed::flatbuffer::GetLayoutCommand *command);
  void execute(
      uint64_t commandId,
      const ::tt::runtime::distributed::flatbuffer::ToLayoutCommand *command);
  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::SubmitCommand *command);
  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::ToHostCommand *command);
  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::MemcpyCommand *command);
  void execute(
      uint64_t commandId,
      const ::tt::runtime::distributed::flatbuffer::ShutdownCommand *command);

  void executeCommand(
      const ::tt::runtime::distributed::flatbuffer::Command *command);

  void sendResponse(::flatbuffers::FlatBufferBuilder &responseBuilder);
  void handleShutdown();
};

} // namespace tt::runtime::distributed::client
#endif // TT_RUNTIME_DETAIL_DISTRIBUTED_CLIENT_COMMAND_EXECUTOR_H
