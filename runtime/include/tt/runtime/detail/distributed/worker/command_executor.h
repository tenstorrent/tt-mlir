// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_WORKER_COMMAND_EXECUTOR_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_WORKER_COMMAND_EXECUTOR_H

#include <atomic>
#include <thread>
#include <unordered_map>

#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/detail/distributed/types/spsc_queue.h"
#include "tt/runtime/types.h"

namespace tt::runtime::distributed::worker {

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
  std::unique_ptr<WorkerSocket> workerSocket_;

  SPSCQueue<SizedBuffer> commandQueue_;
  std::thread commandReceiverThread_;

  SPSCQueue<std::unique_ptr<::flatbuffers::FlatBufferBuilder>> responseQueue_;
  std::thread responseSenderThread_;

  std::unordered_map<uint32_t, ::tt::runtime::Device> devicePool_;
  std::unordered_map<uint64_t, ::tt::runtime::Binary> binaryPool_;
  std::unordered_map<uint64_t, ::tt::runtime::Layout> layoutPool_;
  std::unordered_map<uint64_t, ::tt::runtime::Tensor> tensorPool_;

  ::tt::runtime::Binary
  getOrCreateBinary(const flatbuffers::Vector<uint8_t> *binary,
                    uint64_t binaryId);

  void launchCommandReceiver();
  void receiveCommands();

  void launchResponseSender();
  void sendResponses();

  void execute(uint64_t commandId,
               const ::tt::runtime::distributed::flatbuffer::
                   ConfigureRuntimeContextCommand *command);

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::GetSystemDescCommand
              *command);

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::SetFabricConfigCommand
              *command);

  void execute(uint64_t commandId,
               const ::tt::runtime::distributed::flatbuffer::
                   GetNumAvailableDevicesCommand *command);

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::OpenMeshDeviceCommand
              *command);

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::CloseMeshDeviceCommand
              *command);

  void execute(
      uint64_t commandId,
      const ::tt::runtime::distributed::flatbuffer::CreateSubMeshDeviceCommand
          *command);

  void execute(
      uint64_t commandId,
      const ::tt::runtime::distributed::flatbuffer::ReleaseSubMeshDeviceCommand
          *command);

  void execute(uint64_t commandId,
               const ::tt::runtime::distributed::flatbuffer::GetMeshShapeCommand
                   *command);

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::CreateHostTensorCommand
              *command);

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::IsTensorAllocatedCommand
              *command);

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::GetTensorVolumeCommand
              *command);

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::GetTensorRetainCommand
              *command);

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::SetTensorRetainCommand
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

  void execute(uint64_t commandId,
               const ::tt::runtime::distributed::flatbuffer::GetNumShardsCommand
                   *command);

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::ToHostCommand *command);

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::MemcpyCommand *command);

  void
  execute(uint64_t commandId,
          const ::tt::runtime::distributed::flatbuffer::DeallocateTensorCommand
              *command);

  void execute(
      uint64_t commandId,
      const ::tt::runtime::distributed::flatbuffer::ShutdownCommand *command);

  void executeCommand(
      const ::tt::runtime::distributed::flatbuffer::Command *command);
};

} // namespace tt::runtime::distributed::worker
#endif // TT_RUNTIME_DETAIL_DISTRIBUTED_WORKER_COMMAND_EXECUTOR_H
