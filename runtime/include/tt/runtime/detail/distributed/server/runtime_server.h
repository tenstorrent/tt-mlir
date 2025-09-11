// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_SERVER_RUNTIME_SERVER_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_SERVER_RUNTIME_SERVER_H

#include "flatbuffers/flatbuffers.h"
#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/distributed/types/spsc_queue.h"
#include <thread>

namespace tt::runtime::distributed::server {

class RuntimeServer {
public:
  RuntimeServer() = default;
  ~RuntimeServer();

  RuntimeServer(const RuntimeServer &) = delete;
  RuntimeServer &operator=(const RuntimeServer &) = delete;
  RuntimeServer(RuntimeServer &&) = delete;
  RuntimeServer &operator=(RuntimeServer &&) = delete;

  void launch(uint16_t port, size_t numClients);

  void shutdown();

private:
  std::atomic<bool> shutdownRequested_{false};

  std::unique_ptr<ServerSocket> serverSocket_;
  std::vector<std::unique_ptr<Socket>> clientConnections_;

  std::thread commandDispatcherThread_;
  SPSCQueue<std::unique_ptr<::flatbuffers::FlatBufferBuilder>> commandQueue_;

  std::thread responseParserThread_;
  SPSCQueue<SizedBuffer> responseQueue_;

  void launchCommandDispatcher();
  void dispatchCommands();

  void launchResponseParser();
  void parseResponses();
};

} // namespace tt::runtime::distributed::server

#endif
