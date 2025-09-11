// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/server/runtime_server.h"
#include "tt/runtime/detail/common/logger.h"

namespace tt::runtime::distributed::server {

RuntimeServer::~RuntimeServer() {
  if (!shutdownRequested_.load(std::memory_order_relaxed)) {
    shutdown();
  }
}

void RuntimeServer::launch(uint16_t port, size_t numClients) {
  LOG_ASSERT(!serverSocket_, "RuntimeServer already launched");
  serverSocket_ = std::make_unique<ServerSocket>(port);
  clientConnections_ = serverSocket_->connectToClients(numClients);
  launchCommandDispatcher();
  launchResponseParser();
}

void RuntimeServer::launchCommandDispatcher() {
  LOG_ASSERT(!commandDispatcherThread_.joinable(),
             "Command dispatcher thread already running");
  commandDispatcherThread_ = std::thread([this]() { dispatchCommands(); });
}

void RuntimeServer::dispatchCommands() {
  std::vector<std::future<ssize_t>> writeFutures;
  writeFutures.reserve(clientConnections_.size());
  while (!(shutdownRequested_.load(std::memory_order_acquire) &&
           commandQueue_.empty())) {
    std::optional<std::unique_ptr<::flatbuffers::FlatBufferBuilder>>
        commandBuilderOpt = commandQueue_.popWithTimeout();
    if (!commandBuilderOpt.has_value()) {
      continue;
    }

    std::unique_ptr<::flatbuffers::FlatBufferBuilder> commandBuilder =
        std::move(commandBuilderOpt.value());
    size_t commandSize = commandBuilder->GetSize();
    LOG_ASSERT(commandSize > 0, "Unexpected empty command");

    // Currently broadcasting command to all clients
    // Will need updates later to support multi mesh and MPMD
    for (auto &clientConnection : clientConnections_) {
      writeFutures.push_back(clientConnection->sizePrefixedWriteAsync(
          commandBuilder->GetBufferPointer(), commandSize));
    }
    for (std::future<ssize_t> &writeFuture : writeFutures) {
      writeFuture.wait();
    }
    writeFutures.clear();
  }
}

void RuntimeServer::shutdown() {
  if (shutdownRequested_.load(std::memory_order_relaxed)) {
    LOG_WARNING("Calling shutdown() on already shutdown RuntimeServer, "
                "returning immediately");
    return;
  }
  shutdownRequested_.store(true, std::memory_order_release);
  commandDispatcherThread_.join();
  responseParserThread_.join();
}

} // namespace tt::runtime::distributed::server
