// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_SOCKET_H
#define TT_RUNTIME_DETAIL_COMMON_SOCKET_H

#include "tt/runtime/types.h"
#include <memory>
#include <string>

namespace tt::runtime {

class ServerSocket {
public:
  explicit ServerSocket(int port);
  ~ServerSocket();

  ServerSocket(const ServerSocket &) = delete;
  ServerSocket &operator=(const ServerSocket &) = delete;
  ServerSocket(ServerSocket &&) = default;
  ServerSocket &operator=(ServerSocket &&) = default;

  std::vector<Connection> connectToClients(size_t numClients);
  void disconnectFromClient(Connection clientConnection);

  std::shared_ptr<void> sizePrefixedRead(Connection clientConnection);

  uint32_t sizePrefixedWrite(Connection clientConnection,
                             const std::shared_ptr<void> &msg,
                             uint32_t msgSize);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

class ClientSocket {
public:
  explicit ClientSocket(const std::string &host, int port);
  ~ClientSocket();

  ClientSocket(const ClientSocket &) = delete;
  ClientSocket &operator=(const ClientSocket &) = delete;
  ClientSocket(ClientSocket &&) = default;
  ClientSocket &operator=(ClientSocket &&) = default;

  void connect();
  void disconnect();
  bool isConnected() const;

  std::shared_ptr<void> sizePrefixedRead();

  uint32_t sizePrefixedWrite(const std::shared_ptr<void> &msg,
                             uint32_t msgSize) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace tt::runtime

#endif // TT_RUNTIME_DETAIL_COMMON_SOCKET_H
