// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_SOCKET_H
#define TT_RUNTIME_DETAIL_COMMON_SOCKET_H

#include <memory>
#include <string>
#include <vector>

namespace tt::runtime {

using SocketFd = int;

class Socket {
public:
  explicit Socket(SocketFd fd = -1) : fd_(fd) {}
  explicit Socket(int domain, int type, int protocol = 0);
  ~Socket();

  Socket(const Socket &) = delete;
  Socket &operator=(const Socket &) = delete;
  Socket(Socket &&) = delete;
  Socket &operator=(Socket &&) = delete;

  SocketFd fd() const { return fd_; }
  bool valid() const { return fd_ >= 0; }
  explicit operator bool() const { return valid(); }
  bool close();

  ssize_t readExact(void *buf, size_t nbytes);
  ssize_t writeExact(const void *buf, size_t nbytes);
  std::shared_ptr<void> sizePrefixedRead();
  ssize_t sizePrefixedWrite(const void *msg, uint32_t msgSize);

private:
  SocketFd fd_;
};

class ServerSocket {
public:
  explicit ServerSocket(uint16_t port);
  ~ServerSocket() = default;

  ServerSocket(const ServerSocket &) = delete;
  ServerSocket &operator=(const ServerSocket &) = delete;

  ServerSocket(ServerSocket &&) = default;
  ServerSocket &operator=(ServerSocket &&) = default;

  std::vector<std::unique_ptr<Socket>> connectToClients(size_t numClients);

private:
  std::unique_ptr<Socket> listenSocket_;
};

class ClientSocket {
public:
  explicit ClientSocket(std::string_view host, uint16_t port);
  ~ClientSocket() = default;

  ClientSocket(const ClientSocket &) = delete;
  ClientSocket &operator=(const ClientSocket &) = delete;

  ClientSocket(ClientSocket &&) = default;
  ClientSocket &operator=(ClientSocket &&) = default;

  void connect();
  void disconnect();
  bool isConnected() const;

  std::shared_ptr<void> sizePrefixedRead();

  uint32_t sizePrefixedWrite(const void *msg, uint32_t msgSize);

private:
  std::string host_;
  int port_;
  std::unique_ptr<Socket> clientSocket_;
};

} // namespace tt::runtime

#endif // TT_RUNTIME_DETAIL_COMMON_SOCKET_H
