// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_SOCKET_H
#define TT_RUNTIME_DETAIL_COMMON_SOCKET_H

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace tt::runtime {

using SocketFd = int;

class SizedBuffer {
public:
  SizedBuffer() : data_(nullptr), size_(0) {}
  SizedBuffer(std::shared_ptr<void> data, size_t size)
      : data_(std::move(data)), size_(size) {}

  const uint8_t *data() const {
    return static_cast<const uint8_t *>(data_.get());
  }
  size_t size() const { return size_; }

private:
  std::shared_ptr<void> data_;
  size_t size_;
};

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

  bool hasDataToRead(const std::chrono::milliseconds &timeout =
                         std::chrono::milliseconds(100)) const;
  ssize_t readExact(void *buf, size_t nbytes);
  ssize_t writeExact(const void *buf, size_t nbytes);
  SizedBuffer sizePrefixedRead();
  ssize_t sizePrefixedWrite(const void *msg, uint32_t msgSize);

private:
  SocketFd fd_;
};

class ServerSocket {
public:
  explicit ServerSocket(uint16_t port = 0);
  ~ServerSocket() = default;

  ServerSocket(const ServerSocket &) = delete;
  ServerSocket &operator=(const ServerSocket &) = delete;

  ServerSocket(ServerSocket &&) = default;
  ServerSocket &operator=(ServerSocket &&) = default;

  uint16_t port() const { return port_; }
  std::vector<std::unique_ptr<Socket>> connectToClients(size_t numClients);

private:
  std::unique_ptr<Socket> listenSocket_;
  uint16_t port_ = 0;
};

class ClientSocket {
public:
  explicit ClientSocket(const std::string &host, uint16_t port);
  ~ClientSocket() = default;

  ClientSocket(const ClientSocket &) = delete;
  ClientSocket &operator=(const ClientSocket &) = delete;

  ClientSocket(ClientSocket &&) = default;
  ClientSocket &operator=(ClientSocket &&) = default;

  void connect();
  void disconnect();
  bool isConnected() const;

  bool hasDataToRead(const std::chrono::milliseconds &timeout =
                         std::chrono::milliseconds(100)) const;

  SizedBuffer sizePrefixedRead();

  ssize_t sizePrefixedWrite(const void *msg, uint32_t msgSize);

private:
  std::unique_ptr<Socket> socket_;
};

} // namespace tt::runtime

#endif // TT_RUNTIME_DETAIL_COMMON_SOCKET_H
