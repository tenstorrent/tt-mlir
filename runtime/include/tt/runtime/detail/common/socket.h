// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_SOCKET_H
#define TT_RUNTIME_DETAIL_COMMON_SOCKET_H

#include <chrono>
#include <future>
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

  ssize_t readExact(void *buf, size_t nbytes) const;
  ssize_t writeExact(const void *buf, size_t nbytes) const;

  SizedBuffer sizePrefixedRead() const;
  ssize_t sizePrefixedWrite(const void *msg, uint32_t msgSize) const;

  std::future<SizedBuffer> sizePrefixedReadAsync();
  std::future<ssize_t> sizePrefixedWriteAsync(const void *msg,
                                              uint32_t msgSize) const;

private:
  SocketFd fd_;
};

class ControllerSocket {
public:
  explicit ControllerSocket(uint16_t port = 0);
  ~ControllerSocket() = default;

  ControllerSocket(const ControllerSocket &) = delete;
  ControllerSocket &operator=(const ControllerSocket &) = delete;

  ControllerSocket(ControllerSocket &&) = default;
  ControllerSocket &operator=(ControllerSocket &&) = default;

  uint16_t port() const { return port_; }
  std::vector<std::unique_ptr<Socket>>
  connectToWorkers(size_t numWorkers,
                   const std::chrono::milliseconds &timeout) const;

private:
  std::unique_ptr<Socket> listenSocket_;
  uint16_t port_ = 0;
};

class WorkerSocket {
public:
  explicit WorkerSocket(const std::string &host, uint16_t port);
  ~WorkerSocket() = default;

  WorkerSocket(const WorkerSocket &) = delete;
  WorkerSocket &operator=(const WorkerSocket &) = delete;

  WorkerSocket(WorkerSocket &&) = default;
  WorkerSocket &operator=(WorkerSocket &&) = default;

  void connect();
  void disconnect();
  bool isConnected() const;

  bool hasDataToRead(const std::chrono::milliseconds &timeout =
                         std::chrono::milliseconds(100)) const;

  SizedBuffer sizePrefixedRead() const;

  ssize_t sizePrefixedWrite(const void *msg, uint32_t msgSize) const;

private:
  std::unique_ptr<Socket> socket_;
};

} // namespace tt::runtime

#endif // TT_RUNTIME_DETAIL_COMMON_SOCKET_H
