// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/types.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace tt::runtime {

//
// Socket
//
Socket::Socket(int domain, int type, int protocol) {
  fd_ = ::socket(domain, type, protocol);
  if (fd_ < 0) {
    LOG_ERROR("Failed to create socket: ", std::strerror(errno));
  }
}

Socket::~Socket() { close(); }

bool Socket::close() {
  if (valid()) {
    ::close(fd_);
    fd_ = -1;
    return true;
  }
  return false;
}

ssize_t Socket::readExact(void *buf, size_t nbytes) {
  std::byte *dst = static_cast<std::byte *>(buf);
  size_t total = 0;
  while (total < nbytes) {
    ssize_t r = ::read(fd_, dst + total, nbytes - total);
    if (r == 0) {
      // peer closed, only return partial read if any was performed
      return static_cast<ssize_t>(total);
    }
    if (r < 0) {
      if (errno == EINTR) {
        continue;
      }
      return r;
    }
    total += static_cast<size_t>(r);
  }
  return static_cast<ssize_t>(total);
}

ssize_t Socket::writeExact(const void *buf, size_t nbytes) {
  const std::byte *src = static_cast<const std::byte *>(buf);
  size_t total = 0;
  while (total < nbytes) {
    ssize_t w = ::write(fd_, src + total, nbytes - total);
    if (w < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    total += static_cast<size_t>(w);
  }
  return static_cast<ssize_t>(total);
}

std::shared_ptr<void> Socket::sizePrefixedRead() {
  if (!valid()) {
    LOG_ERROR(__FUNCTION__, ": Invalid socket file descriptor");
    return nullptr;
  }

  uint32_t netSize = 0;
  ssize_t readResult = readExact(&netSize, sizeof(netSize));
  if (readResult < 0) {
    LOG_ERROR("Socket read size failed: ", std::strerror(errno));
    return nullptr;
  }

  uint32_t msgSize = ntohl(netSize);
  if (msgSize == 0) {
    return ::tt::runtime::utils::malloc_shared(0);
  }
  std::shared_ptr<void> msgBuffer =
      ::tt::runtime::utils::malloc_shared(msgSize);
  readResult = readExact(msgBuffer.get(), msgSize);
  if (readResult < 0) {
    LOG_ERROR("Socket read payload failed: ", std::strerror(errno));
    return nullptr;
  }
  return msgBuffer;
}

ssize_t Socket::sizePrefixedWrite(const void *msg, uint32_t msgSize) {
  if (!valid()) {
    LOG_ERROR(__FUNCTION__, ": Invalid socket file descriptor");
    return 0;
  }

  uint32_t netSize = htonl(msgSize);
  ssize_t writeResult = writeExact(&netSize, sizeof(netSize));
  if (writeResult < 0) {
    LOG_ERROR("Socket write size failed: ", std::strerror(errno));
    return 0;
  }

  writeResult = writeExact(msg, msgSize);
  if (writeResult < 0) {
    LOG_ERROR("Socket write payload failed: ", std::strerror(errno));
    return 0;
  }
  return writeResult;
}

//
// ServerSocket
//
ServerSocket::ServerSocket(uint16_t port) {
  std::unique_ptr<Socket> socket =
      std::make_unique<Socket>(AF_INET, SOCK_STREAM, 0);
  if (!socket->valid()) {
    LOG_FATAL("Failed to create server socket on port ", port, ": ",
              std::strerror(errno));
  }

  int opt = 1;
  ::setsockopt(socket->fd(), SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);

  if (::bind(socket->fd(), reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) <
      0) {
    LOG_FATAL("Failed to bind ServerSocket on port ", port, ": ",
              std::strerror(errno));
  }

  if (::listen(socket->fd(), SOMAXCONN) < 0) {
    LOG_FATAL("Failed to listen on port ", port, ": ", std::strerror(errno));
  }

  listenSocket_ = std::move(socket);
  LOG_INFO("ServerSocket bound to port ", port);
}

std::vector<std::unique_ptr<Socket>>
ServerSocket::connectToClients(size_t numClients) {
  std::vector<std::unique_ptr<Socket>> clientConnections;
  clientConnections.reserve(numClients);
  for (size_t i = 0; i < numClients; i++) {
    sockaddr_in clientAddr{};
    socklen_t clientLen = sizeof(clientAddr);
    std::unique_ptr<Socket> clientSocket = std::make_unique<Socket>(
        ::accept(listenSocket_->fd(), reinterpret_cast<sockaddr *>(&clientAddr),
                 &clientLen));
    if (!clientSocket->valid()) {
      LOG_FATAL("Accept failed: ", std::strerror(errno));
    }

    char ipStr[INET_ADDRSTRLEN] = {0};
    ::inet_ntop(AF_INET, &clientAddr.sin_addr, ipStr, sizeof(ipStr));
    uint16_t clientPort = ntohs(clientAddr.sin_port);

    LOG_DEBUG("Connected to client ", i, " at ", ipStr, ":", clientPort);

    clientConnections.emplace_back(std::move(clientSocket));
  }
  return clientConnections;
}

//
// ClientSocket
//
ClientSocket::ClientSocket(std::string_view host, uint16_t port)
    : host_(host), port_(port) {};

void ClientSocket::connect() {
  if (isConnected()) {
    LOG_ERROR("ClientSocket is already connected");
    return;
  }

  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = 0;
  hints.ai_protocol = 0;

  addrinfo *endpoints = nullptr;
  std::string portStr = std::to_string(port_);
  int r = ::getaddrinfo(host_.c_str(), portStr.c_str(), &hints, &endpoints);
  if (r != 0) {
    LOG_FATAL("Client socket getaddrinfo failed: ", ::gai_strerror(r));
  }

  for (addrinfo *endpoint = endpoints; endpoint != nullptr;
       endpoint = endpoint->ai_next) {
    clientSocket_ = std::make_unique<Socket>(
        endpoint->ai_family, endpoint->ai_socktype, endpoint->ai_protocol);
    if (!clientSocket_->valid()) {
      clientSocket_.reset();
      continue;
    }
    if (::connect(clientSocket_->fd(), endpoint->ai_addr,
                  endpoint->ai_addrlen) == 0) {
      break; // success
    }
  }

  ::freeaddrinfo(endpoints);
  if (!clientSocket_ || !clientSocket_->valid()) {
    LOG_FATAL("Client socket connect failed with error: ",
              std::strerror(errno));
  }
}

bool ClientSocket::isConnected() const { return clientSocket_->valid(); }

void ClientSocket::disconnect() { clientSocket_.reset(); }

std::shared_ptr<void> ClientSocket::sizePrefixedRead() {
  LOG_ASSERT(isConnected(), "ClientSocket is not connected");
  return clientSocket_->sizePrefixedRead();
}

uint32_t ClientSocket::sizePrefixedWrite(const void *msg, uint32_t msgSize) {
  LOG_ASSERT(isConnected(), "ClientSocket is not connected");
  return clientSocket_->sizePrefixedWrite(msg, msgSize);
}
} // namespace tt::runtime
