// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/utils.h"
#include <boost/asio/buffer.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>

namespace tt::runtime {

class Socket::Impl {
public:
  Impl(const std::string &host, int port)
      : host(host), port(port),
        ioContext(std::make_unique<boost::asio::io_context>()),
        socket(std::make_unique<boost::asio::ip::tcp::socket>(*ioContext)) {}
  ~Impl() = default;

  void connect() {
    LOG_ASSERT(!isConnected(), "Socket is already connected");
    boost::asio::ip::tcp::resolver resolver(*ioContext);
    boost::asio::ip::tcp::resolver::results_type endpoints =
        resolver.resolve(host, std::to_string(port));

    boost::system::error_code ec;
    boost::asio::connect(*socket, endpoints, ec);

    if (ec) {
      LOG_ERROR("Client socket connect failed with error: ", ec.message());
    }
  }

  // Check if the socket is connected by checking if it is open
  // TODO (jnie): A more sophisticated check would be to peek at the socket to
  // see if it is still alive
  bool isConnected() const { return socket && socket->is_open(); }

  void disconnect() { socket.reset(); }

  std::shared_ptr<void> sizePrefixedRead() {
    DEBUG_ASSERT(isConnected(), "Socket is not connected");
    uint32_t msgSize = 0;
    boost::asio::read(*socket, boost::asio::buffer(&msgSize, sizeof(msgSize)));
    msgSize = ntohl(msgSize);
    std::shared_ptr<void> msgBuffer =
        ::tt::runtime::utils::malloc_shared(msgSize);
    boost::asio::read(*socket, boost::asio::buffer(msgBuffer.get(), msgSize));
    return msgBuffer;
  }

  size_t sizePrefixedWrite(std::shared_ptr<void> msg, size_t msgSize) {
    msgSize = htonl(msgSize);
    boost::asio::write(*socket, boost::asio::buffer(&msgSize, sizeof(msgSize)));
    boost::asio::write(*socket, boost::asio::buffer(msg.get(), msgSize));
    return msgSize;
  }

private:
  std::string host;
  int port;
  std::unique_ptr<boost::asio::io_context> ioContext;
  std::unique_ptr<boost::asio::ip::tcp::socket> socket;
};

Socket::Socket(const std::string &host, int port)
    : impl(std::make_unique<Impl>(host, port)) {}

void Socket::connect() { impl->connect(); }

bool Socket::isConnected() const { return impl->isConnected(); }

void Socket::disconnect() { impl->disconnect(); }

std::shared_ptr<void> Socket::sizePrefixedRead() {
  return impl->sizePrefixedRead();
}

size_t Socket::sizePrefixedWrite(const std::shared_ptr<void> &msg,
                                 size_t msgSize) const {
  return impl->sizePrefixedWrite(msg, msgSize);
}
} // namespace tt::runtime
