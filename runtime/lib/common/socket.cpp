// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/types.h"
#include <boost/asio/buffer.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>

namespace tt::runtime {

namespace detail {
static std::shared_ptr<void>
sizePrefixedRead(boost::asio::ip::tcp::socket &socket) {
  uint32_t msgSize = 0;
  boost::asio::read(socket, boost::asio::buffer(&msgSize, sizeof(msgSize)));
  msgSize = ntohl(msgSize);
  std::shared_ptr<void> msgBuffer =
      ::tt::runtime::utils::malloc_shared(msgSize);
  boost::asio::read(socket, boost::asio::buffer(msgBuffer.get(), msgSize));
  return msgBuffer;
}

static uint32_t sizePrefixedWrite(boost::asio::ip::tcp::socket &socket,
                                  std::shared_ptr<void> msg, uint32_t msgSize) {
  msgSize = htonl(msgSize);
  boost::asio::write(socket, boost::asio::buffer(&msgSize, sizeof(msgSize)));
  boost::asio::write(socket, boost::asio::buffer(msg.get(), msgSize));
  return msgSize;
}
} // namespace detail

//
// ServerSocket
//
class ServerSocket::Impl {
public:
  explicit Impl(int port)
      : ioContext(std::make_unique<boost::asio::io_context>()),
        acceptor(std::make_unique<boost::asio::ip::tcp::acceptor>(*ioContext)) {
    try {
      acceptor->open(boost::asio::ip::tcp::v4());
      acceptor->set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
      acceptor->bind(
          boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port));
      acceptor->listen();
      LOG_INFO("ServerSocket bound to port ", port);
    } catch (const std::exception &e) {
      LOG_ERROR("Failed to create ServerSocket on port ", port, ": ", e.what());
      throw;
    }
  }
  ~Impl() = default;

  std::vector<Connection> connectToClients(size_t numClients) {
    std::vector<Connection> clientConnections;
    for (size_t i = 0; i < numClients; i++) {
      auto clientSocket = std::make_shared<boost::asio::ip::tcp::socket>(
          acceptor->get_executor());
      acceptor->accept(*clientSocket);
      LOG_DEBUG("Connected to client ", i, " at ",
                clientSocket->remote_endpoint());
      clientConnections.emplace_back(
          Connection(std::static_pointer_cast<void>(clientSocket)));
    }
    return clientConnections;
  }

  void disconnectFromClient(boost::asio::ip::tcp::socket &socket) {
    socket.close();
  }

  std::shared_ptr<void> sizePrefixedRead(boost::asio::ip::tcp::socket &socket) {
    return detail::sizePrefixedRead(socket);
  }

  uint32_t sizePrefixedWrite(boost::asio::ip::tcp::socket &socket,
                             const std::shared_ptr<void> &msg,
                             uint32_t msgSize) {
    return detail::sizePrefixedWrite(socket, msg, msgSize);
  }

private:
  std::unique_ptr<boost::asio::io_context> ioContext;
  std::unique_ptr<boost::asio::ip::tcp::acceptor> acceptor;
};

ServerSocket::ServerSocket(int port) : impl(std::make_unique<Impl>(port)) {}

std::vector<Connection> ServerSocket::connectToClients(size_t numClients) {
  return impl->connectToClients(numClients);
}

void ServerSocket::disconnectFromClient(Connection clientConnection) {
  impl->disconnectFromClient(
      clientConnection.as<boost::asio::ip::tcp::socket>());
}

std::shared_ptr<void>
ServerSocket::sizePrefixedRead(Connection clientConnection) {
  return impl->sizePrefixedRead(
      clientConnection.as<boost::asio::ip::tcp::socket>());
}

uint32_t ServerSocket::sizePrefixedWrite(Connection clientConnection,
                                         const std::shared_ptr<void> &msg,
                                         uint32_t msgSize) {
  return impl->sizePrefixedWrite(
      clientConnection.as<boost::asio::ip::tcp::socket>(), msg, msgSize);
}

//
// ClientSocket
//
class ClientSocket::Impl {
public:
  explicit Impl(const std::string &host, int port)
      : host(host), port(port),
        ioContext(std::make_unique<boost::asio::io_context>()) {}
  ~Impl() = default;

  void connect() {
    LOG_ASSERT(!isConnected(), "ClientSocket is already connected");
    socket = std::make_unique<boost::asio::ip::tcp::socket>(*ioContext);
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
    DEBUG_ASSERT(isConnected(), "ClientSocket is not connected");
    return detail::sizePrefixedRead(*socket);
  }

  uint32_t sizePrefixedWrite(std::shared_ptr<void> msg, uint32_t msgSize) {
    DEBUG_ASSERT(isConnected(), "ClientSocket is not connected");
    return detail::sizePrefixedWrite(*socket, msg, msgSize);
  }

private:
  std::string host;
  int port;
  std::unique_ptr<boost::asio::io_context> ioContext;
  std::unique_ptr<boost::asio::ip::tcp::socket> socket;
};

ClientSocket::ClientSocket(const std::string &host, int port)
    : impl(std::make_unique<Impl>(host, port)) {}

void ClientSocket::connect() { impl->connect(); }

bool ClientSocket::isConnected() const { return impl->isConnected(); }

void ClientSocket::disconnect() { impl->disconnect(); }

std::shared_ptr<void> ClientSocket::sizePrefixedRead() {
  return impl->sizePrefixedRead();
}

uint32_t ClientSocket::sizePrefixedWrite(const std::shared_ptr<void> &msg,
                                         uint32_t msgSize) const {
  return impl->sizePrefixedWrite(msg, msgSize);
}
} // namespace tt::runtime
