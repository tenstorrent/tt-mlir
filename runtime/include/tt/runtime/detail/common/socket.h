// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_SOCKET_H
#define TT_RUNTIME_DETAIL_COMMON_SOCKET_H

#include <memory>
#include <string>

namespace tt::runtime {
class Socket {
public:
  explicit Socket(const std::string &host, int port);
  ~Socket();

  Socket(const Socket &) = delete;
  Socket &operator=(const Socket &) = delete;
  Socket(Socket &&) = default;
  Socket &operator=(Socket &&) = default;

  void connect();
  void disconnect();
  bool isConnected() const;

  std::shared_ptr<void> sizePrefixedRead();

  size_t sizePrefixedWrite(const std::shared_ptr<void> &msg,
                           size_t msgSize) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace tt::runtime

#endif // TT_RUNTIME_DETAIL_COMMON_SOCKET_H
