// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/server/server.h"

namespace tt::runtime::distributed {

using Server = tt::runtime::distributed::server::Server;

class ServerSingleton {
public:
  static Server &get() {
    if (!server_) {
      server_ = std::make_unique<Server>();
    }
    return *server_;
  }

  static void destroy() { server_.reset(); }

private:
  ServerSingleton() = default;
  ~ServerSingleton() = default;

  static inline std::unique_ptr<Server> server_ = nullptr;
};

} // namespace tt::runtime::distributed
