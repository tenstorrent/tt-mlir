// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/distributed/server/utils/utils.h"
#include <atomic>

namespace tt::runtime::ttnn::distributed::server {
uint64_t nextCommandId() {
  static std::atomic<uint64_t> commandIdCounter = 0;
  return commandIdCounter.fetch_add(1, std::memory_order_relaxed);
}
} // namespace tt::runtime::ttnn::distributed::server
