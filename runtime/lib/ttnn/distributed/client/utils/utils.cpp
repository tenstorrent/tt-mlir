// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/distributed/client/utils/utils.h"
#include <atomic>

namespace tt::runtime::ttnn::distributed::client {
uint64_t nextResponseId() {
  static std::atomic<uint64_t> responseIdCounter = 0;
  return responseIdCounter.fetch_add(1, std::memory_order_relaxed);
}
} // namespace tt::runtime::ttnn::distributed::client
