// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/types.h"
#include <atomic>

namespace tt::runtime {

std::uint32_t Device::nextDeviceGlobalId() {
  static std::atomic<std::uint32_t> globalId = 0;
  return globalId.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t Tensor::nextTensorGlobalId() {
  static std::atomic<std::uint64_t> globalId = 0;
  return globalId.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t Layout::nextLayoutGlobalId() {
  static std::atomic<std::uint64_t> globalId = 0;
  return globalId.fetch_add(1, std::memory_order_relaxed);
}

} // namespace tt::runtime
