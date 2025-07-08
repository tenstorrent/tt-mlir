// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_TYPES_TRACE_CACHE_KEY_H
#define TT_RUNTIME_DETAIL_TTNN_TYPES_TRACE_CACHE_KEY_H

#include <cstdint>
#include <functional>

namespace tt::runtime::ttnn {

class TraceCacheKey {
public:
  uint64_t binaryId;
  uint64_t traceFuncId;

  TraceCacheKey(uint64_t binaryId, uint64_t traceFuncId)
      : binaryId(binaryId), traceFuncId(traceFuncId) {}

  bool operator==(const TraceCacheKey &other) const {
    return (other.binaryId == binaryId) && (other.traceFuncId == traceFuncId);
  }
};

} // namespace tt::runtime::ttnn

namespace std {
template <>
struct hash<::tt::runtime::ttnn::TraceCacheKey> {
  std::size_t operator()(const tt::runtime::ttnn::TraceCacheKey &key) const {
    std::size_t seed = 0;
    seed ^= std::hash<uint64_t>{}(key.binaryId) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^= std::hash<uint64_t>{}(key.traceFuncId) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    return seed;
  }
};
} // namespace std

#endif
