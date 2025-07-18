// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_TYPES_TRACE_CACHE_KEY_H
#define TT_RUNTIME_DETAIL_TTNN_TYPES_TRACE_CACHE_KEY_H

#include <cstdint>
#include <functional>

namespace tt::runtime::ttnn {

class MainProgramKey {
public:
  uint64_t binaryId;
  size_t mainProgramId;

  MainProgramKey(uint64_t binaryId, size_t mainProgramId)
      : binaryId(binaryId), mainProgramId(mainProgramId) {}

  bool operator==(const MainProgramKey &other) const {
    return (other.binaryId == binaryId) &&
           (other.mainProgramId == mainProgramId);
  }
};

class CaptureExecuteProgramKey {
public:
  size_t captureProgramId;
  size_t executeProgramId;

  CaptureExecuteProgramKey(size_t captureProgramId, size_t executeProgramId)
      : captureProgramId(captureProgramId), executeProgramId(executeProgramId) {
  }

  bool operator==(const CaptureExecuteProgramKey &other) const {
    return (other.captureProgramId == captureProgramId) &&
           (other.executeProgramId == executeProgramId);
  }
};

} // namespace tt::runtime::ttnn

namespace std {
template <>
struct hash<::tt::runtime::ttnn::MainProgramKey> {
  std::size_t operator()(const tt::runtime::ttnn::MainProgramKey &key) const {
    std::size_t seed = 0;
    seed ^= std::hash<uint64_t>{}(key.binaryId) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^= std::hash<size_t>{}(key.mainProgramId) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    return seed;
  }
};

template <>
struct hash<::tt::runtime::ttnn::CaptureExecuteProgramKey> {
  std::size_t
  operator()(const tt::runtime::ttnn::CaptureExecuteProgramKey &key) const {
    std::size_t seed = 0;
    seed ^= std::hash<size_t>{}(key.captureProgramId) + 0x9e3779b9 +
            (seed << 6) + (seed >> 2);
    seed ^= std::hash<size_t>{}(key.executeProgramId) + 0x9e3779b9 +
            (seed << 6) + (seed >> 2);
    return seed;
  }
};
} // namespace std

#endif
