// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_FORKCONVERSIONCOST_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_FORKCONVERSIONCOST_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>

namespace mlir::tt::ttnn {

/// Compute the physical buffer footprint without wrapping the byte estimate.
/// Shapes are the encoded shard/grid dimensions, not logical tensor dimensions;
/// elementBytes includes the encoded tile size for tiled buffers.
template <typename ShardShape, typename GridShape>
inline std::optional<uint64_t>
getPhysicalBufferBytes(uint64_t elementBytes, const ShardShape &shardShape,
                       const GridShape &gridShape) {
  uint64_t bytes = elementBytes;
  auto multiply = [&](const auto &shape) {
    for (auto dimension : shape) {
      if (dimension <= 0 || bytes > std::numeric_limits<uint64_t>::max() /
                                        static_cast<uint64_t>(dimension)) {
        return false;
      }
      bytes *= static_cast<uint64_t>(dimension);
    }
    return true;
  };
  if (bytes == 0 || !multiply(shardShape) || !multiply(gridShape)) {
    return std::nullopt;
  }
  return bytes;
}

/// Bound the signed shape arithmetic used when rebuilding a physical layout.
/// Leave room for tile padding, shard ceil-divisions, and scalar tile lifting.
/// The tiled interleaved builder also accumulates its tile count in an int.
template <typename Shape, typename GridShape>
inline bool canRebuildForkLayout(const Shape &shape, const GridShape &gridShape,
                                 bool tiled) {
  if (shape.size() < 2 || gridShape.size() != 2) {
    return false;
  }
  constexpr uint64_t limit = std::numeric_limits<int64_t>::max() / 32;
  uint64_t volume = 1;
  size_t index = 0;
  for (auto dimension : shape) {
    if (dimension <= 0 || static_cast<uint64_t>(dimension) > limit) {
      return false;
    }
    uint64_t padded = dimension;
    if (tiled && index + 2 >= shape.size()) {
      padded = ((padded + 31) / 32) * 32;
    }
    if (volume > limit / padded) {
      return false;
    }
    volume *= padded;
    ++index;
  }
  uint64_t gridVolume = 1;
  for (auto dimension : gridShape) {
    if (dimension <= 0 ||
        gridVolume > limit / static_cast<uint64_t>(dimension)) {
      return false;
    }
    gridVolume *= static_cast<uint64_t>(dimension);
  }
  return !tiled || volume / 1024 <= std::numeric_limits<int>::max();
}

/// Estimate read-plus-write bytes for distinct materialized conversions.
/// The caller supplies the materializer's exact cache key and only groups uses
/// for which a conversion can be shared. This is not a latency or spill model.
template <typename Key, typename KeySet = std::set<Key>>
class ForkConversionCost {
public:
  /// Add a conversion once. Failed arithmetic leaves the accumulator unchanged.
  bool add(const Key &key, uint64_t sourceBytes, uint64_t targetBytes) {
    if (keys.find(key) != keys.end()) {
      return true;
    }
    uint64_t limit = std::numeric_limits<uint64_t>::max();
    if (sourceBytes > limit - targetBytes ||
        bytes > limit - (sourceBytes + targetBytes)) {
      return false;
    }
    bytes += sourceBytes + targetBytes;
    keys.insert(key);
    return true;
  }

  uint64_t getBytes() const { return bytes; }
  std::size_t getConversionCount() const { return keys.size(); }

private:
  uint64_t bytes = 0;
  KeySet keys;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_FORKCONVERSIONCOST_H
