// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_TYPES_GLOBAL_TENSOR_CACHE_H
#define TT_RUNTIME_DETAIL_TTNN_TYPES_GLOBAL_TENSOR_CACHE_H

#include "tt/runtime/types.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration for friend class
namespace ttsl {
template <typename T>
class Indestructible;
} // namespace ttsl

namespace tt::runtime {
struct CacheKey {
  int deviceId;
  std::string funcHash;
  std::vector<uint64_t> inputVersions;

  CacheKey() = default;
  CacheKey(int deviceId, const std::string funcHash,
           const std::vector<uint64_t> inputVersions)
      : deviceId(deviceId), funcHash(std::move(funcHash)),
        inputVersions(std::move(inputVersions)) {}

  bool operator==(const CacheKey &other) const {
    return deviceId == other.deviceId && funcHash == other.funcHash &&
           inputVersions == other.inputVersions;
  }
};

std::string to_string(const CacheKey &key);

struct CacheKeyHasher {
  std::size_t operator()(const CacheKey &k) const {
    // TODO(pilkic): Optimize hash function if needed.
    return std::hash<std::string>{}(to_string(k));
  }
};

/**
 * Global runtime cache for const-eval tensor results.
 * This is a singleton that enables sharing cached results across different
 * binaries that have identical consteval functions.
 * The cache stores tensors indexed by content hash and function name.
 */
class GlobalTensorCache {
public:
  static GlobalTensorCache &getInstance();

  // Get all cached tensors (results of a const-eval'ed function) associated
  // with:
  //  - deviceId: the device id where the const-eval function was executed
  //  - funcHash: the hash of the const-eval function
  //  - inputVersions: the versions of the input tensors at the time of caching
  //
  // Returns nullptr if cache entry does not exist.
  const std::vector<Tensor> *getAll(const CacheKey &key) const;

  void store(CacheKey key, std::vector<tt::runtime::Tensor> &inputTensors,
             const std::vector<tt::runtime::Tensor> &outputTensors);

  // Clear the entire cache
  void clear() {
    std::unique_lock<std::shared_mutex> lock(cacheMutex);
    cache.clear();
  }

  void removeIfExists(const CacheKey &key) {
    std::unique_lock<std::shared_mutex> lock(cacheMutex);
    auto it = cache.find(key);
    if (it == cache.end()) {
      return;
    }
    cache.erase(it);
  }

  // Get the size of the cache (number of entries)
  size_t size() const { return cache.size(); }

private:
  friend class ttsl::Indestructible<GlobalTensorCache>;

  GlobalTensorCache() = default;
  ~GlobalTensorCache() = default;

  GlobalTensorCache(const GlobalTensorCache &) = delete;
  GlobalTensorCache &operator=(const GlobalTensorCache &) = delete;
  GlobalTensorCache(GlobalTensorCache &&) = delete;
  GlobalTensorCache &operator=(GlobalTensorCache &&) = delete;

  mutable std::shared_mutex cacheMutex;
  std::unordered_map<CacheKey, std::vector<Tensor>, CacheKeyHasher> cache;
};

} // namespace tt::runtime

#endif // TT_RUNTIME_DETAIL_TTNN_TYPES_GLOBAL_TENSOR_CACHE_H
