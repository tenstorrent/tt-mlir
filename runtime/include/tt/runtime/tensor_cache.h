// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TENSOR_CACHE_H
#define TT_RUNTIME_TENSOR_CACHE_H

#include "tt/runtime/debug.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt::runtime {

/**
 * Generate a cache key using the device ID and program index.
 * This provides a unique identifier for each program execution context.
 */
inline std::string generateCacheOuterKey(const int deviceId,
                                         const size_t programIndex) {
  return std::to_string(deviceId) + ":" + std::to_string(programIndex);
}

/**
 * A cache value that stores both input tensor versions and output
 * tensors.
 * When input tensor versions change, the cached outputs need to be recomputed.
 */
struct CacheValue {
  // Input tensor versions used to compute the cached values
  std::vector<uint64_t> inputVersions;
  // The cached output tensors
  std::vector<Tensor> tensors;
};

/**
 * Runtime cache for const-eval tensor results.
 * The cache stores tensors indexed by function name. When querying the cache,
 * both the function name and current input tensor versions are checked to
 * determine if the cached value is still valid.
 */
class TensorCache {
public:
  TensorCache() = default;
  ~TensorCache() = default;

  TensorCache(const TensorCache &) = delete;
  TensorCache &operator=(const TensorCache &) = delete;
  TensorCache(TensorCache &&) = delete;
  TensorCache &operator=(TensorCache &&) = delete;

  // Get all cached tensors for a function name if the cache is valid
  // Returns nullptr if cache is invalid or not found
  const std::vector<Tensor> *
  getAll(const std::string &parentFuncName,
         const std::string &constEvalFuncName,
         const std::vector<uint64_t> &inputVersions) const {
    std::shared_lock<std::shared_mutex> lock(cacheMutex);
    auto it = cache.find(parentFuncName);
    if (it == cache.end()) {
      debug::Stats::get().incrementStat("ConstEvalCacheMiss");
      return nullptr;
    }

    auto internalIt = it->second.find(constEvalFuncName);
    if (internalIt == it->second.end()) {
      debug::Stats::get().incrementStat("ConstEvalCacheMiss");
      return nullptr;
    }

    const CacheValue &value = internalIt->second;
    if (value.inputVersions != inputVersions) {
      debug::Stats::get().incrementStat("ConstEvalCacheMiss");
      return nullptr;
    }
    debug::Stats::get().incrementStat("ConstEvalCacheHit");
    return &value.tensors;
  }

  // Store tensors with explicit input versions
  // Note: if ttrt used C++20 we could replace this code with proper
  // concept/constraint.
  template <typename VersionVec,
            typename = std::enable_if_t<std::is_convertible_v<
                std::decay_t<VersionVec>, std::vector<uint64_t>>>>
  void store(const std::string &parentFuncName,
             const std::string &constEvalFuncName, VersionVec &&inputVersions,
             const std::vector<tt::runtime::Tensor> &tensors) {
    std::unique_lock<std::shared_mutex> lock(cacheMutex);
    cache[parentFuncName][constEvalFuncName] =
        CacheValue{std::forward<VersionVec>(inputVersions), tensors};
  }

  // Clear the entire cache
  void clear() {
    std::unique_lock<std::shared_mutex> lock(cacheMutex);
    cache.clear();
  }

  // Get the size of the cache (number of entries)
  size_t size() const { return cache.size(); }

  // Remove all const-eval funcs associated with a given outer key.
  void remove(const std::string &outerKey) {
    std::unique_lock<std::shared_mutex> lock(cacheMutex);
    auto it = cache.find(outerKey);
    assert(it != cache.end() && "Outer key not found in remove() call!");
    cache.erase(it);
  }

  // Remove all const-eval funcs associated with a given device id + program
  // index.
  void remove(const int deviceId, const size_t programIdx) {
    remove(generateCacheOuterKey(deviceId, programIdx));
  }

private:
  mutable std::shared_mutex cacheMutex;
  // Outer key should be combination of device id and program index, created via
  // generateCacheOuterKey. Inner key will be const-eval func name.
  std::unordered_map<std::string, std::unordered_map<std::string, CacheValue>>
      cache;
};

} // namespace tt::runtime

#endif // TT_RUNTIME_TENSOR_CACHE_H
