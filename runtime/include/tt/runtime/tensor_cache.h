// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TENSOR_CACHE_H
#define TT_RUNTIME_TENSOR_CACHE_H

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/types.h"

#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt::runtime {

/**
 * @brief A cache value that stores both input tensor versions and output
 * tensors.
 *
 * When input tensor versions change, the cached outputs need to be recomputed.
 */
struct CacheValue {
  // Input tensor versions used to compute the cached values
  std::vector<uint64_t> inputVersions;
  // The cached output tensors
  std::vector<Tensor> tensors;
};

/**
 * A cache for runtime tensors.
 *
 * The cache stores tensors indexed by function name. When querying the cache,
 * both the function name and current input tensor versions are checked to
 * determine if the cached value is still valid.
 */
class TensorCache {
public:
  TensorCache() = default;
  ~TensorCache() = default;

  // Make the cache copyable and movable
  TensorCache(const TensorCache &) = default;
  TensorCache &operator=(const TensorCache &) = default;
  TensorCache(TensorCache &&) = default;
  TensorCache &operator=(TensorCache &&) = default;

  // Get all cached tensors for a function name if the cache is valid
  // Returns nullptr if cache is invalid or not found
  const std::vector<Tensor> *
  getAll(const std::string &parentFuncName,
         const std::string &constEvalFuncName,
         const std::vector<uint64_t> &inputVersions) const {
    auto it = cache.find(parentFuncName);
    if (it == cache.end()) {
      ++stats["misses"];
      return nullptr;
    }

    auto internalIt = it->second.find(constEvalFuncName);
    if (internalIt == it->second.end()) {
      ++stats["misses"];
      return nullptr;
    }

    const CacheValue &value = internalIt->second;
    if (value.inputVersions != inputVersions) {
      for (size_t i = 0; i < inputVersions.size(); ++i) {
        LOG_INFO("Prev version: ", value.inputVersions[i],
                 " vs given version: ", inputVersions[i]);
      }
      ++stats["misses"];
      return nullptr;
    }
    ++stats["hits"];
    return &value.tensors;
  }

  // Store tensors with explicit input versions
  void store(const std::string &parentFuncName,
             const std::string &constEvalFuncName, std::vector<Tensor> tensors,
             const std::vector<uint64_t> &inputVersions) {
    CacheValue value;
    value.inputVersions = inputVersions;
    value.tensors = std::move(tensors);

    // Replace any existing entry
    cache[parentFuncName][constEvalFuncName] = std::move(value);
  }

  // Clear the entire cache
  void clear() { cache.clear(); }

  // Get the size of the cache (number of entries)
  size_t size() const { return cache.size(); }

  // Get cache statistics
  std::unordered_map<std::string, size_t> getStats() const { return stats; }

  void remove(const std::string &parentFunc) {
    auto it = cache.find(parentFunc);
    if (it == cache.end()) {
      return;
    }
    cache.erase(it);
  }

private:
  std::unordered_map<std::string, std::unordered_map<std::string, CacheValue>>
      cache;

  mutable std::unordered_map<std::string, size_t> stats;
};

} // namespace tt::runtime

#endif // TT_RUNTIME_TENSOR_CACHE_H
