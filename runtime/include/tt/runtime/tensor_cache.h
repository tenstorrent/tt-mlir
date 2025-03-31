// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TENSOR_CACHE_H
#define TT_RUNTIME_TENSOR_CACHE_H

#include "tt/runtime/types.h"

#include <cstdint>
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

  // Check if a function name exists in the cache
  bool contains(const std::string &functionName) const {
    return cache.find(functionName) != cache.end();
  }

  // Check if a cache entry is valid (input tensor versions match)
  bool isValid(const std::string &functionName,
               const std::vector<Tensor> &inputs) const {
    auto it = cache.find(functionName);
    if (it == cache.end()) {
      // If not in cache, it's not valid
      return false;
    }

    // For now, assume all tensors are valid since we're not tracking versions
    return true;
  }

  // Get all cached tensors for a function name if the cache is valid
  // Returns nullptr if cache is invalid or not found
  const std::vector<Tensor> *
  getAll(const std::string &functionName,
         const std::vector<uint64_t> &inputVersions) const {
    auto it = cache.find(functionName);
    if (it == cache.end()) {
      return nullptr;
    }

    // For now, ignore version checking
    return &it->second.tensors;
  }

  // Store tensors and their input versions in the cache
  void store(const std::string &functionName, std::vector<Tensor> tensors,
             const std::vector<Tensor> &inputs) {
    CacheValue value;

    // For now, use zeros for versions since we're not tracking them
    value.inputVersions.resize(inputs.size(), 0);

    value.tensors = std::move(tensors);

    // Replace any existing entry
    cache[functionName] = std::move(value);
  }

  // New overload that takes input versions directly
  void store(const std::string &functionName, std::vector<Tensor> tensors,
             const std::vector<uint64_t> &inputVersions) {
    CacheValue value;
    value.inputVersions = inputVersions;
    value.tensors = std::move(tensors);

    // Replace any existing entry
    cache[functionName] = std::move(value);
  }

  // Remove a function name and its associated tensors from the cache
  void remove(const std::string &functionName) { cache.erase(functionName); }

  // Clear the entire cache
  void clear() { cache.clear(); }

  // Get the size of the cache (number of entries)
  size_t size() const { return cache.size(); }

  // Get cache statistics
  std::unordered_map<std::string, size_t> getStats() const {
    std::unordered_map<std::string, size_t> stats;
    stats["total_entries"] = cache.size();

    size_t total_tensors = 0;
    for (const auto &[key, value] : cache) {
      total_tensors += value.tensors.size();
    }
    stats["total_tensors"] = total_tensors;

    return stats;
  }

private:
  std::unordered_map<std::string, CacheValue> cache;
};

} // namespace tt::runtime

#endif // TT_RUNTIME_TTNN_TENSOR_CACHE_H
