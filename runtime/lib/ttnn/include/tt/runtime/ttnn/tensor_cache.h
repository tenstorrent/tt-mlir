// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_TENSOR_CACHE_H
#define TT_RUNTIME_TTNN_TENSOR_CACHE_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace tt::runtime::ttnn {

/**
 * @brief A cache key used to identify tensors in the cache.
 *
 * The key consists of a function name and a hash of the input tensors.
 */
class CacheKey {
public:
  CacheKey(const std::string &functionName,
           const std::vector<uint32_t> &inputIds)
      : functionName(functionName), inputIds(inputIds) {}

  bool operator==(const CacheKey &other) const {
    return functionName == other.functionName && inputIds == other.inputIds;
  }

  const std::string &getFunctionName() const { return functionName; }
  const std::vector<uint32_t> &getInputIds() const { return inputIds; }

private:
  std::string functionName;
  std::vector<uint32_t> inputIds;
};

// Custom hash function for CacheKey
struct CacheKeyHash {
  std::size_t operator()(const CacheKey &key) const {
    std::size_t h1 = std::hash<std::string>{}(key.getFunctionName());
    std::size_t h2 = 0;
    for (const auto &id : key.getInputIds()) {
      h2 ^= std::hash<uint32_t>{}(id) + 0x9e3779b9 + (h2 << 6) + (h2 >> 2);
    }
    return h1 ^ (h2 << 1);
  }
};

/**
 * @brief A wrapper for ttnn::Tensor that adds reference counting.
 *
 * This class is used to manage the lifetime of tensors in the cache.
 */
class TensorPtrWrapper {
public:
  TensorPtrWrapper(::ttnn::Tensor *tensor) : tensor(tensor), refCount(1) {}

  // Increment reference count
  void addRef() { ++refCount; }

  // Decrement reference count and return true if the tensor should be
  // deallocated
  bool release() { return --refCount == 0; }

  // Get the wrapped tensor
  ::ttnn::Tensor *getTensor() { return tensor; }
  const ::ttnn::Tensor *getTensor() const { return tensor; }

  // Get the current reference count
  uint32_t getRefCount() const { return refCount; }

private:
  ::ttnn::Tensor *tensor;
  uint32_t refCount;
};

/**
 * @brief A cache for tensors that manages their lifetime using reference
 * counting.
 *
 * The cache stores tensors indexed by a CacheKey, which consists of a function
 * name and a hash of the input tensors.
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

  // Check if a key exists in the cache
  bool contains(const CacheKey &key) const {
    return cache.find(key) != cache.end();
  }

  // Get all tensors for a key
  const std::vector<TensorPtrWrapper> &getAll(const CacheKey &key) const {
    static const std::vector<TensorPtrWrapper> empty;
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    }
    return empty;
  }

  // Add a tensor wrapper to the cache under the specified key
  void add(const CacheKey &key, TensorPtrWrapper wrapper) {
    auto it = cache.find(key);
    if (it != cache.end()) {
      // Key exists, add to the vector
      it->second.push_back(std::move(wrapper));
    } else {
      // Key doesn't exist, create a new vector
      std::vector<TensorPtrWrapper> wrappers;
      wrappers.push_back(std::move(wrapper));
      cache.emplace(key, std::move(wrappers));
    }
  }

  // Add a tensor to the cache under the specified key
  void add(const CacheKey &key, ::ttnn::Tensor *tensor) {
    add(key, TensorPtrWrapper(tensor));
  }

  // Add multiple tensor wrappers at once under the specified key
  void addAll(const CacheKey &key, std::vector<TensorPtrWrapper> wrappers) {
    auto it = cache.find(key);
    if (it != cache.end()) {
      // Key exists, move all wrappers to the existing vector
      it->second.reserve(it->second.size() + wrappers.size());
      for (auto &wrapper : wrappers) {
        it->second.push_back(std::move(wrapper));
      }
    } else {
      // Key doesn't exist, move the entire vector
      cache.emplace(key, std::move(wrappers));
    }
  }

  // Remove a key and all its associated tensors from the cache
  void remove(const CacheKey &key) { cache.erase(key); }

  // Clear the entire cache
  void clear() { cache.clear(); }

  // Get the size of the cache (number of entries)
  size_t size() const { return cache.size(); }

  // Get cache statistics
  std::unordered_map<std::string, size_t> getStats() const {
    std::unordered_map<std::string, size_t> stats;
    stats["total_entries"] = cache.size();

    size_t total_tensors = 0;
    for (const auto &[key, wrappers] : cache) {
      total_tensors += wrappers.size();
    }
    stats["total_tensors"] = total_tensors;

    return stats;
  }

private:
  std::unordered_map<CacheKey, std::vector<TensorPtrWrapper>, CacheKeyHash>
      cache;
};

} // namespace tt::runtime::ttnn

#endif // TT_RUNTIME_TTNN_TENSOR_CACHE_H
