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
class TensorWrapper {
public:
  TensorWrapper(::ttnn::Tensor tensor)
      : tensor(std::move(tensor)), refCount(1) {}

  // Increment reference count
  void addRef() { ++refCount; }

  // Decrement reference count and return true if the tensor should be
  // deallocated
  bool release() { return --refCount == 0; }

  // Get the wrapped tensor
  ::ttnn::Tensor &getTensor() { return tensor; }
  const ::ttnn::Tensor &getTensor() const { return tensor; }

  // Get the current reference count
  uint32_t getRefCount() const { return refCount; }

private:
  ::ttnn::Tensor tensor;
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

  // Check if a tensor exists in the cache
  bool contains(const CacheKey &key) const {
    return cache.find(key) != cache.end();
  }

  // Get a tensor from the cache, incrementing its reference count
  ::ttnn::Tensor *get(const CacheKey &key) {
    auto it = cache.find(key);
    if (it != cache.end()) {
      it->second.addRef();
      return &it->second.getTensor();
    }
    return nullptr;
  }

  // Add a tensor to the cache
  void add(const CacheKey &key, ::ttnn::Tensor tensor) {
    cache.emplace(key, TensorWrapper(std::move(tensor)));
  }

  // Release a tensor, removing it from the cache if the reference count reaches
  // zero
  void release(const CacheKey &key) {
    auto it = cache.find(key);
    if (it != cache.end() && it->second.release()) {
      cache.erase(it);
    }
  }

  // Clear the entire cache
  void clear() { cache.clear(); }

private:
  std::unordered_map<CacheKey, TensorWrapper, CacheKeyHash> cache;
};

} // namespace tt::runtime::ttnn

#endif // TT_RUNTIME_TTNN_TENSOR_CACHE_H
