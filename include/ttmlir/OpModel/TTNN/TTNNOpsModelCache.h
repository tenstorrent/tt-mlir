// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNOPSMODELCACHE_H
#define TTMLIR_DIALECT_TTNN_IR_TTNNOPSMODELCACHE_H

// #include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
// #include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
// #include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpConstraints.h"

#include <iostream>
#include <llvm/ADT/DenseMap.h>
#include <mutex>
#include <shared_mutex>

namespace mlir::tt::ttnn {

/// A thread-safe cache for TTNN operation model results.
/// This cache stores the results of getOpConstraints and getOpRuntime calls
/// to avoid redundant computations.
template <typename ValueT>
class TTNNOpModelCache {
public:
  template <typename Callable, typename... Args>
  llvm::Expected<ValueT> getOrCompute(Callable &&computeFunc, Operation *op,
                                      Args &&...args) {
    HashValue hash = llvm::hash_combine(args...);

    // Try to read from cache first (shared lock)
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      typename Cache::iterator cacheIt = cache_.find(op);
      if (cacheIt != cache_.end()) {
        OperationMap &operationMap = cacheIt->second;
        typename OperationMap::iterator operationIt = operationMap.find(hash);
        if (operationIt != operationMap.end()) {
          std::cout << "Found in cache\n";
          // Return the cached value
          return operationIt->second;
        }
      }
    }

    std::cout << "Not found in cache, calculating\n";
    // Compute the value if not in cache
    llvm::Expected<ValueT> result =
        std::forward<Callable>(computeFunc)(std::forward<Args>(args)...);
    // Only cache successful results
    if (result) {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      // Check again in case another thread computed the same value
      typename Cache::iterator cacheIt = cache_.find(op);
      if (cacheIt != cache_.end()) {
        // OperationMap already exists for the op:
        OperationMap &operationMap = cacheIt->second;
        auto [it, inserted] = operationMap.try_emplace(hash, *result);
        if (!inserted) {
          // If another thread beat us to it, use their value
          return it->second;
        }
      } else {
        // OperationMap does not exist for the op, create it:
        OperationMap operationMap;
        auto [it, inserted] = operationMap.try_emplace(hash, *result);
        if (!inserted) {
          // If another thread beat us to it, use their value
          return it->second;
        }
        cache_.try_emplace(op, operationMap);
      }
    }
    return result;
  }

  /// Clear the cache.
  void clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    cache_.clear();
  }

private:
  using HashValue = std::size_t;
  using OperationMap = llvm::DenseMap<HashValue, ValueT>;
  using Cache = llvm::DenseMap<Operation *, OperationMap>;
  Cache cache_;
  mutable std::shared_mutex mutex_;
};

// Global cache instances
static TTNNOpModelCache<op_model::ttnn::OpConstraints> opConstraintsCache;
static TTNNOpModelCache<size_t> opRuntimeCache;

} // namespace mlir::tt::ttnn

namespace llvm {
template <typename T>
hash_code hash_value(const llvm::SmallVector<T> &arg) {
  return hash_combine_range(arg);
}

// hash_value for std::vector<T>
template <typename T>
hash_code hash_value(const std::vector<T> &arg) {
  return hash_combine_range(arg);
}

// template <typename T>
// hash_code hash_value(const mlir::tt::ttnn::TTNNLayoutAttr &arg) {
//   std::cout << "Hashing TTNNLayoutAttr\n";
//   return hash_value(static_cast<const ::mlir::Attribute &>(arg));
// }

} // namespace llvm

namespace mlir::tt::ttnn {
// The definition for hash_value(std::vector<mlir::tt::ttnn::TTNNLayoutAttr>) is
// needed. Therefore, it should be defined in both llvm and mlir::tt::ttnn
// namespaces.
template <typename T>
llvm::hash_code hash_value(const std::vector<T> &arg) {
  return llvm::hash_combine_range(arg);
}
} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_IR_TTNNOPSMODELCACHE_H
