// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPSMODELCACHE_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPSMODELCACHE_H

#include "ttmlir/OpModel/TTNN/TTNNOpConstraints.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <type_traits>

namespace mlir::tt::ttnn {

// Forward declaration of the singleton class.
template <typename ValueT>
class TTNNOpModelCache;

// Singleton accessor functions.
TTNNOpModelCache<op_model::ttnn::OpConstraints> &opConstraintsCache();
TTNNOpModelCache<size_t> &opRuntimeCache();

// A cache for TTNN operation model results. This cache stores the results of
// getOpConstraints and getOpRuntime calls to avoid redundant computations.
// Using this cache results in a 20-30% average compile time reduction.
template <typename ValueT>
class TTNNOpModelCache {
  // It is important to define the singleton accessor functions to prevent
  // multiple instances of the cache to be created.
  friend TTNNOpModelCache<op_model::ttnn::OpConstraints> &opConstraintsCache();
  friend TTNNOpModelCache<size_t> &opRuntimeCache();

public:
  TTNNOpModelCache(const TTNNOpModelCache &) = delete;
  TTNNOpModelCache &operator=(const TTNNOpModelCache &) = delete;

  // Statistics about cache performance.
  struct CacheStats {
    size_t hits = 0;    // Number of cache hits
    size_t misses = 0;  // Number of cache misses
    size_t entries = 0; // Total number of entries in the cache
  };

  // Get current cache statistics.
  CacheStats getStats() const { return stats; }

  // Clear the cache and reset statistics.
  void clear() {
    cache.clear();
    stats = CacheStats{};
  }

  // Get the total number of cached items.
  size_t size() const { return stats.entries; }

  bool empty() const { return size() == 0; }

  // Get cache statistics as a string.
  std::string statsToString() const {
    const size_t total = stats.hits + stats.misses;
    if (total == 0) {
      return "  No cache statistics available (no accesses recorded)\n";
    }

    const double hitRatio = (static_cast<double>(stats.hits) / total) * 100.0;
    const double missRatio =
        (static_cast<double>(stats.misses) / total) * 100.0;

    std::string statsStr = "  Cache Statistics (" + std::to_string(total) +
                           " total accesses):\n" +
                           "    Hits: " + std::to_string(stats.hits) + " (" +
                           std::to_string(hitRatio) + "%)\n" +
                           "    Misses: " + std::to_string(stats.misses) +
                           " (" + std::to_string(missRatio) + "%)\n" +
                           "    Size: " + std::to_string(size()) + "\n";
    return statsStr;
  }

  // Main interface to get a value from cache or compute it if not present.
  template <typename Callable, typename... Args,
            typename = std::enable_if_t<std::conjunction_v<
                std::is_invocable<Callable, Args...>,
                std::is_same<std::invoke_result_t<Callable, Args...>,
                             llvm::Expected<ValueT>>>>>
  llvm::Expected<ValueT> getOrCompute(Callable &&computeFunc, Operation *op,
                                      Args &&...args) {
    assert(op != nullptr);
    // The following line attempts to combine the arguments into a single
    // hash_code. For user-defined types it attempts to call a hash_value
    // overload (via ADL) for the type (provided at the end of this file).
    llvm::hash_code hashValue = llvm::hash_combine(std::forward<Args>(args)...);

    // Try to read from cache first.
    if (auto cached = tryGetFromCache(op, hashValue)) {
      return *cached;
    }

    // Not in cache, compute the value.
    llvm::Expected<ValueT> result =
        std::forward<Callable>(computeFunc)(std::forward<Args>(args)...);

    // If computation was successful, store the result.
    if (result) {
      storeInCache(op, hashValue, *result);
    }

    return result;
  }

private:
  // Private constructor - only accessible by friend functions.
  TTNNOpModelCache() = default;

  std::optional<ValueT> tryGetFromCache(Operation *op, llvm::hash_code hash) {
    mlir::TypeID opTypeID = op->getName().getTypeID();
    typename Cache::iterator cacheIt = cache.find(opTypeID);
    if (cacheIt == cache.end()) {
      stats.misses++;
      return std::nullopt;
    }

    OpCache &opCache = cacheIt->second;
    typename OpCache::iterator opCacheIt = opCache.find(hash);
    if (opCacheIt == opCache.end()) {
      stats.misses++;
      return std::nullopt;
    }

    stats.hits++;
    return opCacheIt->second;
  }

  void storeInCache(Operation *op, llvm::hash_code hash, const ValueT &value) {
    mlir::TypeID opTypeID = op->getName().getTypeID();
    cache[opTypeID][hash] = value;
    stats.entries++;
  }

  // This class uses indirect hashing to enable caching for each op type
  // separately. This helps in reducing the number of cache misses and also
  // enables the compiler to produce more meaningful stats if we want to report
  // cache stats for each op type separately.
  // According to llvm docs, mlir::TypeID is unique for each Operation*
  // (https://mlir.llvm.org/doxygen/classmlir_1_1TypeID.html), so it is safe and
  // efficient to use it as a key in the cache.
  using OpCache = llvm::DenseMap<llvm::hash_code, ValueT>;
  using Cache = llvm::DenseMap<mlir::TypeID, OpCache>;

  Cache cache;
  CacheStats stats;
};

// Singleton accessor implementations
inline TTNNOpModelCache<op_model::ttnn::OpConstraints> &opConstraintsCache() {
  // According to C++11 standards:
  //  §6.7 [stmt.dcl] p4 If control enters the declaration concurrently while
  //  the variable is being initialized, the concurrent execution shall wait for
  //  completion of the initialization.
  static TTNNOpModelCache<op_model::ttnn::OpConstraints> instance;
  return instance;
}

inline TTNNOpModelCache<size_t> &opRuntimeCache() {
  static TTNNOpModelCache<size_t> instance;
  return instance;
}

// =---------------------------------------------------------------------------=
// The definition for hash_value(std::vector<mlir::tt::ttnn::TTNNLayoutAttr>) is
// needed. Therefore, it should be defined in both llvm and mlir::tt::ttnn
// namespaces.
template <typename T>
llvm::hash_code hash_value(const std::vector<T> &arg) {
  return llvm::hash_combine_range(arg);
}

} // namespace mlir::tt::ttnn

namespace llvm {
// The following definitions are not found by compiler in any header file. LLVM
// needs to know how to hash all argument types of TTNN ops.
template <typename T>
hash_code hash_value(const llvm::SmallVector<T> &arg) {
  return hash_combine_range(arg);
}

template <typename T>
hash_code hash_value(const std::vector<T> &arg) {
  return hash_combine_range(arg);
}

} // namespace llvm

#endif // TTMLIR_OPMODEL_TTNN_TTNNOPSMODELCACHE_H
