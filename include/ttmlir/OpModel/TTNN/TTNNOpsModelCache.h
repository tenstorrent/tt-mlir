// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPSMODELCACHE_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPSMODELCACHE_H

#include "ttmlir/Dialect/TTNN/Interfaces/OpModelError.h"
#include "ttmlir/OpModel/TTNN/TTNNOpConstraints.h"
// Self-guards on TTMLIR_ENABLE_OPMODEL (expands to nothing when OpModel is
// disabled); the device-generation lookup below is guarded separately.
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace mlir::tt::ttnn {

// Forward declaration of the singleton class.
template <typename ValueT>
class TTNNOpModelCache;

// Singleton accessor functions.
TTNNOpModelCache<op_model::OpConstraints> &opConstraintsCache();
TTNNOpModelCache<size_t> &opRuntimeCache();

// Trimming down large error messages.
inline std::string trimFailureMessage(std::string message) {
  constexpr int keptLines = 8;
  constexpr size_t keptBytes = 4096;
  std::string bounded =
      ttmlir::utils::firstNLines(std::move(message), keptLines);
  if (bounded.size() > keptBytes) {
    bounded.resize(keptBytes);
  }
  return bounded;
}

// A cache for TTNN operation model results. This cache stores the results of
// getOpConstraints and getOpRuntime calls to avoid redundant computations.
// Using this cache results in a 20-30% average compile time reduction.
//
// Failures are memoized too, not just accepted results.
template <typename ValueT>
class TTNNOpModelCache {
  // It is important to define the singleton accessor functions to prevent
  // multiple instances of the cache to be created.
  friend TTNNOpModelCache<op_model::OpConstraints> &opConstraintsCache();
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
    values.clear();
    failures.clear();
    stats = CacheStats{};
  }

  // Get the total number of cached items, accepted and rejected.
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
    // Cached values may contain MLIR attributes/types, which are owned by the
    // MLIRContext that created them. Drop entries before crossing context
    // boundaries.
    MLIRContext *opContext = op->getContext();
    if (opContext != context) {
      clear();
      context = opContext;
    }
#ifdef TTMLIR_ENABLE_OPMODEL
    // Cached entries are computed against the active device's grid. If the
    // device session's grid changed, drop the now-stale entries. The device
    // context owns the change signal (a generation counter) and knows nothing
    // about this cache.
    const uint64_t deviceGen =
        op_model::SingletonDeviceContext::getInstance().getDeviceGeneration();
    if (deviceGen != generation) {
      clear();
      generation = deviceGen;
    }
#endif
    // The following line attempts to combine the arguments into a single
    // hash_code. For user-defined types it attempts to call a hash_value
    // overload (via ADL) for the type (provided at the end of this file).
    llvm::hash_code hashValue = llvm::hash_combine(std::forward<Args>(args)...);

    if (std::optional<llvm::Expected<ValueT>> cached =
            tryGetFromCache(op, hashValue)) {
      return std::move(*cached);
    }

    // Not in cache, compute the value.
    llvm::Expected<ValueT> result =
        std::forward<Callable>(computeFunc)(std::forward<Args>(args)...);

    if (result) {
      storeInCache(op, hashValue, *result);
      return result;
    }

    llvm::Error error = result.takeError();
    // We don't cache OpNotSupportedError - the caller depends on the exact
    // type.
    if (error.isA<detail::OpNotSupportedError>()) {
      return std::move(error);
    }
    // takeError() consumed the error, so hand back an equivalent one.
    std::string message = trimFailureMessage(llvm::toString(std::move(error)));
    llvm::Error rejection =
        llvm::createStringError(llvm::inconvertibleErrorCode(), message);
    storeInCache(op, hashValue, std::move(message));
    return rejection;
  }

private:
  // Private constructor - only accessible by friend functions.
  TTNNOpModelCache() = default;

  ~TTNNOpModelCache() = default;

  // Returns std::nullopt on a miss.
  std::optional<llvm::Expected<ValueT>> tryGetFromCache(Operation *op,
                                                        llvm::hash_code hash) {
    mlir::TypeID opTypeID = op->getName().getTypeID();
    typename SuccessCacheMap::const_iterator valueOpIt = values.find(opTypeID);
    if (valueOpIt != values.end()) {
      typename SuccessCache::const_iterator it = valueOpIt->second.find(hash);
      if (it != valueOpIt->second.end()) {
        stats.hits++;
        return llvm::Expected<ValueT>(it->second);
      }
    }
    if (!failures.empty()) {
      typename FailureCacheMap::const_iterator rejectionOpIt =
          failures.find(opTypeID);
      if (rejectionOpIt != failures.end()) {
        typename FailureCache::const_iterator it =
            rejectionOpIt->second.find(hash);
        if (it != rejectionOpIt->second.end()) {
          stats.hits++;
          return llvm::Expected<ValueT>(llvm::createStringError(
              llvm::inconvertibleErrorCode(), it->second));
        }
      }
    }
    stats.misses++;
    return std::nullopt;
  }

  // try_emplace rather than operator[]: a re-store of a key already present
  // must not double-count `entries`, which is what size()/empty() report.
  void storeInCache(Operation *op, llvm::hash_code hash, const ValueT &value) {
    mlir::TypeID opTypeID = op->getName().getTypeID();
    if (values[opTypeID].try_emplace(hash, value).second) {
      stats.entries++;
    }
  }

  void storeInCache(Operation *op, llvm::hash_code hash, std::string message) {
    mlir::TypeID opTypeID = op->getName().getTypeID();
    if (failures[opTypeID].try_emplace(hash, std::move(message)).second) {
      stats.entries++;
    }
  }

  // This class uses indirect hashing to enable caching for each op type
  // separately. This helps in reducing the number of cache misses and also
  // enables the compiler to produce more meaningful stats if we want to report
  // cache stats for each op type separately.
  // According to llvm docs, mlir::TypeID is unique for each Operation*
  // (https://mlir.llvm.org/doxygen/classmlir_1_1TypeID.html), so it is safe and
  // efficient to use it as a key in the cache.
  //
  // Kept separate from SuccessCache: llvm::Error/Expected is not suitable for
  // storing in cache, so a failure is a plain string, turned into a fresh
  // Error on each hit.
  using SuccessCache = llvm::DenseMap<llvm::hash_code, ValueT>;
  using SuccessCacheMap = llvm::DenseMap<mlir::TypeID, SuccessCache>;
  using FailureCache = llvm::DenseMap<llvm::hash_code, std::string>;
  using FailureCacheMap = llvm::DenseMap<mlir::TypeID, FailureCache>;

  SuccessCacheMap values;
  FailureCacheMap failures;
  CacheStats stats;
  // MLIR context that owns any attributes/types stored in cached values.
  MLIRContext *context = nullptr;
  // Device generation this cache was last filled under; see getOrCompute.
  uint64_t generation = 0;
};

// Singleton accessor implementations
inline TTNNOpModelCache<op_model::OpConstraints> &opConstraintsCache() {
  // According to C++11 standards:
  //  §6.7 [stmt.dcl] p4 If control enters the declaration concurrently while
  //  the variable is being initialized, the concurrent execution shall wait for
  //  completion of the initialization.
  static TTNNOpModelCache<op_model::OpConstraints> instance;
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
inline hash_code hash_value(mlir::Attribute attr) {
  return hash_value(attr.getAsOpaquePointer());
}

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
