// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPSMODELCACHE_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPSMODELCACHE_H

#include "ttmlir/OpModel/TTNN/TTNNOpConstraints.h"
// Self-guards on TTMLIR_ENABLE_OPMODEL (expands to nothing when OpModel is
// disabled); the device-generation lookup below is guarded separately.
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <type_traits>

namespace mlir::tt::ttnn {

// Forward declaration of the singleton class.
template <typename ValueT>
class TTNNOpModelCache;

// Singleton accessor functions.
TTNNOpModelCache<op_model::OpConstraints> &opConstraintsCache();
TTNNOpModelCache<size_t> &opRuntimeCache();

// Memoize failed queries too, not just successful ones. A layout search rejects
// far more candidates than it accepts, and the rejected ones repeat (a decoder
// stack re-queries the same failing tuples once per identical layer), so without
// this every rejection is recomputed. Opt-in while it is being evaluated: a
// cached rejection that was not reproducible would silently drop a legal layout
// out of the search.
inline bool opModelCacheFailuresEnabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("TTMLIR_OPMODEL_CACHE_FAILURES");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

// Report hit/miss/rejection counts on the way out, so a compile can be measured
// without a pass having to reach into the cache.
inline bool opModelCacheStatsEnabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("TTMLIR_OPMODEL_CACHE_STATS");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

// TODO: enforce the mock-device precondition. It cannot be checked from here:
// TTMLIR_ENABLE_OPMODEL is a PUBLIC compile definition on the OpModel library
// alone (lib/OpModel/TTNN/CMakeLists.txt) and is NOT defined for
// MLIRTTNNInterfaces, the only target that instantiates getOrCompute, so
// SingletonDeviceContext is not even declared here. (The same #ifdef silently
// disables the device-generation invalidation below, which is a separate bug.)
// Enforcing it needs an accessor declared outside that guard and defined in the
// OpModel library. Until then this is env-gated and off by default.
inline bool opModelCacheRejectionsAllowed() { return true; }

// This message means the rejection did not come from ttnn: an exception escaped
// and executeConstraintQuery lost it, because the assert meant to stop us is
// compiled out in a release build. A host bad_alloc or a broken invariant lands
// here, so the verdict is not a property of the key and must not be replayed.
inline bool isUncacheableRejection(const std::string &message) {
  return message.find("<error message not set>") != std::string::npos;
}

// Rejections are only ever displayed truncated, and a six-figure rejection count
// makes the untruncated metal text worth dropping.
inline std::string truncateRejection(std::string message) {
  constexpr int keptLines = 8;
  size_t pos = 0;
  for (int i = 0; i < keptLines; ++i) {
    pos = message.find('\n', pos);
    if (pos == std::string::npos) {
      return message;
    }
    ++pos;
  }
  message.resize(pos);
  return message;
}

// A cache for TTNN operation model results. This cache stores the results of
// getOpConstraints and getOpRuntime calls to avoid redundant computations.
// Using this cache results in a 20-30% average compile time reduction.
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
    // Rejection accounting. `failedComputes` counts backend queries that ran and
    // came back rejected; `failureHits` counts the ones a cached rejection stood
    // in for. The ratio between them is the whole value of caching failures.
    size_t failedComputes = 0;
    size_t failureEntries = 0;
    size_t failureHits = 0;
    size_t clears = 0; // invalidations, so a low hit count can be explained
  };

  // Get current cache statistics.
  CacheStats getStats() const { return stats; }

  // Clear the cache and reset statistics.
  void clear() {
    cache.clear();
    // Query counters describe work already spent rather than what is currently
    // cached, so they outlive an invalidation: a mid-compile clear would
    // otherwise hide most of the compile from the stats dump.
    const CacheStats prev = stats;
    stats = CacheStats{};
    stats.failedComputes = prev.failedComputes;
    stats.failureHits = prev.failureHits;
    stats.clears = prev.clears + 1;
  }

  // Get the total number of cached items.
  size_t size() const { return stats.entries; }

  bool empty() const { return size() == 0; }

  // Get cache statistics as a string.
  std::string statsToString() const {
    const size_t total = stats.hits + stats.misses;
    if (total == 0 && stats.failedComputes == 0 && stats.failureHits == 0) {
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
                           "    Size: " + std::to_string(size()) + "\n" +
                           "    Failed computes: " +
                           std::to_string(stats.failedComputes) + "\n" +
                           "    Failure entries: " +
                           std::to_string(stats.failureEntries) + "\n" +
                           "    Failure hits: " +
                           std::to_string(stats.failureHits) + "\n" +
                           "    Clears: " + std::to_string(stats.clears) + "\n";
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

    // Try to read from cache first.
    if (std::optional<Entry> cached = tryGetFromCache(op, hashValue)) {
      if (cached->rejected) {
        stats.failureHits++;
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       cached->message);
      }
      return cached->value;
    }

    // Not in cache, compute the value.
    llvm::Expected<ValueT> result =
        std::forward<Callable>(computeFunc)(std::forward<Args>(args)...);

    if (result) {
      Entry accepted;
      accepted.value = *result;
      storeInCache(op, hashValue, accepted);
      return result;
    }

    stats.failedComputes++;
    if (!cacheRejections || !opModelCacheFailuresEnabled() ||
        !opModelCacheRejectionsAllowed()) {
      return result;
    }
    // takeError() consumes the error, so hand back an equivalent one. Callers
    // treat a rejection as a boolean plus a message for diagnostics.
    std::string message = llvm::toString(result.takeError());
    if (isUncacheableRejection(message)) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
    }
    Entry entry;
    entry.rejected = true;
    entry.message = truncateRejection(std::move(message));
    storeInCache(op, hashValue, entry);
    stats.failureEntries++;
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   entry.message);
  }

private:
  // Private constructor - only accessible by friend functions.
  explicit TTNNOpModelCache(const char *name, bool cacheRejections)
      : name(name), cacheRejections(cacheRejections) {}

  // Reports on the way out. Uses fprintf rather than llvm::errs(): this is a
  // function-local static whose destruction order against llvm's own stream
  // statics is unspecified.
  ~TTNNOpModelCache() {
    if (opModelCacheStatsEnabled()) {
      std::fprintf(stderr, "TTNNOpModelCache<%s> [failures cached: %s]\n%s",
                   name, opModelCacheFailuresEnabled() ? "yes" : "no",
                   statsToString().c_str());
    }
  }

  // A cached query outcome: either a value the backend accepted, or the message
  // it was rejected with.
  struct Entry {
    ValueT value{};
    bool rejected = false;
    std::string message;
  };

  std::optional<Entry> tryGetFromCache(Operation *op, llvm::hash_code hash) {
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

  void storeInCache(Operation *op, llvm::hash_code hash, const Entry &entry) {
    mlir::TypeID opTypeID = op->getName().getTypeID();
    cache[opTypeID][hash] = entry;
    stats.entries++;
  }

  // This class uses indirect hashing to enable caching for each op type
  // separately. This helps in reducing the number of cache misses and also
  // enables the compiler to produce more meaningful stats if we want to report
  // cache stats for each op type separately.
  // According to llvm docs, mlir::TypeID is unique for each Operation*
  // (https://mlir.llvm.org/doxygen/classmlir_1_1TypeID.html), so it is safe and
  // efficient to use it as a key in the cache.
  using OpCache = llvm::DenseMap<llvm::hash_code, Entry>;
  using Cache = llvm::DenseMap<mlir::TypeID, OpCache>;

  const char *name;
  // getOpRuntime rejects unconditionally under a mock device and otherwise
  // executes for real, so only constraint rejections are memoizable.
  const bool cacheRejections;
  Cache cache;
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
  static TTNNOpModelCache<op_model::OpConstraints> instance(
      "constraints", /*cacheRejections=*/true);
  return instance;
}

inline TTNNOpModelCache<size_t> &opRuntimeCache() {
  static TTNNOpModelCache<size_t> instance("runtime",
                                           /*cacheRejections=*/false);
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
