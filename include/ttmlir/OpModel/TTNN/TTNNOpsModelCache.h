// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPSMODELCACHE_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPSMODELCACHE_H

#include "ttmlir/Dialect/TTNN/Interfaces/OpModelError.h"
#include "ttmlir/OpModel/TTNN/TTNNOpConstraints.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
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

// A stored rejection message is bounded, because a six-figure rejection count
// makes metal's untruncated TT_FATAL text worth dropping. The bound is on bytes
// as well as lines: a single TT_FATAL line is routinely longer than the eight
// lines a caller displays, so a line count alone establishes nothing.
// `firstNLines` is the same helper both downstream consumers use
// (OpConstraintValidation, ShardSolver), so a replayed rejection reads exactly
// like a freshly computed one.
inline constexpr int g_rejectionKeptLines = 8;
inline constexpr size_t g_rejectionKeptBytes = 4096;

inline std::string boundRejectionMessage(llvm::StringRef message) {
  std::string bounded =
      ttmlir::utils::firstNLines(message.str(), g_rejectionKeptLines);
  if (bounded.size() > g_rejectionKeptBytes) {
    bounded.resize(g_rejectionKeptBytes);
  }
  return bounded;
}

// A cache for TTNN operation model results. This cache stores the results of
// getOpConstraints and getOpRuntime calls to avoid redundant computations.
// Using this cache results in a 20-30% average compile time reduction.
//
// Failures are memoized too, not just accepted results, because a layout
// search rejects far more candidates than it accepts and the rejected ones
// repeat: a decoder stack re-queries the same failing tuples once per identical
// layer. The one exception is OpNotSupportedError, which is raised by tt-mlir
// without consulting the backend and whose type callers dispatch on (see
// OpConstraintValidation::checkConstraintsResult); it is passed through
// unmemoized with its type intact.
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
    size_t entries = 0; // Accepted results currently cached
    // Rejection accounting. Both counters below are subsets of the two above --
    // `rejectionHits` of `hits`, `rejectionComputes` of `misses` -- so
    // hits + misses is still the total number of accesses and the printed hit
    // ratio stays meaningful. The ratio between them is the whole value of
    // memoizing rejections.
    size_t rejectionHits = 0;
    size_t rejectionComputes = 0;
    size_t rejectionEntries = 0; // Rejections currently cached
    size_t rejectionBytes = 0;   // Total size of cached rejection messages
    // Invalidations, so a low hit ratio can be explained. Counts only drops
    // that discarded something; the first access has nothing to discard.
    size_t invalidations = 0;
  };

  // Get current cache statistics.
  CacheStats getStats() const { return stats; }

  // Clear the cache and reset statistics.
  void clear() {
    values.clear();
    rejections.clear();
    stats = CacheStats{};
  }

  // Get the total number of cached items, accepted and rejected.
  size_t size() const { return stats.entries + stats.rejectionEntries; }

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

    std::string statsStr =
        "  Cache Statistics (" + std::to_string(total) +
        " total accesses):\n" + "    Hits: " + std::to_string(stats.hits) +
        " (" + std::to_string(hitRatio) + "%)\n" +
        "    Misses: " + std::to_string(stats.misses) + " (" +
        std::to_string(missRatio) + "%)\n" +
        "    Size: " + std::to_string(size()) + "\n" +
        "    Rejection hits: " + std::to_string(stats.rejectionHits) + "\n" +
        "    Rejection computes: " + std::to_string(stats.rejectionComputes) +
        "\n" + "    Rejection entries: " +
        std::to_string(stats.rejectionEntries) + " (" +
        std::to_string(stats.rejectionBytes) + " bytes)\n" +
        "    Invalidations: " + std::to_string(stats.invalidations) + "\n";
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
      invalidate();
      context = opContext;
    }
    // Cached entries are computed against the active device's grid. If the
    // device session's grid changed, drop the now-stale entries. The device
    // context owns the change signal (a generation counter) and knows nothing
    // about this cache.
    const uint64_t deviceGen = op_model::getDeviceGeneration();
    if (deviceGen != generation) {
      invalidate();
      generation = deviceGen;
    }
    // The following line attempts to combine the arguments into a single
    // hash_code. For user-defined types it attempts to call a hash_value
    // overload (via ADL) for the type (provided at the end of this file).
    llvm::hash_code hashValue = llvm::hash_combine(std::forward<Args>(args)...);
    mlir::TypeID opTypeID = op->getName().getTypeID();

    if (const ValueT *cached = findValue(opTypeID, hashValue)) {
      stats.hits++;
      return *cached;
    }
    if (const std::string *cached = findRejection(opTypeID, hashValue)) {
      stats.hits++;
      stats.rejectionHits++;
      return llvm::createStringError(llvm::inconvertibleErrorCode(), *cached);
    }
    stats.misses++;

    // Not in cache, compute the value.
    llvm::Expected<ValueT> result =
        std::forward<Callable>(computeFunc)(std::forward<Args>(args)...);

    if (result) {
      storeValue(opTypeID, hashValue, *result);
      return result;
    }

    stats.rejectionComputes++;
    llvm::Error error = result.takeError();
    // OpNotSupportedError is tt-mlir declining to query the backend at all;
    // callers dispatch on its type to tell "not implemented" from "the backend
    // said no" (see OpConstraintValidation::checkConstraintsResult), so it is
    // passed through unmemoized with its type intact. Every other failure is
    // memoized like a success.
    if (!cacheRejections || error.isA<detail::OpNotSupportedError>()) {
      return std::move(error);
    }
    // takeError() consumed the error, so hand back an equivalent one. The
    // bounded copy is what gets cached; the caller still sees the full message,
    // so memoizing does not shorten first-occurrence diagnostics.
    std::string message = llvm::toString(std::move(error));
    storeRejection(opTypeID, hashValue, boundRejectionMessage(message));
    return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
  }

private:
  // Private constructor - only accessible by friend functions.
  explicit TTNNOpModelCache(bool cacheRejections)
      : cacheRejections(cacheRejections) {}

  ~TTNNOpModelCache() = default;

  // Drop everything computed under a different context or device grid. A first
  // access has nothing to drop, so it is not reported as an invalidation:
  // `invalidations` exists to explain a low hit ratio.
  void invalidate() {
    if (values.empty() && rejections.empty()) {
      return;
    }
    values.clear();
    rejections.clear();
    stats.entries = 0;
    stats.rejectionEntries = 0;
    stats.rejectionBytes = 0;
    stats.invalidations++;
  }

  // Returns nullptr on a miss. A pointer rather than a copy: accepted values
  // carry a SmallVector<TTNNLayoutAttr>, and copying one out of the map on
  // every hit was pure overhead. Safe because nothing is inserted between the
  // lookup and the caller's use of the result.
  const ValueT *findValue(mlir::TypeID opTypeID, llvm::hash_code hash) const {
    typename ValueCacheMap::const_iterator opCacheIt = values.find(opTypeID);
    if (opCacheIt == values.end()) {
      return nullptr;
    }
    typename ValueCache::const_iterator it = opCacheIt->second.find(hash);
    return it == opCacheIt->second.end() ? nullptr : &it->second;
  }

  const std::string *findRejection(mlir::TypeID opTypeID,
                                   llvm::hash_code hash) const {
    if (rejections.empty()) {
      return nullptr;
    }
    typename RejectionCacheMap::const_iterator opCacheIt =
        rejections.find(opTypeID);
    if (opCacheIt == rejections.end()) {
      return nullptr;
    }
    typename RejectionCache::const_iterator it = opCacheIt->second.find(hash);
    return it == opCacheIt->second.end() ? nullptr : &it->second;
  }

  // try_emplace rather than operator[]: a re-store of a key already present
  // must not double-count `entries`, which is what size()/empty() report.
  void storeValue(mlir::TypeID opTypeID, llvm::hash_code hash,
                  const ValueT &value) {
    if (values[opTypeID].try_emplace(hash, value).second) {
      stats.entries++;
    }
  }

  void storeRejection(mlir::TypeID opTypeID, llvm::hash_code hash,
                      std::string message) {
    const size_t messageSize = message.size();
    if (rejections[opTypeID].try_emplace(hash, std::move(message)).second) {
      stats.rejectionEntries++;
      stats.rejectionBytes += messageSize;
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
  // Rejections live in their own map rather than in a widened value type: their
  // message would otherwise be carried by every accepted entry (and every empty
  // slot) of the far more numerous value map, and by the runtime cache, which
  // can never populate it.
  using ValueCache = llvm::DenseMap<llvm::hash_code, ValueT>;
  using ValueCacheMap = llvm::DenseMap<mlir::TypeID, ValueCache>;
  using RejectionCache = llvm::DenseMap<llvm::hash_code, std::string>;
  using RejectionCacheMap = llvm::DenseMap<mlir::TypeID, RejectionCache>;

  // getOpRuntime rejects unconditionally under a mock device and otherwise
  // executes for real, so only constraint rejections are memoizable.
  const bool cacheRejections;
  ValueCacheMap values;
  RejectionCacheMap rejections;
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
      /*cacheRejections=*/true);
  return instance;
}

inline TTNNOpModelCache<size_t> &opRuntimeCache() {
  static TTNNOpModelCache<size_t> instance(/*cacheRejections=*/false);
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
