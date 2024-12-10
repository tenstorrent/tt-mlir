// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_OPCONSTRAINTSCACHE_H
#define TTMLIR_OPMODEL_TTNN_OPCONSTRAINTSCACHE_H

// #include "TTNNOpModelLib_Impl.hpp"

#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

namespace mlir::tt::op_model {

/// Helper class to hash tuples
struct TupleHash {
  template <typename... Args>
  std::size_t operator()(const std::tuple<Args...> &tuple) const {
    return std::apply(
        [](auto &&...args) {
          size_t seed = 0;
          ((seed ^= std::hash<std::decay_t<decltype(args)>>{}(args) +
                    0x9e3779b9 + (seed << 6) + (seed >> 2)),
           ...);
          return seed;
        },
        tuple);
  }
};

/// Class template for an unordered_map with tuple keys
///
/// This class provides a caching mechanism where values are associated with
/// tuple keys. If a value for a given key is not found in the cache, it is
/// created using a provided callable and then stored in the cache.
///
/// @tparam TupleType The type of the keys (tuples) used in the cache.
/// @tparam ValueType The type of the values stored in the cache.
/// @tparam ValueCreatorCallable The type of the callable used to create values
/// when a key is not found.
/// @tparam Hash The type of the hash function used for the keys (default is
/// TupleHash).
///
/// @note This class is non-copyable and non-movable.
template <typename TupleType, typename ValueType, class ValueCreatorCallable,
          class Hash = TupleHash>
class TupleCache {
public:
  /// Type alias for the underlying unordered_map.
  using MapType = std::unordered_map<TupleType, ValueType, Hash>;

  /// Constructor for the TupleCache class.
  ///
  /// @param name The name of the cache.
  /// @param callable The callable used to create values when a key is not
  /// found.
  TupleCache(std::string_view name, ValueCreatorCallable callable)
      : name(name), callable(callable){};

  ~TupleCache() {
    // TODO(mbezulj) Figure out if these stats make sense to have cache at all.
    // and add flag to enable/disable them.
    std::cout << "Cache " << name << " hits: " << hits << ", misses: " << misses
              << std::endl;
  }
  TupleCache(const TupleCache &) = delete;
  TupleCache(TupleCache &&) = delete;
  TupleCache &operator=(const TupleCache &) = delete;
  TupleCache &operator=(TupleCache &&) = delete;

  /// Retrieves a value by key, creates one using provided lambda if not found.
  ///
  /// @param key The key for which to retrieve or create the value.
  /// @return The value associated with the key.
  ValueType getOrCreate(const TupleType &key) {
    auto exists = cache.find(key) != cache.end();
    if (!exists) {
      misses++;
      cache[key] = callable(key);
    } else {
      hits++;
    }
    return cache[key];
  }

  /// Retrieves the name of the cache.
  std::string_view getName() const { return name; }

private:
  std::string_view name;
  MapType cache;
  ValueCreatorCallable callable;
  size_t hits = 0;
  size_t misses = 0;
};

} // namespace mlir::tt::op_model
#endif // TTMLIR_OPMODEL_TTNN_OPCONSTRAINTSCACHE_H