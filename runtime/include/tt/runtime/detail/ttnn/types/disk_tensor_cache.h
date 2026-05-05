// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_TYPES_DISK_TENSOR_CACHE_H
#define TT_RUNTIME_DETAIL_TTNN_TYPES_DISK_TENSOR_CACHE_H

#include "tt/runtime/types.h"

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

// Forward declaration for friend class
namespace ttsl {
template <typename T>
class Indestructible;
} // namespace ttsl

namespace tt::runtime {

/**
 * Singleton cache manager for disk-based tensor caching.
 *
 * This cache stores tensors to disk at paths based on program hash and argument
 * index. On cache hit, tensors are loaded from disk. On cache miss, tensors are
 * written to disk.
 *
 * Cache path format: ./generated/tensorcache/<program_hash>/<arg_index>.tensorbin
 *
 * The cache is gated by the TTMLIR_ENABLE_DISK_CACHE environment variable.
 */
class DiskTensorCache {
public:
  static DiskTensorCache &getInstance();

  // Check if disk cache feature is enabled via env var
  static bool isEnabled();

  // Check if cache entry exists on disk
  bool exists(const std::string &programHash, uint32_t argIndex) const;

  // Get the file path for a cache entry
  std::filesystem::path getCachePath(const std::string &programHash,
                                     uint32_t argIndex) const;

  // Ensure cache directory exists for a given program hash
  void ensureDirectoryExists(const std::string &programHash);

  // Mark cache entry as written (for in-memory tracking)
  void markWritten(const std::string &programHash, uint32_t argIndex);

  // Check if entry has been written this session
  bool hasBeenWritten(const std::string &programHash, uint32_t argIndex) const;

  //
  // Runtime Tensor Cache (L1) - in-memory cache of device tensors
  //

  // Get cached runtime tensor, returns nullptr if not cached
  const ::tt::runtime::Tensor *getRuntimeTensor(const std::string &programHash,
                                                uint32_t argIndex) const;

  // Store runtime tensor in cache
  void storeRuntimeTensor(const std::string &programHash, uint32_t argIndex,
                          const ::tt::runtime::Tensor &tensor);

  // Check if runtime tensor is cached
  bool hasRuntimeTensor(const std::string &programHash,
                        uint32_t argIndex) const;

private:
  friend class ttsl::Indestructible<DiskTensorCache>;

  DiskTensorCache() = default;
  ~DiskTensorCache() = default;

  DiskTensorCache(const DiskTensorCache &) = delete;
  DiskTensorCache &operator=(const DiskTensorCache &) = delete;
  DiskTensorCache(DiskTensorCache &&) = delete;
  DiskTensorCache &operator=(DiskTensorCache &&) = delete;

  static constexpr const char *kCacheBasePath = "./generated/tensorcache";
  static constexpr const char *kEnvVarName = "TTMLIR_ENABLE_DISK_CACHE";

  // In-memory tracking of entries written this session (L2 disk cache)
  mutable std::shared_mutex cacheMutex;
  std::unordered_set<std::string> writtenEntries; // "hash/index" format

  // Runtime tensor cache (L1) - keyed by "programHash/argIndex"
  std::unordered_map<std::string, ::tt::runtime::Tensor> runtimeTensorCache;
};

} // namespace tt::runtime

#endif // TT_RUNTIME_DETAIL_TTNN_TYPES_DISK_TENSOR_CACHE_H
