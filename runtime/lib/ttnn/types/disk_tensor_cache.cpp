// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types/disk_tensor_cache.h"

#include <tt_stl/indestructible.hpp>

#include <cstdlib>

namespace tt::runtime {

DiskTensorCache &DiskTensorCache::getInstance() {
  static ttsl::Indestructible<DiskTensorCache> instance;
  return instance.get();
}

bool DiskTensorCache::isEnabled() {
  static const bool enabled = std::getenv(kEnvVarName) != nullptr;
  return enabled;
}

std::filesystem::path
DiskTensorCache::getCachePath(const std::string &programHash,
                              uint32_t argIndex) const {
  return std::filesystem::path(kCacheBasePath) / programHash /
         (std::to_string(argIndex) + ".tensorbin");
}

bool DiskTensorCache::exists(const std::string &programHash,
                             uint32_t argIndex) const {
  auto path = getCachePath(programHash, argIndex);
  return std::filesystem::exists(path);
}

void DiskTensorCache::ensureDirectoryExists(const std::string &programHash) {
  auto dirPath = std::filesystem::path(kCacheBasePath) / programHash;
  std::filesystem::create_directories(dirPath);
}

void DiskTensorCache::markWritten(const std::string &programHash,
                                  uint32_t argIndex) {
  std::unique_lock<std::shared_mutex> lock(cacheMutex);
  writtenEntries.insert(programHash + "/" + std::to_string(argIndex));
}

bool DiskTensorCache::hasBeenWritten(const std::string &programHash,
                                     uint32_t argIndex) const {
  std::shared_lock<std::shared_mutex> lock(cacheMutex);
  return writtenEntries.count(programHash + "/" + std::to_string(argIndex)) > 0;
}

} // namespace tt::runtime
