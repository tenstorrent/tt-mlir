// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/trace_cache.h"
#include <sstream>

namespace tt::runtime::ttnn {

std::string generateTraceCacheOuterKey(const uint64_t binaryId,
                                       const size_t programId) {
  return std::to_string(binaryId) + ":" + std::to_string(programId);
}

bool TraceCache::contains(uint64_t binaryId, size_t programId,
                          const std::string &traceFuncName) const {
  auto outerKey = generateTraceCacheOuterKey(binaryId, programId);
  auto outerIt = cache.find(outerKey);
  if (outerIt == cache.end()) {
    return false;
  }

  return outerIt->second.contains(traceFuncName);
}

TraceData *TraceCache::get(uint64_t binaryId, size_t programId,
                           const std::string &traceFuncName) {
  auto outerKey = generateTraceCacheOuterKey(binaryId, programId);
  auto outerIt = cache.find(outerKey);
  if (outerIt == cache.end()) {
    return nullptr;
  }

  auto innerIt = outerIt->second.find(traceFuncName);
  if (innerIt == outerIt->second.end()) {
    return nullptr;
  }

  return &innerIt->second;
}

void TraceCache::insert(uint64_t binaryId, size_t programId,
                        const std::string &traceFuncName,
                        const TraceData &traceData) {
  auto outerKey = generateTraceCacheOuterKey(binaryId, programId);
  cache[outerKey][traceFuncName] = traceData;
}

bool TraceCache::erase(uint64_t binaryId, size_t programId) {
  auto outerKey = generateTraceCacheOuterKey(binaryId, programId);
  auto outerIt = cache.find(outerKey);
  if (outerIt == cache.end()) {
    return false;
  }

  std::shared_ptr<::ttnn::MeshDevice> lockedDevice = meshDevice.lock();
  if (lockedDevice && lockedDevice->is_initialized()) {
    for (const auto &[_, traceData] : outerIt->second) {
      ::ttnn::operations::trace::release_trace(lockedDevice.get(),
                                               traceData.traceId);
    }
  }

  cache.erase(outerIt);
  return true;
}

bool TraceCache::erase(uint64_t binaryId, size_t programId,
                       const std::string &traceFuncName) {
  auto outerKey = generateTraceCacheOuterKey(binaryId, programId);
  auto outerIt = cache.find(outerKey);
  if (outerIt == cache.end()) {
    return false;
  }

  auto innerIt = outerIt->second.find(traceFuncName);
  if (innerIt == outerIt->second.end()) {
    return false;
  }

  std::shared_ptr<::ttnn::MeshDevice> lockedDevice = meshDevice.lock();
  if (lockedDevice && lockedDevice->is_initialized()) {
    ::ttnn::operations::trace::release_trace(lockedDevice.get(),
                                             innerIt->second.traceId);
  }

  outerIt->second.erase(innerIt);
  return true;
}

std::optional<size_t>
TraceCache::getDebugStat(const std::string &statName) const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  auto it = stats.find(statName);
  if (it != stats.end()) {
    return it->second;
  }
#endif
  return std::nullopt;
}

void TraceCache::incrementDebugStat(const std::string &statName) const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  stats[statName]++;
#endif
}

void TraceCache::printDebugStats() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  std::ostringstream ss;
  ss << "TraceCache Debug Stats:\n";
  for (const auto &[statName, value] : stats) {
    ss << "  " << statName << ": " << value << "\n";
  }
  LOG_DEBUG(ss.str());
#endif
}

void TraceCache::clearDebugStats() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  stats.clear();
#endif
}

} // namespace tt::runtime::ttnn
