// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types/trace_cache.h"

namespace tt::runtime::ttnn {

bool TraceCache::contains(
    const MainProgramKey &key,
    const CaptureExecuteProgramKey &captureExecuteKey) const {
  auto outerIt = cache.find(key);
  if (outerIt == cache.end()) {
    return false;
  }

  return outerIt->second.contains(captureExecuteKey);
}

TraceData *TraceCache::get(const MainProgramKey &key,
                           const CaptureExecuteProgramKey &captureExecuteKey) {
  auto outerIt = cache.find(key);
  if (outerIt == cache.end()) {
    return nullptr;
  }

  auto innerIt = outerIt->second.find(captureExecuteKey);
  if (innerIt == outerIt->second.end()) {
    return nullptr;
  }

  return &innerIt->second;
}

void TraceCache::insert(const MainProgramKey &key,
                        const CaptureExecuteProgramKey &captureExecuteKey,
                        const TraceData &traceData) {
  cache[key][captureExecuteKey] = traceData;
}

void TraceCache::erase(const MainProgramKey &key) {
  auto outerIt = cache.find(key);
  if (outerIt == cache.end()) {
    return;
  }

  std::shared_ptr<::ttnn::MeshDevice> lockedDevice = meshDevice.lock();
  if (lockedDevice && lockedDevice->is_initialized()) {
    for (const auto &[_, traceData] : outerIt->second) {
      ::ttnn::operations::trace::release_trace(lockedDevice.get(),
                                               traceData.traceId);
    }
  }

  cache.erase(outerIt);
}

void TraceCache::erase(const MainProgramKey &key,
                       const CaptureExecuteProgramKey &captureExecuteKey) {
  auto outerIt = cache.find(key);
  if (outerIt == cache.end()) {
    return;
  }

  auto innerIt = outerIt->second.find(captureExecuteKey);
  if (innerIt == outerIt->second.end()) {
    return;
  }

  std::shared_ptr<::ttnn::MeshDevice> lockedDevice = meshDevice.lock();
  if (lockedDevice && lockedDevice->is_initialized()) {
    ::ttnn::operations::trace::release_trace(lockedDevice.get(),
                                             innerIt->second.traceId);
  }

  outerIt->second.erase(captureExecuteKey);
}

uint32_t TraceCache::getPendingExecutionCount(
    const MainProgramKey &key,
    const CaptureExecuteProgramKey &captureExecuteKey) {
  auto outerIt = pendingExecutionCounts.find(key);
  if (outerIt == pendingExecutionCounts.end()) {
    return 0;
  }

  auto innerIt = outerIt->second.find(captureExecuteKey);
  if (innerIt == outerIt->second.end()) {
    return 0;
  }

  return innerIt->second;
}

void TraceCache::incrementPendingExecutionCount(
    const MainProgramKey &key,
    const CaptureExecuteProgramKey &captureExecuteKey) {
  pendingExecutionCounts[key][captureExecuteKey]++;
}

void TraceCache::erasePendingExecutionCount(
    const MainProgramKey &key,
    const CaptureExecuteProgramKey &captureExecuteKey) {
  auto outerIt = pendingExecutionCounts.find(key);
  if (outerIt == pendingExecutionCounts.end()) {
    return;
  }

  outerIt->second.erase(captureExecuteKey);
  if (outerIt->second.empty()) {
    pendingExecutionCounts.erase(outerIt);
  }
}

} // namespace tt::runtime::ttnn
