// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types/trace_cache.h"
#include <optional>

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
                        TraceData traceData) {
  cache[key][captureExecuteKey] = std::move(traceData);
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

std::optional<TraceData>
TraceCache::erase(const MainProgramKey &key,
                  const CaptureExecuteProgramKey &captureExecuteKey) {
  auto outerIt = cache.find(key);
  if (outerIt == cache.end()) {
    return std::nullopt;
  }

  auto innerIt = outerIt->second.find(captureExecuteKey);
  if (innerIt == outerIt->second.end()) {
    return std::nullopt;
  }

  // Move the trace data out so that we can return it after releasing the trace
  // on the device.
  auto traceData = std::move(innerIt->second);

  std::shared_ptr<::ttnn::MeshDevice> lockedDevice = meshDevice.lock();
  if (lockedDevice && lockedDevice->is_initialized()) {
    ::ttnn::operations::trace::release_trace(lockedDevice.get(),
                                             traceData.traceId);
  }

  outerIt->second.erase(captureExecuteKey);

  return traceData;
}

uint64_t TraceCache::getGenerationId() const { return generation_id; }

void TraceCache::incrementGeneration() { generation_id++; }

} // namespace tt::runtime::ttnn
