// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types/trace_cache.h"

#include "tt/runtime/detail/common/logger.h"

#include <cstdlib>
#include <tt-metalium/distributed.hpp>

namespace tt::runtime::ttnn {

using LogType = ::tt::runtime::logger::LogType;

namespace {
// Gated by TT_RUNTIME_SYNC_BEFORE_TRACE_RELEASE (checked once, startup-set).
// Targets the actual unsafe transition -- a trace buffer's DRAM being freed
// while another trace's replay may still have work in flight against it --
// rather than TT_RUNTIME_SYNC_AFTER_TRACE, which syncs after every single
// trace replay regardless of whether an eviction is happening at all.
void syncBeforeTraceReleaseIfNeeded(::ttnn::MeshDevice *device) {
  static const bool enabled =
      std::getenv("TT_RUNTIME_SYNC_BEFORE_TRACE_RELEASE") != nullptr;
  if (enabled) {
    ::tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    LOG_INFO(LogType::LogRuntimeTTNN, "Synced before trace release");
  }
}
} // namespace

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
    syncBeforeTraceReleaseIfNeeded(lockedDevice.get());
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
    syncBeforeTraceReleaseIfNeeded(lockedDevice.get());
    ::ttnn::operations::trace::release_trace(lockedDevice.get(),
                                             innerIt->second.traceId);
  }

  outerIt->second.erase(captureExecuteKey);
}

} // namespace tt::runtime::ttnn
