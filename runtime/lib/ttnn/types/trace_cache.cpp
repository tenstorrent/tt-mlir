// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types/trace_cache.h"

namespace tt::runtime::ttnn {

bool TraceCache::contains(uint64_t binaryId, size_t mainProgramId,
                          size_t captureProgramId,
                          size_t executeProgramId) const {
  MainProgramKey outerKey(binaryId, mainProgramId);
  auto outerIt = cache.find(outerKey);
  if (outerIt == cache.end()) {
    return false;
  }

  TraceCaptureExecuteKey innerKey(captureProgramId, executeProgramId);
  return outerIt->second.find(innerKey) != outerIt->second.end();
}

TraceData *TraceCache::get(uint64_t binaryId, size_t mainProgramId,
                           size_t captureProgramId, size_t executeProgramId) {
  MainProgramKey outerKey(binaryId, mainProgramId);
  auto outerIt = cache.find(outerKey);
  if (outerIt == cache.end()) {
    return nullptr;
  }

  TraceCaptureExecuteKey innerKey(captureProgramId, executeProgramId);
  if (outerIt->second.find(innerKey) == outerIt->second.end()) {
    return nullptr;
  }

  return &outerIt->second[innerKey];
}

void TraceCache::insert(uint64_t binaryId, size_t mainProgramId,
                        size_t captureProgramId, size_t executeProgramId,
                        const TraceData &traceData) {
  MainProgramKey outerKey(binaryId, mainProgramId);
  TraceCaptureExecuteKey innerKey(captureProgramId, executeProgramId);
  cache[outerKey][innerKey] = traceData;
}

void TraceCache::erase(uint64_t binaryId, size_t mainProgramId) {
  auto outerKey = MainProgramKey(binaryId, mainProgramId);
  auto outerIt = cache.find(outerKey);
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

void TraceCache::erase(uint64_t binaryId, size_t mainProgramId,
                       size_t captureProgramId, size_t executeProgramId) {
  auto outerKey = MainProgramKey(binaryId, mainProgramId);
  auto outerIt = cache.find(outerKey);
  if (outerIt == cache.end()) {
    return;
  }

  TraceCaptureExecuteKey innerKey(captureProgramId, executeProgramId);
  auto innerIt = outerIt->second.find(innerKey);
  if (innerIt == outerIt->second.end()) {
    return;
  }

  std::shared_ptr<::ttnn::MeshDevice> lockedDevice = meshDevice.lock();
  if (lockedDevice && lockedDevice->is_initialized()) {
    ::ttnn::operations::trace::release_trace(lockedDevice.get(),
                                             innerIt->second.traceId);
  }

  outerIt->second.erase(innerIt);
}

} // namespace tt::runtime::ttnn
