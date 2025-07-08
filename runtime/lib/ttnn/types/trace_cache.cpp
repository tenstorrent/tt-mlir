// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types/trace_cache.h"

namespace tt::runtime::ttnn {

bool TraceCache::contains(uint64_t binaryId, uint64_t traceFuncId) const {
  TraceCacheKey key{binaryId, traceFuncId};
  return cache.contains(key);
}

TraceData *TraceCache::get(uint64_t binaryId, uint64_t traceFuncId) {
  TraceCacheKey key{binaryId, traceFuncId};
  auto it = cache.find(key);
  if (it == cache.end()) {
    return nullptr;
  }

  return &it->second;
}

void TraceCache::insert(uint64_t binaryId, uint64_t traceFuncId,
                        const TraceData &traceData) {
  TraceCacheKey key{binaryId, traceFuncId};
  cache[key] = traceData;
}

void TraceCache::erase(uint64_t binaryId, uint64_t traceFuncId) {
  TraceCacheKey key{binaryId, traceFuncId};
  auto it = cache.find(key);
  if (it == cache.end()) {
    return;
  }

  std::shared_ptr<::ttnn::MeshDevice> lockedDevice = meshDevice.lock();
  if (lockedDevice && lockedDevice->is_initialized()) {
    ::ttnn::operations::trace::release_trace(lockedDevice.get(),
                                             it->second.traceId);
  }

  cache.erase(it);
}

} // namespace tt::runtime::ttnn
