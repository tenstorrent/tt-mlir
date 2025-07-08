// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_TYPES_TRACE_CACHE_H
#define TT_RUNTIME_DETAIL_TTNN_TYPES_TRACE_CACHE_H

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/types/trace_cache_key.h"

namespace tt::runtime::ttnn {

struct TraceData {
  ::ttnn::MeshTraceId traceId;
  // Input tensor buffers to write the input tensors to
  std::vector<::tt::runtime::Tensor> inputTensors;
  // Output tensor buffers to write the output tensors to
  std::vector<::tt::runtime::Tensor> outputTensors;
};

class TraceCache {
public:
  TraceCache(std::shared_ptr<::ttnn::MeshDevice> meshDevice)
      : meshDevice(meshDevice) {}

  TraceCache(const TraceCache &) = delete;
  TraceCache &operator=(const TraceCache &) = delete;

  TraceCache(TraceCache &&) = delete;
  TraceCache &operator=(TraceCache &&) = delete;

  bool contains(uint64_t binaryId, uint64_t traceFuncId) const;
  TraceData *get(uint64_t binaryId, uint64_t traceFuncId);
  void insert(uint64_t binaryId, uint64_t traceFuncId,
              const TraceData &traceData);
  void erase(uint64_t binaryId, uint64_t traceFuncId);

private:
  std::weak_ptr<::ttnn::MeshDevice> meshDevice;
  std::unordered_map<TraceCacheKey, TraceData> cache;
};
} // namespace tt::runtime::ttnn

#endif
