// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_TYPES_TRACE_CACHE_H
#define TT_RUNTIME_DETAIL_TTNN_TYPES_TRACE_CACHE_H

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/types/trace_cache_key.h"

namespace tt::runtime::ttnn {

struct TraceData {
  ::ttnn::MeshTraceId traceId;
  // Input tensor buffers to write the input tensors to
  std::vector<::tt::runtime::Tensor> inputTensors;
  // Output tensor buffers to write the output tensors to
  std::vector<::tt::runtime::Tensor> outputTensors;
  // Device generation at capture time — used to detect stale traces
  uint64_t capturedAtGeneration = 0;
};

class TraceCache {
public:
  TraceCache(std::shared_ptr<::ttnn::MeshDevice> meshDevice)
      : meshDevice(meshDevice) {}

  TraceCache(const TraceCache &) = delete;
  TraceCache &operator=(const TraceCache &) = delete;

  TraceCache(TraceCache &&) = delete;
  TraceCache &operator=(TraceCache &&) = delete;

  bool contains(const MainProgramKey &key,
                const CaptureExecuteProgramKey &captureExecuteKey) const;
  TraceData *get(const MainProgramKey &key,
                 const CaptureExecuteProgramKey &captureExecuteKey);
  void insert(const MainProgramKey &key,
              const CaptureExecuteProgramKey &captureExecuteKey,
              TraceData traceData);
  void erase(const MainProgramKey &key);
  void erase(const MainProgramKey &key,
             const CaptureExecuteProgramKey &captureExecuteKey);
  // Move TraceData out of the cache and return it. The entry is removed.
  // Asserts if the key doesn't exist. The trace is NOT released — caller
  // is responsible.
  TraceData extract(const MainProgramKey &key,
                    const CaptureExecuteProgramKey &captureExecuteKey);

  // Device generation tracking for trace staleness detection.
  // Incremented when new device memory is allocated outside of trace lifecycle.
  uint64_t getDeviceGeneration() const;
  void incrementDeviceGeneration();

private:
  std::weak_ptr<::ttnn::MeshDevice> meshDevice;
  std::unordered_map<MainProgramKey,
                     std::unordered_map<CaptureExecuteProgramKey, TraceData>>
      cache;
  // Monotonic counter incremented on device memory allocations outside trace
  // lifecycle. Used to detect stale captured traces.
  uint64_t deviceGeneration = 0;
};
} // namespace tt::runtime::ttnn

#endif
