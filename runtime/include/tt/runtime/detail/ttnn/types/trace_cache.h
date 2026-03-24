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
  /// Input tensor buffers to write the input tensors to.
  std::vector<::tt::runtime::Tensor> inputTensors;
  /// Output tensor buffers to write the output tensors to.
  std::vector<::tt::runtime::Tensor> outputTensors;
  /// Trace cache generation id at capture time - used to detect stale traces.
  uint64_t generationId = 0;
};

/// Cache for storing captured traces.
///
/// The captured traces are stored by a two-level key:
/// - the first level key identifies the main program (binary id + main program
/// index)
/// - the second level key identifies the capture and execute program IDs for
/// the trace.
///
/// The cache also maintains a generation id to allow detection of stale traces.
/// Traces are inherently unsafe to reuse after certain operations (e.g. after a
/// new graph is captured). This is because the old trace may overwrite memory
/// allocated for the new graph - those allocations were not present at
/// the time of the capture, the captured graph could have used that memory for
/// its intermediate buffers.
///
/// To handle those cases, the cache provides the generation id - which should
/// be incremented by runtime whenever it performs an operation that may
/// invalidate existing traces.
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

  uint64_t getGenerationId() const { return generationId; }

  /// Increments the generation id.
  /// Should be called by runtime whenever it performs an operation that may
  /// invalidate existing traces (e.g. when a new graph is captured).
  void incrementGeneration() { generationId++; }

private:
  std::weak_ptr<::ttnn::MeshDevice> meshDevice;
  std::unordered_map<MainProgramKey,
                     std::unordered_map<CaptureExecuteProgramKey, TraceData>>
      cache;
  uint64_t generationId = 0;
};
} // namespace tt::runtime::ttnn

#endif
