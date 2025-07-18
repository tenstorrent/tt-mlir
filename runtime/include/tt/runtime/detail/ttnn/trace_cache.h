// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_TRACE_CACHE_H
#define TT_RUNTIME_DETAIL_TTNN_TRACE_CACHE_H

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

namespace tt::runtime::ttnn {

std::string generateTraceCacheOuterKey(const uint64_t binaryId,
                                       const size_t programId);

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

  bool contains(uint64_t binaryId, size_t programId,
                const std::string &traceFuncName) const;
  TraceData *get(uint64_t binaryId, size_t programId,
                 const std::string &traceFuncName);
  void insert(uint64_t binaryId, size_t programId,
              const std::string &traceFuncName, const TraceData &traceData);
  bool erase(uint64_t binaryId, size_t programId);
  bool erase(uint64_t binaryId, size_t programId,
             const std::string &traceFuncName);

private:
  std::weak_ptr<::ttnn::MeshDevice> meshDevice;
  // Outer key should be combination of device id and program index, created via
  // generateTraceCacheOuterKey. Inner key will be trace func name.
  std::unordered_map<std::string, std::unordered_map<std::string, TraceData>>
      cache;
};
} // namespace tt::runtime::ttnn

#endif
