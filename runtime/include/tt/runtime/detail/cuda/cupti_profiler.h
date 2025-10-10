// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_CUDA_CUPTI_PROFILER_H
#define TT_RUNTIME_DETAIL_CUDA_CUPTI_PROFILER_H

#ifdef TTMLIR_ENABLE_CUDA

#include <cuda.h>
#include <cupti.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt::runtime::cuda {

struct ProfilerMetrics {
  double kernelExecutionTime = 0.0; // milliseconds
  double memoryTransferTime = 0.0;  // milliseconds
  size_t memoryUsage = 0;           // bytes
  double smUtilization = 0.0;       // percentage
  double memoryBandwidth = 0.0;     // GB/s
  uint32_t kernelCount = 0;
  uint32_t memoryTransferCount = 0;
};

class CuptiProfiler {
public:
  CuptiProfiler();
  ~CuptiProfiler();

  // Initialize CUPTI profiling
  bool initialize();

  // Start profiling session
  void startProfiling();

  // Stop profiling session and collect metrics
  void stopProfiling();

  // Get collected metrics
  ProfilerMetrics getMetrics() const;

  // Print detailed profiling report
  void printReport() const;

  // Enable specific metric collection
  void enableMetric(const std::string &metricName);

  // Get list of available metrics
  std::vector<std::string> getAvailableMetrics() const;

private:
  bool initialized_ = false;
  bool profiling_ = false;
  ProfilerMetrics metrics_;

  // CUPTI event group and context
  CUpti_EventGroup eventGroup_;
  CUcontext context_;

  // Activity buffers
  std::vector<uint8_t> activityBuffer_;
  static constexpr size_t ACTIVITY_BUFFER_SIZE = 8 * 1024 * 1024; // 8MB

  // Callback functions
  static void CUPTIAPI activityCallback(uint8_t *buffer, size_t size,
                                        size_t validSize);
  static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                                       size_t *maxNumRecords);
  static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
                                       uint8_t *buffer, size_t size,
                                       size_t validSize);

  // Helper methods
  void processActivityBuffer(uint8_t *buffer, size_t validSize);
  void processKernelActivity(CUpti_Activity *record);
  void processMemcpyActivity(CUpti_Activity *record);

  // Static instance for callbacks
  static CuptiProfiler *instance_;
};

} // namespace tt::runtime::cuda

#endif // TTMLIR_ENABLE_CUDA

#endif // TT_RUNTIME_DETAIL_CUDA_CUPTI_PROFILER_H
