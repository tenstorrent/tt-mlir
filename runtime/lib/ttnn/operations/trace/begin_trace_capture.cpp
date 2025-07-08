// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/trace/begin_trace_capture.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/types/trace_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/types.h"

namespace tt::runtime::ttnn::operations::trace {

void run(const ::tt::target::ttnn::BeginTraceCaptureOp *op,
         ProgramContext &context) {
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  ::ttnn::QueueId ttnnCqId(op->cq_id());

  LOG_ASSERT(meshDevice.get_program_cache().is_enabled(),
             "Program cache must be enabled");
  LOG_ASSERT(meshDevice.allocator()->get_config().trace_region_size > 0,
             "Trace region size must be greater than 0");

  Binary &executableHandle = context.getExecutableHandle();

  std::shared_ptr<::tt::runtime::ttnn::TraceCache> traceCache =
      deviceHandle.getTraceCache()
          ->asSharedPtr<::tt::runtime::ttnn::TraceCache>(DeviceRuntime::TTNN);
  LOG_ASSERT(traceCache, "TraceCache must be initialized in DeviceHandle");

  uint64_t binaryId = executableHandle.id();
  uint64_t traceFuncId = op->trace_func_id();
  LOG_ASSERT(!traceCache->contains(binaryId, traceFuncId),
             "Trace already exists for binary_id ", binaryId, " traceFuncId ",
             traceFuncId);

  ::ttnn::MeshTraceId traceId =
      ::ttnn::operations::trace::begin_trace_capture(&meshDevice, ttnnCqId);

  TraceData traceData{
      .traceId = traceId, .inputTensors = {}, .outputTensors = {}};
  traceCache->insert(binaryId, traceFuncId, traceData);
}

} // namespace tt::runtime::ttnn::operations::trace
