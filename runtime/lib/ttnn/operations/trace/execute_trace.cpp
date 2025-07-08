// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/trace/execute_trace.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/types/trace_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"

namespace tt::runtime::ttnn::operations::trace {

void run(const ::tt::target::ttnn::ExecuteTraceOp *op,
         ProgramContext &context) {
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  std::shared_ptr<::tt::runtime::ttnn::TraceCache> traceCache =
      deviceHandle.getTraceCache()
          ->asSharedPtr<::tt::runtime::ttnn::TraceCache>(DeviceRuntime::TTNN);
  LOG_ASSERT(traceCache, "TraceCache must be initialized in DeviceHandle");

  uint64_t binaryId = context.getExecutableHandle().id();
  uint64_t traceFuncId = op->trace_func_id();
  TraceData *traceData = traceCache->get(binaryId, traceFuncId);
  LOG_ASSERT(traceData, "Trace does not exist for binary_id ", binaryId,
             " traceFuncId ", traceFuncId);

  ::ttnn::QueueId ttnnCqId = ::ttnn::QueueId(op->cq_id());
  // Execute the trace
  ::ttnn::operations::trace::execute_trace(&meshDevice, traceData->traceId,
                                           ttnnCqId, op->blocking());
}

} // namespace tt::runtime::ttnn::operations::trace
