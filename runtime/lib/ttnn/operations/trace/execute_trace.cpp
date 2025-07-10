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
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  ::ttnn::QueueId ttnnCqId = ::ttnn::QueueId(op->cq_id());

  const ::ttnn::Tensor &traceIdTensor =
      context.getTensorPool().getTTNNTensorAndValidate(op->trace_id());
  std::vector<uint32_t> traceIdVec = traceIdTensor.to_vector<uint32_t>();
  LOG_ASSERT(traceIdVec.size() == 1, "Trace ID must be a single value");
  ::ttnn::MeshTraceId traceId(traceIdVec[0]);

  // Execute the trace
  ::ttnn::operations::trace::execute_trace(&meshDevice, traceId, ttnnCqId,
                                           op->blocking());
}

} // namespace tt::runtime::ttnn::operations::trace
