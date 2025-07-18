// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/trace/execute_trace.h"
#include "tt/runtime/detail/common/logger.h"
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
  LOG_ASSERT(traceIdTensor.dtype() == ::ttnn::DataType::UINT32,
             "Trace ID must be UINT32");

  uint32_t traceId = utils::getScalarFromTensor<uint32_t>(traceIdTensor);
  ::ttnn::MeshTraceId meshTraceId(traceId);

  // Execute the trace
  ::ttnn::operations::trace::execute_trace(&meshDevice, meshTraceId, ttnnCqId,
                                           op->blocking());
}

} // namespace tt::runtime::ttnn::operations::trace
