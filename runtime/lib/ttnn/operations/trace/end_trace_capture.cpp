// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/trace/end_trace_capture.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/types/trace_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"

namespace tt::runtime::ttnn::operations::trace {

void run(const ::tt::target::ttnn::EndTraceCaptureOp *op,
         ProgramContext &context) {
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  ::ttnn::QueueId ttnnCqId(op->cq_id());

  LOG_ASSERT(meshDevice.get_program_cache().is_enabled(),
             "Program cache must be enabled");

  const ::ttnn::Tensor &traceIdTensor =
      context.getTensorPool().getTTNNTensorAndValidate(op->trace_id());
  LOG_ASSERT(traceIdTensor.dtype() == ::ttnn::DataType::UINT32,
             "Trace ID must be UINT32");

  uint32_t traceId = utils::getScalarFromTensor<uint32_t>(traceIdTensor);
  ::ttnn::MeshTraceId meshTraceId(traceId);

  ::ttnn::operations::trace::end_trace_capture(&meshDevice, meshTraceId,
                                               ttnnCqId);

  // Restore top-down allocation after the trace ends so that any EmptyOps
  // following end_trace (e.g., output slot allocations) stay at high addresses.
  context.setAllocateTopDown(true);
}

} // namespace tt::runtime::ttnn::operations::trace
