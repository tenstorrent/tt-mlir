// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/trace/end_trace_capture.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/types/trace_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/types.h"

namespace tt::runtime::ttnn::operations::trace {

void run(const ::tt::target::ttnn::EndTraceCaptureOp *op,
         ProgramContext &context) {
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  ::ttnn::QueueId ttnnCqId(op->cq_id());

  LOG_ASSERT(meshDevice.get_program_cache().is_enabled(),
             "Program cache must be enabled");
  LOG_ASSERT(meshDevice.allocator()->get_config().trace_region_size > 0,
             "Trace region size must be greater than 0");

  const ::ttnn::Tensor &traceIdTensor =
      context.getTensorPool().getTTNNTensorAndValidate(op->trace_id());

  std::vector<uint32_t> traceIdVec = traceIdTensor.to_vector<uint32_t>();
  LOG_ASSERT(traceIdVec.size() == 1, "Trace ID must be a single value");
  ::ttnn::MeshTraceId traceId(traceIdVec[0]);

  ::ttnn::operations::trace::end_trace_capture(&meshDevice, traceId, ttnnCqId);
}

} // namespace tt::runtime::ttnn::operations::trace
