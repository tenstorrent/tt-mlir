// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/trace/begin_trace_capture.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::trace {

void run(const ::tt::target::ttnn::BeginTraceCaptureOp *op,
         ProgramContext &context) {
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  ::ttnn::QueueId ttnnCqId(op->cq_id());

  LOG_ASSERT(meshDevice.get_program_cache().is_enabled(),
             "Program cache must be enabled");

  LOG_ASSERT(meshDevice.allocator()
                     ->get_statistics(::ttnn::BufferType::TRACE)
                     .total_allocatable_size_bytes > 0,
             "Trace region size must be greater than 0");

  ::ttnn::MeshTraceId traceId =
      ::ttnn::operations::trace::begin_trace_capture(&meshDevice, ttnnCqId);
  ::ttnn::Tensor traceIdTensor =
      ::tt::runtime::ttnn::utils::createTTNNTensor<uint32_t>(
          &traceId.get(), ::ttnn::Shape(), ::ttnn::DataType::UINT32);
  context.getTensorPool().insertTTNNTensorAndValidate(op->trace_id(),
                                                      traceIdTensor);
}

} // namespace tt::runtime::ttnn::operations::trace
