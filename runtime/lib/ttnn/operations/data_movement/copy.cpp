// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/copy.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::CopyOp *op, ProgramContext &context) {
  LOG_ASSERT(::tt::runtime::ttnn::utils::inDeviceMemory(op->src()),
             "Source tensor must be in device memory");
  LOG_ASSERT(::tt::runtime::ttnn::utils::inDeviceMemory(op->dst()),
             "Destination tensor must be in device memory");

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &src = tensorPool.getTTNNTensorAndValidate(op->src());
  ::ttnn::Tensor &dst = tensorPool.getTTNNTensorAndValidate(op->dst());

  // ttnn::copy requires the input and output layouts to match. Since
  // tt-metal#54212 ported argmax to Metal 2.0, ttnn::argmax can hand back a
  // ROW_MAJOR tensor while the trace output slot it is copied into is TILE,
  // which trips the layout assert in CopyDeviceOperation. Reconcile the layouts
  // the same way updateTensorInPool does. Remove once tt-metal#54987 is fixed.
  if (src.layout() != dst.layout()) {
    ::ttnn::copy(::ttnn::to_layout(src, dst.layout()), dst);
    return;
  }

  ::ttnn::copy(src, dst);
}
} // namespace tt::runtime::ttnn::operations::data_movement
