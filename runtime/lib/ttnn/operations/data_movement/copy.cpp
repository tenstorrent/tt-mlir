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

  ::ttnn::copy(src, dst);

  // The destination keeps its global id but no longer holds the contents anyone
  // may have cached from it. See ProgramContext::getCachedHostScalar.
  tensorPool.getTTNNTensorWrapperAndValidate(op->dst()).updateVersion();
}
} // namespace tt::runtime::ttnn::operations::data_movement
