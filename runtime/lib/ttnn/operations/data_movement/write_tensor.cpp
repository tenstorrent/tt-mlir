// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/write_tensor.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::WriteTensorOp *op, ProgramContext &context) {
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->host_tensor()),
             "Host tensor must be in system memory");
  LOG_ASSERT(::tt::runtime::ttnn::utils::inDeviceMemory(op->device_tensor()),
             "Device tensor must be in device memory");

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &hostTensor =
      tensorPool.getTTNNTensorAndValidate(op->host_tensor());
  ::ttnn::Tensor &deviceTensor =
      tensorPool.getTTNNTensorAndValidate(op->device_tensor());
  bool blocking = op->blocking();
  ::ttnn::QueueId ttnnCqId = ::ttnn::QueueId(op->cq_id());

  ::tt::tt_metal::write_tensor(hostTensor, deviceTensor, blocking, ttnnCqId);
}
} // namespace tt::runtime::ttnn::operations::data_movement
