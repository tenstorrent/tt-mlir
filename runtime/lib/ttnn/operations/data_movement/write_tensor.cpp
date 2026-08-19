// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/write_tensor.h"
#include "tt/runtime/detail/common/logger.h"
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
  TTNNTensorWrapper &deviceTensorWrapper =
      tensorPool.getTTNNTensorWrapperAndValidate(op->device_tensor());
  ::ttnn::QueueId ttnnCqId = ::ttnn::QueueId(op->cq_id());

  // Note: copy_to_device replaced write_tensor and does not have a blocking
  // parameter. The operation is always blocking.
  ::ttnn::copy_to_device(hostTensor, deviceTensorWrapper.getTensor(), ttnnCqId);

  // The destination keeps its global id but no longer holds the contents anyone
  // may have cached from it. See ProgramContext::getCachedHostScalar.
  deviceTensorWrapper.updateVersion();
}
} // namespace tt::runtime::ttnn::operations::data_movement
