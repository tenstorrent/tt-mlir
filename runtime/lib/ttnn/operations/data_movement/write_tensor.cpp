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
  ::ttnn::Tensor &deviceTensor =
      tensorPool.getTTNNTensorAndValidate(op->device_tensor());
  ::ttnn::QueueId ttnnCqId = ::ttnn::QueueId(op->cq_id());

  // Note: copy_to_device replaced write_tensor and does not have a blocking
  // parameter. The operation is always blocking.
  if (hostTensor.logical_shape() != deviceTensor.logical_shape()) {
    // Staged trace input slot (same bytes, different page layout): write the
    // bytes without re-deriving the slot's spec from the host tensor.
    const auto hostBuffer = hostTensor.host_tensor().buffer().get_shard(
        ::tt::tt_metal::distributed::MeshCoordinate(0, 0));
    LOG_ASSERT(hostBuffer.has_value(), "Host tensor has no shard at (0, 0)");
    auto bytes = hostBuffer->view_bytes();
    LOG_ASSERT(bytes.size() ==
                   deviceTensor.tensor_spec().compute_packed_buffer_size_bytes(),
               "Host bytes (", bytes.size(),
               ") do not match the device tensor size");
    ::ttnn::copy_to_device(deviceTensor.device()->mesh_command_queue(*ttnnCqId),
                           bytes.data(), deviceTensor, std::nullopt);
    return;
  }
  ::ttnn::copy_to_device(hostTensor, deviceTensor, ttnnCqId);
}
} // namespace tt::runtime::ttnn::operations::data_movement
