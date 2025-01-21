// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/to_device.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::ToDeviceOp *op, ProgramContext &context) {
  LOG_ASSERT(op->device(), "ToDeviceOp must have a device");
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(inputTensor.is_allocated());
  DEBUG_ASSERT(::tt::runtime::ttnn::utils::isOnHost(inputTensor.storage_type()),
               "Calling ttnn::to_device on a device tensor");
  std::optional<::ttnn::MemoryConfig> memoryConfig = std::nullopt;

  if (op->memcfg()) {
    memoryConfig = utils::createMemoryConfig(op->memcfg(), op->out());
  }
  DeviceVariant targetDevice =
      context.getTargetDevice(op->device()->global_id());
  ::ttnn::Tensor out = std::visit(
      [&](auto &&targetDevice) -> ::ttnn::Tensor {
        return ::ttnn::to_device(inputTensor, &(targetDevice.get()),
                                 memoryConfig);
      },
      targetDevice);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
