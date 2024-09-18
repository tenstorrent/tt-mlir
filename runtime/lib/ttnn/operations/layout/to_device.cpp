// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_device.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::ToDeviceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.tensorPool;
  DeviceMap &devicePool = context.devicePool;
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());
  assert((utils::isOnHost(inputTensor) or utils::isOnDevice(inputTensor)) &&
         "Unsupported storage type");

  ::ttnn::MemoryConfig memoryConfig =
      utils::createMemoryConfig(op->memcfg(), op->out());
  ::ttnn::Device &device = utils::getDevice(op->device(), devicePool);
  ::ttnn::Tensor out = ::ttnn::to_device(inputTensor, &device, memoryConfig);

  tensorPool.try_emplace(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
