// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_device.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::ToDeviceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  // TODO (jnie): Update this once we support multi device tensors
  ::ttnn::Device &device =
      context.getDeviceFromView(op->device()->global_id(), 0);
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());
  assert(utils::isOnHost(inputTensor) &&
         "Calling ttnn::to_device on a device tensor");

  ::ttnn::MemoryConfig memoryConfig =
      utils::createMemoryConfig(op->memcfg(), op->out());

  ::ttnn::Tensor out = ::ttnn::to_device(inputTensor, &device, memoryConfig);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
