// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_device.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::FromDeviceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.tensorPool;
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());
  assert(utils::isOnDevice(inputTensor) &&
         "Calling ttnn::from_device on a host tensor");

  ::ttnn::Tensor out = ::ttnn::from_device(inputTensor);

  tensorPool.try_emplace(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
