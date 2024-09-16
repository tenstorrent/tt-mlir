// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "maxpool2d.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::pool {
void run(const ::tt::target::ttnn::MaxPool2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.tensorPool;
  DeviceMap &devicePool = context.devicePool;
  const ::ttnn::Tensor &input = tensorPool.at(op->in()->global_id());
  const ::ttnn::operations::pool::MaxPool2DOp operation =
      ::ttnn::operations::pool::MaxPool2DOp();

  ::ttnn::Device &device = utils::getDevice(op->device(), devicePool);
  ::ttnn::Tensor out = operation.invoke(
      0, input, op->batch_size(), op->input_height(), op->input_width(),
      op->channels(), {op->kernel_height(), op->kernel_width()},
      {op->stride_height(), op->stride_width()},
      {op->padding_height(), op->padding_width()},
      {op->dilation_height(), op->dilation_width()}, &device);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
  return;
}
} // namespace tt::runtime::ttnn::operations::pool
