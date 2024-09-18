// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "maxpool2d.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::pool {

static ::ttnn::Tensor
preshardForMaxPool2d(const ::ttnn::Tensor &input,
                     const ::tt::target::ttnn::MaxPool2dOp *op,
                     ::ttnn::Device &device) {
  uint32_t output_height =
      1 + (op->input_height() + 2 * op->padding_height() -
           op->dilation_height() * (op->kernel_height() - 1) - 1) /
              op->stride_height();
  uint32_t output_width =
      1 + (op->input_width() + 2 * op->padding_width() -
           op->dilation_width() * (op->kernel_width() - 1) - 1) /
              op->stride_width();

  auto parallel_config =
      ::ttnn::operations::conv::conv2d::determine_parallel_config(
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, op->batch_size(),
          op->channels(), output_height, output_width, op->channels(), &device,
          ShardOrientation::ROW_MAJOR);
  auto sharded_memory_config = ::ttnn::operations::conv::conv2d::
      create_sharded_memory_config_from_parallel_config(input.shape(),
                                                        parallel_config, 1);
  return ::ttnn::to_memory_config(input, sharded_memory_config, std::nullopt);
}

void run(const ::tt::target::ttnn::MaxPool2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.tensorPool;
  DeviceMap &devicePool = context.devicePool;
  const ::ttnn::Tensor &input = tensorPool.at(op->in()->global_id());
  const ::ttnn::operations::pool::MaxPool2DOp operation =
      ::ttnn::operations::pool::MaxPool2DOp();

  ::ttnn::Device &device = utils::getDevice(op->device(), devicePool);

  const ::ttnn::Tensor preShardedInput =
      preshardForMaxPool2d(input, op, device);
  ::ttnn::Tensor out =
      operation.invoke(0, preShardedInput, op->batch_size(), op->input_height(),
                       op->input_width(), op->channels(),
                       {op->kernel_height(), op->kernel_width()},
                       {op->stride_height(), op->stride_width()},
                       {op->padding_height(), op->padding_width()},
                       {op->dilation_height(), op->dilation_width()}, &device);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
  return;
}
} // namespace tt::runtime::ttnn::operations::pool
