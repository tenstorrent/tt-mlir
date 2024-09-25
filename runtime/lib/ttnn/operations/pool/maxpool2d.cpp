// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "maxpool2d.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::pool {

static ::ttnn::Tensor
preshardForMaxPool2d(const ::tt::target::ttnn::MaxPool2dOp *op,
                     ::ttnn::Device &device, ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &input = tensorPool.at(op->in()->global_id());
  const ::ttnn::Shape inputShape = ::ttnn::Shape(::tt::tt_metal::LegacyShape(
      ::tt::runtime::ttnn::utils::toShapeFromFBShape(
          *op->in()->desc()->shape())));
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
      create_sharded_memory_config_from_parallel_config(inputShape,
                                                        parallel_config, 1);
  return ::ttnn::to_memory_config(input, sharded_memory_config, std::nullopt);
}

void run(const ::tt::target::ttnn::MaxPool2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  // TODO (jnie): Update this once we support multi device tensors
  // Investigate how to handle multi device in maxpool2d
  ::ttnn::Device &device =
      context.getDeviceFromView(op->device()->global_id(), 0);
  const ::ttnn::operations::pool::MaxPool2DOp operation =
      ::ttnn::operations::pool::MaxPool2DOp();

  const ::ttnn::Tensor preShardedInput =
      preshardForMaxPool2d(op, device, tensorPool);

  ::ttnn::Tensor out =
      operation.invoke(0, preShardedInput, op->batch_size(), op->input_height(),
                       op->input_width(), op->channels(),
                       {op->kernel_height(), op->kernel_width()},
                       {op->stride_height(), op->stride_width()},
                       {op->padding_height(), op->padding_width()},
                       {op->dilation_height(), op->dilation_width()}, &device);

  auto new_memconfig = out.memory_config();
  new_memconfig.memory_layout = TensorMemoryLayout::INTERLEAVED;
  new_memconfig.buffer_type = BufferType::DRAM;
  out = ::ttnn::to_memory_config(out, new_memconfig, std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
  return;
}
} // namespace tt::runtime::ttnn::operations::pool
