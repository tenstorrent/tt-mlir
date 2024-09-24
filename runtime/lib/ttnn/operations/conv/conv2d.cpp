// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::Conv2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  // TODO (jnie): Update this once we support multi device tensors
  // Investigate how to handle multi device in conv2d
  ::ttnn::Device &device =
      context.getDeviceFromView(op->device()->global_id(), 0);
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());
  std::optional<::ttnn::Tensor> bias =
      op->bias() ? std::make_optional(tensorPool.at(op->bias()->global_id()))
                 : std::nullopt;
  auto config = ::ttnn::operations::conv::conv2d::Conv2dConfig();
  config.dtype = utils::getDataType(op->input());
  config.weights_dtype = utils::getDataType(op->weight());
  ::ttnn::Tensor out =
      std::get<0>(::ttnn::operations::conv::conv2d::conv2d<::ttnn::Device>(
          input, weight, &device, op->in_channels(), op->out_channels(),
          op->batch_size(), op->input_height(), op->input_width(),
          {op->kernel_height(), op->kernel_width()},
          {op->stride_height(), op->stride_width()},
          {op->padding_height(), op->padding_width()},
          {op->dilation_height(), op->dilation_width()}, op->groups(), bias,
          config));
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
