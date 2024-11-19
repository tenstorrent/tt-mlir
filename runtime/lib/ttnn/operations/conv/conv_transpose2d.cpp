// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_transpose2d.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp"
#include "ttnn/types.hpp"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::ConvTranspose2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());
  DEBUG_ASSERT(input.is_allocated());
  DEBUG_ASSERT(weight.is_allocated());

  std::optional<::ttnn::Tensor> bias =
      op->bias() ? std::make_optional(tensorPool.at(op->bias()->global_id()))
                 : std::nullopt;
  auto config = ::ttnn::operations::conv::conv2d::Conv2dConfig();
  config.dtype = utils::getDataType(op->input());
  config.weights_dtype = utils::getDataType(op->weight());
  ::ttnn::MemoryConfig outMemConfig = utils::createMemoryConfig(op->out());
  DeviceVariant targetDevice =
      context.getTargetDevice(op->device()->global_id());
  ::ttnn::Tensor out = std::visit(
      [&](auto &&targetDevice) -> ::ttnn::Tensor {
        return std::get<0>(::ttnn::operations::conv::conv_transpose2d::ConvTranpose2dOperation::invoke(
            ::ttnn::DefaultQueueId,
            input,
            weight, 
            &(targetDevice.get()),
            op->in_channels(),
            op->out_channels(),
            op->batch_size(),
            op->input_height(),
            op->input_width(),
            {op->kernel_height(), op->kernel_width()},
            {op->stride_height(), op->stride_width()},
            {op->padding_height(), op->padding_width()},
            {op->output_padding_height(), op->output_padding_width()},
            {op->dilation_height(), op->dilation_width()},
            op->groups(),
            bias,
            config));
      },
      targetDevice);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::conv
