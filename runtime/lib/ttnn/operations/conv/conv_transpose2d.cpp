// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv_transpose2d.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp"
#include "ttnn/types.hpp"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::ConvTranspose2dOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());
  DEBUG_ASSERT(input.is_allocated());
  DEBUG_ASSERT(weight.is_allocated());

  std::optional<::ttnn::Tensor> bias =
      op->bias() ? std::make_optional(tensorPool.at(op->bias()->global_id()))
                 : std::nullopt;

  std::array<uint32_t, 2> kernelSize, stride, padding, outputPadding, dilation;
  std::copy(op->kernel_size()->begin(), op->kernel_size()->end(),
            kernelSize.begin());
  std::copy(op->stride()->begin(), op->stride()->end(), stride.begin());
  std::copy(op->padding()->begin(), op->padding()->end(), padding.begin());
  std::copy(op->output_padding()->begin(), op->output_padding()->end(),
            outputPadding.begin());
  std::copy(op->dilation()->begin(), op->dilation()->end(), dilation.begin());

  auto config = ::ttnn::operations::conv::Conv2dConfig();
  config.dtype = utils::getDataType(op->input());
  config.weights_dtype = utils::getDataType(op->weight());
  config.shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED;
  ::ttnn::MemoryConfig outMemConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());

  DeviceVariant targetDevice =
      context.getTargetDevice(op->device()->global_id());
  ::ttnn::Tensor out = std::visit(
      [&](auto &&targetDevice) -> ::ttnn::Tensor {
        return std::get<0>(::ttnn::conv_transpose2d(
            ::ttnn::DefaultQueueId, input, weight, &(targetDevice.get()),
            op->in_channels(), op->out_channels(), op->batch_size(),
            op->input_height(), op->input_width(), kernelSize, stride, padding,
            outputPadding, dilation, op->groups(), bias, config));
      },
      targetDevice);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::conv
