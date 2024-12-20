// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_transpose2d.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/types.hpp"
#include <ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp>

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

  auto copyArray = [](auto source, auto &destination) {
    std::copy(source->begin(), source->end(), destination.begin());
  };

  std::array<uint32_t, 2> kernelSize, stride, padding, outputPadding, dilation;
  copyArray(op->kernel_size(), kernelSize);
  copyArray(op->stride(), stride);
  copyArray(op->padding(), padding);
  copyArray(op->output_padding(), outputPadding);
  copyArray(op->dilation(), dilation);

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
