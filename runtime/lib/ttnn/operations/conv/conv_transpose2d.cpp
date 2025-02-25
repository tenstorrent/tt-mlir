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

  LOG_ASSERT(op->kernel_size()->size() == 2,
             "Kernel size expected to have 2 elements");
  LOG_ASSERT(op->stride()->size() == 2, "Stride expected to have 2 elements");
  LOG_ASSERT(op->padding()->size() == 2, "Padding expected to have 2 elements");
  LOG_ASSERT(op->output_padding()->size() == 2,
             "Output padding expected to have 2 elements");
  LOG_ASSERT(op->dilation()->size() == 2,
             "Dilation expected to have 2 elements");

  std::array<uint32_t, 2> kernelSize, stride, padding, outputPadding, dilation;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());
  std::copy_n(op->padding()->begin(), 2, padding.begin());
  std::copy_n(op->output_padding()->begin(), 2, outputPadding.begin());
  std::copy_n(op->dilation()->begin(), 2, dilation.begin());

  auto config = ::ttnn::operations::conv::Conv2dConfig();
  config.dtype = utils::getDataType(op->input());
  config.weights_dtype = utils::getDataType(op->weight());
  config.shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED;

  DeviceVariant targetDevice =
      context.getTargetDevice(op->device()->global_id());
  ::ttnn::Tensor out = std::visit(
      [&](auto &&targetDevice) -> ::ttnn::Tensor {
        return std::get<0>(::ttnn::conv_transpose2d(
            input, weight, &targetDevice.get(), op->in_channels(),
            op->out_channels(), op->batch_size(), op->input_height(),
            op->input_width(), kernelSize, stride, padding, outputPadding,
            dilation, op->groups(), bias, config));
      },
      targetDevice);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::conv
