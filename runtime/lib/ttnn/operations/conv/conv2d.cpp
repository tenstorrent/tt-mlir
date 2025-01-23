// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv2d.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/types.hpp"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::Conv2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());
  DEBUG_ASSERT(input.is_allocated());
  DEBUG_ASSERT(weight.is_allocated());

  std::optional<::ttnn::Tensor> bias =
      op->bias() ? std::make_optional(tensorPool.at(op->bias()->global_id()))
                 : std::nullopt;

  std::array<uint32_t, 2> kernelSize, stride, padding, dilation;
  std::copy(op->kernel_size()->begin(), op->kernel_size()->end(),
            kernelSize.begin());
  std::copy(op->stride()->begin(), op->stride()->end(), stride.begin());
  std::copy(op->padding()->begin(), op->padding()->end(), padding.begin());
  std::copy(op->dilation()->begin(), op->dilation()->end(), dilation.begin());

  std::optional<::ttnn::operations::conv::Conv2dConfig> conv2dConfig =
      op->conv2d_config()
          ? std::make_optional(
                ::tt::runtime::ttnn::operations::utils::createConv2dConfig(
                    op->conv2d_config()))
          : std::nullopt;

  // Use defaults for now, until compiler drives this.
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig = std::nullopt;

  ::ttnn::MemoryConfig outMemConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());
  DeviceVariant targetDevice =
      context.getTargetDevice(op->device()->global_id());
  ::ttnn::Tensor out = std::visit(
      [&](auto &&targetDevice) -> ::ttnn::Tensor {
        return std::get<0>(::ttnn::operations::conv::conv2d::conv2d(
            input, weight, &(targetDevice.get()), op->in_channels(),
            op->out_channels(), op->batch_size(), op->input_height(),
            op->input_width(), kernelSize, stride, padding, dilation,
            op->groups(), bias, conv2dConfig, computeConfig, outMemConfig));
      },
      targetDevice);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
