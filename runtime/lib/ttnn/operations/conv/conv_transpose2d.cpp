// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv_transpose2d.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp"
#include "ttnn/types.hpp"

namespace tt::runtime::ttnn::operations::conv {
using ::ttnn::operations::conv::conv2d::ResultWithOptions;
void run(const ::tt::target::ttnn::ConvTranspose2dOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias =
      op->bias()
          ? std::make_optional(tensorPool.getTTNNTensorAndValidate(op->bias()))
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

  std::optional<::ttnn::DataType> outputDtype;
  if (op->output_dtype()) {
    outputDtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
  }

  auto conv2dConfig = ::ttnn::operations::conv::Conv2dConfig();
  if (op->conv2d_config()) {
    conv2dConfig = utils::createConv2dConfig(op->conv2d_config());
  } else {
    // TODO (azecevic): Has to be set explicitly to false, otherwise it will
    // assert for flattened Conv2dOp.
    // https://github.com/tenstorrent/tt-metal/issues/30985
    conv2dConfig.enable_kernel_stride_folding = false;
  }

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ResultWithOptions result = ::ttnn::conv_transpose2d(
      input, weight, &targetDevice, op->in_channels(), op->out_channels(),
      op->batch_size(), op->input_height(), op->input_width(), kernelSize,
      stride, padding, outputPadding, dilation, op->groups(), outputDtype, bias,
      conv2dConfig, computeConfig, memoryConfig);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result));

  ::ttnn::Tensor out = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::conv
