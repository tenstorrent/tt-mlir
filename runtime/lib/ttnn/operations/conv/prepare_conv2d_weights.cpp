// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/prepare_conv2d_weights.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::PrepareConv2dWeightsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &weightTensor =
      tensorPool.getTTNNTensorAndValidate(op->weight_tensor());

  std::optional<::ttnn::MemoryConfig> inputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->input_memory_config());
  LOG_ASSERT(inputMemoryConfig.has_value(),
             "Input memory expected to have a value");

  LOG_ASSERT(op->kernel_size()->size() == 2,
             "Kernel size expected to have 2 elements");
  LOG_ASSERT(op->stride()->size() == 2, "Stride expected to have 2 elements");
  LOG_ASSERT(op->padding()->size() == 2 || op->padding()->size() == 4,
             "Padding expected to have 2 or 4 elements");
  LOG_ASSERT(op->dilation()->size() == 2,
             "Dilation expected to have 2 elements");

  std::array<uint32_t, 2> kernelSize, stride, dilation;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());
  std::copy_n(op->dilation()->begin(), 2, dilation.begin());

  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  if (op->padding()->size() == 2) {
    std::array<uint32_t, 2> symPadding;
    std::copy_n(op->padding()->begin(), 2, symPadding.begin());
    padding = symPadding;
  } else {
    std::array<uint32_t, 4> asymPadding;
    std::copy_n(op->padding()->begin(), 4, asymPadding.begin());
    padding = asymPadding;
  }

  ::ttnn::DataType inputDtype =
      ::tt::runtime::ttnn::utils::toTTNNDataType(op->input_dtype());

  std::optional<::ttnn::DataType> outputDtype;
  if (op->output_dtype()) {
    outputDtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
  }

  std::optional<::ttnn::operations::conv::Conv2dConfig> conv2dConfig;
  if (op->conv2d_config()) {
    conv2dConfig = utils::createConv2dConfig(op->conv2d_config());
  }

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  std::optional<::ttnn::operations::conv::conv2d::Conv2dSliceConfig>
      sliceConfig;
  if (op->conv2d_slice_config()) {
    sliceConfig = utils::createConv2dSliceConfig(op->conv2d_slice_config());
  }

  ::ttnn::Tensor out = ::ttnn::operations::conv::conv2d::prepare_conv_weights(
      weightTensor, *inputMemoryConfig,
      ::tt::runtime::ttnn::utils::toTTNNLayout(op->input_tensor_layout()),
      op->weights_format()->str(), op->in_channels(), op->out_channels(),
      op->batch_size(), op->input_height(), op->input_width(), kernelSize,
      stride, padding, dilation, op->has_bias(), op->groups(), &targetDevice,
      inputDtype, outputDtype, conv2dConfig, std::nullopt, sliceConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
