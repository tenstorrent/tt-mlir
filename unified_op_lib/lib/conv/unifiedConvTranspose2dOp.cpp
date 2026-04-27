// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv/unifiedConvTranspose2dOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "utils/utils.h"
#include <variant>

namespace unifiedOpLib {

ConvTranspose2dResolvedParams
resolveConvTranspose2dParams(const ::tt::target::ttnn::ConvTranspose2dOpT &opT,
                             CallType callType) {
  LOG_ASSERT(opT.kernel_size.size() == 2,
             "Kernel size expected to have 2 elements");
  LOG_ASSERT(opT.stride.size() == 2, "Stride expected to have 2 elements");
  LOG_ASSERT(opT.padding.size() == 2, "Padding expected to have 2 elements");
  LOG_ASSERT(opT.output_padding.size() == 2,
             "Output padding expected to have 2 elements");
  LOG_ASSERT(opT.dilation.size() == 2, "Dilation expected to have 2 elements");

  ConvTranspose2dResolvedParams params;

  std::copy_n(opT.kernel_size.begin(), 2, params.kernelSize.begin());
  std::copy_n(opT.stride.begin(), 2, params.stride.begin());
  std::copy_n(opT.dilation.begin(), 2, params.dilation.begin());
  std::copy_n(opT.padding.begin(), 2, params.padding.begin());
  std::copy_n(opT.output_padding.begin(), 2, params.outputPadding.begin());

  if (opT.output_dtype.has_value()) {
    params.outputDtype = operations::utils::toTTNNDataType(*opT.output_dtype);
  } else if (opT.out) {
    params.outputDtype = operations::utils::getDataType(*opT.out);
  }

  if (opT.memory_config) {
    params.memoryConfig = operations::utils::createMemoryConfigIfNeeded(
        *opT.memory_config, callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*opT.out) ||
                   params.memoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  if (opT.conv2d_config) {
    params.conv2dConfig =
        operations::utils::createConv2dConfig(*opT.conv2d_config);
  }

  if (opT.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*opT.compute_config);
  }

  if (opT.conv2d_slice_config) {
    params.sliceConfig =
        operations::utils::createConv2dSliceConfig(*opT.conv2d_slice_config);
  }

  return params;
}

ConvTranspose2dOpResult callConvTranspose2d(
    CallType callType, const ::tt::target::ttnn::ConvTranspose2dOpT &opT,
    TensorArg input, TensorArg weight, std::optional<TensorArg> bias,
    ::ttnn::MeshDevice &targetDevice) {

  ConvTranspose2dResolvedParams params =
      resolveConvTranspose2dParams(opT, callType);

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::conv_transpose2d, &targetDevice,
        std::get<::ttnn::TensorSpec>(input),
        std::get<::ttnn::TensorSpec>(weight), &targetDevice, opT.in_channels,
        opT.out_channels, opT.batch_size, opT.input_height, opT.input_width,
        params.kernelSize, params.stride, params.padding, params.outputPadding,
        params.dilation, opT.groups, params.outputDtype,
        bias ? std::optional<::ttnn::TensorSpec>(
                   std::get<::ttnn::TensorSpec>(*bias))
             : std::nullopt,
        params.conv2dConfig, params.computeConfig, params.memoryConfig,
        params.sliceConfig,
        /*mirror_kernel=*/true,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/false);
  case CallType::QUERY_OP_RUNTIME:
    return QUERY_OP_RUNTIME(
        ::ttnn::conv_transpose2d, &targetDevice,
        std::get<::ttnn::TensorSpec>(input),
        std::get<::ttnn::TensorSpec>(weight), &targetDevice, opT.in_channels,
        opT.out_channels, opT.batch_size, opT.input_height, opT.input_width,
        params.kernelSize, params.stride, params.padding, params.outputPadding,
        params.dilation, opT.groups, params.outputDtype,
        bias ? std::optional<::ttnn::TensorSpec>(
                   std::get<::ttnn::TensorSpec>(*bias))
             : std::nullopt,
        params.conv2dConfig, params.computeConfig, params.memoryConfig,
        params.sliceConfig,
        /*mirror_kernel=*/true,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/false);
  case CallType::EXECUTE: {
    const auto &in = *std::get<const ::ttnn::Tensor *>(input);
    const auto &wt = *std::get<const ::ttnn::Tensor *>(weight);
    std::optional<::ttnn::Tensor> b;
    if (bias) {
      b = *std::get<const ::ttnn::Tensor *>(*bias);
    }
    return ::ttnn::conv_transpose2d(
        in, wt, &targetDevice, opT.in_channels, opT.out_channels,
        opT.batch_size, opT.input_height, opT.input_width, params.kernelSize,
        params.stride, params.padding, params.outputPadding, params.dilation,
        opT.groups, params.outputDtype, b, params.conv2dConfig,
        params.computeConfig, params.memoryConfig, params.sliceConfig,
        /*mirror_kernel=*/true,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/false);
  }
  }
}

} // namespace unifiedOpLib
