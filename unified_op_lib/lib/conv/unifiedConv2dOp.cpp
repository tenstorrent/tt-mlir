// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv/unifiedConv2dOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "utils/utils.h"
#include <operations/conv/conv2d/conv2d.hpp>
#include <operations/functions.hpp>
#include <variant>

namespace unifiedOpLib {

Conv2dResolvedParams
resolveConv2dParams(const ::tt::target::ttnn::Conv2dOpT &conv2dOpT) {
  LOG_ASSERT(conv2dOpT.kernel_size.size() == 2,
             "Kernel size expected to have 2 elements");
  LOG_ASSERT(conv2dOpT.stride.size() == 2,
             "Stride expected to have 2 elements");
  LOG_ASSERT(conv2dOpT.padding.size() == 2 || conv2dOpT.padding.size() == 4,
             "Padding expected to have 2 or 4 elements");
  LOG_ASSERT(conv2dOpT.dilation.size() == 2,
             "Dilation expected to have 2 elements");

  Conv2dResolvedParams params;

  std::copy_n(conv2dOpT.kernel_size.begin(), 2, params.kernelSize.begin());
  std::copy_n(conv2dOpT.stride.begin(), 2, params.stride.begin());
  std::copy_n(conv2dOpT.dilation.begin(), 2, params.dilation.begin());

  if (conv2dOpT.padding.size() == 2) {
    std::array<uint32_t, 2> symPadding;
    std::copy_n(conv2dOpT.padding.begin(), 2, symPadding.begin());
    params.padding = symPadding;
  } else {
    std::array<uint32_t, 4> asymPadding;
    std::copy_n(conv2dOpT.padding.begin(), 4, asymPadding.begin());
    params.padding = asymPadding;
  }

  if (conv2dOpT.output_dtype.has_value()) {
    params.outputDtype =
        operations::utils::toTTNNDataType(*conv2dOpT.output_dtype);
  }

  if (conv2dOpT.conv2d_config) {
    params.conv2dConfig =
        operations::utils::createConv2dConfig(*conv2dOpT.conv2d_config);
  }

  if (conv2dOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *conv2dOpT.compute_config);
  }

  if (conv2dOpT.conv2d_slice_config) {
    params.sliceConfig = operations::utils::createConv2dSliceConfig(
        *conv2dOpT.conv2d_slice_config);
  }

  if (conv2dOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*conv2dOpT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*conv2dOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

Conv2dOpResult
callConv2d(CallType callType, const ::tt::target::ttnn::Conv2dOpT &conv2dOpT,
           TensorArg input, TensorArg weight, std::optional<TensorArg> bias,
           ::ttnn::MeshDevice &targetDevice,
           std::optional<::ttnn::MemoryConfig> outputMemoryConfig) {

  Conv2dResolvedParams params = resolveConv2dParams(conv2dOpT);
  if (outputMemoryConfig.has_value()) {
    params.outputMemoryConfig = outputMemoryConfig;
  }

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::conv2d, &targetDevice, std::get<::ttnn::TensorSpec>(input),
        std::get<::ttnn::TensorSpec>(weight), &targetDevice,
        conv2dOpT.in_channels, conv2dOpT.out_channels, conv2dOpT.batch_size,
        conv2dOpT.input_height, conv2dOpT.input_width, params.kernelSize,
        params.stride, params.padding, params.dilation, conv2dOpT.groups,
        params.outputDtype,
        bias ? std::optional<::ttnn::TensorSpec>(
                   std::get<::ttnn::TensorSpec>(*bias))
             : std::nullopt,
        params.conv2dConfig, params.computeConfig, params.outputMemoryConfig,
        params.sliceConfig,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/false);
  case CallType::QUERY_OP_RUNTIME:
    return QUERY_OP_RUNTIME(
        ::ttnn::conv2d, &targetDevice, std::get<::ttnn::TensorSpec>(input),
        std::get<::ttnn::TensorSpec>(weight), &targetDevice,
        conv2dOpT.in_channels, conv2dOpT.out_channels, conv2dOpT.batch_size,
        conv2dOpT.input_height, conv2dOpT.input_width, params.kernelSize,
        params.stride, params.padding, params.dilation, conv2dOpT.groups,
        params.outputDtype,
        bias ? std::optional<::ttnn::TensorSpec>(
                   std::get<::ttnn::TensorSpec>(*bias))
             : std::nullopt,
        params.conv2dConfig, params.computeConfig, params.outputMemoryConfig,
        params.sliceConfig,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/false);
  case CallType::EXECUTE: {
    const auto &in = *std::get<const ::ttnn::Tensor *>(input);
    const auto &wt = *std::get<const ::ttnn::Tensor *>(weight);
    std::optional<::ttnn::Tensor> b;
    if (bias) {
      b = *std::get<const ::ttnn::Tensor *>(*bias);
    }

    return ::ttnn::conv2d(
        in, wt, &targetDevice, conv2dOpT.in_channels, conv2dOpT.out_channels,
        conv2dOpT.batch_size, conv2dOpT.input_height, conv2dOpT.input_width,
        params.kernelSize, params.stride, params.padding, params.dilation,
        conv2dOpT.groups, params.outputDtype, b, params.conv2dConfig,
        params.computeConfig, params.outputMemoryConfig, params.sliceConfig);
  }
  }
}

} // namespace unifiedOpLib
