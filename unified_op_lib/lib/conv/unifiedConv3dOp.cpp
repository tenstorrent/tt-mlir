// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv/unifiedConv3dOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttnn/operations/experimental/conv3d/conv3d.hpp"
#include "utils/utils.h"
#include <variant>

namespace unifiedOpLib {

Conv3dResolvedParams
resolveConv3dParams(const ::tt::target::ttnn::Conv3dOpT &conv3dOpT,
                    CallType callType, ::ttnn::MeshDevice &targetDevice) {
  LOG_ASSERT(conv3dOpT.kernel_size.size() == 3,
             "Kernel size expected to have 3 elements");
  LOG_ASSERT(conv3dOpT.stride.size() == 3,
             "Stride expected to have 3 elements");
  LOG_ASSERT(conv3dOpT.padding.size() == 3,
             "Padding expected to have 3 elements");

  Conv3dResolvedParams params;

  std::copy_n(conv3dOpT.kernel_size.begin(), 3, params.kernelSize.begin());
  std::copy_n(conv3dOpT.stride.begin(), 3, params.stride.begin());
  std::copy_n(conv3dOpT.padding.begin(), 3, params.padding.begin());

  params.dilation = std::array<uint32_t, 3>{1, 1, 1};

  params.paddingMode = conv3dOpT.padding_mode;

  params.outputDtype = ::ttnn::DataType::BFLOAT16;
  if (conv3dOpT.output_dtype.has_value()) {
    params.outputDtype =
        operations::utils::toTTNNDataType(*conv3dOpT.output_dtype);
  } else if (conv3dOpT.out) {
    params.outputDtype = operations::utils::getDataType(*conv3dOpT.out);
  }

  if (conv3dOpT.conv3d_config) {
    const auto &fbConfig = *conv3dOpT.conv3d_config;
    ::ttnn::experimental::prim::Conv3dConfig config;
    if (fbConfig.weights_dtype.has_value()) {
      config.weights_dtype =
          operations::utils::toTTNNDataType(*fbConfig.weights_dtype);
    }
    if (fbConfig.t_out_block.has_value()) {
      config.T_out_block = *fbConfig.t_out_block;
    }
    if (fbConfig.w_out_block.has_value()) {
      config.W_out_block = *fbConfig.w_out_block;
    }
    if (fbConfig.h_out_block.has_value()) {
      config.H_out_block = *fbConfig.h_out_block;
    }
    if (fbConfig.c_out_block.has_value()) {
      config.C_out_block = *fbConfig.c_out_block;
    }
    if (fbConfig.c_in_block.has_value()) {
      config.C_in_block = *fbConfig.c_in_block;
    }
    if (conv3dOpT.conv3d_config->compute_with_storage_grid_size) {
      params.conv3dConfig->compute_with_storage_grid_size =
          operations::utils::toTTNNCoreCoord(
              *conv3dOpT.conv3d_config->compute_with_storage_grid_size);
    } else {
      params.conv3dConfig->compute_with_storage_grid_size =
          targetDevice.compute_with_storage_grid_size();
    }
    params.conv3dConfig = config;
  }

  if (conv3dOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *conv3dOpT.compute_config);
  }

  // auto deviceComputeConfig = ::ttnn::init_device_compute_kernel_config(
  //   targetDevice.arch(), computeConfig, ::tt::tt_metal::MathFidelity::HiFi4,
  //   true, true, false);

  if (conv3dOpT.memory_config) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        *conv3dOpT.memory_config, callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*conv3dOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

Conv3dOpResult callConv3d(CallType callType,
                          const ::tt::target::ttnn::Conv3dOpT &conv3dOpT,
                          TensorArg input, TensorArg weight,
                          std::optional<TensorArg> bias,
                          ::ttnn::MeshDevice &targetDevice) {

  Conv3dResolvedParams params =
      resolveConv3dParams(conv3dOpT, callType, targetDevice);

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::experimental::conv3d, &targetDevice,
        std::get<::ttnn::TensorSpec>(input),
        std::get<::ttnn::TensorSpec>(weight),
        std::optional<::ttnn::MeshDevice *>(&targetDevice),
        bias ? std::optional<::ttnn::TensorSpec>(
                   std::get<::ttnn::TensorSpec>(*bias))
             : std::nullopt,
        params.conv3dConfig, params.outputDtype, conv3dOpT.out_channels,
        params.kernelSize, params.stride, params.padding, params.dilation,
        params.paddingMode, conv3dOpT.groups, params.outputMemoryConfig,
        params.computeConfig);
  case CallType::QUERY_OP_RUNTIME:
    return QUERY_OP_RUNTIME(::ttnn::experimental::conv3d, &targetDevice,
                            std::get<::ttnn::TensorSpec>(input),
                            std::get<::ttnn::TensorSpec>(weight),
                            std::optional<::ttnn::MeshDevice *>(&targetDevice),
                            bias ? std::optional<::ttnn::TensorSpec>(
                                       std::get<::ttnn::TensorSpec>(*bias))
                                 : std::nullopt,
                            params.conv3dConfig, params.outputDtype,
                            conv3dOpT.out_channels, params.kernelSize,
                            params.stride, params.padding, params.dilation,
                            params.paddingMode, conv3dOpT.groups,
                            params.outputMemoryConfig, params.computeConfig);
  case CallType::EXECUTE: {
    const auto &in = *std::get<const ::ttnn::Tensor *>(input);
    const auto &wt = *std::get<const ::ttnn::Tensor *>(weight);
    std::optional<::ttnn::Tensor> b;
    if (bias) {
      b = *std::get<const ::ttnn::Tensor *>(*bias);
    }
    return ::ttnn::experimental::conv3d(
        in, wt, &targetDevice, b, params.conv3dConfig, params.outputDtype,
        conv3dOpT.out_channels, params.kernelSize, params.stride,
        params.padding, params.dilation, params.paddingMode, conv3dOpT.groups,
        params.outputMemoryConfig, params.computeConfig);
  }
  }
}

} // namespace unifiedOpLib
