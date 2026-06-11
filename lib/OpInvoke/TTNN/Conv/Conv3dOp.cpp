// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Conv/Conv3dOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/experimental/conv3d/conv3d.hpp"

#include <optional>

namespace ttnn_op_invoke {

Conv3dResolvedParams
resolveConv3dParams(const ::tt::target::ttnn::Conv3dOpT &conv3dOpT,
                    ::ttnn::MeshDevice *device) {
  TT_INVOKE_ASSERT(conv3dOpT.kernel_size.size() == 3,
                   "Kernel size expected to have 3 elements");
  TT_INVOKE_ASSERT(conv3dOpT.stride.size() == 3,
                   "Stride expected to have 3 elements");
  TT_INVOKE_ASSERT(conv3dOpT.padding.size() == 3,
                   "Padding expected to have 3 elements");

  Conv3dResolvedParams params;
  std::copy_n(conv3dOpT.kernel_size.begin(), 3, params.kernelSize.begin());
  std::copy_n(conv3dOpT.stride.begin(), 3, params.stride.begin());
  std::copy_n(conv3dOpT.padding.begin(), 3, params.padding.begin());

  params.outputDtype = ::ttnn::DataType::BFLOAT16;
  if (conv3dOpT.out) {
    params.outputDtype = operations::utils::getDataType(*conv3dOpT.out);
  } else if (conv3dOpT.output_dtype.has_value()) {
    params.outputDtype =
        operations::utils::toTTNNDataType(*conv3dOpT.output_dtype);
  }

  if (conv3dOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *conv3dOpT.compute_config);
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
          device->compute_with_storage_grid_size();
    }
    params.conv3dConfig = config;
  }

  if (conv3dOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*conv3dOpT.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*conv3dOpT.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createConv3dTuple(Tag tag, const ::tt::target::ttnn::Conv3dOpT &conv3dOpT,
                       TensorArg input, TensorArg weight,
                       std::optional<TensorArg> bias,
                       ::ttnn::MeshDevice *device,
                       Conv3dResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), resolveTensorArg(weight, tag),
      std::optional<::ttnn::MeshDevice *>(device),
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      params.conv3dConfig, params.outputDtype, conv3dOpT.out_channels,
      params.kernelSize, params.stride, params.padding,
      std::array<uint32_t, 3>{1, 1, 1}, conv3dOpT.padding_mode,
      conv3dOpT.groups, params.outputMemoryConfig, params.computeConfig);
}

Conv3dOpResult callConv3d(CallType callType,
                          const ::tt::target::ttnn::Conv3dOpT &conv3dOpT,
                          TensorArg input, TensorArg weight,
                          std::optional<TensorArg> bias,
                          ::ttnn::MeshDevice *device) {
  Conv3dResolvedParams params = resolveConv3dParams(conv3dOpT, device);

  auto makeTuple = [&](auto tag) {
    return createConv3dTuple(tag, conv3dOpT, input, weight, bias, device,
                             params);
  };

  return callOp<Conv3dOpResult>(WRAP_OP(::ttnn::experimental::conv3d), callType,
                                makeTuple, device);
}

} // namespace ttnn_op_invoke
