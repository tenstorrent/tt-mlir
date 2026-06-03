// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Conv/ConvTranspose2dOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

ConvTranspose2dResolvedParams resolveConvTranspose2dParams(
    const ::tt::target::ttnn::ConvTranspose2dOpT &convTranspose2dOpT) {
  LOG_ASSERT(convTranspose2dOpT.kernel_size.size() == 2,
             "Kernel size expected to have 2 elements");
  LOG_ASSERT(convTranspose2dOpT.stride.size() == 2,
             "Stride expected to have 2 elements");
  LOG_ASSERT(convTranspose2dOpT.padding.size() == 2,
             "Padding expected to have 2 elements");
  LOG_ASSERT(convTranspose2dOpT.output_padding.size() == 2,
             "Output padding expected to have 2 elements");
  LOG_ASSERT(convTranspose2dOpT.dilation.size() == 2,
             "Dilation expected to have 2 elements");

  ConvTranspose2dResolvedParams params;

  std::copy_n(convTranspose2dOpT.kernel_size.begin(), 2,
              params.kernelSize.begin());
  std::copy_n(convTranspose2dOpT.stride.begin(), 2, params.stride.begin());
  std::copy_n(convTranspose2dOpT.padding.begin(), 2, params.padding.begin());
  std::copy_n(convTranspose2dOpT.output_padding.begin(), 2,
              params.outputPadding.begin());
  std::copy_n(convTranspose2dOpT.dilation.begin(), 2, params.dilation.begin());

  if (convTranspose2dOpT.out) {
    params.outputDtype =
        operations::utils::getDataType(*convTranspose2dOpT.out);
  } else if (convTranspose2dOpT.output_dtype.has_value()) {
    params.outputDtype =
        operations::utils::toTTNNDataType(*convTranspose2dOpT.output_dtype);
  }

  if (convTranspose2dOpT.conv2d_config) {
    params.conv2dConfig = operations::utils::createConv2dConfig(
        *convTranspose2dOpT.conv2d_config);
  }

  if (convTranspose2dOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *convTranspose2dOpT.compute_config);
  }

  if (convTranspose2dOpT.conv2d_slice_config) {
    params.sliceConfig = operations::utils::createConv2dSliceConfig(
        *convTranspose2dOpT.conv2d_slice_config);
  }

  if (convTranspose2dOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*convTranspose2dOpT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*convTranspose2dOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createConvTranspose2dTuple(
    Tag tag, const ::tt::target::ttnn::ConvTranspose2dOpT &convTranspose2dOpT,
    TensorArg input, TensorArg weight, std::optional<TensorArg> bias,
    ::ttnn::MeshDevice *device,
    const ConvTranspose2dResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), resolveTensorArg(weight, tag),
      device, convTranspose2dOpT.in_channels,
      convTranspose2dOpT.out_channels, convTranspose2dOpT.batch_size,
      convTranspose2dOpT.input_height, convTranspose2dOpT.input_width,
      params.kernelSize, params.stride, params.padding, params.outputPadding,
      params.dilation, convTranspose2dOpT.groups, params.outputDtype,
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      params.conv2dConfig, params.computeConfig, params.outputMemoryConfig,
      params.sliceConfig, /*mirror_kernel=*/true, /*return_output_dim*/ false,
      /*return_weights_and_bias*/ false);
}

ConvTranspose2dOpResult callConvTranspose2d(
    CallType callType,
    const ::tt::target::ttnn::ConvTranspose2dOpT &convTranspose2dOpT,
    TensorArg input, TensorArg weight, std::optional<TensorArg> bias,
    ::ttnn::MeshDevice *device) {

  ConvTranspose2dResolvedParams params =
      resolveConvTranspose2dParams(convTranspose2dOpT);

  auto makeTuple = [&](auto tag) {
    return createConvTranspose2dTuple(tag, convTranspose2dOpT, input, weight,
                                      bias, device, params);
  };

  callOp(::ttnn::conv_transpose2d);
}

} // namespace ttnn_op_invoke
