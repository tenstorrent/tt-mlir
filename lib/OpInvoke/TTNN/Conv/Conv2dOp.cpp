// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Conv/Conv2dOp.h"
#include "operations/conv/conv2d/conv2d.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

Conv2dResolvedParams
resolveConv2dParams(const ::tt::target::ttnn::Conv2dOpT &conv2dOp) {
  TT_INVOKE_ASSERT(conv2dOp.kernel_size.size() == 2,
                   "Kernel size expected to have 2 elements");
  TT_INVOKE_ASSERT(conv2dOp.stride.size() == 2,
                   "Stride expected to have 2 elements");
  TT_INVOKE_ASSERT(conv2dOp.padding.size() == 2 || conv2dOp.padding.size() == 4,
                   "Padding expected to have 2 or 4 elements");
  TT_INVOKE_ASSERT(conv2dOp.dilation.size() == 2,
                   "Dilation expected to have 2 elements");

  Conv2dResolvedParams params;

  std::copy_n(conv2dOp.kernel_size.begin(), 2, params.kernelSize.begin());
  std::copy_n(conv2dOp.stride.begin(), 2, params.stride.begin());
  std::copy_n(conv2dOp.dilation.begin(), 2, params.dilation.begin());

  if (conv2dOp.padding.size() == 2) {
    std::array<uint32_t, 2> symPadding;
    std::copy_n(conv2dOp.padding.begin(), 2, symPadding.begin());
    params.padding = symPadding;
  } else {
    std::array<uint32_t, 4> asymPadding;
    std::copy_n(conv2dOp.padding.begin(), 4, asymPadding.begin());
    params.padding = asymPadding;
  }

  if (conv2dOp.out) {
    params.outputDtype = operations::utils::getDataType(*conv2dOp.out);
  } else if (conv2dOp.output_dtype.has_value()) {
    params.outputDtype =
        operations::utils::toTTNNDataType(*conv2dOp.output_dtype);
  }

  if (conv2dOp.conv2d_config) {
    params.conv2dConfig =
        operations::utils::createConv2dConfig(*conv2dOp.conv2d_config);
  }

  if (conv2dOp.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *conv2dOp.compute_config);
  }

  if (conv2dOp.conv2d_slice_config) {
    params.sliceConfig = operations::utils::createConv2dSliceConfig(
        *conv2dOp.conv2d_slice_config);
  }

  if (conv2dOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*conv2dOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*conv2dOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createConv2dTuple(Tag tag, const ::tt::target::ttnn::Conv2dOpT &conv2dOp,
                       TensorArg input, TensorArg weight,
                       std::optional<TensorArg> bias,
                       ::ttnn::MeshDevice *targetDevice,
                       const Conv2dResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), resolveTensorArg(weight, tag), targetDevice,
      conv2dOp.in_channels, conv2dOp.out_channels, conv2dOp.batch_size,
      conv2dOp.input_height, conv2dOp.input_width, params.kernelSize,
      params.stride, params.padding, params.dilation, conv2dOp.groups,
      params.outputDtype,
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      params.conv2dConfig, params.computeConfig, params.outputMemoryConfig,
      params.sliceConfig, false, false);
}

Conv2dOpResult callConv2d(CallType callType,
                          const ::tt::target::ttnn::Conv2dOpT &conv2dOp,
                          TensorArg input, TensorArg weight,
                          std::optional<TensorArg> bias,
                          ::ttnn::MeshDevice *device) {

  Conv2dResolvedParams params = resolveConv2dParams(conv2dOp);

  auto makeTuple = [&](auto tag) {
    return createConv2dTuple(tag, conv2dOp, input, weight, bias, device,
                             params);
  };

  return callOp<Conv2dOpResult>(WRAP_OP(::ttnn::conv2d), callType, makeTuple,
                                device);
}

} // namespace ttnn_op_invoke
