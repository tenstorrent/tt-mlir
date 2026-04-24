// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv/unifiedPrepareConv2dWeightsOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "utils/utils.h"
#include <variant>

namespace unifiedOpLib {

PrepareConv2dWeightsResolvedParams resolvePrepareConv2dWeightsParams(
    const ::tt::target::ttnn::PrepareConv2dWeightsOpT &opT, CallType callType) {
  LOG_ASSERT(opT.kernel_size.size() == 2,
             "Kernel size expected to have 2 elements");
  LOG_ASSERT(opT.stride.size() == 2, "Stride expected to have 2 elements");
  LOG_ASSERT(opT.padding.size() == 2 || opT.padding.size() == 4,
             "Padding expected to have 2 or 4 elements");
  LOG_ASSERT(opT.dilation.size() == 2, "Dilation expected to have 2 elements");

  PrepareConv2dWeightsResolvedParams params;

  std::copy_n(opT.kernel_size.begin(), 2, params.kernelSize.begin());
  std::copy_n(opT.stride.begin(), 2, params.stride.begin());
  std::copy_n(opT.dilation.begin(), 2, params.dilation.begin());

  if (opT.padding.size() == 2) {
    std::array<uint32_t, 2> symPadding;
    std::copy_n(opT.padding.begin(), 2, symPadding.begin());
    params.padding = symPadding;
  } else {
    std::array<uint32_t, 4> asymPadding;
    std::copy_n(opT.padding.begin(), 4, asymPadding.begin());
    params.padding = asymPadding;
  }

  params.inputDtype = operations::utils::toTTNNDataType(opT.input_dtype);

  if (opT.output_dtype.has_value()) {
    params.outputDtype = operations::utils::toTTNNDataType(*opT.output_dtype);
  } else if (opT.out) {
    params.outputDtype = operations::utils::getDataType(*opT.out);
  }

  LOG_ASSERT(opT.input_memory_config,
             "Input memory config must be present for PrepareConv2dWeights");
  auto inputMemCfg = operations::utils::createMemoryConfigIfNeeded(
      *opT.input_memory_config, callType);
  LOG_ASSERT(inputMemCfg.has_value(), "Input memory config must have a value");
  params.inputMemoryConfig = *inputMemCfg;

  params.inputLayout = operations::utils::toTTNNLayout(opT.input_tensor_layout);

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

PrepareConv2dWeightsOpResult
callPrepareConv2dWeights(CallType callType,
                         const ::tt::target::ttnn::PrepareConv2dWeightsOpT &opT,
                         TensorArg weight, ::ttnn::MeshDevice &targetDevice) {

  PrepareConv2dWeightsResolvedParams params =
      resolvePrepareConv2dWeightsParams(opT, callType);

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::operations::conv::conv2d::prepare_conv_weights, &targetDevice,
        *std::get<const ::ttnn::Tensor *>(weight), params.inputMemoryConfig,
        params.inputLayout, opT.weights_format, opT.in_channels,
        opT.out_channels, opT.batch_size, opT.input_height, opT.input_width,
        params.kernelSize, params.stride, params.padding, params.dilation,
        opT.has_bias, opT.groups, &targetDevice, params.inputDtype,
        params.outputDtype, params.conv2dConfig, params.computeConfig,
        params.sliceConfig);
  case CallType::QUERY_OP_RUNTIME: {
    ::ttnn::graph::RuntimeQueryResponse response;
    response.error_message =
        "Runtime query not implemented for PrepareConv2dWeightsOp yet";
    return response;
  }
  case CallType::EXECUTE:
    return ::ttnn::operations::conv::conv2d::prepare_conv_weights(
        *std::get<const ::ttnn::Tensor *>(weight), params.inputMemoryConfig,
        params.inputLayout, opT.weights_format, opT.in_channels,
        opT.out_channels, opT.batch_size, opT.input_height, opT.input_width,
        params.kernelSize, params.stride, params.padding, params.dilation,
        opT.has_bias, opT.groups, &targetDevice, params.inputDtype,
        params.outputDtype, params.conv2dConfig, params.computeConfig,
        params.sliceConfig);
  }
}

} // namespace unifiedOpLib
