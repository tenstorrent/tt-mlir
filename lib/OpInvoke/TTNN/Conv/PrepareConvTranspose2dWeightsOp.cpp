// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Conv/PrepareConvTranspose2dWeightsOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/conv/conv_transpose2d/prepare_conv_transpose2d_weights.hpp"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

PrepareConvTranspose2dWeightsResolvedParams
resolvePrepareConvTranspose2dWeightsParams(
    const ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT &op) {
  TT_INVOKE_ASSERT(op.kernel_size.size() == 2,
                   "Kernel size expected to have 2 elements");
  TT_INVOKE_ASSERT(op.stride.size() == 2, "Stride expected to have 2 elements");
  TT_INVOKE_ASSERT(op.padding.size() == 2 || op.padding.size() == 4,
                   "Padding expected to have 2 or 4 elements");
  TT_INVOKE_ASSERT(op.dilation.size() == 2,
                   "Dilation expected to have 2 elements");

  PrepareConvTranspose2dWeightsResolvedParams params;

  std::copy_n(op.kernel_size.begin(), 2, params.kernelSize.begin());
  std::copy_n(op.stride.begin(), 2, params.stride.begin());
  std::copy_n(op.dilation.begin(), 2, params.dilation.begin());

  if (op.padding.size() == 2) {
    std::array<uint32_t, 2> symPadding;
    std::copy_n(op.padding.begin(), 2, symPadding.begin());
    params.padding = symPadding;
  } else {
    std::array<uint32_t, 4> asymPadding;
    std::copy_n(op.padding.begin(), 4, asymPadding.begin());
    params.padding = asymPadding;
  }

  params.inputDtype = operations::utils::toTTNNDataType(op.input_dtype);

  if (op.out) {
    params.outputDtype = operations::utils::getDataType(*op.out);
  } else if (op.output_dtype.has_value()) {
    params.outputDtype = operations::utils::toTTNNDataType(*op.output_dtype);
  }

  params.inputLayout = operations::utils::toTTNNLayout(op.input_tensor_layout);

  TT_INVOKE_ASSERT(
      op.input_memory_config,
      "Input memory config is required for prepare_conv_transpose2d_weights");
  std::optional<::ttnn::MemoryConfig> inputMemoryConfig =
      operations::utils::createMemoryConfigIfNeeded(*op.input_memory_config);
  TT_INVOKE_ASSERT(inputMemoryConfig.has_value(),
                   "Input memory config expected to have a value");
  params.inputMemoryConfig = *inputMemoryConfig;

  if (op.conv2d_config) {
    params.conv2dConfig =
        operations::utils::createConv2dConfig(*op.conv2d_config);
  }

  if (op.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*op.compute_config);
  }

  if (op.conv2d_slice_config) {
    params.sliceConfig =
        operations::utils::createConv2dSliceConfig(*op.conv2d_slice_config);
  }

  return params;
}

template <typename Tag>
auto createPrepareConvTranspose2dWeightsTuple(
    Tag tag, const ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT &op,
    TensorArg weightTensor, ::ttnn::MeshDevice *device,
    const PrepareConvTranspose2dWeightsResolvedParams &params) {
  return std::make_tuple(
      *std::get<const ::ttnn::Tensor *>(weightTensor), params.inputMemoryConfig,
      params.inputLayout, op.weights_format, op.in_channels, op.out_channels,
      op.batch_size, op.input_height, op.input_width, params.kernelSize,
      params.stride, params.padding, params.dilation, op.has_bias, op.groups,
      device, params.inputDtype, params.outputDtype, params.conv2dConfig,
      params.computeConfig, params.sliceConfig, op.mirror_kernel);
}

PrepareConvTranspose2dWeightsOpResult callPrepareConvTranspose2dWeights(
    CallType callType,
    const ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT &op,
    TensorArg weightTensor, ::ttnn::MeshDevice *device) {

  PrepareConvTranspose2dWeightsResolvedParams params =
      resolvePrepareConvTranspose2dWeightsParams(op);

  auto makeTuple = [&](auto tag) {
    return createPrepareConvTranspose2dWeightsTuple(tag, op, weightTensor,
                                                    device, params);
  };

  return callOp<PrepareConvTranspose2dWeightsOpResult, true, false>(
      WRAP_OP(::ttnn::operations::conv::conv_transpose2d::
                  prepare_conv_transpose2d_weights),
      callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
