// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Conv/PrepareConvTranspose2dBiasOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/conv/conv_transpose2d/prepare_conv_transpose2d_weights.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

PrepareConvTranspose2dBiasResolvedParams
resolvePrepareConvTranspose2dBiasParams(
    const ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT &opT) {
  LOG_ASSERT(opT.kernel_size.size() == 2,
             "Kernel size expected to have 2 elements");
  LOG_ASSERT(opT.stride.size() == 2, "Stride expected to have 2 elements");
  LOG_ASSERT(opT.padding.size() == 2 || opT.padding.size() == 4,
             "Padding expected to have 2 or 4 elements");
  LOG_ASSERT(opT.dilation.size() == 2, "Dilation expected to have 2 elements");

  PrepareConvTranspose2dBiasResolvedParams params;

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

  if (opT.out) {
    params.outputDtype = operations::utils::getDataType(*opT.out);
  } else if (opT.output_dtype.has_value()) {
    params.outputDtype = operations::utils::toTTNNDataType(*opT.output_dtype);
  }

  params.inputLayout = operations::utils::toTTNNLayout(opT.input_tensor_layout);

  LOG_ASSERT(
      opT.input_memory_config,
      "Input memory config is required for prepare_conv_transpose2d_bias");
  std::optional<::ttnn::MemoryConfig> inputMemoryConfig =
      operations::utils::createMemoryConfigIfNeeded(*opT.input_memory_config);
  LOG_ASSERT(inputMemoryConfig.has_value(),
             "Input memory config expected to have a value");
  params.inputMemoryConfig = *inputMemoryConfig;

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

template <typename Tag>
auto createPrepareConvTranspose2dBiasTuple(
    Tag tag, const ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT &opT,
    TensorArg biasTensor, ::ttnn::MeshDevice *device,
    const PrepareConvTranspose2dBiasResolvedParams &params) {
  return std::make_tuple(
      *std::get<const ::ttnn::Tensor *>(biasTensor), params.inputMemoryConfig,
      params.inputLayout, opT.in_channels, opT.out_channels, opT.batch_size,
      opT.input_height, opT.input_width, params.kernelSize, params.stride,
      params.padding, params.dilation, opT.groups, device,
      params.inputDtype, params.outputDtype, params.conv2dConfig,
      params.computeConfig, params.sliceConfig);
}

PrepareConvTranspose2dBiasOpResult callPrepareConvTranspose2dBias(
    CallType callType,
    const ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT &opT,
    TensorArg biasTensor, ::ttnn::MeshDevice *device) {

  PrepareConvTranspose2dBiasResolvedParams params =
      resolvePrepareConvTranspose2dBiasParams(opT);

  auto makeTuple = [&](auto tag) {
    return createPrepareConvTranspose2dBiasTuple(tag, opT, biasTensor, device,
                                                 params);
  };

  tryCallingOp(::ttnn::operations::conv::conv_transpose2d::
                   prepare_conv_transpose2d_bias,
               true, false, "PrepareConvTranspose2dBiasOp");
}

} // namespace ttnn_op_invoke
