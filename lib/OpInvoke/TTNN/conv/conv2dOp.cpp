// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/conv/conv2dOp.h"
#include "operations/conv/conv2d/conv2d.hpp"
#include "operations/functions.hpp"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

Conv2dResolvedParams
resolveConv2dParams(const ::tt::target::ttnn::Conv2dOpT &conv2dOpT,
                    CallType callType) {
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

  if (conv2dOpT.out) {
    params.outputDtype = operations::utils::getDataType(*conv2dOpT.out);
  } else if (conv2dOpT.output_dtype.has_value()) {
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
        operations::utils::getTensorRefMemoryConfig(*conv2dOpT.out), callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*conv2dOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createConv2dTuple(Tag tag, const ::tt::target::ttnn::Conv2dOpT &conv2dOpT,
                       TensorArg input, TensorArg weight,
                       std::optional<TensorArg> bias,
                       ::ttnn::MeshDevice &targetDevice,
                       const Conv2dResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), resolveTensorArg(weight, tag),
      &targetDevice, conv2dOpT.in_channels, conv2dOpT.out_channels,
      conv2dOpT.batch_size, conv2dOpT.input_height, conv2dOpT.input_width,
      params.kernelSize, params.stride, params.padding, params.dilation,
      conv2dOpT.groups, params.outputDtype,
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      params.conv2dConfig, params.computeConfig, params.outputMemoryConfig,
      params.sliceConfig, false, false);
}

Conv2dOpResult callConv2d(CallType callType,
                          const ::tt::target::ttnn::Conv2dOpT &conv2dOpT,
                          TensorArg input, TensorArg weight,
                          std::optional<TensorArg> bias,
                          ::ttnn::MeshDevice &targetDevice) {

  Conv2dResolvedParams params = resolveConv2dParams(conv2dOpT, callType);

  auto makeTuple = [&](auto tag) {
    return createConv2dTuple(tag, conv2dOpT, input, weight, bias, targetDevice,
                             params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_CONSTRAINTS(::ttnn::conv2d, &targetDevice,
                                      std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_RUNTIME(::ttnn::conv2d, &targetDevice,
                                  std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::conv2d(std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

} // namespace ttnn_op_invoke
